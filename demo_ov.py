import string
import argparse

import torch
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate

from openvino.runtime import Core, get_version

def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    print('Model input parameters:')
    opt_dict = vars(opt)
    for key, value in opt_dict.items():
        print(f"{key}: {value}")

    print(f"\nOpenVINO Version: {get_version()}")

    # load model
    print(f"Loading pretrained model from {opt.saved_model}")
    model_xml = opt.saved_model

    core = Core()
    model = core.read_model(model=model_xml)
    input_layer = model.input(0)
    model.reshape({input_layer:[opt.batch_size,opt.input_channel,opt.imgH,opt.imgW]}) #{"onnx::Sub_0":[1,1,32,100]}

    compiled_model = core.compile_model(model=model, device_name=opt.device.upper())

    output_layer = compiled_model.output(0)
    print(f"Input Layer: {input_layer}")
    print(f"Output Layer: {output_layer}")

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    for image_tensors, image_path_list in demo_loader:
        batch_size = image_tensors.size(0)
        image = image_tensors.to(opt.device)
        # For max length prediction
        length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(opt.device)
        text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(opt.device)

        if 'CTC' in opt.Prediction:
            preds = compiled_model(image, text_for_pred)[output_layer]
            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            # preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            # preds = compiled_model((image, text_for_pred))[output_layer]
            preds = compiled_model(image)[output_layer]
            # print(f"preds shape: {preds.shape}")
            preds = torch.from_numpy(preds)
            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)


        log = open(f'./log_demo_result.txt', 'a')
        dashed_line = '-' * 80
        head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'

        print(f'{dashed_line}\n{head}\n{dashed_line}')
        log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
            if 'Attn' in opt.Prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            # calculate confidence score (= multiply of pred_max_prob)
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]

            print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
            log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n')

        log.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', default='models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml', required=False, help="path to OV XML to evaluation")
    parser.add_argument('--image_folder', default='demo_image/', required=False, help='path to image_folder which contains text images')
    parser.add_argument('--eval_data', default='evaluation/', required=False, help='path to evaluation dataset')
    parser.add_argument('--device', default='cpu', required=False, help='Target device to run on')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation',default="TPS", type=str, required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', default="ResNet", type=str, required=False, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', default='BiLSTM', type=str, required=False, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', default='Attn', type=str, required=False, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    demo(opt)
