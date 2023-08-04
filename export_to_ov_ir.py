import string
import argparse
from pathlib import Path
from utils import CTCLabelConverter, AttnLabelConverter
from model import Model

import torch
from openvino.runtime import serialize
from openvino.tools import mo

def export_to_ov_ir(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)

    print('Model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    model = torch.nn.DataParallel(model).to(opt.device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=opt.device))
    model = model.module # Unwrap the model from DataParallel using the .module attribute

    # predict
    model.eval()

    # Create dummy inputs
    dummy_input = torch.randn(1, opt.input_channel, opt.imgH, opt.imgW)
    dummy_input_txt = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0)
    print(f"dummy_input shape: {dummy_input.shape}, dummy_input_txt: {dummy_input_txt.shape}")

    # Save FP32 ONNX and IR model
    torch.onnx.export(model, (dummy_input, dummy_input_txt), opt.fp32_onnx_path, opset_version=16)
    model_ir = mo.convert_model(input_model=opt.fp32_onnx_path)
    serialize(model_ir, str(opt.fp32_ir_path))

    # from openvino.tools.mo import convert_model
    # ov_model = convert_model(model, example_input=(dummy_input, dummy_input_txt) )
    # serialize(ov_model, str(opt.fp32_ir_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model', default='models/TPS-ResNet-BiLSTM-Attn.pth', required=False, help="path to saved_model")
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

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {opt.device}")

    output_dir = Path("models-exported")  # Update OUTPUT_DIR to opt.output_dir
    base_model_name = opt.saved_model.split('/')[-1].split('.')[0] # Get the filename only, ex: TPS-ResNet-BiLSTM-Attn
    output_dir.mkdir(exist_ok=True)

    # Paths where ONNX and OpenVINO IR models will be stored.
    opt.fp32_onnx_path = output_dir / Path(base_model_name + "_fp32").with_suffix(".onnx")
    opt.fp32_ir_path = output_dir / Path(base_model_name + "_fp32").with_suffix(".xml")

    export_to_ov_ir(opt)

    print()
    print(f"ONNX model saved at: {opt.fp32_onnx_path}")
    print(f"OpenVINO IR model saved at: {opt.fp32_ir_path}")
