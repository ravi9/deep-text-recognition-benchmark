# Deep-Text-Recoginition-Benchmark w OpenVINO

## Prerequisites

Install Python 3.8 in a virtual environment (e.g. using pyenv or conda). 

```bash
conda create -n deeptext python=3.8 -y
#  After installing, activate conda env
conda activate deeptext
```

Then, install requirements:

```bash
pip install openvino-dev[pytorch,onnx]==2023.0.1
pip install lmdb pillow torchvision nltk natsort
```

## Export to ONNX and then to OV IR
```bash
python export_to_ov_ir.py
```

The above execution will generate:
```
ONNX model saved at: models-exported/TPS-ResNet-BiLSTM-Attn_fp32.onnx
OpenVINO IR model saved at: models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml
```

## Benchmark OpenVINO IR using benchmark_app
```bash
benchmark_app -m models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml \
-shape="1[1,1,32,100],2[1,26]"  \
-ip f32 -t 5 -nstreams 1 -hint none
```

Error:
```
$ benchmark_app -m models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml -
shape="1[1,1,32,100],2[1,26]"  -ip f32 -t 5 -nstreams 1 -hint none
[Step 1/11] Parsing and validating input arguments
[ INFO ] Parsing input parameters
[Step 2/11] Loading OpenVINO Runtime
[ INFO ] OpenVINO:
[ INFO ] Build ................................. 2023.0.1-11005-fa1c41994f3-releases/2023/0
[ INFO ] 
[ INFO ] Device info:
[ INFO ] CPU
[ INFO ] Build ................................. 2023.0.1-11005-fa1c41994f3-releases/2023/0
[ INFO ] 
[ INFO ] 
[Step 3/11] Setting device configuration
[Step 4/11] Reading model files
[ INFO ] Loading model files
[ INFO ] Read model took 47.09 ms
[ INFO ] Original model I/O parameters:
[ INFO ] Model inputs:
[ INFO ]     input.1 (node: input.1) : f32 / [...] / [1,1,32,100]
[ INFO ]     onnx::Gather_1 (node: onnx::Gather_1) : i64 / [...] / [1,26]
[ INFO ] Model outputs:
[ INFO ]     3116 (node: 3116) : f32 / [...] / [1,26,38]
[Step 5/11] Resizing model to match image sizes and given batch
[ INFO ] Model batch size: 1
[Step 6/11] Configuring input of the model
[ INFO ] Model inputs:
[ INFO ]     input.1 (node: input.1) : f32 / [N,C,H,W] / [1,1,32,100]
[ INFO ]     onnx::Gather_1 (node: onnx::Gather_1) : f32 / [...] / [1,26]
[ INFO ] Model outputs:
[ INFO ]     3116 (node: 3116) : f32 / [...] / [1,26,38]
[Step 7/11] Loading the model to the device
[ INFO ] Compile model took 667.03 ms
[Step 8/11] Querying optimal runtime parameters
[ INFO ] Model:
[ INFO ]   NETWORK_NAME: torch_jit
[ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
[ INFO ]   NUM_STREAMS: 1
[ INFO ]   AFFINITY: Affinity.CORE
[ INFO ]   INFERENCE_NUM_THREADS: 48
[ INFO ]   PERF_COUNT: False
[ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
[ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
[ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
[ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
[ INFO ]   ENABLE_CPU_PINNING: True
[ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
[ INFO ]   ENABLE_HYPER_THREADING: False
[ INFO ]   EXECUTION_DEVICES: ['CPU']
[Step 9/11] Creating infer requests and preparing input tensors
[ WARNING ] No input files were given for input 'input.1'!. This input will be filled with random values!
[ WARNING ] No input files were given for input 'onnx::Gather_1'!. This input will be filled with random values!
[ INFO ] Fill input 'input.1' with random values 
[ INFO ] Fill input 'onnx::Gather_1' with random values 
[Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests using 1 streams for CPU, limits: 5000 ms duration)
[ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
[ ERROR ] Check 'false' failed at src/inference/src/infer_request.cpp:239:
Check 'false' failed at src/inference/src/dev/converter_utils.cpp:652:
Check 'false' failed at src/bindings/python/src/pyopenvino/core/async_infer_queue.cpp:107:
ScatterElementsUpdate node with name '/Prediction/ScatterElements_25' have indices value that points to non-existing output tensor element


Traceback (most recent call last):
  File "/home/rpanchum/miniconda3/envs/ebay-enc/lib/python3.8/site-packages/openvino/tools/benchmark/main.py", line 558, in main
    duration_ms = f"{benchmark.first_infer(requests):.2f}"
  File "/home/rpanchum/miniconda3/envs/ebay-enc/lib/python3.8/site-packages/openvino/tools/benchmark/benchmark.py", line 86, in first_infer
    requests.wait_all()
RuntimeError: Check 'false' failed at src/inference/src/infer_request.cpp:239:
Check 'false' failed at src/inference/src/dev/converter_utils.cpp:652:
Check 'false' failed at src/bindings/python/src/pyopenvino/core/async_infer_queue.cpp:107:
ScatterElementsUpdate node with name '/Prediction/ScatterElements_25' have indices value that points to non-existing output tensor element

```

## Try with OpenVINO api's to test the model.

```
python demo_ov.py --batch_size 1 --image_folder demo_image
```

Output observed, but accuracy is bad.

```
$ python demo_ov.py --batch_size 1 --image_folder demo_image
Device: cpu
model input parameters 32 100 20 1 512 256 38 25 TPS ResNet BiLSTM Attn
loading pretrained model from models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml
Inference Precision: <Type: 'float32'>
Input Layer: <ConstOutput: names[input.1] shape[1,1,32,100] type: f32>
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_1.png         avaiilable                      0.4550
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_2.jpg         shaaasshacc                     0.1499
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_3.png         london                          0.8993
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_4.png         greensttaad                     0.3400
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_5.png         toas                            0.9938
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_6.png         meeerry                         0.2124
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_7.png         undergroundd                    0.3090
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_8.jpg         rroaaldd                        0.1309
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_9.jpg         bally                           0.1565
preds shape: (1, 26, 38)
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image_1/demo_10.jpg        univversity                     0.7729
```

