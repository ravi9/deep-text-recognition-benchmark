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
# To export models for OVMS:
python export_to_ov_ir.py --output_dir models-exported-ovms --ovms

# To export models to use with demo_ov.py
python export_to_ov_ir.py
```

The above execution will generate:
```
ONNX model saved at: models-exported/TPS-ResNet-BiLSTM-Attn_fp32.onnx
OpenVINO IR model saved at: models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml
```

## Try with OpenVINO api's to test the model.

```
python demo_ov.py --batch_size 1 --image_folder demo_image
```

```
$ python demo_ov.py --batch_size 1 --image_folder demo_image
Device: cpu
model input parameters 32 100 20 1 512 256 38 25 TPS ResNet BiLSTM Attn
loading pretrained model from models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml
Inference Precision: <Type: 'float32'>
Input Layer: <ConstOutput: names[input.1] shape[1,1,32,100] type: f32>
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_1.png         available                       0.9999
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_2.jpg         shakeshack                      0.9530
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_3.png         london                          0.9840
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_4.png         greenstead                      0.9985
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_5.png         toast                           0.9961
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_6.png         merry                           0.9975
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_7.png         underground                     1.0000
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_8.jpg         ronaldo                         0.8386
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_9.jpg         bally                           0.7493
--------------------------------------------------------------------------------
image_path                      predicted_labels                confidence score
--------------------------------------------------------------------------------
demo_image/demo_10.jpg        university                      0.9998
```

## Benchmark OpenVINO IR using benchmark_app
```bash
benchmark_app -m models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml \
-t 5 -nstreams 1 -hint none
```

```
$ benchmark_app -m models-exported/TPS-ResNet-BiLSTM-Attn_fp32.xml -
t 5 -nstreams 1 -hint none
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
[ INFO ] Read model took 48.75 ms
[ INFO ] Original model I/O parameters:
[ INFO ] Model inputs:
[ INFO ]     input.1 (node: input.1) : f32 / [...] / [1,1,32,100]
[ INFO ] Model outputs:
[ INFO ]     3136 (node: 3136) : f32 / [...] / [1,26,38]
[Step 5/11] Resizing model to match image sizes and given batch
[ INFO ] Model batch size: 1
[Step 6/11] Configuring input of the model
[ INFO ] Model inputs:
[ INFO ]     input.1 (node: input.1) : f32 / [N,C,H,W] / [1,1,32,100]
[ INFO ] Model outputs:
[ INFO ]     3136 (node: 3136) : f32 / [...] / [1,26,38]
[Step 7/11] Loading the model to the device
[ INFO ] Compile model took 669.77 ms
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
[ INFO ] Fill input 'input.1' with random values 
[Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests using 1 streams for CPU, limits: 5000 ms duration)
[ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
[ INFO ] First inference took 67.65 ms
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices:['CPU']
[ INFO ] Count:            287 iterations
[ INFO ] Duration:         5025.64 ms
[ INFO ] Latency:
[ INFO ]    Median:        16.85 ms
[ INFO ]    Average:       17.42 ms
[ INFO ]    Min:           16.10 ms
[ INFO ]    Max:           74.41 ms
[ INFO ] Throughput:   57.11 FPS

```


