## Definitions
Model profiling requires each op's input and output tensor shapes, which means users need to perform
a shape_infer before calling profile function of onnx_tool.Graph.

Here are profiling items:
1. Forward MACs(or FLOPs) and its percentage
2. Backward MACs(or FLOPs) and its percentage(experimental)
3. Memory usage (in bytes) and its percent
4. Parameter number(static tensor size) and its percent
5. Input tensor shape(index=0) and output tensor shape(index=0)
6. Sparse pattern, sparse block ratio and sparse ratio.

### MACs
1. 1 MAC = float(a) * float(b) + float(c)  
This formula is implemented as one instruction on most hardware platform.  
  
2. Treat other instructions as 1 MAC, like cmp, add, and mov. Although these instructions are much lighter than float
computation instruction.
  
3. Special function's MACs are not accurate and quite different between different platforms. For example, there is no 
instruction for exp on x86 device, you need to implement this by a sequence of instructions. The total instruction 
number depends on how accurate result you want to get. You can treat this as the affect of fast-math.

4. onnx_tool does not consider computation optimizations. The zero padding area will not be ignored from MACs statistics. 
Also, onnx_tool won't assume that some ops will be fused, like the batch normalization op after the convolution op.

### Memory
Layer memory usage = Weight tensor memory + Output tensor memory   
The same weight tensor will be only count once.

### Parameters
Parameter number stands for the model space size. The total size of weight tensor.  
The same weight tensor will be count once like memory.

### Sparsity
1. Sparse block pattern 1x4 means that tensor values are grouped into 1x4 blocks, the block is  
a zero block if all 4 values are zeros.
2. Sparse ratio represents the zero value element ratio, e.g. a tensor is [1,0,0,0], the sparse ratio is 75%
3. Sparse block ratio is the biggest sparse block shape which meets the limited sparse ratio loss. e.g. a tensor  
is [1,0,0,0], it can be sparse ratio=75% with block=1x1, also can be sparse ratio=50% with block=1x2.

## How to use

* python usage  
please refer to profile_model, dynamic_input_shapes and custom_layer_register in [benchmark/samples.py](https://github.com/ThanatosShinji/onnx-tool/blob/main/benchmark/samples.py)
* cli usage  

    ```shell
    python -m onnx_tool -h # print usage
    python -m onnx_tool -i 'resnet50-v1-12.onnx' -o 'resnet50_shapes.onnx'
    ```    
    ```shell
    python -m onnx_tool -i 'resnet50-v1-12.onnx' --names resnetv17_stage1_conv3_fwd resnetv17_stage1_conv2_fwd  -o 'resnet50_shapes.onnx'
  #add hidden tensors to graph's output tensors
    ```    
    ```shell
    python -m onnx_tool -i rvm_mobilenetv3_fp32.onnx --mode profile --dynamic_inputs \
    src:f32:1x3x1080x1920 r1i:f32:1x16x135x240 r2i:f32:1x20x68x120 r3i:f32:1x40x34x60 r4i:f32:1x64x17x30 downsample_ratio:f32:-1:0.25
    #dynamic_inputs string format:  <tensor name>:<data type>:<shape>[:<data>]
    ```   
