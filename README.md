<h5>This fast convolution implementation accompanies the following paper</h5>

```
Forrest N. Iandola, David Sheffield, Michael Anderson, Phitchaya Mangpo Phothilimthana, and Kurt Keutzer. 
"Communication-Minimizing 2D Convolution in GPU Registers."
To appear in IEEE International Conference on Image Processing (ICIP), 2013. 
```
[Paper (PDF)](http://www.forrestiandola.com/PREPRINT_convolution_2d_gpu_registers.pdf)

<h3>Recommended System Configuration</h3>
- Ubuntu 12.04
- NVIDIA Fermi or Kepler GPU (compute capability 2.0 or higher)
- CUDA 5.0
- Install **OpenCV** for file I/O


<h3>Getting started</h3>
``` C++
make
./main
```

<h3>Code structure</h3>

<h5>The crux our convolution implementation `convolution.cu)</h5>
``` C++
//memoryLayout = {global, global_register, texCache, or texCache_register}
//filter dims and size to load = {2x2, 3x3, ...}
__global__ convolutionKernel_memoryLayout_filter3x3_size4x4(...) //one of many implementations that we generate
    load 4x4 window to registers
    
    compute 4 convolution windows, 3x3 each.
    
    save results to global memory
```

<h5>Autotuning</h5>
`doCodegen.sh` produces code for many convolution filter sizes and many levels of unrolling. 
Feel free to modify `doCodegen.sh `and `codeGen_*.sh` to explore alternative filter sizes, unrolling levels, and memory layouts. 
Dive into this at your own risk though -- some 1337ness with bash may be required.

<h3>FAQ</h3>
<h5>Can I use this code as a library in my own system?</h5>
This code is designed as a research prototype. 
Feel free to try to use it as a library, but our main goal is to illustrate the performance of our communication-minimizing convolution technique.

<h5>How do I set the convolution filter?</h5>
Our autotuner uses a hard-coded blur kernel. To plug in your own filter with no computational performance penalty, use constant memory:

``` C++
//adapted from NVIDIA_CUDA-5.0_Samples/3_Imaging/convolutionSeparable/convolutionSeparable.cu
#define KERNEL_LENGTH 9
__constant__ float c_Kernel[KERNEL_LENGTH]; //constant memory on GPU (global variable in C++)

extern "C" void initialize_constant_memory() //call this before launching a convolution on your GPU
{
    float h_Kernel[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1}; //example filter (you can set this dynamically)
    CHECK_CUDART(cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float)));
}

//code from our autotuner, with the addition of a user-specified kernel:
__device__ void convolutionDevice_size4x4_kernel3x3(float in[4][4], float *out, int startX, int startY)
{
    float tmp=
        in[startY+0][startX+0] * c_Kernel[0] + //c_Kernel[] replaces hard-coded 'filter' parameter in provided code.
        in[startY+0][startX+1] * c_Kernel[1] +
        in[startY+0][startX+2] * c_Kernel[2] +

        in[startY+1][startX+0] * c_Kernel[3] +
        in[startY+1][startX+1] * c_Kernel[4] +
        in[startY+1][startX+2] * c_Kernel[5] +

        in[startY+2][startX+0] * c_Kernel[6] +
        in[startY+2][startX+1] * c_Kernel[7] +
        in[startY+2][startX+2] * c_Kernel[8]
    *out = tmp;
}
```
<h5>What data formats are supported?</h5>
Our code supports 1-channel 32-bit floating-point images. 
We find that this format is very common for feature extraction algorithms like HOG and SIFT. 
You may be able to implement our techniques for other formats, such as 8-bit multi-channel images.

