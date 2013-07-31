#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__
#include "cuda.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "helpers.h"
using namespace std;

float convolutionWrapper(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename="convolved.png");
float convolutionWrapper_texCache(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename="convolved.png");

__global__ void convolutionKernel_global_register_size2x2_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size3x3_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size4x4_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size5x5_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size6x6_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size7x7_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size3x3_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size4x4_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size5x5_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size6x6_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size7x7_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size4x4_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size5x5_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size6x6_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size7x7_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size5x5_kernel5x5(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size6x6_kernel5x5(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size7x7_kernel5x5(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size6x6_kernel6x6(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size7x7_kernel6x6(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_global_register_size7x7_kernel7x7(const float* in, float* out, const int width, const int height);

__global__ void convolutionKernel_texCache_register_size2x2_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size3x3_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size4x4_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size5x5_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size6x6_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size7x7_kernel2x2(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size3x3_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size4x4_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size5x5_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size6x6_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size7x7_kernel3x3(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size4x4_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size5x5_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size6x6_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size7x7_kernel4x4(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size5x5_kernel5x5(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size6x6_kernel5x5(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size7x7_kernel5x5(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size6x6_kernel6x6(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size7x7_kernel6x6(const float* in, float* out, const int width, const int height);
__global__ void convolutionKernel_texCache_register_size7x7_kernel7x7(const float* in, float* out, const int width, const int height);

__global__ void convolutionKernel_global_only_kernel3x3(const float* in, float* out, const int width, const int height);
#endif
