#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "convolution.h"
//#include "helpers.h"
using namespace std;
//using namespace cv;

texture<float, 2, cudaReadModeElementType> tex;

static inline __device__ int clamp(int x, int l, int r)
{
    //max and min are gpu intrinsics
    return max(l, min(x,r));
}

static inline __device__ int clamp_addr(int x, int y, int mx, int my)
{
    int xaddr = clamp(x, 0, (mx-1));
    int yaddr = clamp(y, 0, (my-1));
    return (yaddr*mx) + xaddr;
}
//Load from DRAM to register
__device__ void load_global_register_size2x2(const float* in, float registers[2][2], int startX, int startY, int width, int height)
{
    for(int i=0; i<2; i++)
    {
        registers[i][0] = in[clamp_addr(startX+0, startY+i, width, height)];
        registers[i][1] = in[clamp_addr(startX+1, startY+i, width, height)];
    }
}
__device__ void load_global_register_size3x3(const float* in, float registers[3][3], int startX, int startY, int width, int height)
{
    for(int i=0; i<3; i++)
    {
        registers[i][0] = in[clamp_addr(startX+0, startY+i, width, height)];
        registers[i][1] = in[clamp_addr(startX+1, startY+i, width, height)];
        registers[i][2] = in[clamp_addr(startX+2, startY+i, width, height)];
    }
}
__device__ void load_global_register_size4x4(const float* in, float registers[4][4], int startX, int startY, int width, int height)
{
    for(int i=0; i<4; i++)
    {
        registers[i][0] = in[clamp_addr(startX+0, startY+i, width, height)];
        registers[i][1] = in[clamp_addr(startX+1, startY+i, width, height)];
        registers[i][2] = in[clamp_addr(startX+2, startY+i, width, height)];
        registers[i][3] = in[clamp_addr(startX+3, startY+i, width, height)];
    }
}
__device__ void load_global_register_size5x5(const float* in, float registers[5][5], int startX, int startY, int width, int height)
{
    for(int i=0; i<5; i++)
    {
        registers[i][0] = in[clamp_addr(startX+0, startY+i, width, height)];
        registers[i][1] = in[clamp_addr(startX+1, startY+i, width, height)];
        registers[i][2] = in[clamp_addr(startX+2, startY+i, width, height)];
        registers[i][3] = in[clamp_addr(startX+3, startY+i, width, height)];
        registers[i][4] = in[clamp_addr(startX+4, startY+i, width, height)];
    }
}
__device__ void load_global_register_size6x6(const float* in, float registers[6][6], int startX, int startY, int width, int height)
{
    for(int i=0; i<6; i++)
    {
        registers[i][0] = in[clamp_addr(startX+0, startY+i, width, height)];
        registers[i][1] = in[clamp_addr(startX+1, startY+i, width, height)];
        registers[i][2] = in[clamp_addr(startX+2, startY+i, width, height)];
        registers[i][3] = in[clamp_addr(startX+3, startY+i, width, height)];
        registers[i][4] = in[clamp_addr(startX+4, startY+i, width, height)];
        registers[i][5] = in[clamp_addr(startX+5, startY+i, width, height)];
    }
}
__device__ void load_global_register_size7x7(const float* in, float registers[7][7], int startX, int startY, int width, int height)
{
    for(int i=0; i<7; i++)
    {
        registers[i][0] = in[clamp_addr(startX+0, startY+i, width, height)];
        registers[i][1] = in[clamp_addr(startX+1, startY+i, width, height)];
        registers[i][2] = in[clamp_addr(startX+2, startY+i, width, height)];
        registers[i][3] = in[clamp_addr(startX+3, startY+i, width, height)];
        registers[i][4] = in[clamp_addr(startX+4, startY+i, width, height)];
        registers[i][5] = in[clamp_addr(startX+5, startY+i, width, height)];
        registers[i][6] = in[clamp_addr(startX+6, startY+i, width, height)];
    }
}
__device__ void load_texCache_register_size2x2(const float* in, float registers[2][2], int startX, int startY, int width, int height)
{
    for(int i=0; i<2; i++)
    {
        registers[i][0] =  tex2D(tex, float(startX+0)+0.5, float(startY+i)+0.5);
        registers[i][1] =  tex2D(tex, float(startX+1)+0.5, float(startY+i)+0.5);
    }
}
__device__ void load_texCache_register_size3x3(const float* in, float registers[3][3], int startX, int startY, int width, int height)
{
    for(int i=0; i<3; i++)
    {
        registers[i][0] =  tex2D(tex, float(startX+0)+0.5, float(startY+i)+0.5);
        registers[i][1] =  tex2D(tex, float(startX+1)+0.5, float(startY+i)+0.5);
        registers[i][2] =  tex2D(tex, float(startX+2)+0.5, float(startY+i)+0.5);
    }
}
__device__ void load_texCache_register_size4x4(const float* in, float registers[4][4], int startX, int startY, int width, int height)
{
    for(int i=0; i<4; i++)
    {
        registers[i][0] =  tex2D(tex, float(startX+0)+0.5, float(startY+i)+0.5);
        registers[i][1] =  tex2D(tex, float(startX+1)+0.5, float(startY+i)+0.5);
        registers[i][2] =  tex2D(tex, float(startX+2)+0.5, float(startY+i)+0.5);
        registers[i][3] =  tex2D(tex, float(startX+3)+0.5, float(startY+i)+0.5);
    }
}
__device__ void load_texCache_register_size5x5(const float* in, float registers[5][5], int startX, int startY, int width, int height)
{
    for(int i=0; i<5; i++)
    {
        registers[i][0] =  tex2D(tex, float(startX+0)+0.5, float(startY+i)+0.5);
        registers[i][1] =  tex2D(tex, float(startX+1)+0.5, float(startY+i)+0.5);
        registers[i][2] =  tex2D(tex, float(startX+2)+0.5, float(startY+i)+0.5);
        registers[i][3] =  tex2D(tex, float(startX+3)+0.5, float(startY+i)+0.5);
        registers[i][4] =  tex2D(tex, float(startX+4)+0.5, float(startY+i)+0.5);
    }
}
__device__ void load_texCache_register_size6x6(const float* in, float registers[6][6], int startX, int startY, int width, int height)
{
    for(int i=0; i<6; i++)
    {
        registers[i][0] =  tex2D(tex, float(startX+0)+0.5, float(startY+i)+0.5);
        registers[i][1] =  tex2D(tex, float(startX+1)+0.5, float(startY+i)+0.5);
        registers[i][2] =  tex2D(tex, float(startX+2)+0.5, float(startY+i)+0.5);
        registers[i][3] =  tex2D(tex, float(startX+3)+0.5, float(startY+i)+0.5);
        registers[i][4] =  tex2D(tex, float(startX+4)+0.5, float(startY+i)+0.5);
        registers[i][5] =  tex2D(tex, float(startX+5)+0.5, float(startY+i)+0.5);
    }
}
__device__ void load_texCache_register_size7x7(const float* in, float registers[7][7], int startX, int startY, int width, int height)
{
    for(int i=0; i<7; i++)
    {
        registers[i][0] =  tex2D(tex, float(startX+0)+0.5, float(startY+i)+0.5);
        registers[i][1] =  tex2D(tex, float(startX+1)+0.5, float(startY+i)+0.5);
        registers[i][2] =  tex2D(tex, float(startX+2)+0.5, float(startY+i)+0.5);
        registers[i][3] =  tex2D(tex, float(startX+3)+0.5, float(startY+i)+0.5);
        registers[i][4] =  tex2D(tex, float(startX+4)+0.5, float(startY+i)+0.5);
        registers[i][5] =  tex2D(tex, float(startX+5)+0.5, float(startY+i)+0.5);
        registers[i][6] =  tex2D(tex, float(startX+6)+0.5, float(startY+i)+0.5);
    }
}
//Store from register to DRAM

__device__ void store_register_global_size1x1(float* out, float registers[1][1], int startX, int startY, int width, int height)
{
    for(int i=0; i<1; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
    }
}

__device__ void store_register_global_size2x2(float* out, float registers[2][2], int startX, int startY, int width, int height)
{
    for(int i=0; i<2; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
        out[(startY+i)*width + startX+1] = registers[i][1];
    }
}

__device__ void store_register_global_size3x3(float* out, float registers[3][3], int startX, int startY, int width, int height)
{
    for(int i=0; i<3; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
        out[(startY+i)*width + startX+1] = registers[i][1];
        out[(startY+i)*width + startX+2] = registers[i][2];
    }
}

__device__ void store_register_global_size4x4(float* out, float registers[4][4], int startX, int startY, int width, int height)
{
    for(int i=0; i<4; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
        out[(startY+i)*width + startX+1] = registers[i][1];
        out[(startY+i)*width + startX+2] = registers[i][2];
        out[(startY+i)*width + startX+3] = registers[i][3];
    }
}

__device__ void store_register_global_size5x5(float* out, float registers[5][5], int startX, int startY, int width, int height)
{
    for(int i=0; i<5; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
        out[(startY+i)*width + startX+1] = registers[i][1];
        out[(startY+i)*width + startX+2] = registers[i][2];
        out[(startY+i)*width + startX+3] = registers[i][3];
        out[(startY+i)*width + startX+4] = registers[i][4];
    }
}

__device__ void store_register_global_size6x6(float* out, float registers[6][6], int startX, int startY, int width, int height)
{
    for(int i=0; i<6; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
        out[(startY+i)*width + startX+1] = registers[i][1];
        out[(startY+i)*width + startX+2] = registers[i][2];
        out[(startY+i)*width + startX+3] = registers[i][3];
        out[(startY+i)*width + startX+4] = registers[i][4];
        out[(startY+i)*width + startX+5] = registers[i][5];
    }
}

__device__ void store_register_global_size7x7(float* out, float registers[7][7], int startX, int startY, int width, int height)
{
    for(int i=0; i<7; i++)
    {
        out[(startY+i)*width + startX+0] = registers[i][0];
        out[(startY+i)*width + startX+1] = registers[i][1];
        out[(startY+i)*width + startX+2] = registers[i][2];
        out[(startY+i)*width + startX+3] = registers[i][3];
        out[(startY+i)*width + startX+4] = registers[i][4];
        out[(startY+i)*width + startX+5] = registers[i][5];
        out[(startY+i)*width + startX+6] = registers[i][6];
    }
}
//Convolution -- hard-coded 2D arrays

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size2x2_kernel2x2(float in[2][2], float *out, int startX, int startY)
{
    const float filter = .2500; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size3x3_kernel2x2(float in[3][3], float *out, int startX, int startY)
{
    const float filter = .2500; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size3x3_kernel3x3(float in[3][3], float *out, int startX, int startY)
{
    const float filter = 0.1111; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size4x4_kernel2x2(float in[4][4], float *out, int startX, int startY)
{
    const float filter = .2500; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size4x4_kernel3x3(float in[4][4], float *out, int startX, int startY)
{
    const float filter = 0.1111; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size4x4_kernel4x4(float in[4][4], float *out, int startX, int startY)
{
    const float filter = 0.0625; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size5x5_kernel2x2(float in[5][5], float *out, int startX, int startY)
{
    const float filter = .2500; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size5x5_kernel3x3(float in[5][5], float *out, int startX, int startY)
{
    const float filter = 0.1111; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size5x5_kernel4x4(float in[5][5], float *out, int startX, int startY)
{
    const float filter = 0.0625; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size5x5_kernel5x5(float in[5][5], float *out, int startX, int startY)
{
    const float filter = 0.0400; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +
        in[startY+0][startX+4] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +
        in[startY+1][startX+4] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +
        in[startY+2][startX+4] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +
        in[startY+3][startX+4] * filter +

        in[startY+4][startX+0] * filter +
        in[startY+4][startX+1] * filter +
        in[startY+4][startX+2] * filter +
        in[startY+4][startX+3] * filter +
        in[startY+4][startX+4] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size6x6_kernel2x2(float in[6][6], float *out, int startX, int startY)
{
    const float filter = .2500; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size6x6_kernel3x3(float in[6][6], float *out, int startX, int startY)
{
    const float filter = 0.1111; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size6x6_kernel4x4(float in[6][6], float *out, int startX, int startY)
{
    const float filter = 0.0625; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size6x6_kernel5x5(float in[6][6], float *out, int startX, int startY)
{
    const float filter = 0.0400; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +
        in[startY+0][startX+4] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +
        in[startY+1][startX+4] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +
        in[startY+2][startX+4] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +
        in[startY+3][startX+4] * filter +

        in[startY+4][startX+0] * filter +
        in[startY+4][startX+1] * filter +
        in[startY+4][startX+2] * filter +
        in[startY+4][startX+3] * filter +
        in[startY+4][startX+4] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size6x6_kernel6x6(float in[6][6], float *out, int startX, int startY)
{
    const float filter = 0.0278; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +
        in[startY+0][startX+4] * filter +
        in[startY+0][startX+5] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +
        in[startY+1][startX+4] * filter +
        in[startY+1][startX+5] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +
        in[startY+2][startX+4] * filter +
        in[startY+2][startX+5] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +
        in[startY+3][startX+4] * filter +
        in[startY+3][startX+5] * filter +

        in[startY+4][startX+0] * filter +
        in[startY+4][startX+1] * filter +
        in[startY+4][startX+2] * filter +
        in[startY+4][startX+3] * filter +
        in[startY+4][startX+4] * filter +
        in[startY+4][startX+5] * filter +

        in[startY+5][startX+0] * filter +
        in[startY+5][startX+1] * filter +
        in[startY+5][startX+2] * filter +
        in[startY+5][startX+3] * filter +
        in[startY+5][startX+4] * filter +
        in[startY+5][startX+5] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size7x7_kernel2x2(float in[7][7], float *out, int startX, int startY)
{
    const float filter = .2500; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size7x7_kernel3x3(float in[7][7], float *out, int startX, int startY)
{
    const float filter = 0.1111; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size7x7_kernel4x4(float in[7][7], float *out, int startX, int startY)
{
    const float filter = 0.0625; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size7x7_kernel5x5(float in[7][7], float *out, int startX, int startY)
{
    const float filter = 0.0400; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +
        in[startY+0][startX+4] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +
        in[startY+1][startX+4] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +
        in[startY+2][startX+4] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +
        in[startY+3][startX+4] * filter +

        in[startY+4][startX+0] * filter +
        in[startY+4][startX+1] * filter +
        in[startY+4][startX+2] * filter +
        in[startY+4][startX+3] * filter +
        in[startY+4][startX+4] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size7x7_kernel6x6(float in[7][7], float *out, int startX, int startY)
{
    const float filter = 0.0278; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +
        in[startY+0][startX+4] * filter +
        in[startY+0][startX+5] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +
        in[startY+1][startX+4] * filter +
        in[startY+1][startX+5] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +
        in[startY+2][startX+4] * filter +
        in[startY+2][startX+5] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +
        in[startY+3][startX+4] * filter +
        in[startY+3][startX+5] * filter +

        in[startY+4][startX+0] * filter +
        in[startY+4][startX+1] * filter +
        in[startY+4][startX+2] * filter +
        in[startY+4][startX+3] * filter +
        in[startY+4][startX+4] * filter +
        in[startY+4][startX+5] * filter +

        in[startY+5][startX+0] * filter +
        in[startY+5][startX+1] * filter +
        in[startY+5][startX+2] * filter +
        in[startY+5][startX+3] * filter +
        in[startY+5][startX+4] * filter +
        in[startY+5][startX+5] * filter +

        0;
    *out = tmp;
}

//@param height,width = dims of 'in' array
__device__ void convolutionDevice_size7x7_kernel7x7(float in[7][7], float *out, int startX, int startY)
{
    const float filter = 0.0204; // 1/(kernelSize 2D)
    float tmp=
        in[startY+0][startX+0] * filter +
        in[startY+0][startX+1] * filter +
        in[startY+0][startX+2] * filter +
        in[startY+0][startX+3] * filter +
        in[startY+0][startX+4] * filter +
        in[startY+0][startX+5] * filter +
        in[startY+0][startX+6] * filter +

        in[startY+1][startX+0] * filter +
        in[startY+1][startX+1] * filter +
        in[startY+1][startX+2] * filter +
        in[startY+1][startX+3] * filter +
        in[startY+1][startX+4] * filter +
        in[startY+1][startX+5] * filter +
        in[startY+1][startX+6] * filter +

        in[startY+2][startX+0] * filter +
        in[startY+2][startX+1] * filter +
        in[startY+2][startX+2] * filter +
        in[startY+2][startX+3] * filter +
        in[startY+2][startX+4] * filter +
        in[startY+2][startX+5] * filter +
        in[startY+2][startX+6] * filter +

        in[startY+3][startX+0] * filter +
        in[startY+3][startX+1] * filter +
        in[startY+3][startX+2] * filter +
        in[startY+3][startX+3] * filter +
        in[startY+3][startX+4] * filter +
        in[startY+3][startX+5] * filter +
        in[startY+3][startX+6] * filter +

        in[startY+4][startX+0] * filter +
        in[startY+4][startX+1] * filter +
        in[startY+4][startX+2] * filter +
        in[startY+4][startX+3] * filter +
        in[startY+4][startX+4] * filter +
        in[startY+4][startX+5] * filter +
        in[startY+4][startX+6] * filter +

        in[startY+5][startX+0] * filter +
        in[startY+5][startX+1] * filter +
        in[startY+5][startX+2] * filter +
        in[startY+5][startX+3] * filter +
        in[startY+5][startX+4] * filter +
        in[startY+5][startX+5] * filter +
        in[startY+5][startX+6] * filter +

        in[startY+6][startX+0] * filter +
        in[startY+6][startX+1] * filter +
        in[startY+6][startX+2] * filter +
        in[startY+6][startX+3] * filter +
        in[startY+6][startX+4] * filter +
        in[startY+6][startX+5] * filter +
        in[startY+6][startX+6] * filter +

        0;
    *out = tmp;
}
//Kernels that load, convolve, and store.
__global__ void convolutionKernel_global_register_size2x2_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[2][2];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size2x2(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size2x2_kernel2x2(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size3x3_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[3][3];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_global_register_size3x3(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size3x3_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size3x3_kernel2x2(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size4x4_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_global_register_size4x4(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size4x4_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size4x4_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size4x4_kernel2x2(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size5x5_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[4][4];
    if(globalX < width && globalY < height)
    {
        load_global_register_size5x5(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<4; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][3], 3, y);
        }
        store_register_global_size4x4(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size6x6_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[5][5];
    if(globalX < width && globalY < height)
    {
        load_global_register_size6x6(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<5; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][3], 3, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][4], 4, y);
        }
        store_register_global_size5x5(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size7x7_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[6][6];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<6; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][3], 3, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][4], 4, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][5], 5, y);
        }
        store_register_global_size6x6(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size3x3_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[3][3];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size3x3(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size3x3_kernel3x3(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size4x4_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_global_register_size4x4(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size4x4_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size4x4_kernel3x3(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size5x5_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_global_register_size5x5(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size5x5_kernel3x3(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size5x5_kernel3x3(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size6x6_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[4][4];
    if(globalX < width && globalY < height)
    {
        load_global_register_size6x6(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<4; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][3], 3, y);
        }
        store_register_global_size4x4(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size7x7_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[5][5];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<5; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][3], 3, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][4], 4, y);
        }
        store_register_global_size5x5(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size4x4_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size4x4(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size4x4_kernel4x4(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size5x5_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_global_register_size5x5(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel4x4(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size5x5_kernel4x4(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size6x6_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_global_register_size6x6(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel4x4(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel4x4(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size6x6_kernel4x4(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size7x7_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[4][4];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<4; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][3], 3, y);
        }
        store_register_global_size4x4(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size5x5_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size5x5(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel5x5(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size6x6_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_global_register_size6x6(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel5x5(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel5x5(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size7x7_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel5x5(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel5x5(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel5x5(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size6x6_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size6x6(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel6x6(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size7x7_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel6x6(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel6x6(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_global_register_size7x7_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-3, globalY-3, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel7x7(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size2x2_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[2][2];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size2x2(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size2x2_kernel2x2(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size3x3_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[3][3];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size3x3(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size3x3_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size3x3_kernel2x2(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size4x4_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size4x4(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size4x4_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size4x4_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size4x4_kernel2x2(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size5x5_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[4][4];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size5x5(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<4; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size5x5_kernel2x2(registers, &outRegisters[y][3], 3, y);
        }
        store_register_global_size4x4(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size6x6_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[5][5];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size6x6(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<5; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][3], 3, y);
           convolutionDevice_size6x6_kernel2x2(registers, &outRegisters[y][4], 4, y);
        }
        store_register_global_size5x5(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size7x7_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[6][6];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<6; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][3], 3, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][4], 4, y);
           convolutionDevice_size7x7_kernel2x2(registers, &outRegisters[y][5], 5, y);
        }
        store_register_global_size6x6(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size3x3_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[3][3];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size3x3(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size3x3_kernel3x3(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size4x4_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size4x4(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size4x4_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size4x4_kernel3x3(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size5x5_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size5x5(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size5x5_kernel3x3(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size5x5_kernel3x3(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size6x6_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[4][4];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size6x6(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<4; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size6x6_kernel3x3(registers, &outRegisters[y][3], 3, y);
        }
        store_register_global_size4x4(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size7x7_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[5][5];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<5; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][3], 3, y);
           convolutionDevice_size7x7_kernel3x3(registers, &outRegisters[y][4], 4, y);
        }
        store_register_global_size5x5(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size4x4_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size4x4(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size4x4_kernel4x4(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size5x5_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size5x5(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel4x4(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size5x5_kernel4x4(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size6x6_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size6x6(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel4x4(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel4x4(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size6x6_kernel4x4(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size7x7_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[4][4];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<4; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][2], 2, y);
           convolutionDevice_size7x7_kernel4x4(registers, &outRegisters[y][3], 3, y);
        }
        store_register_global_size4x4(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size5x5_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size5x5(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size5x5_kernel5x5(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size6x6_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size6x6(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel5x5(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size6x6_kernel5x5(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size7x7_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[3][3];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<3; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel5x5(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel5x5(registers, &outRegisters[y][1], 1, y);
           convolutionDevice_size7x7_kernel5x5(registers, &outRegisters[y][2], 2, y);
        }
        store_register_global_size3x3(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size6x6_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size6x6(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size6x6_kernel6x6(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size7x7_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[2][2];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<2; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel6x6(registers, &outRegisters[y][0], 0, y);
           convolutionDevice_size7x7_kernel6x6(registers, &outRegisters[y][1], 1, y);
        }
        store_register_global_size2x2(out, outRegisters, globalX, globalY, width, height);
    }
}
__global__ void convolutionKernel_texCache_register_size7x7_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-3, globalY-3, width, height); //includes offset so ghost zone is loaded
        for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        {
           convolutionDevice_size7x7_kernel7x7(registers, &outRegisters[y][0], 0, y);
        }
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
    }
}
//Kernels that load, convolve, and store.
__global__ void convolutionKernel_global_register_BenchmarkComputeTime_size2x2_kernel2x2(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[2][2];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size2x2(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size2x2_kernel2x2(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_global_register_BenchmarkComputeTime_size3x3_kernel3x3(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[3][3];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size3x3(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size3x3_kernel3x3(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_global_register_BenchmarkComputeTime_size4x4_kernel4x4(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size4x4(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size4x4_kernel4x4(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_global_register_BenchmarkComputeTime_size5x5_kernel5x5(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size5x5(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size5x5_kernel5x5(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_global_register_BenchmarkComputeTime_size6x6_kernel6x6(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size6x6(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size6x6_kernel6x6(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_global_register_BenchmarkComputeTime_size7x7_kernel7x7(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_global_register_size7x7(in, registers, globalX-3, globalY-3, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size7x7_kernel7x7(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_texCache_register_BenchmarkComputeTime_size2x2_kernel2x2(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[2][2];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size2x2(in, registers, globalX-0, globalY-0, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size2x2_kernel2x2(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_texCache_register_BenchmarkComputeTime_size3x3_kernel3x3(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[3][3];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size3x3(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size3x3_kernel3x3(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_texCache_register_BenchmarkComputeTime_size4x4_kernel4x4(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[4][4];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size4x4(in, registers, globalX-1, globalY-1, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size4x4_kernel4x4(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_texCache_register_BenchmarkComputeTime_size5x5_kernel5x5(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[5][5];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size5x5(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size5x5_kernel5x5(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_texCache_register_BenchmarkComputeTime_size6x6_kernel6x6(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[6][6];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size6x6(in, registers, globalX-2, globalY-2, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size6x6_kernel6x6(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}
__global__ void convolutionKernel_texCache_register_BenchmarkComputeTime_size7x7_kernel7x7(const float* in, float* out, const int width, const int height, float* fpuTime)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    float registers[7][7];
    float outRegisters[1][1];
    if(globalX < width && globalY < height)
    {
        load_texCache_register_size7x7(in, registers, globalX-3, globalY-3, width, height); //includes offset so ghost zone is loaded
        __syncthreads();
        double start = clock64();
        //for(int y=0; y<1; y++) //(int, y) = top left of region to convolve 
        int y=0;
        for(int iter=0; iter<5; iter++)
        {
           convolutionDevice_size7x7_kernel7x7(registers, &outRegisters[y][0], 0, y);
        }
        __syncthreads();
        double myTime = clock64() - start;
        store_register_global_size1x1(out, outRegisters, globalX, globalY, width, height);
        fpuTime[globalY*width + globalX] = myTime;
        //fpuTime[globalY*width + globalX] = globalY*width + globalX; //test
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size2x2_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 1; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size3x3_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 2; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size4x4_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 3; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size5x5_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 4; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size6x6_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 5; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size7x7_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 6; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size8x8_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = 7*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 7*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        for(int yy=0; yy < 7; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+6;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size3x3_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 1; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size4x4_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 2; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size5x5_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 3; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size6x6_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 4; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size7x7_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 5; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size8x8_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 6; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size9x9_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = 7*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 7*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        for(int yy=0; yy < 7; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
            
                outIdxX = globalX+6;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size4x4_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 1; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size5x5_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 2; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size6x6_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 3; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size7x7_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 4; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size8x8_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 5; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size9x9_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 6; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size10x10_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = 7*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 7*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        for(int yy=0; yy < 7; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+6;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size5x5_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 1; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size6x6_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 2; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size7x7_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 3; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size8x8_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 4; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size9x9_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 5; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size10x10_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 6; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size11x11_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = 7*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 7*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        for(int yy=0; yy < 7; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
            
                outIdxX = globalX+6;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size6x6_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 1; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size7x7_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 2; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size8x8_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 3; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size9x9_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 4; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size10x10_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 5; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size11x11_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 6; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size12x12_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = 7*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 7*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        for(int yy=0; yy < 7; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+6;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size7x7_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 1*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 1*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 1; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size8x8_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 2*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 2*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 2; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size9x9_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 3*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 3*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 3; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size10x10_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 4*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 4*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 4; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size11x11_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 5*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 5*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 5; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size12x12_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 6*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 6*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 6; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_global_only_size13x13_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = 7*(blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = 7*(blockIdx.y*blockDim.y + threadIdx.y);
    int outIdxX;
    int outIdxY;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        for(int yy=0; yy < 7; yy++)
        {
            
                outIdxX = globalX+0;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+1;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+2;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+3;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+4;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+5;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
            
                outIdxX = globalX+6;
                outIdxY = globalY+yy;
                out[outIdxY*width + outIdxX]=
                    in[clamp_addr(outIdxX+-3, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-3, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+-1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+-1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+0, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+0, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+1, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+1, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+2, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+2, width, height)] * filter +

                    in[clamp_addr(outIdxX+-3, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+-1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+0, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+1, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+2, outIdxY+3, width, height)] * filter +
                    in[clamp_addr(outIdxX+3, outIdxY+3, width, height)] * filter +

                    0;
 
        }
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_texCache_only_size2x2_kernel2x2(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - 0;
    int startY = globalY - 0;

    if(globalX < width && globalY < height)
    {
        const float filter = .2500; // 1/(kernelSize 2D)
        float tmp=
            tex2D(tex, float(startX+0)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+0)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+1)+0.5) * filter +

            0;
        out[globalY*width + globalX] = tmp;
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_texCache_only_size3x3_kernel3x3(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - 1;
    int startY = globalY - 1;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.1111; // 1/(kernelSize 2D)
        float tmp=
            tex2D(tex, float(startX+0)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+0)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+1)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+2)+0.5) * filter +

            0;
        out[globalY*width + globalX] = tmp;
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_texCache_only_size4x4_kernel4x4(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - 1;
    int startY = globalY - 1;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0625; // 1/(kernelSize 2D)
        float tmp=
            tex2D(tex, float(startX+0)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+0)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+1)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+2)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+3)+0.5) * filter +

            0;
        out[globalY*width + globalX] = tmp;
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_texCache_only_size5x5_kernel5x5(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - 2;
    int startY = globalY - 2;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0400; // 1/(kernelSize 2D)
        float tmp=
            tex2D(tex, float(startX+0)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+0)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+1)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+2)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+3)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+4)+0.5) * filter +

            0;
        out[globalY*width + globalX] = tmp;
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_texCache_only_size6x6_kernel6x6(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - 2;
    int startY = globalY - 2;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0278; // 1/(kernelSize 2D)
        float tmp=
            tex2D(tex, float(startX+0)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+0)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+1)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+2)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+3)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+4)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+5)+0.5) * filter +

            0;
        out[globalY*width + globalX] = tmp;
    }
}

//@param height,width = dims of 'in' array
__global__ void convolutionKernel_texCache_only_size7x7_kernel7x7(const float* in, float* out, const int width, const int height)
{
    int globalX = (blockIdx.x*blockDim.x + threadIdx.x);
    int globalY = (blockIdx.y*blockDim.y + threadIdx.y);
    int startX = globalX - 3;
    int startY = globalY - 3;

    if(globalX < width && globalY < height)
    {
        const float filter = 0.0204; // 1/(kernelSize 2D)
        float tmp=
            tex2D(tex, float(startX+0)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+0)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+0)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+1)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+1)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+2)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+2)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+3)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+3)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+4)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+4)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+5)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+5)+0.5) * filter +

            tex2D(tex, float(startX+0)+0.5, float(startY+6)+0.5) * filter +
            tex2D(tex, float(startX+1)+0.5, float(startY+6)+0.5) * filter +
            tex2D(tex, float(startX+2)+0.5, float(startY+6)+0.5) * filter +
            tex2D(tex, float(startX+3)+0.5, float(startY+6)+0.5) * filter +
            tex2D(tex, float(startX+4)+0.5, float(startY+6)+0.5) * filter +
            tex2D(tex, float(startX+5)+0.5, float(startY+6)+0.5) * filter +
            tex2D(tex, float(startX+6)+0.5, float(startY+6)+0.5) * filter +

            0;
        out[globalY*width + globalX] = tmp;
    }
}
//configuration directly pulled from simpleTexture in nvidia sdk
void setTexCacheParams()
{
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = false;
}
float convolutionWrapper(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename)
{
    dim3 grid;
    dim3 block;
    block.x = 16;
    block.y = 16;
    int sqrtConvsPerThread = amountToLoad-kernelSize+1;
    int nx = width / (block.x*sqrtConvsPerThread); //magic number is for persistant kernels (e.g. if persistant6x6, divide by 4)
    int ny = height / (block.y*sqrtConvsPerThread);
    grid.x = (width % block.x == 0) ? nx : nx+1;
    grid.y = (height % block.y == 0) ? ny : ny+1;

    float *dImg;
    CHECK_CUDART(cudaMalloc((void**)&dImg, sizeof(float)*width*height)); //for input data (possibly use texture cache?)
    CHECK_CUDART(cudaMemcpy(dImg, hImg, sizeof(float)*width*height, cudaMemcpyHostToDevice));
    float* dResult; //device memory for output
    CHECK_CUDART(cudaMalloc((void**)&dResult, sizeof(float)*width*height));

    float* dFpuTime;
    CHECK_CUDART(cudaMalloc((void**)&dFpuTime, sizeof(float)*width*height));
    CHECK_CUDART(cudaMemset(dFpuTime, 0, sizeof(float)*width*height));

    double start = read_timer();
    if(amountToLoad == 2 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size2x2_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size2x2_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size3x3_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size3x3_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size4x4_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size4x4_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size5x5_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size5x5_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size6x6_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size6x6_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size7x7_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size7x7_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 8 && kernelSize == 2 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size8x8_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size8x8_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size3x3_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size3x3_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size4x4_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size4x4_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size5x5_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size5x5_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size6x6_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size6x6_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size7x7_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size7x7_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 8 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size8x8_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size8x8_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 9 && kernelSize == 3 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size9x9_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size9x9_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size4x4_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size4x4_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size5x5_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size5x5_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size6x6_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size6x6_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size7x7_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size7x7_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 8 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size8x8_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size8x8_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 9 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size9x9_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size9x9_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 10 && kernelSize == 4 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size10x10_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size10x10_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size5x5_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size5x5_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size6x6_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size6x6_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size7x7_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size7x7_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 8 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size8x8_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size8x8_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 9 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size9x9_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size9x9_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 10 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size10x10_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size10x10_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 11 && kernelSize == 5 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size11x11_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size11x11_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size6x6_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size6x6_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size7x7_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size7x7_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 8 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size8x8_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size8x8_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 9 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size9x9_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size9x9_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 10 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size10x10_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size10x10_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 11 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size11x11_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size11x11_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 12 && kernelSize == 6 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size12x12_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size12x12_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size7x7_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size7x7_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 8 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size8x8_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size8x8_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 9 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size9x9_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size9x9_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 10 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size10x10_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size10x10_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 11 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size11x11_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size11x11_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 12 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size12x12_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size12x12_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 13 && kernelSize == 7 && memoryScheme == "global_only") {
        cudaFuncSetCacheConfig(convolutionKernel_global_only_size13x13_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_only_size13x13_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 2 && kernelSize == 2 && memoryScheme == "global_register_BenchmarkComputeTime") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_BenchmarkComputeTime_size2x2_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_BenchmarkComputeTime_size2x2_kernel2x2<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }
    if(amountToLoad == 3 && kernelSize == 3 && memoryScheme == "global_register_BenchmarkComputeTime") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_BenchmarkComputeTime_size3x3_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_register_BenchmarkComputeTime_size3x3_kernel3x3<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }
    if(amountToLoad == 4 && kernelSize == 4 && memoryScheme == "global_register_BenchmarkComputeTime") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_BenchmarkComputeTime_size4x4_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_register_BenchmarkComputeTime_size4x4_kernel4x4<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }
    if(amountToLoad == 5 && kernelSize == 5 && memoryScheme == "global_register_BenchmarkComputeTime") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_BenchmarkComputeTime_size5x5_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_register_BenchmarkComputeTime_size5x5_kernel5x5<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }
    if(amountToLoad == 6 && kernelSize == 6 && memoryScheme == "global_register_BenchmarkComputeTime") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_BenchmarkComputeTime_size6x6_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_register_BenchmarkComputeTime_size6x6_kernel6x6<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }
    if(amountToLoad == 7 && kernelSize == 7 && memoryScheme == "global_register_BenchmarkComputeTime") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_BenchmarkComputeTime_size7x7_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_register_BenchmarkComputeTime_size7x7_kernel7x7<<<grid, block>>>(dImg, dResult, width, height, dFpuTime);
    }
    if(amountToLoad == 2 && kernelSize == 2 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size2x2_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size2x2_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 2 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size3x3_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size3x3_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 2 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size4x4_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size4x4_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 2 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size5x5_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size5x5_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 2 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size6x6_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size6x6_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 2 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size7x7_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size7x7_kernel2x2<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 3 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size3x3_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size3x3_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 3 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size4x4_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size4x4_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 3 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size5x5_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size5x5_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 3 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size6x6_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size6x6_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 3 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size7x7_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size7x7_kernel3x3<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 4 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size4x4_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size4x4_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 4 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size5x5_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size5x5_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 4 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size6x6_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size6x6_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 4 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size7x7_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size7x7_kernel4x4<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 5 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size5x5_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size5x5_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 5 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size6x6_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size6x6_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 5 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size7x7_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size7x7_kernel5x5<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 6 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size6x6_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size6x6_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 6 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size7x7_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size7x7_kernel6x6<<<grid, block>>>(dImg, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 7 && memoryScheme == "global_register") {
        cudaFuncSetCacheConfig(convolutionKernel_global_register_size7x7_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_global_register_size7x7_kernel7x7<<<grid, block>>>(dImg, dResult, width, height);
    }
    cudaDeviceSynchronize();
    double responseTime = read_timer() - start;
    
    CHECK_CUDART(cudaFree(dImg));
    if(outputImgFlag)
    {
        float* hResult = (float*)malloc(sizeof(float)*width*height);
        CHECK_CUDART(cudaMemcpy(hResult, dResult, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
        outputProcessedImageFloat(hResult, width, height, outFilename);
        free(hResult);
    }
    if(memoryScheme.find("BenchmarkComputeTime") != string::npos)
    {
        float* hFpuTime = (float*)malloc(sizeof(float)*width*height);
        CHECK_CUDART(cudaMemcpy(hFpuTime, dFpuTime, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
        printf("COMPUTE time on GPU, %dx%d filter: \n", kernelSize, kernelSize);
        for(int i=0; i<(width*height); i++)
        {
            printf("%f\n", hFpuTime[i]);
        }
    }
    CHECK_CUDART(cudaFree(dResult));
    CHECK_CUDART(cudaFree(dFpuTime));
    return responseTime;
}
float convolutionWrapper_texCache(float* hImg, const int width, const int height, int amountToLoad, int kernelSize, string memoryScheme, bool outputImgFlag, string outFilename)
{
    dim3 grid;
    dim3 block;
    block.x = 16;
    block.y = 16;
    int sqrtConvsPerThread = amountToLoad-kernelSize+1;
    int nx = width / (block.x*sqrtConvsPerThread); //magic number is for persistant kernels (e.g. if persistant6x6, divide by 4)
    int ny = height / (block.y*sqrtConvsPerThread);
    grid.x = (width % block.x == 0) ? nx : nx+1;
    grid.y = (height % block.y == 0) ? ny : ny+1;

    cudaArray *dImg; //cudaArray*, not float*
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    CHECK_CUDART(cudaMallocArray(&dImg, &channelDesc, width, height));
    CHECK_CUDART(cudaMemcpyToArray(dImg, 0, 0, hImg, width*height*sizeof(float), cudaMemcpyHostToDevice));
    setTexCacheParams();
    CHECK_CUDART(cudaBindTextureToArray(tex, dImg, channelDesc));
    float* dummy = NULL; //my texCache code doesn't need input data ptr, but it was easier to have one in code generation anyway.
    float* dResult; //device memory for output
    CHECK_CUDART(cudaMalloc((void**)&dResult, sizeof(float)*width*height));

    double start = read_timer();
    if(amountToLoad == 2 && kernelSize == 2 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size2x2_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size2x2_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 2 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size3x3_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size3x3_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 2 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size4x4_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size4x4_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 2 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size5x5_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size5x5_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 2 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size6x6_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size6x6_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 2 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size7x7_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size7x7_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 3 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size3x3_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size3x3_kernel3x3<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 3 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size4x4_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size4x4_kernel3x3<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 3 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size5x5_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size5x5_kernel3x3<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 3 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size6x6_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size6x6_kernel3x3<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 3 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size7x7_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size7x7_kernel3x3<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 4 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size4x4_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size4x4_kernel4x4<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 4 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size5x5_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size5x5_kernel4x4<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 4 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size6x6_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size6x6_kernel4x4<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 4 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size7x7_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size7x7_kernel4x4<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 5 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size5x5_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size5x5_kernel5x5<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 5 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size6x6_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size6x6_kernel5x5<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 5 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size7x7_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size7x7_kernel5x5<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 6 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size6x6_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size6x6_kernel6x6<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 6 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size7x7_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size7x7_kernel6x6<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 7 && memoryScheme == "texCache_register") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_register_size7x7_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_texCache_register_size7x7_kernel7x7<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 2 && kernelSize == 2 && memoryScheme == "texCache_only") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_only_size2x2_kernel2x2, cudaFuncCachePreferL1);
        convolutionKernel_texCache_only_size2x2_kernel2x2<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 3 && kernelSize == 3 && memoryScheme == "texCache_only") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_only_size3x3_kernel3x3, cudaFuncCachePreferL1);
        convolutionKernel_texCache_only_size3x3_kernel3x3<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 4 && kernelSize == 4 && memoryScheme == "texCache_only") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_only_size4x4_kernel4x4, cudaFuncCachePreferL1);
        convolutionKernel_texCache_only_size4x4_kernel4x4<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 5 && kernelSize == 5 && memoryScheme == "texCache_only") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_only_size5x5_kernel5x5, cudaFuncCachePreferL1);
        convolutionKernel_texCache_only_size5x5_kernel5x5<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 6 && kernelSize == 6 && memoryScheme == "texCache_only") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_only_size6x6_kernel6x6, cudaFuncCachePreferL1);
        convolutionKernel_texCache_only_size6x6_kernel6x6<<<grid, block>>>(dummy, dResult, width, height);
    }
    if(amountToLoad == 7 && kernelSize == 7 && memoryScheme == "texCache_only") {
        cudaFuncSetCacheConfig(convolutionKernel_texCache_only_size7x7_kernel7x7, cudaFuncCachePreferL1);
        convolutionKernel_texCache_only_size7x7_kernel7x7<<<grid, block>>>(dummy, dResult, width, height);
    }

    cudaDeviceSynchronize();
    double responseTime = read_timer() - start;
   
    CHECK_CUDART(cudaFreeArray(dImg));
    if(outputImgFlag)
    {
        float* hResult = (float*)malloc(sizeof(float)*width*height);
        CHECK_CUDART(cudaMemcpy(hResult, dResult, sizeof(float)*width*height, cudaMemcpyDeviceToHost));
        outputProcessedImageFloat(hResult, width, height, outFilename);
        free(hResult);
    }
    CHECK_CUDART(cudaFree(dResult));
    return responseTime;
}
