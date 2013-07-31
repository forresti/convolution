#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "helpers.h"
using namespace std;
using namespace cv;
void testConvolution()
{
    cv::Mat img = getRawImage("./Lena.pgm");
    img.convertTo(img, CV_32FC1);
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 2, 2, "global_only", true, "results/kernel2x2_size2x2_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 3, 2, "global_only", true, "results/kernel2x2_size3x3_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 4, 2, "global_only", true, "results/kernel2x2_size4x4_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 2, "global_only", true, "results/kernel2x2_size5x5_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 2, "global_only", true, "results/kernel2x2_size6x6_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 2, "global_only", true, "results/kernel2x2_size7x7_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 8, 2, "global_only", true, "results/kernel2x2_size8x8_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 3, 3, "global_only", true, "results/kernel3x3_size3x3_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 4, 3, "global_only", true, "results/kernel3x3_size4x4_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 3, "global_only", true, "results/kernel3x3_size5x5_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 3, "global_only", true, "results/kernel3x3_size6x6_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 3, "global_only", true, "results/kernel3x3_size7x7_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 8, 3, "global_only", true, "results/kernel3x3_size8x8_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 9, 3, "global_only", true, "results/kernel3x3_size9x9_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 4, 4, "global_only", true, "results/kernel4x4_size4x4_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 4, "global_only", true, "results/kernel4x4_size5x5_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 4, "global_only", true, "results/kernel4x4_size6x6_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 4, "global_only", true, "results/kernel4x4_size7x7_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 8, 4, "global_only", true, "results/kernel4x4_size8x8_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 9, 4, "global_only", true, "results/kernel4x4_size9x9_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 10, 4, "global_only", true, "results/kernel4x4_size10x10_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 5, "global_only", true, "results/kernel5x5_size5x5_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 5, "global_only", true, "results/kernel5x5_size6x6_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 5, "global_only", true, "results/kernel5x5_size7x7_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 8, 5, "global_only", true, "results/kernel5x5_size8x8_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 9, 5, "global_only", true, "results/kernel5x5_size9x9_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 10, 5, "global_only", true, "results/kernel5x5_size10x10_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 11, 5, "global_only", true, "results/kernel5x5_size11x11_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 6, "global_only", true, "results/kernel6x6_size6x6_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 6, "global_only", true, "results/kernel6x6_size7x7_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 8, 6, "global_only", true, "results/kernel6x6_size8x8_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 9, 6, "global_only", true, "results/kernel6x6_size9x9_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 10, 6, "global_only", true, "results/kernel6x6_size10x10_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 11, 6, "global_only", true, "results/kernel6x6_size11x11_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 12, 6, "global_only", true, "results/kernel6x6_size12x12_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 7, "global_only", true, "results/kernel7x7_size7x7_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 8, 7, "global_only", true, "results/kernel7x7_size8x8_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 9, 7, "global_only", true, "results/kernel7x7_size9x9_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 10, 7, "global_only", true, "results/kernel7x7_size10x10_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 11, 7, "global_only", true, "results/kernel7x7_size11x11_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 12, 7, "global_only", true, "results/kernel7x7_size12x12_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 13, 7, "global_only", true, "results/kernel7x7_size13x13_global_only.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 2, 2, "global_register", true, "results/kernel2x2_size2x2_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 3, 2, "global_register", true, "results/kernel2x2_size3x3_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 4, 2, "global_register", true, "results/kernel2x2_size4x4_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 2, "global_register", true, "results/kernel2x2_size5x5_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 2, "global_register", true, "results/kernel2x2_size6x6_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 2, "global_register", true, "results/kernel2x2_size7x7_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 3, 3, "global_register", true, "results/kernel3x3_size3x3_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 4, 3, "global_register", true, "results/kernel3x3_size4x4_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 3, "global_register", true, "results/kernel3x3_size5x5_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 3, "global_register", true, "results/kernel3x3_size6x6_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 3, "global_register", true, "results/kernel3x3_size7x7_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 4, 4, "global_register", true, "results/kernel4x4_size4x4_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 4, "global_register", true, "results/kernel4x4_size5x5_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 4, "global_register", true, "results/kernel4x4_size6x6_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 4, "global_register", true, "results/kernel4x4_size7x7_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 5, 5, "global_register", true, "results/kernel5x5_size5x5_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 5, "global_register", true, "results/kernel5x5_size6x6_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 5, "global_register", true, "results/kernel5x5_size7x7_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 6, 6, "global_register", true, "results/kernel6x6_size6x6_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 6, "global_register", true, "results/kernel6x6_size7x7_global_register.png");
    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, 7, 7, "global_register", true, "results/kernel7x7_size7x7_global_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 2, 2, "texCache_only", true, "results/kernel2x2_size2x2_texCache_only.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 3, 3, "texCache_only", true, "results/kernel3x3_size3x3_texCache_only.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 4, 4, "texCache_only", true, "results/kernel4x4_size4x4_texCache_only.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 5, 5, "texCache_only", true, "results/kernel5x5_size5x5_texCache_only.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 6, 6, "texCache_only", true, "results/kernel6x6_size6x6_texCache_only.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 7, "texCache_only", true, "results/kernel7x7_size7x7_texCache_only.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 2, 2, "texCache_register", true, "results/kernel2x2_size2x2_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 3, 2, "texCache_register", true, "results/kernel2x2_size3x3_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 4, 2, "texCache_register", true, "results/kernel2x2_size4x4_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 5, 2, "texCache_register", true, "results/kernel2x2_size5x5_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 6, 2, "texCache_register", true, "results/kernel2x2_size6x6_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 2, "texCache_register", true, "results/kernel2x2_size7x7_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 3, 3, "texCache_register", true, "results/kernel3x3_size3x3_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 4, 3, "texCache_register", true, "results/kernel3x3_size4x4_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 5, 3, "texCache_register", true, "results/kernel3x3_size5x5_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 6, 3, "texCache_register", true, "results/kernel3x3_size6x6_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 3, "texCache_register", true, "results/kernel3x3_size7x7_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 4, 4, "texCache_register", true, "results/kernel4x4_size4x4_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 5, 4, "texCache_register", true, "results/kernel4x4_size5x5_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 6, 4, "texCache_register", true, "results/kernel4x4_size6x6_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 4, "texCache_register", true, "results/kernel4x4_size7x7_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 5, 5, "texCache_register", true, "results/kernel5x5_size5x5_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 6, 5, "texCache_register", true, "results/kernel5x5_size6x6_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 5, "texCache_register", true, "results/kernel5x5_size7x7_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 6, 6, "texCache_register", true, "results/kernel6x6_size6x6_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 6, "texCache_register", true, "results/kernel6x6_size7x7_texCache_register.png");
    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, 7, 7, "texCache_register", true, "results/kernel7x7_size7x7_texCache_register.png");
}

void testConvolution_withDummyImg(int height, int width)
{
    float* img = getDummyImg(height, width);
    FILE * pFile = fopen("perf.txt", "w");
    fprintf(pFile, "kernelSize amountToLoad memoryScheme responseTime\n");
    int nRuns = 10;
    float responseTime = 0;
    responseTime = convolutionWrapper(img, width, height, 3, 3, "global_register", false); //warmup
    printf("memoryScheme = %s \n", "global_only");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        for(int sqrtConvsPerThread=1; sqrtConvsPerThread<8; sqrtConvsPerThread++)
        {
            int amountToLoad = sqrtConvsPerThread+kernelSize-1; //actually, prefetching nothing in this version
            responseTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                float tmpTime = convolutionWrapper(img, width, height, amountToLoad, kernelSize, "global_only", false);
                responseTime += tmpTime;
            }
            responseTime = responseTime/nRuns;
            fprintf(pFile, "%d, %d, %s, %f \n", kernelSize, amountToLoad, "global_only", responseTime);
            printf("kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n", kernelSize, amountToLoad, responseTime);
            cudaDeviceSynchronize();
        }
        printf("\n");
    }
    printf("memoryScheme = %s \n", "global_register");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        for(int amountToLoad=kernelSize; amountToLoad<8; amountToLoad++)
        {
            responseTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                float tmpTime = convolutionWrapper(img, width, height, amountToLoad, kernelSize, "global_register", false);
                responseTime += tmpTime;
            }
            responseTime = responseTime/nRuns;
            fprintf(pFile, "%d, %d, %s, %f \n", kernelSize, amountToLoad, "global_register", responseTime);
            printf("kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n", kernelSize, amountToLoad, responseTime);
            cudaDeviceSynchronize();
        }
        printf("\n");
    }
    printf("memoryScheme = %s \n", "texCache_only");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        int amountToLoad = kernelSize;
        responseTime = 0;
        for(int i=0; i<nRuns; i++)
        {
            float tmpTime = convolutionWrapper_texCache(img, width, height, amountToLoad, kernelSize, "texCache_only", false);
            responseTime += tmpTime;
        }
        responseTime = responseTime/nRuns;
        fprintf(pFile, "%d, %d, %s, %f \n", kernelSize, amountToLoad, "texCache_only", responseTime);
        printf("kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n", kernelSize, amountToLoad, responseTime);
        cudaDeviceSynchronize();
        printf("\n");
    }
    printf("memoryScheme = %s \n", "texCache_register");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        for(int amountToLoad=kernelSize; amountToLoad<8; amountToLoad++)
        {
            responseTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                float tmpTime = convolutionWrapper_texCache(img, width, height, amountToLoad, kernelSize, "texCache_register", false);
                responseTime += tmpTime;
            }
            responseTime = responseTime/nRuns;
            fprintf(pFile, "%d, %d, %s, %f \n", kernelSize, amountToLoad, "texCache_register", responseTime);
            printf("kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n", kernelSize, amountToLoad, responseTime);
            cudaDeviceSynchronize();
        }
        printf("\n");
    }
    fclose(pFile);
}
