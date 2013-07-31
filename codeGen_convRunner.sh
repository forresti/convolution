echo '#include "cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include "convolution.h"
#include "helpers.h"
using namespace std;
using namespace cv;'

echo 'void testConvolution()
{
    cv::Mat img = getRawImage("./Lena.pgm");
    img.convertTo(img, CV_32FC1);'
for memoryScheme in "global_only"
do
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    for((sqrtConvsPerThread=1; sqrtConvsPerThread<8; sqrtConvsPerThread++)) do
    amountToLoad=$(($sqrtConvsPerThread+$kernelSize-1))
        echo "    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, $amountToLoad, $kernelSize, \"$memoryScheme\", true, \"results/kernel${kernelSize}x${kernelSize}_size${amountToLoad}x${amountToLoad}_${memoryScheme}.png\");"
    done
    done
done
for memoryScheme in "global_register" #"global_shared_register"
do
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    for((amountToLoad=$kernelSize; amountToLoad<8; amountToLoad++)) do
        echo "    convolutionWrapper((float*)&img.data[0], img.cols, img.rows, $amountToLoad, $kernelSize, \"$memoryScheme\", true, \"results/kernel${kernelSize}x${kernelSize}_size${amountToLoad}x${amountToLoad}_${memoryScheme}.png\");"
    done
    done
done
for memoryScheme in "texCache_only" 
do
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    amountToLoad=$kernelSize
        echo "    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, $amountToLoad, $kernelSize, \"$memoryScheme\", true, \"results/kernel${kernelSize}x${kernelSize}_size${amountToLoad}x${amountToLoad}_${memoryScheme}.png\");"
    done
done
for memoryScheme in "texCache_register" 
do
    for((kernelSize=2; kernelSize<8; kernelSize++)) do
    for((amountToLoad=$kernelSize; amountToLoad<8; amountToLoad++)) do
        echo "    convolutionWrapper_texCache((float*)&img.data[0], img.cols, img.rows, $amountToLoad, $kernelSize, \"$memoryScheme\", true, \"results/kernel${kernelSize}x${kernelSize}_size${amountToLoad}x${amountToLoad}_${memoryScheme}.png\");"
    done
    done
done
echo '}

void testConvolution_withDummyImg(int height, int width)
{
    float* img = getDummyImg(height, width);
    FILE * pFile = fopen("perf.txt", "w");
    fprintf(pFile, "kernelSize amountToLoad memoryScheme responseTime\n");
    int nRuns = 10;
    float responseTime = 0;
    responseTime = convolutionWrapper(img, width, height, 3, 3, "global_register", false); //warmup'
for memoryScheme in "global_only" #"global_register_BenchmarkComputeTime"
do
echo "    printf(\"memoryScheme = %s \n\", \"$memoryScheme\");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        for(int sqrtConvsPerThread=1; sqrtConvsPerThread<8; sqrtConvsPerThread++)
        {
            int amountToLoad = sqrtConvsPerThread+kernelSize-1; //actually, prefetching nothing in this version
            responseTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                float tmpTime = convolutionWrapper(img, width, height, amountToLoad, kernelSize, \"$memoryScheme\", false);
                responseTime += tmpTime;
            }
            responseTime = responseTime/nRuns;
            fprintf(pFile, \"%d, %d, %s, %f \\n\", kernelSize, amountToLoad, \"$memoryScheme\", responseTime);
            printf(\"kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n\", kernelSize, amountToLoad, responseTime);
            cudaDeviceSynchronize();
        }
        printf(\"\n\");
    }"
done
for memoryScheme in "global_register" #"global_shared_register"
do
echo "    printf(\"memoryScheme = %s \n\", \"$memoryScheme\");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        for(int amountToLoad=kernelSize; amountToLoad<8; amountToLoad++)
        {
            responseTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                float tmpTime = convolutionWrapper(img, width, height, amountToLoad, kernelSize, \"$memoryScheme\", false);
                responseTime += tmpTime;
            }
            responseTime = responseTime/nRuns;
            fprintf(pFile, \"%d, %d, %s, %f \\n\", kernelSize, amountToLoad, \"$memoryScheme\", responseTime);
            printf(\"kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n\", kernelSize, amountToLoad, responseTime);
            cudaDeviceSynchronize();
        }
        printf(\"\n\");
    }"
done
for memoryScheme in "texCache_only"
do
echo "    printf(\"memoryScheme = %s \n\", \"$memoryScheme\");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        int amountToLoad = kernelSize;
        responseTime = 0;
        for(int i=0; i<nRuns; i++)
        {
            float tmpTime = convolutionWrapper_texCache(img, width, height, amountToLoad, kernelSize, \"$memoryScheme\", false);
            responseTime += tmpTime;
        }
        responseTime = responseTime/nRuns;
        fprintf(pFile, \"%d, %d, %s, %f \\n\", kernelSize, amountToLoad, \"$memoryScheme\", responseTime);
        printf(\"kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n\", kernelSize, amountToLoad, responseTime);
        cudaDeviceSynchronize();
        printf(\"\n\");
    }"
done
for memoryScheme in "texCache_register" #"texCache_shared_register"
do
echo "    printf(\"memoryScheme = %s \n\", \"$memoryScheme\");
    for(int kernelSize=2; kernelSize<8; kernelSize++)   
    {
        for(int amountToLoad=kernelSize; amountToLoad<8; amountToLoad++)
        {
            responseTime = 0;
            for(int i=0; i<nRuns; i++)
            {
                float tmpTime = convolutionWrapper_texCache(img, width, height, amountToLoad, kernelSize, \"$memoryScheme\", false);
                responseTime += tmpTime;
            }
            responseTime = responseTime/nRuns;
            fprintf(pFile, \"%d, %d, %s, %f \\n\", kernelSize, amountToLoad, \"$memoryScheme\", responseTime);
            printf(\"kernelSize = %d. amountToLoad = %d. time per Convolution = %f seconds \n\", kernelSize, amountToLoad, responseTime);
            cudaDeviceSynchronize();
        }
        printf(\"\n\");
    }
    fclose(pFile);"
done

echo '}'


