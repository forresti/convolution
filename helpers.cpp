#include "helpers.h"

double read_timer()
{
    struct timeval start;
    gettimeofday( &start, NULL );
    return (double)((start.tv_sec) + 1.0e-6 * (start.tv_usec)); //in seconds
}

// designed for use with UW Madison raw image data
// http://pages.cs.wisc.edu/~dyer/cs534-fall11/hw-toc.html
cv::Mat getRawImage(string in_filename)
{
    cv::Mat img = cv::imread(in_filename);
    cv::cvtColor(img, img, CV_RGB2GRAY);
    int dim = min(img.rows, img.cols);
    img = img.colRange(0, dim).rowRange(0, dim); //make the image square (cut off bottom right if necessary)
    //printf("in getRawImage(). rows = %d, cols = %d \n", img.rows, img.cols); 
    return img;
}

void outputProcessedImage(unsigned int* processedImg, int width, int height, string out_filename)
{
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            uchar* px = (uchar*)&processedImg[i*width + j];
            img.at<uchar>(i,j) = px[0]; //just grab the 1st of the 4 pixel spaces in a uchar4
       }
    }

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite(out_filename, img, compression_params);
}

void outputProcessedImageFloat(float* processedImg, int width, int height, string out_filename)
{
    //cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat img = cv::Mat::zeros(height, width, CV_32FC1);
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            img.at<uchar>(i,j) = (uchar)processedImg[i*width + j];
       }
    }
    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite(out_filename, img, compression_params);
}

void outputProcessedImageUchar(uchar* processedImg, int width, int height, string out_filename)
{
    cv::Mat img = cv::Mat::zeros(height, width, CV_8UC1);
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            img.at<uchar>(i,j) = processedImg[i*width + j]; 
       }
    }

    vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    cv::imwrite(out_filename, img, compression_params);
}

float* getDummyImg(int height, int width)
{
    float* img = (float*)malloc(sizeof(float)*width*height);
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            img[i*width + j] = (i+j)%256; //arbitrary  
        }
    }
    return img;
}

uchar* getDummyImgUchar(int height, int width)
{
    uchar* img = (uchar*)malloc(sizeof(uchar)*width*height);
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            img[i*width + j] = (i+j)%256; //arbitrary  
        }
    }
    return img;
}

float getResponseTime(cudaEvent_t start, cudaEvent_t stop)
{
    float responseTime;
    cudaEventElapsedTime(&responseTime, start, stop);
    return responseTime/1000; //convert ms to seconds
}



