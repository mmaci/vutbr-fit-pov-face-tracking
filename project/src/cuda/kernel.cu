       
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <fstream>

#include "header.h"

cudaError_t run(unsigned char*);
__device__ void preprocess_image();
__device__ void detect();

__global__ void runKernel()
{
    int threadId = threadIdx.x;

	preprocess_image();

	__syncthreads();

	detect();
}

__device__ void preprocess_image()
{

}

__device__ void detect()
{

}

int main(int argc, char** argv)
{
	///////////////////////////////////////////////////////////////////
	// init phase

	std::string inputFilename;
	bool dataset = false;
	for (int i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == "-i" && i + 1 < argc) {
			inputFilename = argv[++i];
		}
		if (std::string(argv[i]) == "-d" && i + 1 < argc) {
			dataset = true;
			inputFilename = argv[++i];
		}
		else {
			std::cerr << "Usage: " << argv[0] << " -i [input file]" << std::endl;
			return EXIT_FAILURE;
		}
	}

	std::vector<cv::Mat> images;
	if (dataset) {
		std::ifstream in;
		in.open(inputFilename);

		std::string filename;
		while (!in.eof()) {
			std::getline(in, filename);
			cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);

			if (!image.data)			
				std::cerr << "[" << LIBNAME << "]: " << "Could not open or find the image: " << filename << ")" << std::endl;

			images.push_back(image);
		}
	}
	else {
		cv::Mat image = cv::imread(inputFilename.c_str(), CV_LOAD_IMAGE_COLOR);

		if (!image.data)		
			std::cerr << "[" << LIBNAME << "]: " << "Could not open or find the image (filename: " << inputFilename << ")" << std::endl;

		images.push_back(image);
	}
	
	for (std::vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat image = *it;

		cudaError_t cudaStatus = run(image.data);
		if (cudaStatus != cudaSuccess) {
			std::cerr << "[" << LIBNAME << "]: " << "CUDA runtime error" << std::endl;;
			return EXIT_FAILURE;
		}

		// needed for profiling - NSight
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			std::cerr << "[" << LIBNAME << "]: " << "cudaDeviceReset failed" << std::endl;;
			return EXIT_FAILURE;
		}
	}	

    return EXIT_SUCCESS;
}


cudaError_t run(unsigned char* data)
{    
    cudaError_t cudaStatus;
 
    cudaStatus = cudaSetDevice(0);   
 
    runKernel<<<1, 128>>>();
 
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
		std::cerr << "[" << LIBNAME << "]: " << cudaGetErrorString(cudaStatus) << std::endl;        
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
		std::cerr << "[" << LIBNAME << "]: " << "cudaDeviceSynchronize failed (error code: " << cudaStatus << ")" << std::endl;        
    }

    return cudaStatus;
}
