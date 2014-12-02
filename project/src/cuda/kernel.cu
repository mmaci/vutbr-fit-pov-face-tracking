       
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <vector>
#include <fstream>

#include "header.h"
#include "detector.h"

/// wrapper to call kernels
cudaError_t runKernelWrapper(uint8* /* device image */, Detection* /* device detection buffer */, uint32* /* device detection count */, cudaTextureObject_t* /* ptr to txt memory with alphas*/);

/// runs object detectin on gpu itself
__device__ void detect(uint8* /* device image */, Detection* /* detections */, uint32* /* detection count */, cudaTextureObject_t* /* alphas */);
/// gpu bilinear interpolation
__device__ void bilinearInterpolation(uint8* /* input image */, uint8* /* output image */, const float /* scale */);
/// builds a pyramid image with parameters set in header.h
__device__ void buildPyramid(uint8* /* device image */);

/// detector stages
__constant__ Stage stages[STAGE_COUNT];
/// detector parameters
__constant__ DetectorInfo detectorInfo[1];

__global__ void runKernel(uint8* imageData, Detection* detections, uint32* detectionCount, cudaTextureObject_t* alphas)
{
	buildPyramid(imageData);

	__syncthreads();
	
	detect(imageData, detections, detectionCount, alphas);
}

__device__ void buildPyramid(uint8* imageData)
{	
	float scale = 1.0f;
	uint32 image_width = detectorInfo[0].imageWidth;
	uint32 image_height = detectorInfo[0].imageHeight;

	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < (image_width-1) && y < (image_height-1))
	{		
		int offset = (detectorInfo[0].imageWidth * detectorInfo[0].imageHeight);
		for (int level = 0; level < PYRAMID_IMAGE_COUNT; ++level)
		{
			scale *= SCALE_FACTOR;
			image_width /= SCALE_FACTOR;
			image_height /= SCALE_FACTOR;

			bilinearInterpolation(imageData, imageData + offset, scale);

			offset += image_height * detectorInfo[0].imageWidth;
		}
	}	
}

__device__ void bilinearInterpolation(uint8* inImage, uint8* outImage, const float scale)
{	
	const int origX = blockIdx.x*blockDim.x + threadIdx.x;
	const int origY = blockIdx.y*blockDim.y + threadIdx.y;

	const int x = origX / scale;
	const int y = origY / scale;

	float dx = origX / scale - x;
	float dy = origY / scale - y;

	uint8 a = inImage[origY * detectorInfo[0].imageWidth + origX];
	uint8 b = inImage[origY * detectorInfo[0].imageWidth + origX + 1];
	uint8 c = inImage[(origY + 1) * detectorInfo[0].imageWidth + origX];
	uint8 d = inImage[(origY + 1) * detectorInfo[0].imageWidth + origX + 1];

	uint8 res = (a * (1.0f - dx) * (1.0f - dy))
		+ (b * dx * (1.0f - dy))
		+ (c * (1.0f - dx) * dy)
		+ (d * dx * dy);

	outImage[y * detectorInfo[0].pyramidImageWidth + x] = res;
}

__device__ void sumRegions(uint8* imageData, Stage* stage, uint32* values)
{
	uint32 baseX = blockIdx.x * blockDim.x + threadIdx.x + stage->x;
	uint32 baseY = blockIdx.y * blockDim.y + threadIdx.y + stage->y;

	for (uint32 i = 0; i < 9; ++i) {
		uint32 startX = baseX + (i % 3) * stage->width;
		uint32 startY = baseY + (i / 3) * stage->height;

		uint32 acc = 0;
		for (uint32 x = startX; x < startX + stage->width; ++x) {
			for (uint32 y = startY; y < startY + stage->height; ++y) {
				acc += imageData[y * detectorInfo[0].pyramidImageWidth + x];
			}
		}

		values[i] = acc;
	}
}

__device__ float evalLBP(uint8* data, Stage* stage, cudaTextureObject_t* alphas)
{
	const int LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

	uint32 values[9];

	sumRegions(data, stage, values);

	int code = 0;
	for (int i = 0; i < 8; ++i)
		code |= (values[LBPOrder[i]] > values[4]) << i;

	return tex1Dfetch<float>(*alphas, stage->alphaOffset + code);
}

__device__ bool eval(uint8* imageData, float* response, cudaTextureObject_t* alphas)
{	
	for (int i = 0; i < detectorInfo[0].stageCount; ++i) {		
		Stage* stage = &stages[i];
		*response += evalLBP(imageData, stage, alphas);
		if (*response < stage->thetaB) {
			return false;
		}
	}	
	return true;
}

__device__ void detect(uint8* data, Detection* detections, uint32* detectionCount, cudaTextureObject_t* alphas)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < detectorInfo[0].pyramidImageWidth - CLASSIFIER_WIDTH && y < detectorInfo[0].pyramidImageHeight - CLASSIFIER_HEIGHT) {
		for (int i = 0; i < detectorInfo[0].stageCount; ++i) {
			float response = 0.0f;
			if (eval(data, &response, alphas)) {
				uint32 id = atomicInc(detectionCount, 1);

				if (id > MAX_DETECTIONS)
					return;

				Detection d;
				d.x = x;
				d.y = y;
				d.width = CLASSIFIER_WIDTH;
				d.height = CLASSIFIER_HEIGHT;
				d.response = response;
				detections[id] = d;
			}
		}
	}
}

cudaError_t runKernelWrapper(uint8* imageData, Detection* detections, uint32* detectionCount, cudaTextureObject_t* alphas)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	dim3 grid(16, 16);
	dim3 block(32, 32);
	runKernel <<<grid, block>>>(imageData, detections, detectionCount, alphas);

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

int main(int argc, char** argv)
{
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
				std::cerr << "[" << LIBNAME << "]: " << "Could not open or find the image (filename: " << filename << ")" << std::endl;

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
		cv::Mat image_bw;

		// TODO: do b&w conversion on GPU
		cvtColor(image, image_bw, CV_RGB2GRAY);				
		
		// TODO: rewrite this
		const size_t ORIG_IMAGE_SIZE = image_bw.cols * image_bw.rows * sizeof(uint8);		
		const size_t PYRAMID_IMAGE_HEIGHT = image_bw.rows * PYRAMID_IMAGE_COUNT;
		const size_t PYRAMID_IMAGE_WIDTH = image_bw.cols;
		const size_t PYRAMID_IMAGE_SIZE = PYRAMID_IMAGE_HEIGHT * PYRAMID_IMAGE_WIDTH;
					

		// ********* DEVICE VARIABLES **********
		float* devAlphaBuffer;
		uint8* devImageData;
		uint32* devDetectionCount;
		Detection* devDetections;

		// ********* HOST VARIABLES *********
		uint8* hostImageData;
		hostImageData = (uint8*)malloc(sizeof(uint8) * PYRAMID_IMAGE_SIZE);
		uint32 hostDetectionCount = 0;
		Detection hostDetections[MAX_DETECTIONS];

		// ********* CONSTANTS **********
		DetectorInfo hostDetectorInfo[1];
		hostDetectorInfo[0].imageWidth = image_bw.cols;
		hostDetectorInfo[0].imageHeight = image_bw.rows;
		hostDetectorInfo[0].pyramidImageWidth = PYRAMID_IMAGE_WIDTH;
		hostDetectorInfo[0].pyramidImageHeight = PYRAMID_IMAGE_HEIGHT;
		hostDetectorInfo[0].classifierWidth = CLASSIFIER_WIDTH;
		hostDetectorInfo[0].classifierHeight = CLASSIFIER_HEIGHT;
		hostDetectorInfo[0].alphaCount = ALPHA_COUNT;
		hostDetectorInfo[0].stageCount = STAGE_COUNT;
		
		// ********* GPU MEMORY ALLOCATION-COPY **********		
		// constant memory
		cudaMemcpyToSymbol(stages, hostStages, sizeof(Stage) * STAGE_COUNT);
		cudaMemcpyToSymbol(detectorInfo, hostDetectorInfo, sizeof(DetectorInfo));
		
		// texture memory		
		cudaMalloc(&devAlphaBuffer, STAGE_COUNT * ALPHA_COUNT * sizeof(float));		

		// create texture object
		// resource params
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeLinear;
		resDesc.res.linear.devPtr = devAlphaBuffer;
		resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
		resDesc.res.linear.desc.x = sizeof(float) * 8; // in bits
		resDesc.res.linear.sizeInBytes = STAGE_COUNT * ALPHA_COUNT * sizeof(float);

		// texture params
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.readMode = cudaReadModeElementType;

		cudaTextureObject_t texAlphas;
		cudaCreateTextureObject(&texAlphas, &resDesc, &texDesc, nullptr);

		// global memory
		cudaMalloc((void**)&devImageData, PYRAMID_IMAGE_SIZE * sizeof(uint8));
		cudaMalloc((void**)&devDetectionCount, sizeof(uint32));		
		cudaMalloc((void**)&devDetections, MAX_DETECTIONS * sizeof(Detection));

		cudaMemcpy(devImageData, image_bw.data, ORIG_IMAGE_SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(devDetectionCount, &hostDetectionCount, sizeof(uint32), cudaMemcpyHostToDevice);		

		// ********* RUN ALL THEM KERNELS! **********		

		cudaError_t cudaStatus = runKernelWrapper(
			devImageData, 
			devDetections, 
			devDetectionCount,
			&texAlphas
		);

		// ********* COPY RESULTS FROM GPU *********
		
		cudaMemcpy(&hostDetectionCount, devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost);
		cudaMemcpy(hostDetections, devDetections, hostDetectionCount * MAX_DETECTIONS, cudaMemcpyDeviceToHost);
		cudaMemcpy(hostImageData, devImageData, sizeof(uint8) * PYRAMID_IMAGE_SIZE, cudaMemcpyDeviceToHost);

		// ********* FREE CUDA MEMORY *********
		cudaFree(devDetectionCount);
		cudaFree(devImageData);
		cudaFree(devDetections);
		cudaFree(devAlphaBuffer);
		cudaDestroyTextureObject(texAlphas);

		// ********* SHOW RESULTS *********

		// pyramid image
		cv::Mat pyramidImage(cv::Size(PYRAMID_IMAGE_WIDTH, PYRAMID_IMAGE_HEIGHT), CV_8U);
		pyramidImage.data = hostImageData;

		cv::imshow("Pyramid Image", pyramidImage);
		cv::waitKey();

		// show detections
		std::cout << "Detection count: " << hostDetectionCount << std::endl;

		for (uint32 i = 0; i < hostDetectionCount; ++i) {
			cv::rectangle(pyramidImage, cvPoint(hostDetections[i].x, hostDetections[i].y), cvPoint(hostDetections[i].x + hostDetections[i].width, hostDetections[i].y + hostDetections[i].height), CV_RGB(0, 0, 0), 3);
			cv::rectangle(pyramidImage, cvPoint(hostDetections[i].x, hostDetections[i].y), cvPoint(hostDetections[i].x + hostDetections[i].width, hostDetections[i].y + hostDetections[i].height), CV_RGB(255, 255, 255));
		}
		
		cv::imshow("Detections", pyramidImage);
		cv::waitKey();

		// ******** FREE HOST MEMORY *********
		free(hostImageData);

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



