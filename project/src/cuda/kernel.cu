
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>

#include "header.h"
#include "detector.h"
#include "alphas.h"

#include <algorithm>
#include <vector>

struct Person {
	bool active;
	cv::MatND hist;
	uint8 color[3];
	size_t id;
	Detection det;
	CvPoint lastPoint;
};

/// wrapper to call kernels
cudaError_t runKernelWrapper(uint8* /* device image */, Detection* /* device detection buffer */, uint32* /* device detection count */, Bounds*, const size_t);

/// runs object detectin on gpu itself
__device__ void detect(uint8* /* device image */, Detection* /* detections */, uint32* /* detection count */, uint16 /* starting stage */, uint16 /* ending stage */, Bounds*);
/// gpu bilinear interpolation
__device__ void bilinearInterpolation(uint8* /* output image */, const float /* scale */);
/// builds a pyramid image with parameters set in header.h
__device__ void buildPyramid(uint8* /* device image */, uint32, uint32, uint32, uint32, Bounds*, uint32, uint32);

/// detector stages
__constant__ Stage stages[STAGE_COUNT];
/// detector parameters
__constant__ DetectorInfo detectorInfo[1];

/// pyramid kernel

texture<uint8> textureOriginalImage;
texture<uint8> texturePyramidImage;
texture<float> textureAlphas;

/// tracking
std::vector<Person> persons;
//Person *persons[MAX_PERSONS];
//uint8 personsCount = 0;
uint32 param = OPT_ALL;

__global__ void pyramidImageKernel(uint8* imageData, Bounds* bounds)
{
	buildPyramid(imageData, 320, 240, 48, 48, bounds, 8, 4);
}

__device__ void buildPyramid(uint8* imageData, uint32 max_x, uint32 max_y, uint32 min_x, uint32 min_y, Bounds* bounds, uint32 octaves, uint32 levels_per_octave)
{
	// coords in the original image
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// only index data in the original image
	if (x < (detectorInfo[0].imageWidth - 1) && y < (detectorInfo[0].imageHeight - 1))
	{

		float scaling_factor = pow(2.0f, 1.0f / levels_per_octave);
		bool is_landscape = detectorInfo[0].imageWidth > detectorInfo[0].imageHeight;

		uint32 init_offset = detectorInfo[0].pyramidImageWidth * detectorInfo[0].imageHeight;
		uint32 init_y_offset = detectorInfo[0].imageHeight;
		uint32 init_x_offset = 0;

		uint32 offset, y_offset = init_y_offset, x_offset;
		for (uint8 octave = 0; octave < octaves; ++octave)
		{
			uint32 max_width = max_x / (octave + 1);
			uint32 max_height = max_y / (octave + 1);

			// box to which fit the resized image
			float current_scale = is_landscape ? (float)detectorInfo[0].imageWidth / (float)max_width : (float)detectorInfo[0].imageHeight / (float)max_height;

			uint32 image_width = detectorInfo[0].imageWidth / current_scale;
			uint32 image_height = detectorInfo[0].imageHeight / current_scale;

			// set current X-offset to the beginning and total offset based on current octave
			x_offset = init_x_offset;
			offset = init_offset;
			for (uint8 i = 0; i < octave; ++i)
				offset += (max_y / (i + 1)) * detectorInfo[0].pyramidImageWidth;

			// set starting scale based on current octave		
			uint32 final_y_offset = image_height;

			// process all levels of the pyramid
			for (uint8 level = 0; level < levels_per_octave; ++level)
			{
				bilinearInterpolation(imageData + offset, current_scale);

				if (x == 0 && y == 0) {
					uint32 bounds_id = levels_per_octave * octave + level;
					bounds[bounds_id].offset = offset;
					bounds[bounds_id].y_offset = y_offset;
					bounds[bounds_id].x_offset = x_offset;
					bounds[bounds_id].width = image_width;
					bounds[bounds_id].height = image_height;
					bounds[bounds_id].scale = current_scale;
				}

				current_scale *= scaling_factor;
				x_offset += image_width;
				offset += image_width;

				image_width = (float)detectorInfo[0].imageWidth / current_scale;
				image_height = (float)detectorInfo[0].imageHeight / current_scale;

				if (image_width < min_x || image_height < min_y)
					break;
			}

			y_offset += final_y_offset;
		}
	}
}


/// detection kernels

__global__ void detectionKernel1(uint8* imageData, Detection* detections, uint32* detectionCount, Bounds* bounds)
{
	detect(imageData, detections, detectionCount, 0, 2048, bounds);
}

__device__ void bilinearInterpolation(uint8* outImage, float scale)
{
	const int origX = blockIdx.x*blockDim.x + threadIdx.x;
	const int origY = blockIdx.y*blockDim.y + threadIdx.y;

	const int x = (float)origX / scale;
	const int y = (float)origY / scale;

	uint8 res = tex1Dfetch(textureOriginalImage, origY * detectorInfo[0].imageWidth + origX);

	outImage[y * detectorInfo[0].pyramidImageWidth + x] = res;
}

__device__ void sumRegions(uint8* imageData, uint32 x, uint32 y, Stage* stage, uint32* values)
{
	values[0] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	x += stage->width;
	values[1] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	x += stage->width;
	values[2] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	y += stage->height;
	values[5] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	y += stage->height;
	values[8] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	x -= stage->width;
	values[7] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	x -= stage->width;
	values[6] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	y -= stage->height;
	values[3] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
	x += stage->width;
	values[4] = tex1Dfetch(texturePyramidImage, y * detectorInfo[0].imageWidth + x);
}

__device__ float evalLBP(uint8* data, uint32 x, uint32 y, Stage* stage)
{
	const uint8 LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

	uint32 values[9];

	sumRegions(data, x, y, stage, values);

	uint8 code = 0;
	for (uint8 i = 0; i < 8; ++i)
		code |= (values[LBPOrder[i]] > values[4]) << i;

	return tex1Dfetch(textureAlphas, stage->alphaOffset + code);
}

__device__ bool eval(uint8* imageData, uint32 x, uint32 y, float* response, uint16 startStage, uint16 endStage)
{
	for (uint16 i = startStage; i < endStage; ++i) {
		Stage stage = stages[i];
		*response += evalLBP(imageData, x + stage.x, y + stage.y, &stage);
		if (*response < stage.thetaB) {
			return false;
		}
	}

	// final waldboost threshold
	return *response > FINAL_THRESHOLD;
}

__device__ void detect(uint8* imageData, Detection* detections, uint32* detectionCount, uint16 startStage, uint16 endStage, Bounds* bounds)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if (x < (detectorInfo[0].pyramidImageWidth - detectorInfo[0].classifierWidth) && y < (detectorInfo[0].pyramidImageHeight - detectorInfo[0].classifierHeight))
	{
		float response = 0.0f;
		if (eval(imageData, x, y, &response, startStage, endStage)) {

			Bounds b;
			for (uint8 i = 0; i < 8 * 3; ++i) {
				if (x >= bounds[i].x_offset && x < (bounds[i].x_offset + bounds[i].width) &&
					y >= bounds[i].y_offset && y < (bounds[i].y_offset + bounds[i].height)) {
					b = bounds[i];
					break;
				}
			}

			uint32 pos = atomicInc(detectionCount, 2048);
			detections[pos].x = (float)(x - b.x_offset) * b.scale;
			detections[pos].y = (float)(y - b.y_offset) * b.scale;
			detections[pos].width = detectorInfo[0].classifierWidth * b.scale;
			detections[pos].height = detectorInfo[0].classifierHeight * b.scale;
			detections[pos].response = response;
		}

	}
}

cudaError_t runKernelWrapper(uint8* imageData, Detection* detections, uint32* detectionCount, Bounds* bounds, const size_t pyramidImageSize)
{
	cudaEvent_t start_detection, stop_detection, start_pyramid, stop_pyramid;
	cudaEventCreate(&start_detection);
	cudaEventCreate(&stop_detection);
	cudaEventCreate(&start_pyramid);
	cudaEventCreate(&stop_pyramid);

	float pyramid_time = 0.f, detection_time = 0.f;

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);

	dim3 grid(32, 128);
	dim3 block(32, 32);

	if (param & OPT_TIMER)
		cudaEventRecord(start_pyramid);

	pyramidImageKernel << <grid, block >> > (imageData, bounds);

	if (param & OPT_TIMER)
	{
		cudaEventRecord(stop_pyramid);
		cudaEventSynchronize(stop_pyramid);
		cudaEventElapsedTime(&pyramid_time, start_pyramid, stop_pyramid);
		printf("PyramidKernel time: %f ms\n", pyramid_time);
	}

	cudaThreadSynchronize();

	// bind created pyramid to texture memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8>();
	cudaBindTexture(nullptr, &texturePyramidImage, imageData, &channelDesc, sizeof(uint8) * pyramidImageSize);

	cudaEventRecord(start_detection);
	detectionKernel1 << <grid, block >> >(imageData, detections, detectionCount, bounds);

	cudaUnbindTexture(texturePyramidImage);

	cudaEventRecord(stop_detection);
	cudaEventSynchronize(stop_detection);
	cudaEventElapsedTime(&detection_time, start_detection, stop_detection);

	if (param & OPT_TIMER)
	{
		printf("DetectionKernel time: %f ms\n", detection_time);
		printf("Total time: %f ms \n", pyramid_time + detection_time);
	}	

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

bool checkAndOverwrite(Detection* check, Detection* target)
{
	int32 x_overlap = std::max(0, (int32)(std::min(check->x + check->width, target->x + target->width) - std::max(check->x, target->x)));
	int32 y_overlap = std::max(0, (int32)(std::min(check->y + check->height, target->y + target->height) - std::max(check->y, target->y)));
	//std::cout << x_overlap << " a " <<  y_overlap << std::endl;

	//std::cout << (x_overlap*y_overlap) << " check " << (check->width*check->height) << " target " << (target->width*target->height) << std::endl;
	uint32 s = (x_overlap*y_overlap);
	if (((s / (float)(check->width*check->height)) > OVERLAY) || ((s / (float)(target->width*target->height)) > OVERLAY)){
		if (target->response < check->response){
			target->x = check->x;
			target->y = check->y;
			target->width = check->width;
			target->height = check->height;
			target->response = check->response;
			//target = check;

		}
		return true;
	}
	else
	{
		return false; //neprekryvaji se dostatecne
	}

}

void separateDetections(Detection *detections, uint32 count, std::vector<Detection> *separate)
{
	//int32 spCount = 0;
	if (count > 0){
		//prvni priradim
		separate->push_back(detections[0]);
		for (uint32 i = 1; i < count; ++i) {
			//porovnani s predchazejicima jestli se neprekryvaji
			bool overwrite = false;
			for (uint32 j = 0; j < separate->size(); j++){
				//pokud prekryvaji vyberu ten s vetsim
				//Detection* det = &separate->at(j);
				if (checkAndOverwrite(&detections[i], &separate->at(j))){
					overwrite = true;
					break;
				}
			}
			//pokud neprekryvaji ulozim na novou pozici
			if (!overwrite)	separate->push_back(detections[i]);
		}

	}
}

double getScore(cv::MatND *hist, uint32 personID, Detection det, uint32 cols, uint32 rows){
	double score = compareHist(*hist, persons[personID].hist, CV_COMP_BHATTACHARYYA);
	//score += (1 - compareHist(hist, persons[k].hist, CV_COMP_CORREL));
	uint32 x1, x2, y1, y2;
	x1 = persons[personID].lastPoint.x;
	x2 = det.x + det.width / 2;
	y1 = persons[personID].lastPoint.y;
	y2 = det.y + det.height / 2;
	//std::cout << x1 << " " << y1 << " ; " << x2 << " " << y2 << "(x-x)^2=" << pow((int32)(x1 - x2), 2) << "(y-y)^2=" << pow((int32)(y1 - y2), 2) << std::endl;
	
	return score + sqrt(pow((int32)(x1 - x2), 2) + pow((int32)(y1 - y2), 2)) / (2 * sqrt(pow(cols, 2) + pow(rows, 2)));  //vzdalenost (hodnoty 0 - 0,5)
}

void addNewPerson(cv::Mat *image, cv::MatND *hist, Detection det)
{
	Person p;
	p.active = true;
	p.color[0] = rand() % 255;
	p.color[1] = rand() % 255;
	p.color[2] = rand() % 255;

	p.id = persons.size();
	//p.descriptor = desc;
	p.hist = *hist;
	p.det = det;
	persons.push_back(p);
	if (param & OPT_OUTPUT_FACES)
	{
		cv::Mat imageROI = cv::Mat(*image, cv::Rect(det.x, det.y, det.width, det.height));
		char filename[256];
		sprintf(filename, "face%i.jpg", p.id);
		cv::imwrite(filename, imageROI);
	}

}

void getHistogram(cv::Mat *image, cv::MatND *hist, Detection det){
	cv::Mat faceImg = image->clone();

	faceImg = cv::Mat(faceImg, (cv::Rect(det.x, det.y, det.width, det.height)));//vyber ROI
	faceImg.copyTo(faceImg); //zkopirovani jen ROI

	cvtColor(faceImg, faceImg, cv::COLOR_BGR2HSV);
	int h_bins = 60; int s_bins = 60;// int v_bins = 50;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 360 };
	float s_ranges[] = { 0, 256 };
	//float v_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	cv::calcHist(&faceImg, 1, channels, cv::Mat(), *hist, 2, histSize, ranges, true, false);
	cv::normalize(*hist, *hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
	
}

bool runDetector(cv::Mat* image, std::ofstream *output)
{
	cv::Mat image_bw;

	// TODO: do b&w conversion on GPU
	cvtColor(*image, image_bw, CV_BGR2GRAY);

	// TODO: rewrite this
	const size_t ORIG_IMAGE_SIZE = image_bw.cols * image_bw.rows * sizeof(uint8);
	const size_t PYRAMID_IMAGE_HEIGHT = image_bw.rows * 3;
	const size_t PYRAMID_IMAGE_WIDTH = image_bw.cols;
	const size_t PYRAMID_IMAGE_SIZE = PYRAMID_IMAGE_HEIGHT * PYRAMID_IMAGE_WIDTH;


	// ********* DEVICE VARIABLES **********
	float* devAlphaBuffer;
	uint8* devImageData, *devOriginalImage;
	uint32* devDetectionCount;
	Detection* devDetections;
	Bounds* devBounds;

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
	cudaMemcpy(devAlphaBuffer, alphas, STAGE_COUNT * ALPHA_COUNT * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&devImageData, PYRAMID_IMAGE_SIZE * sizeof(uint8));
	cudaMalloc(&devOriginalImage, ORIG_IMAGE_SIZE * sizeof(uint8));
	cudaMalloc(&devDetectionCount, sizeof(uint32));
	cudaMalloc(&devDetections, MAX_DETECTIONS * sizeof(Detection));
	cudaMalloc(&devBounds, PYRAMID_IMAGE_COUNT * sizeof(Bounds));

	uint8* clean = (uint8*)malloc(PYRAMID_IMAGE_SIZE * sizeof(uint8));
	memset(clean, 0, PYRAMID_IMAGE_SIZE * sizeof(uint8));
	cudaMemcpy(devImageData, clean, PYRAMID_IMAGE_SIZE * sizeof(uint8), cudaMemcpyHostToDevice);
	free(clean);

	cudaMemcpy(devImageData, image_bw.data, ORIG_IMAGE_SIZE * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaMemcpy(devOriginalImage, image_bw.data, ORIG_IMAGE_SIZE * sizeof(uint8), cudaMemcpyHostToDevice);
	cudaMemcpy(devDetectionCount, &hostDetectionCount, sizeof(uint32), cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8>();
	cudaBindTexture(nullptr, &textureOriginalImage, devOriginalImage, &channelDesc, sizeof(uint8) * ORIG_IMAGE_SIZE);

	cudaChannelFormatDesc alphaChannelDesc = cudaCreateChannelDesc<float>();
	cudaBindTexture(nullptr, &textureAlphas, devAlphaBuffer, &alphaChannelDesc, STAGE_COUNT * ALPHA_COUNT * sizeof(float));

	// ********* RUN ALL THEM KERNELS! **********		

	cudaError_t cudaStatus = runKernelWrapper(
		devImageData,
		devDetections,
		devDetectionCount,
		devBounds,
		PYRAMID_IMAGE_SIZE
		);

	// ********* COPY RESULTS FROM GPU *********

	cudaMemcpy(&hostDetectionCount, devDetectionCount, sizeof(uint32), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostDetections, devDetections, hostDetectionCount * sizeof(Detection), cudaMemcpyDeviceToHost);
	cudaMemcpy(hostImageData, devImageData, sizeof(uint8) * PYRAMID_IMAGE_SIZE, cudaMemcpyDeviceToHost);

	// ********* FREE CUDA MEMORY *********
	cudaUnbindTexture(textureOriginalImage);
	cudaUnbindTexture(textureAlphas);
		
	cudaFree(devImageData);
	cudaFree(devOriginalImage);	
	cudaFree(devDetections);
	cudaFree(devDetectionCount);
	cudaFree(devAlphaBuffer);
	cudaFree(devBounds);

	// ********* SHOW RESULTS *********	

	if (param & OPT_VERBOSE)
		std::cout << "Detection count: " << hostDetectionCount << std::endl;


	if (param & OPT_TRACKING)
	{
		std::vector<Detection> separate;
		separateDetections(hostDetections, hostDetectionCount, &separate);
		bool emptyPersons = persons.size() == 0;

		for (uint32 i = 0; i < separate.size(); i++){
			// TODO: remove this
			// temporary fix for some weird asymetrical detections
			// just ignore these detections
			if ((separate[i].width != 0 || separate[i].height != 0) && separate[i].width == separate[i].height)
			{
				if (separate[i].x + separate[i].width >= (uint32)image->cols){
					separate[i].width = image->cols - separate[i].x;
				}
				if (separate[i].y + separate[i].height >= (uint32)image->rows){
					separate[i].height = image->rows - separate[i].y;
				}
				//std::cout << "pred obrazkem " << i << " x: " << separate[i].x << " y: " << separate[i].y << " wi: " << separate[i].width << " he: " << separate[i].height << " cols " << image->cols << " rows " << image->rows << std::endl;
				cv::MatND hist;
				getHistogram(image, &hist, separate[i]);



				//desc.convertTo(desc, CV_32F);
				if (emptyPersons){ //prvni snimek s detekci
					//create new one and random color
					//std::cout << "novy " << i << std::endl;
					addNewPerson(image, &hist, separate[i]);
				}
				else //if (false)  //porovnani
				{

					//std::cout << "porovnani" << i << std::endl;

					double minS = 1000.0;
					size_t minIndex = persons.size();
					for (uint32 k = 0; k < persons.size(); ++k) {
						if (!persons[k].active){
							double score = getScore(&hist, k, separate[i], image->cols, image->rows);
							if (score < minS){
								minS = score;
								minIndex = k;
							}
						}
					}
					//std::cout << "score " << minS << std::endl;
					if ((minS < MAX_SCORE) && (minIndex < persons.size())){
						persons[minIndex].active = true;
						persons[minIndex].hist = hist;
						persons[minIndex].det = separate[i];
						//nalezeno = true;
					}
					else {
						//std::cout << "Nenalezeno " << i << std::endl;
						addNewPerson(image, &hist, separate[i]);
					}
				}
			}//if ((separate[i].width != 0 || 
		}//for (uint32 i = 0; i < separate.size();
		for (uint32 i = 0; i < persons.size(); ++i)
		{
			if (param & OPT_VERBOSE)
				std::cout << "[" << persons[i].det.x << "," << persons[i].det.y << "," << persons[i].det.width << "," << persons[i].det.height << "] " << persons[i].det.response << ", ";

			if (persons[i].active){
				//zapis do souboru a vykresleni
				persons[i].lastPoint = cvPoint(persons[i].det.x + persons[i].det.width / 2, persons[i].det.y + persons[i].det.height / 2);
				if (output->is_open())//pokud zadam output file
				{
					//        id osoby               stred pozice (x)                     y                                             
					*output << i << ";" << persons[i].lastPoint.x << ";" << persons[i].lastPoint.y << ";" << persons[i].det.response << std::endl;
				}
				cv::rectangle(*image, cvPoint(persons[i].det.x, persons[i].det.y), cvPoint(persons[i].det.x + persons[i].det.width, persons[i].det.y + persons[i].det.height), CV_RGB(persons[i].color[0], persons[i].color[1], persons[i].color[2]), 1);

				persons[i].active = false;
			}
		}
		if (output->is_open())//pokud zadam output file
		{
			//konec snimku  
			*output << "------end of frame-------" << std::endl;
		}

	}
	else
	{
		for (uint32 i = 0; i < hostDetectionCount; ++i)
		{
			if (param & OPT_VERBOSE)
				std::cout << "[" << hostDetections[i].x << "," << hostDetections[i].y << "," << hostDetections[i].width << "," << hostDetections[i].height << "] " << hostDetections[i].response << ", ";

			if (param & OPT_VISUAL_OUTPUT)
				cv::rectangle(*image, cvPoint(hostDetections[i].x, hostDetections[i].y), cvPoint(hostDetections[i].x + hostDetections[i].width, hostDetections[i].y + hostDetections[i].height), CV_RGB(0, 255, 0), 1);
		}
	}
	// ******** FREE HOST MEMORY *********
	free(hostImageData);

	if (cudaStatus != cudaSuccess) {
		std::cerr << "[" << LIBNAME << "]: " << "CUDA runtime error" << std::endl;;
		return false;
	}

	// needed for profiling - NSight
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::cerr << "[" << LIBNAME << "]: " << "cudaDeviceReset failed" << std::endl;;
		return false;
	}

	return true;
}

bool process(std::string inFilename, Filetypes inFileType, std::string outFilename) {
	
	std::ofstream output;
	if (!outFilename.empty())
	{		
		output.open(outFilename, std::ios::out);
		if (!output.is_open())
		{
			std::cerr << "Could not open output file (filename: " << outFilename << ")" << std::endl;
			return false;
		}
		else
		{
			//        id osoby               stred pozice (x)                                         y                         
			output << "id;pozice x; pozice y; response" << std::endl;
		}
	}

	cv::Mat image;
	switch (inFileType)
	{
	case INPUT_IMAGE:
	{
		image = cv::imread(inFilename.c_str(), CV_LOAD_IMAGE_COLOR);

		if (!image.data)
			std::cerr << "[" << LIBNAME << "]: " << "Could not open or find the image (filename: " << inFilename << ")" << std::endl;

		runDetector(&image, &output);

		if (param & OPT_VISUAL_OUTPUT)
		{
			cv::imshow(LIBNAME, image);
			cv::waitKey(WAIT_DELAY);
		}

		break;
	}
	case INPUT_DATASET:
	{
		std::ifstream in;
		in.open(inFilename);
		std::string file;
		while (!in.eof())
		{
			std::getline(in, file);
			image = cv::imread(file.c_str(), CV_LOAD_IMAGE_COLOR);

			if (!image.data)
			{
				std::cerr << "[" << LIBNAME << "]: " << "Could not open or find the image (inFilename: " << file.c_str() << ")" << std::endl;
				continue;
			}

			runDetector(&image, &output);

			if (param & OPT_VISUAL_OUTPUT)
			{
				cv::imshow(LIBNAME, image);
				cv::waitKey(WAIT_DELAY);
			}
		}
		break;
	}
	case INPUT_VIDEO:
	{
		cv::VideoCapture video;

		video.open(inFilename);
		while (true) {
			video >> image;

			if (image.empty())
				break;

			runDetector(&image, &output);

			if (param & OPT_VISUAL_OUTPUT)
			{
				cv::imshow(LIBNAME, image);
				cv::waitKey(WAIT_DELAY);
			}
		}
		video.release();
		break;
	}
	default:
		return false;
	}

	if (output.is_open())
	{
		output << "List of faces in video/dataset: " << std::endl;
		for (uint32 i = 0; i < persons.size(); i++)
			output << i << ";(" << (int)persons[i].color[0] << "," << (int)persons[i].color[1] << "," << (int)persons[i].color[2] << ")" << std::endl;		
		
		output.close();
	}	

	return true;
}

int main(int argc, char** argv)
{
	std::string inputFilename;
	std::string outputFilename;
	Filetypes mode;
	for (int i = 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == "-ii" && i + 1 < argc) {
			mode = INPUT_IMAGE;
			inputFilename = argv[++i];
		}
		else if (std::string(argv[i]) == "-di" && i + 1 < argc) {
			mode = INPUT_DATASET;
			inputFilename = argv[++i];
		}
		else if (std::string(argv[i]) == "-iv" && i + 1 < argc) {
			mode = INPUT_VIDEO;
			inputFilename = argv[++i];
		}
		else if (std::string(argv[i]) == "-ot" && i + 1 < argc) {

			outputFilename = argv[++i];
		}
		else {
			std::cerr << "Usage: " << argv[0] << " -ii [input file] or -di [dataset] or -iv [input video] and -ot [output track info]" << std::endl;
			return EXIT_FAILURE;
		}
	}

	process(inputFilename, mode, outputFilename);

	return EXIT_SUCCESS;
}



