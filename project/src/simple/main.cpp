#include "header.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <fstream>
#include "detector.h"

#include "SimpleWaldboost.h"

int main(int argc, char** argv) {

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
			{
				std::cerr << "Could not open or find the image: " << filename << std::endl;				
			}

			images.push_back(image);
		}
	}
	else {
		cv::Mat image = cv::imread(inputFilename.c_str(), CV_LOAD_IMAGE_COLOR);

		if (!image.data)
		{
			std::cerr << "Could not open or find the image: " << inputFilename << std::endl;			
		}

		images.push_back(image);
	}	

	SimpleWaldboost w;
	w.init();

	for (std::vector<cv::Mat>::const_iterator it = images.begin(); it != images.end(); ++it)
	{
		cv::Mat image = *it;

		w.setImage(image);
		w.createPyramids(cvSize(320, 240), cvSize(48, 48), 8, 4);

		// preloaded classfier, we dont have to load it from xml
		// and we do not support it anyways
		w.setClassifier(&detector);

		///////////////////////////////////////////////////////////////////
		// detection phase	

		std::vector<Detection> detections;
		int n = w.detect(&detections);

		// draw rectangles in place of detections
		for (std::vector<Detection>::const_iterator it = detections.begin(); it != detections.end(); ++it)
		{
			cv::rectangle(image, cvPoint(it->x, it->y), cvPoint(it->x + it->width, it->y + it->height), CV_RGB(0, 0, 0), 3);
			cv::rectangle(image, cvPoint(it->x, it->y), cvPoint(it->x + it->width, it->y + it->height), CV_RGB(255, 255, 255));
		}

		cv::imshow("Result", image);
		cv::waitKey();
	}
		
	return EXIT_SUCCESS;
}
