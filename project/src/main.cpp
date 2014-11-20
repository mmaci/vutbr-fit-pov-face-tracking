#include "header.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <aboninterface.h>
#include "facedetector.h"

#include "SimpleWaldboost.h"

int main(int argc, char** argv) {
	
	///////////////////////////////////////////////////////////////////
	// init phase

	if (argc != 2)
	{
		std::cerr << " Usage: object-detection [filename]" << std::endl;
		return EXIT_FAILURE;
	}

	cv::Mat image;
	image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
		
	if (!image.data)
	{
		std::cerr << "Could not open or find the image" << std::endl;
		return EXIT_FAILURE;
	}
	cv::imshow("Test", image);
	cv::waitKey();

	SimpleWaldboost w;
	w.init();
	w.setImage(image);
	w.setPyramid(w.createPyramid(cvSize(320, 240), cvSize(48, 48), 8, 4));
	
	// preloaded classfier, we dont have to load it from xml
	// and we do not support it anyways
	w.setClassifier(facedetector);		

	///////////////////////////////////////////////////////////////////
	// detection phase	

	std::vector<abonDetection> detections(MAX_DETECTIONS);

	// detect
	float scale = float(image.size().width) / 320;
	int n = w.detect(&detections, MAX_DETECTIONS, scale);

	// draw rectangles in place of detections
	for (std::vector<abonDetection>::const_iterator it = detections.begin(); it != detections.end(); ++it) 
	{
		cv::rectangle(image, cvPoint(it->x, it->y), cvPoint(it->x + it->width, it->y + it->height), CV_RGB(0, 0, 0), 3);
		cv::rectangle(image, cvPoint(it->x, it->y), cvPoint(it->x + it->width, it->y + it->height), CV_RGB(255, 255, 255));
	}

	cv::imshow("FINAL", image);
	cv::waitKey();
		
	return EXIT_SUCCESS;
}
