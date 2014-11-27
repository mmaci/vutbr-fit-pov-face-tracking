#ifndef H_HEADER
#define H_HEADER

#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define MAX_DETECTIONS 2000
#define DEBUG 1

struct PyramidImage {
	cv::Mat image;
	float scale;	
};

typedef std::vector< PyramidImage > Pyramid;
typedef std::vector< Pyramid > PyramidContainer;

struct Stage {
	unsigned int x, y;
	unsigned int width, height;
	float thetaB;
	float* alpha;
};

enum ClassifierType {
	LBP = 1,
};

struct Classifier {
	ClassifierType type;
	unsigned int stageCount;
	unsigned int alphaCount;
	float threshhold;
	unsigned int width;
	unsigned int height;
	Stage* stages;
	float* alphas;
};

struct Detection {
	Detection(){};
	Detection(unsigned int posx, unsigned int posy, unsigned int w, unsigned int h, float r) :
		x(posx), y(posy), width(w), height(h), response(r)
	{};

	unsigned int x, y;
	unsigned int width, height;
	float response;
};

#endif