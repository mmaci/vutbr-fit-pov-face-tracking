#ifndef H_SIMPLE_WALDBOOST
#define H_SIMPLE_WALDBOOST

#include "header.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class SimpleWaldboost
{
public:	
	SimpleWaldboost() { };
	SimpleWaldboost(Classifier* c) : _classifier(c) { };
	~SimpleWaldboost();	

	void setImage(cv::Mat const& i);
	cv::Mat getImage() const { return _image; }

	void setClassifier(Classifier* classifier) { _classifier = classifier; }
	Classifier* getClassifier() const { return _classifier; }

	cv::Mat resizeImage(int width, int height);
	cv::Mat resizeImage(cv::Size size) { return resizeImage(size.width, size.height); };
	
	void init() { };
	void createPyramids(cv::Size base, cv::Size min, int octaves, int levelsPerOctave);
	
	unsigned int detect(std::vector<Detection>* results);

	bool eval(cv::Mat* image, unsigned int x, unsigned int y, float* response);
	void sumRegions(cv::Mat* image, unsigned int baseX, unsigned baseY, unsigned int width, unsigned int height, std::vector<unsigned int>& result);
	float evalLBP(cv::Mat* image, unsigned x, unsigned y, Stage s);
	

private:
	Classifier* _classifier;

	cv::Mat _image;
	PyramidContainer _pyramids;
};

#endif
