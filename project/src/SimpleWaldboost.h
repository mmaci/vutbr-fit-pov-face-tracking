#ifndef H_SIMPLE_WALDBOOST
#define H_SIMPLE_WALDBOOST

#include "header.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <aboninterface.h>

#include "WeakClassifier.h"

class SimpleWaldboost
{
public:	
	SimpleWaldboost() { };
	SimpleWaldboost(WeakClassifier c) : _classifier(c) { };
	~SimpleWaldboost();	

	void setScale(float const& s) { _scale = s; }
	float getScale() const { return _scale; }

	void setLevels(int const& l) { _levels = l; }
	int getLevels() const { return _levels; }	

	void setImage(cv::Mat const& i);
	cv::Mat getImage() const { return _image; }

	void setPyramid(std::vector<cv::Mat> const& p){ _pyramid = p; }
	std::vector<cv::Mat> getPyramid() const { return _pyramid; }

	void setClassifier(WeakClassifier classifier) { _classifier = classifier; }
	WeakClassifier getClassifier() const { return _classifier; }

	std::vector<cv::Mat> createPyramid(cv::Size base, cv::Size min, int octaves, int levelsPerOctave);

	cv::Mat resizeImage(int width, int height);
	cv::Mat resizeImage(cv::Size size) { return resizeImage(size.width, size.height); };
	
	void init() { };
	
	unsigned int detect(std::vector<abonDetection>* results, unsigned int size, float scale);
	

private:
	WeakClassifier _classifier;
	float _scale;
	int _levels;
	cv::Mat _image;
	std::vector<cv::Mat> _pyramid;
};

#endif
