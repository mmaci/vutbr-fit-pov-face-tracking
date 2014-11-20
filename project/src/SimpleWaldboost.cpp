#include "SimpleWaldboost.h"

SimpleWaldboost::~SimpleWaldboost()
{
}


std::vector<cv::Mat> SimpleWaldboost::createPyramid(cv::Size base, cv::Size min, int octaves, int levelsPerOctave)
{
	float scale = pow(2.0f, 1.0f / levelsPerOctave);
	std::vector<cv::Mat> result;

	for (int i = 0; i < octaves; ++i) 
	{
		cv::Size size = base;
		size.width /= static_cast<float>(i + 1);
		size.height /= static_cast<float>(i + 1);

		for (int j = 0; j < levelsPerOctave; ++j)
		{
			cv::Mat image = resizeImage(size);
			result.push_back(image);

			#ifdef DEBUG
				std::cout << "Creating resized image [" << size.width << "," << size.height << "]" << std::endl;
			#endif

			size.width /= scale;
			size.height /= scale;

			if (size.width <= min.width || size.height <= min.height)
				return result;
		}
	}
	return result;
}

unsigned int SimpleWaldboost::detect(std::vector<abonDetection>* results, unsigned int size, float scale)
{
	for (std::vector<cv::Mat>::const_iterator it = _pyramid.begin(); it != _pyramid.end(); ++it)
	{
		#ifdef DEBUG
			std::cout << "Running detection on pyramid image [" << it->cols << "," << it->rows << "]" << std::endl;
		#endif
		cv::Mat image = *it;
		for (unsigned int y = 0; y < image.cols - _classifier.data().width; ++y) 
		{
			for (unsigned int x = 0; x < image.rows - _classifier.data().height; ++x)
			{
				float response;
				bool res = _classifier.eval(image, x, y, &response);
				if (res) {
					results->push_back(abonDetection(x, y, _classifier.data().width, _classifier.data().height, response, 0.0f));
					if (results->size() >= MAX_DETECTIONS)
						return results->size();
				}

			}
		}
	}
	return results->size();
}
cv::Mat SimpleWaldboost::resizeImage(int width, int height)
{
	cv::Mat image;
	cv::resize(getImage(), image, cv::Size(height, width));
	return image;
}

void SimpleWaldboost::setImage(cv::Mat const& i) {
	cv::Mat gray;
	cv::cvtColor(i, gray, CV_RGB2GRAY);
	_image = gray;
}
