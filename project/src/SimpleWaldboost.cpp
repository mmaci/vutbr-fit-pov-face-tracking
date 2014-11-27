#include "SimpleWaldboost.h"
#include "header.h"

SimpleWaldboost::~SimpleWaldboost()
{
}

void SimpleWaldboost::createPyramids(cv::Size base, cv::Size min, int octaves, int levelsPerOctave)
{
	float scale = pow(2.0f, 1.0f / levelsPerOctave);
	_pyramids.clear();

	bool isLandscape = _image.cols > _image.rows;
	cv::Size newSize;

	for (int i = 0; i < octaves; ++i) 
	{
		cv::Size size = base;
		size.width /= static_cast<float>(i + 1);
		size.height /= static_cast<float>(i + 1);

		float scale = isLandscape ? _image.cols / size.width : _image.rows / size.width;									

		newSize.width = _image.cols / scale;
		newSize.height = _image.rows / scale;		
		
		Pyramid p;		
		for (int j = 0; j < levelsPerOctave; ++j)
		{
			PyramidImage pi;
							
			cv::resize(_image, pi.image, newSize);	
			pi.scale = scale;

			p.push_back(pi);
		
			size.width /= scale;
			size.height /= scale;

			if (size.width <= min.width || size.height <= min.height)
				break;
		}

		_pyramids.push_back(p);
	}
	
}

bool SimpleWaldboost::eval(cv::Mat* image, unsigned int x, unsigned int y, float* response)
{
	for (unsigned int i = 0; i < _classifier->stageCount; ++i)
	{
		Stage stage = _classifier->stages[i];
		*response += evalLBP(image, x, y, stage);
		if (*response < stage.thetaB)
			return false;
	}

	return *response > _classifier->threshhold;
}


/// Sum of regions in 3x3 grid.
/// Sums values in regions and stores the results in a vector.
void SimpleWaldboost::sumRegions(cv::Mat* image, unsigned int baseX, unsigned baseY, unsigned int width, unsigned int height, std::vector<unsigned int>& result)
{
	for (unsigned int i = 0; i < 9; ++i) {
		unsigned int startX = baseX + (i % 3) * width;
		unsigned int startY = baseY + (i / 3) * height;

		unsigned int acc = 0;
		for (unsigned int x = startX; x < startX + width; ++x) {
			for (unsigned int y = startY; y < startY + height; ++y) {
				acc += image->at<unsigned char>(y, x);
			}
		}

		result[i] = acc;
	}
}

/// evaluates a single LBP stage
float SimpleWaldboost::evalLBP(cv::Mat* image, unsigned x, unsigned y, Stage s) {
	static const int LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

	std::vector<unsigned int> values(9);

	sumRegions(image, x + s.x, y + s.y, s.width, s.height, values);

	int code = 0;
	for (int i = 0; i < 8; ++i)
	{
		code |= (values[LBPOrder[i]] > values[4]) << i;
	}

	return s.alpha[code];
}

unsigned int SimpleWaldboost::detect(std::vector<Detection>* results)
{	
	for (PyramidContainer::const_iterator ii = _pyramids.begin(); ii != _pyramids.end(); ++ii)
	{
		for (Pyramid::const_iterator ij = ii->begin(); ij != ii->end(); ++ij)
		{
			cv::Mat image = ij->image;
			float scale = ij->scale;

			float response = 0.0f;
			for (unsigned int x = 0; x < image.cols - _classifier->width; ++x)
			{
				for (unsigned int y = 0; y < image.rows - _classifier->height; ++y)
				{			
					response = 0.0f;
					if (eval(&image, x, y, &response))
					{					
						results->push_back(Detection(x * scale, y * scale, _classifier->width * scale, _classifier->height * scale, response));
						if (results->size() >= MAX_DETECTIONS)
							return results->size();
					}

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
