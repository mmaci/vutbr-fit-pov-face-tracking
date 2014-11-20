#ifndef H_WEAK_CLASSIFIER
#define H_WEAK_CLASSIFIER

#include <string>

class WeakClassifier
{
public:
	// constructors
	WeakClassifier() { };
	WeakClassifier(std::string filename) : _filename(filename)
	{
		set(filename);
	};
	WeakClassifier(TClassifier classifier) : _classifier(classifier) { }

	// setters, getters
	void set(std::string filename)
	{
		// TODO: load classifier from xml file
	}

	void set(ClassifierType type, unsigned int stageCount, unsigned int evaluatedStageCount, unsigned int alphaCount, float threshold, unsigned int width, unsigned int height, void* stages, float* alphas)
	{
		TClassifier c = { type, stageCount, evaluatedStageCount, alphaCount, threshold, width, height, stages, alphas };
		_classifier = c;
	};
	void set(TClassifier const& classifier) { _classifier = classifier; }

	TClassifier data() const { return _classifier; }

	// evaluation methods

	/// evaluates the whole image
	bool eval(cv::Mat image, unsigned int x, unsigned int y, float* response) {
		// only implement LBP atm.
//		if (_classifier.tp == LBP) {
			TStage* s = reinterpret_cast<TStage*>(_classifier.stage);
			for (TStage* stage = s; stage < s + _classifier.stageCount; ++stage) {
				*response += evalLBP(image, x, y, stage);
				if (*response < stage->theta_b)
					return false;
			}
	//	}

		return (*response > _classifier.threshold) ? true : false;
	}
	
	/// Sum of regions in 3x3 grid.
	/// Sums values in regions and stores the results in a vector.
	static void sumRegions3x3(unsigned char* data, int width, int height, unsigned int widthStep, int* result)
	{
		unsigned blockStep = height * widthStep;
		widthStep -= width; // move ptr to the beginning of nextline window (we are at the end of the window, when we move)

		// Prepare pointer array
		unsigned char* base[9] = {
			data,							data + width,							data + width + width,
			data + blockStep,				data + blockStep + width,				data + blockStep + width + width,
			data + blockStep + blockStep,	data + blockStep + blockStep + width,	data + blockStep + blockStep + width + width
		};

		for (int y = 0; y < height; ++y)
		{			
			for (int x = 0; x < width; ++x)
			{
				for (int i = 0; i < 9; ++i)
				{
					result[i] += *base[i];
					// move pointer to the next element
					// used next x iteration
					++base[i];
				}				
			}
			// set pointers to next line 
			// widthStep is decreased by width
			// because we increment base[i] every iteration
			for (int i = 0; i < 9; ++i)
				base[i] += widthStep;
		}
	}

	/// evaluates a single LBP stage
	float evalLBP(cv::Mat image, unsigned x, unsigned y, void* s) {
		TStage* stage = reinterpret_cast<TStage*>(s);

		static const int LBPOrder[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

		int values[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

		unsigned char* base = image.data + y * image.rows + x;
		sumRegions3x3(base, stage->w, stage->h, image.cols, values);
		
		int code = 0;
		for (int i = 0; i < 8; ++i)
		{
			code |= (values[LBPOrder[i]] > values[4]) << i;
		}
		
		return _classifier.alpha[code];
	}	

private:
	std::string _filename;
	TClassifier _classifier;
};

#endif
