#ifndef H_HEADER
#define H_HEADER

#include <string>

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;

typedef char int8;
typedef short int16;
typedef int int32;

const std::string LIBNAME = "object-detector";

struct Stage {
	uint8 x, y;
	uint8 width, height;
	float thetaB;
	uint32 alphaOffset;
};

struct Detection {
	uint32 x, y;
	uint32 width, height;
	float response;
};

struct DetectorInfo {
	uint32 width, height;
	uint32 imageWidth, imageHeight;
	uint32 pyramidImageWidth, pyramidImageHeight;
	uint8 classifierWidth, classifierHeight;
	uint16 alphaCount, stageCount;
};

const uint32 MAX_DETECTIONS = 2048;
const uint32 ALPHA_COUNT = 256;
const uint32 STAGE_COUNT = 2048;
const uint8 CLASSIFIER_WIDTH = 26;
const uint8 CLASSIFIER_HEIGHT = 26;

#define PYRAMID_IMAGE_COUNT 5
#define SCALE_FACTOR 1.3f

#endif
