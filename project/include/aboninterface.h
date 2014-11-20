//-----------------------------------------------------------------------------
// aboninterface.h
// Interface to adaboost evaluator plugin
// Jiri Havel, DCGM FIT BUT, Brno
// $Id: aboninterface.h 161 2009-08-28 09:06:15Z ihavel $
//-----------------------------------------------------------------------------
#ifndef _ABON_INTERFACE_H
#define _ABON_INTERFACE_H

#include "classifier.h"
#include "timer.h"

#ifdef __cplusplus
extern "C" {
#endif

enum abonErrorCode
{
    aboneOk = 0,
    aboneParam, //spatny parametr
    aboneMemory, //nedostatek pamìti
    aboneSize, //obrázek je jinak velký, než bylo nastavené abonSetSize
    aboneError, //obecna chyba
    aboneNumErrorCodes
};

enum abonTimeStamps
{
    abontsTotal,
    abontsPyramid,
    abontsConvolution,
    abontsDetection,
    abontsAux1,
    abontsAux2,
    abontsNumTimeStamps // totok musi byt posledni
};

//Forward deklarace struktur

struct abonDetector;
struct abonImageData;
struct abonTimeStamp;
struct abonDetection;

//Callbacky

typedef abonImageData *abonCbGetImage(void *param);
typedef abonErrorCode abonCbPutImage(void *param, abonImageData *image);

//Rozhranní knihovny

//Knihovna exportuje jen abonCreate
typedef abonDetector *abonFunCreate();

//Ukazatele na tyhle funkce nacpe abonInit do struktury abonDetector
typedef abonErrorCode abonFunSetVerbosity(abonDetector *detector, unsigned verbosity);
typedef abonErrorCode abonFunSetTimer(abonDetector *detector, Timer *timer);
typedef abonErrorCode abonFunSetClassifier(abonDetector *detector, TClassifier *classifier);
typedef abonErrorCode abonFunSetStep(abonDetector *detector, unsigned hstep, unsigned vstep);
typedef abonErrorCode abonFunSetSize(abonDetector *detector, unsigned width, unsigned height, unsigned pitch);
typedef abonErrorCode abonFunSetScale(abonDetector *detector, float start, float step, unsigned count);
typedef abonErrorCode abonFunSetRotation(abonDetector *detector, float start, float step, unsigned count);
typedef abonErrorCode abonFunSetArguments(abonDetector *detector, unsigned argc, const char *argv[]);
typedef abonErrorCode abonFunAttach(abonDetector *detector, abonDetector *attachment);
typedef abonErrorCode abonFunProcess(abonDetector *detector, abonImageData *image);
typedef abonErrorCode abonFunStart(abonDetector *detector, abonCbGetImage *get, abonCbPutImage *put, void *param);
typedef abonErrorCode abonFunRelease(abonDetector *detector);
typedef abonErrorCode abonFunTimestampDescription(abonDetector *detector, abonTimeStamps *timestamp, const char **description);

struct abonDetector
{
    void *library;//< dll/so handle, set by abonLoad
    abonFunCreate *create;
    //nastavení detektoru
    abonFunSetVerbosity *setVerbosity;
    abonFunSetTimer *setTimer;
    abonFunSetClassifier *setClassifier;
    abonFunSetStep *setStep;
    abonFunSetSize *setSize;
    abonFunSetScale *setScale;
    abonFunSetRotation *setRotation;
    abonFunSetArguments *setArguments;
    abonFunAttach *attach;
    //zpracování
    abonFunProcess *process;
    abonFunStart *start;
    //úklid
    abonFunRelease *release;
    abonFunTimestampDescription *timestampDescription;
    //nastavené hodnoty
    unsigned verbosity;
    Timer *timer;
    TClassifier *classifier;
    unsigned hstep, vstep;
    unsigned width, height, pitch;
    //vlastní vìci si knihovna pøidá za konec
};

/// abonDetector initial value. Use instead of memset in plugins.
const abonDetector abonDetectorInitializer = {
    NULL, NULL,//handle, create
    NULL, NULL,//verbosity, timer
    NULL, NULL, NULL, NULL, NULL, NULL, NULL,//klasifikátor, velikost, krok, scale, rotace, argumenty, attach
    NULL, NULL,//process, start
    NULL, NULL,//úklid, popis
    //hodnoty
    0, NULL,//verbosita, timer
    NULL,//klasifikátor
    1, 1,//krok
    0, 0, 0,//velikost
};

struct abonTimeStamp
{
    double start;
    double end;

#ifdef __cplusplus
    abonTimeStamp() throw() {}
    abonTimeStamp(const double _s, const double _e) throw()
    : start(_s), end(_e) {}
#endif 
};

struct abonDetection
{
    unsigned x,y;//levy horni roh
    unsigned width,height;
    float response;
    float angle;

#ifdef __cplusplus
    abonDetection() throw() {}
    abonDetection(const unsigned _x, const unsigned _y, const unsigned _w, const unsigned _h, const float _r, const float _a) throw()
    : x(_x), y(_y), width(_w), height(_h), response(_r), angle(_a) {}
#endif
};

struct abonImageData
{
    // image data
    unsigned width, height, pitch;
    char *data;

    // other stuff
    unsigned imageID;

    // timestamps
    abonTimeStamp timestamps[abontsNumTimeStamps];

    //pocet vyhodnocenych slabych klasifikatoru
    //pokud detektor nemeri, necha beze zmeny
    unsigned classifiers, weakClassifiers;

    // vysledne detekce
    unsigned detectionCount;
    abonDetection *detections, *detectionsEnd;//like begin/end iterators
    bool nmsDone;//true - non-maxima suppression done

    // dalsi veci???
};

#ifdef __cplusplus
}//extern "C"
#endif

#endif//_ABON_INTERFACE_H_
