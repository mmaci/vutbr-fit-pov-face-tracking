//-----------------------------------------------------------------------------
// abon.h
// Adaboost evaluator plugin loader and wrapper
// Jiri Havel, DCGM FIT BUT, Brno
// $Id: abon.h 159 2009-08-27 11:04:06Z ihavel $
//-----------------------------------------------------------------------------
#ifndef _ABON_H_
#define _ABON_H_

#include <aboninterface.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
void abonSetOpenGLUse(int flag); // 1 - smi se pouzit, 0 - nech ho byt
void abonSetMaxDetectionsPerFrame(int max);
void abonSetXYZ();
// ...
*/

//zabalení volání do rozumnìjšího tvaru

static inline abonErrorCode abonSetVerbosity(abonDetector *detector, unsigned verbosity)
{
    return detector->setVerbosity(detector, verbosity);
}
static inline abonErrorCode abonSetTimer(abonDetector *detector, Timer *timer)
{
    return detector->setTimer(detector, timer);
}
static inline abonErrorCode abonSetClassifier(abonDetector *detector, TClassifier *classifier)
{
    return detector->setClassifier(detector, classifier);
}
static inline abonErrorCode abonSetStep(abonDetector *detector, unsigned hstep, unsigned vstep)
{
    return detector->setStep(detector, hstep, vstep);
}
static inline abonErrorCode abonSetSize(abonDetector *detector, unsigned width, unsigned height, unsigned pitch)
{
    return detector->setSize(detector, width, height, pitch);
}
static inline abonErrorCode abonSetScale(abonDetector *detector, float start, float step, unsigned count)
{
    return detector->setScale(detector, start, step, count);
}
static inline abonErrorCode abonSetRotation(abonDetector *detector, float start, float step, unsigned count)
{
    return detector->setRotation(detector, start, step, count);
}
static inline abonErrorCode abonSetArguments(abonDetector *detector, unsigned argc, const char *argv[])
{
    return detector->setArguments(detector, argc, argv);
}
static inline abonErrorCode abonAttach(abonDetector *detector, abonDetector *attachment)
{
    return detector->attach(detector, attachment);
}
static inline abonErrorCode abonProcess(abonDetector *detector, abonImageData *image)
{
    return detector->process(detector, image);
}
static inline abonErrorCode abonStart(abonDetector *detector, abonCbGetImage *get, abonCbPutImage *put, void *param)
{
    return detector->start(detector, get, put, param);
}

//funkce pro nacteni pluginu

/// Load plugin dll/so and init detector
abonDetector *abonLoad(const char *engine);
/// Kill detector and release dll/so
/// nop if detector == NULL
void abonUnload(abonDetector *detector);

//ostatní funkce

/// Translate abonErrorCode to human readable form
const char *abonTranslateError(abonErrorCode error);
/// Translate abonTimeStamp to human readable form
const char *abonTranslateTimeStamp(abonTimeStamps timestamp);

#ifdef __cplusplus
}//extern "C"
#endif

#ifdef __cplusplus
/// C++ wrapper for abonDetector
/// currently only unloads
class DetectorPtr
{
private :
    abonDetector *detector;
public :
    DetectorPtr(abonDetector *d) throw() : detector(d) {}
    ~DetectorPtr() throw() { abonUnload(detector); }

    void set(abonDetector *d) throw() { detector = d; }
};
#endif//__cplusplus

#endif//_ABON_H_
