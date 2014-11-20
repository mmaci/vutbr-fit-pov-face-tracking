//-----------------------------------------------------------------------------
// timer.h
// Wrapper for plattform timers
// Jiri Havel, DCGM FIT BUT, Brno
// $Id: timer.h 88 2009-04-22 17:19:02Z ihavel $
//-----------------------------------------------------------------------------
#ifndef _ABON_TIMER_H_
#define _ABON_TIMER_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void Timer;
	
Timer *timerCreate();

double timerTime(const Timer *timer);

void timerReset(Timer *timer);

void timerRelease(Timer *timer);

#ifdef __cplusplus
}//extern "C"
#endif

#ifdef __cplusplus
class TimerPtr
{
public :
    Timer *timer;

    double time() const { return timerTime(timer); }
    void reset() { timerReset(timer); }

    TimerPtr() : timer(timerCreate()) {}
    ~TimerPtr() { timerRelease(timer); }
};
#endif//__cplusplus

#endif//_ABON_TIMER_H_
