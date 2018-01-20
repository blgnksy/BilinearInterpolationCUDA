#ifndef TIMER_HELPER_H
#define TIMER_HELPER_H
#include <windows.h>
#include <iostream>

class FastTimer
{
public:
	double PCFreq;
	__int64 CounterStart;
	FastTimer();
	void StartCounter();
	double GetCounter();
	~FastTimer();
private:
};




#endif