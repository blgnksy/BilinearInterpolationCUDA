#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// CUDA error checking Macro.
#define CUDA_CALL(x,y) {if((x) != cudaSuccess){ \
							printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
							printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
							exit(EXIT_FAILURE);}\
						else{\
							printf("CUDA Success at %d. (%s)\n",__LINE__,y); \
							}\
						}

#endif
