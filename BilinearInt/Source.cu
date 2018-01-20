#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "npp.h"
#include <math.h>
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include "device_launch_parameters.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN

// CUDA error checking Macro.
#define CUDA_CALL(x,y) {if((x) != cudaSuccess){ \
  printf("CUDA error at %s:%d\n",__FILE__,__LINE__); \
  printf("  %s\n", cudaGetErrorString(cudaGetLastError())); \
  exit(EXIT_FAILURE);}\
  else{printf("CUDA Success at %d. (%s)\n",__LINE__,y); }}

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		std::cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}

// Function Protypes.
uint8_t *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);

void
WritePGM(char * sFileName, uint8_t * pDst_Host, int nWidth, int nHeight, int nMaxGray);

__global__ void
TransformKernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, uint8_t* __restrict const d_out, const int neww);

void InterpolateSum(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh);

int main()
{
	// Host parameter declarations.	
	int   nWidth, nHeight, nMaxGray, nNormalizer;

	// Load image to the host.
	std::cout << "Loading PGM file." << std::endl;
	auto pSrc_HostA = LoadPGM("lena_beforeB.pgm", nWidth, nHeight, nMaxGray);


	constexpr int oldw = 256;
	constexpr int oldh = 256;
	constexpr int neww = static_cast<int>(static_cast<double>(oldw) * 8);
	constexpr int newh = static_cast<int>(static_cast<double>(oldh) * 8);
	const size_t total = static_cast<size_t>(neww)*static_cast<size_t>(newh);


	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);

	cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	cudaArray* d_img_arr;

	CUDA_CALL(cudaMallocArray(&d_img_arr, &chandesc_img, oldw, oldh, cudaArrayTextureGather),"Memory Allocation.");
	CUDA_CALL(cudaMemcpyToArray(d_img_arr, 0, 0, pSrc_HostA, oldh * oldw, cudaMemcpyHostToDevice), "Memory Cpoied to Array.");

	struct cudaResourceDesc resdesc_img;
	memset(&resdesc_img, 0, sizeof(resdesc_img));
	resdesc_img.resType = cudaResourceTypeArray;
	resdesc_img.res.array.array = d_img_arr;

	struct cudaTextureDesc texdesc_img;
	memset(&texdesc_img, 0, sizeof(texdesc_img));
	texdesc_img.addressMode[0] = cudaAddressModeClamp;
	texdesc_img.addressMode[1] = cudaAddressModeClamp;
	texdesc_img.readMode = cudaReadModeNormalizedFloat;
	texdesc_img.filterMode = cudaFilterModePoint;
	texdesc_img.normalizedCoords = 0;

	cudaTextureObject_t d_img_tex = 0;
	CUDA_CALL(cudaCreateTextureObject(&d_img_tex, &resdesc_img, &texdesc_img, nullptr),"Texture Object Created.");

	uint8_t* pDst_Dev = nullptr;
	CUDA_CALL(cudaMalloc(&pDst_Dev, total),"Memory Allocated.");

	StartCounter();
	InterpolateSum(d_img_tex, oldw, oldh, pDst_Dev, neww, newh);
	std::cout << "Process finished in " << GetCounter() << std::endl;

	auto pDst_Host = new uint8_t[neww * newh];
	CUDA_CALL(cudaMemcpy(pDst_Host, pDst_Dev, total, cudaMemcpyDeviceToHost),"Memory Copied.");

	// Output the result image.
	std::cout << "Output the PGM file." << std::endl;
	WritePGM("lena_after.pgm", pDst_Host, neww, newh, nMaxGray);


	getchar();
}

__global__ void
TransformKernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, uint8_t* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;
	const float fy = (y + 0.5f)*gys - 0.5f;
	const float wt_y = fy - floor(fy);
	const float invwt_y = 1.0f - wt_y;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fx = (x + 0.5f)*gxs - 0.5f;
		const float4 f = tex2Dgather<float4>(d_img_tex, fx + 0.5f, fy + 0.5f);
		const float wt_x = fx - floor(fx);
		const float invwt_x = 1.0f - wt_x;
		const float xa = invwt_x*f.w + wt_x*f.z;
		const float xb = invwt_x*f.x + wt_x*f.y;
		const float res = 255.0f*(invwt_y*xa + wt_y*xb) + 0.5f;
		if (x < neww) d_out[y*neww + x] = res;
	}
}

void InterpolateSum(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
	TransformKernel << <{((neww - 1) >> 9) + 1, newh}, 256 >> >(d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )

// Load PGM file.
uint8_t *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{
	char aLine[256];
	FILE * fInput = fopen(sFileName, "r");
	if (fInput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	// First line: version
	fgets(aLine, 256, fInput);
	std::cout << "\tVersion: " << aLine;
	// Second line: comment
	fgets(aLine, 256, fInput);
	std::cout << "\tComment: " << aLine;
	fseek(fInput, -1, SEEK_CUR);
	// Third line: size
	fscanf(fInput, "%d", &nWidth);
	std::cout << "\tWidth: " << nWidth;
	fscanf(fInput, "%d", &nHeight);
	std::cout << " Height: " << nHeight << std::endl;
	// Fourth line: max value
	fscanf(fInput, "%d", &nMaxGray);
	std::cout << "\tMax value: " << nMaxGray << std::endl;
	while (getc(fInput) != '\n');
	// Following lines: data
	uint8_t *pSrc_Host = new uint8_t[nWidth * nHeight];
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			pSrc_Host[i*nWidth + j] = fgetc(fInput);
	fclose(fInput);

	return pSrc_Host;
}


// Write PGM image.
void
WritePGM(char *sFileName, uint8_t *pDst_Host, int nWidth, int nHeight, int nMaxGray)
{
	FILE * fOutput = fopen(sFileName, "w+");
	if (fOutput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	char * aComment = "# Created by NPP";
	fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			fputc(pDst_Host[i*nWidth + j], fOutput);
	fclose(fOutput);
}