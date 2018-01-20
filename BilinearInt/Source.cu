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

//Global  declaration
#define DIM 512

// Function Protypes.
Npp8u *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);

void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray);


__global__ void
TransformKernel(Npp8u * pDst_Dev, Npp8u * pSrc_Dev, const int nWidth);




// Main function.
int
main(int argc, char ** argv)
{
	// Host parameter declarations.	
	Npp8u * pSrc_Host, *pDst_Host;
	int   nWidth, nHeight, nMaxGray, nNormalizer;

	std::cout << "GPU VERSION" << std::endl;

	// Load image to the host.
	std::cout << "Load PGM file." << std::endl;
	pSrc_Host = LoadPGM("lena_before.pgm", nWidth, nHeight, nMaxGray);
	pDst_Host = new Npp8u[nWidth * nHeight * 8];

	// Device parameter declarations.
	Npp8u	 * pSrc_Dev, *pDst_Dev;
	int		 nSrcStep_Dev, nDstStep_Dev;

	StartCounter();

	// Allocate Device variables and copy the image from the host to GPU
	pSrc_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nSrcStep_Dev);
	pDst_Dev = nppiMalloc_8u_C1(nWidth, nHeight, &nDstStep_Dev);
	std::cout << "Copy image from host to device." << std::endl;
	CUDA_CALL(cudaMemcpy(pSrc_Dev, pSrc_Host, nWidth * nHeight * sizeof(Npp8u), cudaMemcpyHostToDevice), "Memory copied.(HostToDevice)");

	std::cout << "Process the image on GPU." << std::endl;


	dim3 dimGrid(512);
	dim3 dimBlock(1024);

	TransformKernel << <dimGrid, dimBlock, 0, 0 >> > (pDst_Dev, pSrc_Dev, nWidth);

	// Copy result back to the host.
	std::cout << "Work done! Copy the result back to host." << std::endl;
	CUDA_CALL(cudaMemcpy(pDst_Host, pDst_Dev, nWidth * nHeight * sizeof(Npp8u), cudaMemcpyDeviceToHost), "Memory copied.(DeviceToHost)");

	std::cout << "Time to calculate results(GPU Time): " << GetCounter() << std::endl;
	// Output the result image.
	std::cout << "Output the PGM file." << std::endl;
	WritePGM("lena_after.pgm", pDst_Host, nWidth*4, nHeight, nMaxGray);

	// Clean up.
	std::cout << "Clean up." << std::endl;
	delete[] pSrc_Host;
	delete[] pDst_Host;

	nppiFree(pSrc_Dev);
	nppiFree(pDst_Dev);
	printf("All done. Press Any Key to Continue...");
	getchar();
	return 0;
}

// Disable reporting warnings on functions that were marked with deprecated.
#pragma warning( disable : 4996 )

// Load PGM file.
Npp8u *
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
	Npp8u * pSrc_Host = new Npp8u[nWidth * nHeight];
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			pSrc_Host[i*nWidth + j] = fgetc(fInput);
	fclose(fInput);

	return pSrc_Host;
}

// Write PGM image.
void
WritePGM(char * sFileName, Npp8u * pDst_Host, int nWidth, int nHeight, int nMaxGray)
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


__global__ void
TransformKernel(Npp8u * pDst_Dev, Npp8u * pSrc_Dev, const int nWitdh)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	Npp8u d00 = pSrc_Dev[i];
	Npp8u d01 = pSrc_Dev[i + 1];
	Npp8u d10 = pSrc_Dev[i + nWitdh];
	Npp8u d11 = pSrc_Dev[i + 1 + nWitdh];

	pDst_Dev[i] = pSrc_Dev[i];
	pDst_Dev[i + 1] = static_cast<Npp8u>(d00 / 1 + d01 / 7);
	pDst_Dev[i + 2] = static_cast<Npp8u>(d00 / 2 + d01 / 6);
	pDst_Dev[i + 3] = static_cast<Npp8u>(d00 / 3 + d01 / 5);
	pDst_Dev[i + 4] = static_cast<Npp8u>(d00 / 4 + d01 / 4);
	pDst_Dev[i + 5] = static_cast<Npp8u>(d00 / 5 + d01 / 3);
	pDst_Dev[i + 6] = static_cast<Npp8u>(d00 / 6 + d01 / 2);
	pDst_Dev[i + 7] = static_cast<Npp8u>(d00 / 7 + d01 / 1);
}

