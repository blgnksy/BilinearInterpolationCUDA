
#include <stdio.h>
#include <windows.h>
#include "cuda_helper.h"
#include "timer_helper.h"
#include "pgm_helper.h"

// Function Protypes.
uint8_t *
LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);

void
WritePGM(char * sFileName, uint8_t * pDst_Host, int nWidth, int nHeight, int nMaxGray);

__global__ void
TransformKernel(const cudaTextureObject_t d_img_texA, const cudaTextureObject_t d_img_texB, const cudaTextureObject_t d_img_texC, 
				const float gxs, const float gys, 
				const float gxsB, const float gysB,
				const float gxsC, const float gysC, 
				uint8_t* __restrict const d_out, const int neww);

void InterpolateSum(const cudaTextureObject_t d_img_texA, const cudaTextureObject_t d_img_texB, const cudaTextureObject_t d_img_texC, const int oldw, const int oldh, uint8_t* __restrict const d_out, const uint32_t neww, const uint32_t newh);

int main()
{
#pragma region Variable Declaritions
	// Host parameter declarations.	
	int   nWidth, nHeight, nMaxGray;
	// Device parameter declaration.
	uint8_t* pDst_Dev = nullptr;
#pragma endregion

#pragma region Load image to the host
	std::cout << "Loading PGM file." << std::endl;
	auto pSrc_HostB = LoadPGM((char *)"./data/lena_beforeB.pgm", nWidth, nHeight, nMaxGray);
	auto pSrc_HostC = LoadPGM((char *)"./data/lena_beforeC.pgm", nWidth, nHeight, nMaxGray);
	auto pSrc_HostA = LoadPGM((char *)"./data/lena_beforeA.pgm", nWidth, nHeight, nMaxGray);
#pragma endregion

#pragma region Size Parameter Definitions
	int initial_width = nWidth;
	int initial_height = nHeight;
	int final_width = initial_width * 8;
	int final_heigth = initial_height * 8;
	size_t total = final_width*final_heigth;
#pragma endregion

	//Channel Description
	cudaChannelFormatDesc chandesc_img = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

#pragma region Array-Resource-Texture For Image A 
	cudaArray* d_img_arrA;
	CUDA_CALL(cudaMallocArray(&d_img_arrA, &chandesc_img, initial_width, initial_height, cudaArrayTextureGather), "Memory Allocation.");
	CUDA_CALL(cudaMemcpyToArray(d_img_arrA, 0, 0, pSrc_HostA, initial_width * initial_height, cudaMemcpyHostToDevice), "Memory Cpoied to Array.");

	struct cudaResourceDesc resdesc_imgA;
	memset(&resdesc_imgA, 0, sizeof(resdesc_imgA));
	resdesc_imgA.resType = cudaResourceTypeArray;
	resdesc_imgA.res.array.array = d_img_arrA;

	struct cudaTextureDesc texdesc_imgA;
	memset(&texdesc_imgA, 0, sizeof(texdesc_imgA));
	texdesc_imgA.addressMode[0] = cudaAddressModeClamp;
	texdesc_imgA.addressMode[1] = cudaAddressModeClamp;
	texdesc_imgA.readMode = cudaReadModeNormalizedFloat;
	texdesc_imgA.filterMode = cudaFilterModePoint;
	texdesc_imgA.normalizedCoords = 0;

	cudaTextureObject_t d_img_texA = 0;
	CUDA_CALL(cudaCreateTextureObject(&d_img_texA, &resdesc_imgA, &texdesc_imgA, nullptr), "Texture Object A Created.");
#pragma endregion

#pragma region Array-Resource-Texture For Image B
	cudaArray* d_img_arrB;
	CUDA_CALL(cudaMallocArray(&d_img_arrB, &chandesc_img, initial_width/2, initial_height/2, cudaArrayTextureGather), "Memory Allocation.");
	CUDA_CALL(cudaMemcpyToArray(d_img_arrB, 0, 0, pSrc_HostB, initial_width * initial_height / 4, cudaMemcpyHostToDevice), "Memory Cpoied to Array.");

	struct cudaResourceDesc resdesc_imgB;
	memset(&resdesc_imgB, 0, sizeof(resdesc_imgB));
	resdesc_imgB.resType = cudaResourceTypeArray;
	resdesc_imgB.res.array.array = d_img_arrB;

	struct cudaTextureDesc texdesc_imgB;
	memset(&texdesc_imgB, 0, sizeof(texdesc_imgB));
	texdesc_imgB.addressMode[0] = cudaAddressModeClamp;
	texdesc_imgB.addressMode[1] = cudaAddressModeClamp;
	texdesc_imgB.readMode = cudaReadModeNormalizedFloat;
	texdesc_imgB.filterMode = cudaFilterModePoint;
	texdesc_imgB.normalizedCoords = 0;

	cudaTextureObject_t d_img_texB = 0;
	CUDA_CALL(cudaCreateTextureObject(&d_img_texB, &resdesc_imgB, &texdesc_imgB, nullptr), "Texture Object B Created.");
#pragma endregion

#pragma region Array-Resource-Texture For Image C
	cudaArray* d_img_arrC;
	CUDA_CALL(cudaMallocArray(&d_img_arrC, &chandesc_img, initial_width/4, initial_height/4, cudaArrayTextureGather), "Memory Allocation.");
	CUDA_CALL(cudaMemcpyToArray(d_img_arrC, 0, 0, pSrc_HostC, initial_width * initial_height / 16, cudaMemcpyHostToDevice), "Memory Copied to Array.");

	struct cudaResourceDesc resdesc_imgC;
	memset(&resdesc_imgC, 0, sizeof(resdesc_imgC));
	resdesc_imgC.resType = cudaResourceTypeArray;
	resdesc_imgC.res.array.array = d_img_arrC;

	struct cudaTextureDesc texdesc_imgC;
	memset(&texdesc_imgC, 0, sizeof(texdesc_imgC));
	texdesc_imgC.addressMode[0] = cudaAddressModeClamp;
	texdesc_imgC.addressMode[1] = cudaAddressModeClamp;
	texdesc_imgC.readMode = cudaReadModeNormalizedFloat;
	texdesc_imgC.filterMode = cudaFilterModePoint;
	texdesc_imgC.normalizedCoords = 0;

	cudaTextureObject_t d_img_texC = 0;
	CUDA_CALL(cudaCreateTextureObject(&d_img_texC, &resdesc_imgC, &texdesc_imgC, nullptr), "Texture Object B Created.");
#pragma endregion

	//Device Output Memory Ops
	CUDA_CALL(cudaMalloc(&pDst_Dev, total), "Memory Allocated for Device Output.");

	FastTimer ft; //Timer 
	ft.StartCounter();
	InterpolateSum(d_img_texA, d_img_texB, d_img_texC, initial_width, initial_height, pDst_Dev, final_width, final_heigth);
	std::cout << "Process finished in " << ft.GetCounter() << " ms." << std::endl;

	//Device Output Memory Ops
	auto pDst_Host = new uint8_t[final_width * final_heigth];
	CUDA_CALL(cudaMemcpy(pDst_Host, pDst_Dev, total, cudaMemcpyDeviceToHost), "Output Copied from Device to Host.");

	// Output the result image.
	std::cout << "Output the PGM file." << std::endl;
	WritePGM((char *)"./output/lena_after.pgm", pDst_Host, final_width, final_heigth, nMaxGray);

	getchar();
}
//  Adopted from https://github.com/komrad36/CUDALERP
__global__ void
TransformKernel(const cudaTextureObject_t d_img_texA, const cudaTextureObject_t d_img_texB, const cudaTextureObject_t d_img_texC, 
				const float gxsA, const float gysA, 
				const float gxsB, const float gysB,
				const float gxsC, const float gysC,
				uint8_t* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;

#pragma region Bilinear Interpolation of Image A
	const float fyA = (y + 0.5f) * gysA - 0.5f;
	const float wt_yA = fyA - floor(fyA);
	const float invwt_yA = 1.0f - wt_yA;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fxA = (x + 0.5f)  * gxsA - 0.5f;
		// fA carries the d00, d01, d10, d11
		const float4 fA = tex2Dgather<float4>(d_img_texA, fxA + 0.5f, fyA + 0.5f); 
		const float wt_xA = fxA - floor(fxA); 
		const float invwt_xA = 1.0f - wt_xA;
		const float xaA = invwt_xA*fA.w + wt_xA*fA.z;
		const float xbA = invwt_xA*fA.x + wt_xA*fA.y;
		const float resA = 255.0f*(invwt_yA*xaA + wt_yA*xbA) + 0.5f;
		if (x < neww) d_out[y*neww + x] = (resA *0.34);
	}
#pragma endregion
#pragma region Bilinear Interpolation of Image B
	const float fyB = (y + 0.5f)*gysB - 0.5f;
	const float wt_yB = fyB - floor(fyB);
	const float invwt_yB = 1.0f - wt_yB;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fxB = (x + 0.5f)*gxsB - 0.5f;
		const float4 fB = tex2Dgather<float4>(d_img_texB, fxB + 0.5f, fyB + 0.5f);
		const float wt_xB = fxB - floor(fxB);
		const float invwt_xB = 1.0f - wt_xB;
		const float xaB = invwt_xB*fB.w + wt_xB*fB.z;
		const float xbB = invwt_xB*fB.x + wt_xB*fB.y;
		const float resB = 255.0f*(invwt_yB*xaB + wt_yB*xbB) + 0.5f;
		if (x < neww) d_out[y*neww + x] += (resB * 0.33);
	}
#pragma endregion

#pragma region Bilinear Interpolation of Image C
	const float fyC = (y + 0.5f)*gysC - 0.5f;
	const float wt_yC = fyC - floor(fyC);
	const float invwt_yC = 1.0f - wt_yC;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fxC = (x + 0.5f)*gxsC - 0.5f;
		const float4 fC = tex2Dgather<float4>(d_img_texC, fxC + 0.5f, fyC + 0.5f);
		const float wt_xC = fxC - floor(fxC);
		const float invwt_xC = 1.0f - wt_xC;
		const float xaC = invwt_xC*fC.w + wt_xC*fC.z;
		const float xbC = invwt_xC*fC.x + wt_xC*fC.y;
		const float resC = 255.0f*(invwt_yC*xaC + wt_yC*xbC) + 0.5f;
		if (x < neww) d_out[y*neww + x] += (resC * 0.33);
	}
#pragma endregion
}

void InterpolateSum(const cudaTextureObject_t d_img_texA, const cudaTextureObject_t d_img_texB, const cudaTextureObject_t d_img_texC, const int initial_width, const int initial_height, uint8_t* __restrict const d_out, const uint32_t final_width, const uint32_t final_height) {
	const float gxsA = static_cast<float>(initial_width) / static_cast<float>(final_width);
	const float gysA = static_cast<float>(initial_height) / static_cast<float>(final_height);
	const float gxsB = static_cast<float>(initial_width / 2) / static_cast<float>(final_width);
	const float gysB = static_cast<float>(initial_height / 2) / static_cast<float>(final_height);
	const float gxsC = static_cast<float>(initial_width / 4) / static_cast<float>(final_width);
	const float gysC = static_cast<float>(initial_height / 4) / static_cast<float>(final_height);
	TransformKernel << < {((final_width - 1) >> 9) + 1, final_height}, 256 >> > (d_img_texA, d_img_texB, d_img_texC, gxsA, gysA, gxsB, gysB, gxsC, gysC, d_out, final_width);
	cudaDeviceSynchronize();
}