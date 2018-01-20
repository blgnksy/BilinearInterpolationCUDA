#include "pgm_helper.h"


// Load PGM file.
uint8_t *LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray)
{
	char aLine[256];
	FILE * fInput = fopen(sFileName, "rb");
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
void WritePGM(char *sFileName, uint8_t *pDst_Host, int nWidth, int nHeight, int nMaxGray)
{
	FILE * fOutput = fopen(sFileName, "wb");
	if (fOutput == 0)
	{
		perror("Cannot open file to read");
		exit(EXIT_FAILURE);
	}
	char *aComment = (char *)"# Created by NPP";
	fprintf(fOutput, "P5\n%s\n%d %d\n%d\n", aComment, nWidth, nHeight, nMaxGray);
	for (int i = 0; i < nHeight; ++i)
		for (int j = 0; j < nWidth; ++j)
			fputc(pDst_Host[i*nWidth + j], fOutput);
	fclose(fOutput);
}