#ifndef PGM_HELPER_H
#define PGM_HELPER_H
#include <stdint.h>
#include <iostream>
void WritePGM(char *sFileName, uint8_t *pDst_Host, int nWidth, int nHeight, int nMaxGray);
uint8_t *LoadPGM(char * sFileName, int & nWidth, int & nHeight, int & nMaxGray);
#endif 



