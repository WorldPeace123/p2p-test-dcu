#include <stdio.h>
#include <stdlib.h>

using namespace std;

//define the data sesize (cubic volume)
#define DATAXSIZE 1024
#define DATAYSIZE 1024
#define DATAZSIZE 1024
//#define DATZSIZE 129
//define the chunk sizes that each threadblock will work on
#define BLKZSIZE 1
#define BLKYSIZE 16
#define BLKXSIZE 16
#define THREADSperBLOCK 256
// define parameters
/*
#define A 0.03
#define EPS  0.00028
#define TOLERANCE 1.0E-7
*/
//#define DIVCONST 1111.11111
#define A 0.015
#define EPS 0.00007
#define MASS 1.0
#define TOLERANCE 1.0E-9
#define DIVCONST 4444.444444444444
/*=1 / ((A) * (A))*/

#define SEP 2.0

#define hipCheckErrors(msg)                                                                        \
    do {                                                                                           \
        hipError_t __err = hipGetLastError();                                                      \
        if (__err != hipSuccess) {                                                                 \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, hipGetErrorString(__err),      \
                    __FILE__, __LINE__);                                                           \
            fprintf(stderr, "*** FAILED - ABORTING\n");                                            \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
