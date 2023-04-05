#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <string.h> 
#include <time.h> 
#include <math.h> 

#define BLOCKSIZE	16384
#define THREADSIZE	128
#define Round		24
#define ENDIAN_CHANGE(val) (((val <<32) & 0xffffffff00000000)|(val >>32))
#define chi(A,B,C) ((A) ^ ((~B) & C))

typedef unsigned char u8;
typedef unsigned int u32;
typedef unsigned long long u64;

static void HandleError(cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
