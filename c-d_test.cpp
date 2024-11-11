#include <stdio.h> 
#include <hip/hip_runtime.h> 

#define NSTREAM 2 
#define BDIM 512 

void initialData(float *ip, int size) 
{
	int i;
	for (i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF) / 10.0f;         
		//printf("%f\n", ip[i]);     
	} 
} 

void sumArraysOnHost(float *A, float *B, float *C, const int N) 
{
	for (int idx = 0; idx < N; idx++)         
		C[idx] = A[idx] + B[idx];
}

__global__ void sumArrays(float *A, float *B, float *C, const int N) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) 
	{ 
		for (int j = 0; j < 60; j++) 
		{ 
			C[idx] = A[idx] + B[idx]; 
		} 
	}
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
	double epsilon = 1.0E-8;     
	bool match = 1;

	for (int i = 0; i < N; i++) 
	{ 
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) 
		{ match = 0;             
		printf("Arrays do not match!\n");             
		printf("host %5.2f gpu %5.2f at %d\n", hostRef[i], gpuRef[i], i);             
		break; 
		} 
	}

	if (match) printf("Arrays match.\n\n");
}

int main(int argc, char **argv) 
{
	printf("> %s Starting...\n", argv[0]);

	int dev = 0;     
	hipSetDevice(dev);     
	hipDeviceProp_t deviceProp;     
	hipGetDeviceProperties(&deviceProp, dev);
	printf("> Using Device %d: %s\n", dev, deviceProp.name);

	// set up data size of vectors     
	//int nElem = 1 << 2;     
	int nElem = 1 << 24;     
	printf("> vector size = %d\n", nElem);     
	size_t nBytes = nElem * sizeof(float);     
	printf("> size nBytes = %ld MB\n", nBytes/1024/1024); 

	float *h_A, *h_B, *h_C;     
	hipHostMalloc((void**)&h_A, nBytes, hipHostMallocDefault);     
	hipHostMalloc((void**)&h_B, nBytes, hipHostMallocDefault);     
	hipHostMalloc((void**)&h_C, nBytes, hipHostMallocDefault);

	initialData(h_A, nElem);     
	initialData(h_B, nElem);     
	memset(h_C, 0, nBytes);

	//sumArraysOnHost(h_A, h_B, hostRef, nElem); 

	float *d_A, *d_B, *d_C;     
	hipMalloc((float**)&d_A, nBytes);     
	hipMalloc((float**)&d_B, nBytes);     
	hipMalloc((float**)&d_C, nBytes);

	hipEvent_t start, stop;     
	hipEventCreate(&start);     
	hipEventCreate(&stop);

	dim3 block(BDIM);     
	dim3 grid((nElem + block.x - 1) / block.x);     
	printf("> grid (%d,%d) block (%d,%d)\n", grid.x, grid.y, block.x, block.y);


	hipMemcpy(d_A, h_A, nBytes, hipMemcpyHostToDevice);     
	hipMemcpy(d_B, h_B, nBytes, hipMemcpyHostToDevice);    
	hipLaunchKernelGGL(sumArrays, dim3(grid), dim3(block), 0, 0, d_A, d_B, d_C, nElem);     
	hipMemcpy(h_C, d_C, nBytes, hipMemcpyDeviceToHost);


	hipEventRecord(start, 0);     
	hipMemcpy(d_A, h_A, nBytes, hipMemcpyHostToDevice);     
	hipMemcpy(d_B, h_B, nBytes, hipMemcpyHostToDevice);     
	hipEventRecord(stop, 0);     
	hipEventSynchronize(stop);     
	float memcpy_h2d_time;     
	hipEventElapsedTime(&memcpy_h2d_time, start, stop);


	hipEventRecord(start, 0);     
	hipLaunchKernelGGL(sumArrays, dim3(grid), dim3(block), 0, 0, d_A, d_B, d_C, nElem);     
	hipEventRecord(stop, 0);     
	hipEventSynchronize(stop);     
	float kernel_time;     
	hipEventElapsedTime(&kernel_time, start, stop);

	hipEventRecord(start, 0);     
	hipMemcpy(h_C, d_C, nBytes, hipMemcpyDeviceToHost);     
	hipEventRecord(stop, 0);     
	hipEventSynchronize(stop);     
	float memcpy_d2h_time;     
	hipEventElapsedTime(&memcpy_d2h_time, start, stop);

	printf("Measured timings (throughput):\n");     
	printf(" Memcpy host to device\t: %f ms (%f GB/s)\n", memcpy_h2d_time, (2 * nBytes * 1e-6) / memcpy_h2d_time);     
	printf(" Memcpy device to host\t: %f ms (%f GB/s)\n", memcpy_d2h_time, (nBytes * 1e-6) / memcpy_d2h_time);     
	printf(" Kernel time: %f ms\n", kernel_time);     
	float total_time = memcpy_h2d_time + memcpy_d2h_time + kernel_time;     
	printf(" Total time: %f ms\n", total_time);

	

	//check device results     
	//checkResult(hostRef, gpuRef, nElem); 

	 // free device global memory     
	hipFree(d_A); 
	hipFree(d_B);     
	hipFree(d_C);

	// free host memory     
	hipHostFree(h_A);     
	hipHostFree(h_B);     
	hipHostFree(h_C); 

	hipEventDestroy(start);     
	hipEventDestroy(stop);
	

	hipDeviceReset();     
	return 0; 
}

