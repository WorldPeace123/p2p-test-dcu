#include "hip/hip_runtime.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

/*
 * This example demonstrates P2P ping-ponging of data from one GPU to another,
 * within the same node. By enabling peer-to-peer transfers, you ensure that
 * copies between GPUs go directly over the PCIe bus. If P2P is not enabled,
 * host memory must be used as a staging area for GPU-to-GPU cudaMemcpys.
 */

__global__ void iKernel(float *src, float *dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx] * 2.0f;
}

inline bool isCapableP2P(int ngpus)
{
    hipDeviceProp_t prop[ngpus];
    int iCount = 0;

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(hipGetDeviceProperties(&prop[i], i));

        if (prop[i].major >= 2) iCount++;

        printf("> GPU%d: %s %s capable of Peer-to-Peer access\n", i,
                prop[i].name, (prop[i].major >= 2 ? "is" : "not"));
    }

    if(iCount != ngpus)
    {
        printf("> no enough device to run this application\n");
    }

    return (iCount == ngpus);
}

/*
 * enable P2P memcopies between GPUs (all GPUs must be compute capability 2.0 or
 * later (Fermi or later)).
 */
inline void enableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        CHECK(hipSetDevice(i));

        for(int j = 0; j < ngpus; j++)
        {
            if(i == j) continue;

            int peer_access_available = 0;
            CHECK(hipDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available)
            {
                CHECK(hipDeviceEnablePeerAccess(j, 0));
                printf("> GPU%d enabled direct access to GPU%d\n", i, j);
            }
            else
            {
                printf("(%d, %d)\n", i, j );
            }
        }
    }
}

inline void disableP2P (int ngpus)
{
    for( int i = 0; i < ngpus; i++ )
    {
        CHECK(hipSetDevice(i));

        for(int j = 0; j < ngpus; j++)
        {
            if( i == j ) continue;

            int peer_access_available = 0;
            CHECK(hipDeviceCanAccessPeer( &peer_access_available, i, j) );

            if( peer_access_available )
            {
                CHECK(hipDeviceDisablePeerAccess(j));
                printf("> GPU%d disabled direct access to GPU%d\n", i, j);
            }
        }
    }
}

void initialData(float *ip, int size)
{
    for(int i = 0; i < size; i++)
    {
        ip[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc, char **argv)
{
    int ngpus;

    // check device count
    CHECK(hipGetDeviceCount(&ngpus));
    printf("> CUDA-capable device count: %i\n", ngpus);

    // check p2p capability
    isCapableP2P(ngpus);

    // get ngpus from command line
    if (argc > 1)
    {
        if (atoi(argv[1]) > ngpus)
        {
            fprintf(stderr, "Invalid number of GPUs specified: %d is greater "
                    "than the total number of GPUs in this platform (%d)\n",
                    atoi(argv[1]), ngpus);
            return 1;
        }

        ngpus  = atoi(argv[1]);
    }

    if (ngpus < 2)
    {
        fprintf(stderr, "No more than 2 GPUs supported\n");
        return 1;
    }

    if (ngpus > 1) enableP2P(ngpus);

    // Allocate buffers
    int iSize = 1<<24;
    const size_t iBytes = iSize * sizeof(float);
    printf("\nAllocating buffers (%iMB on each GPU and CPU Host)...\n",
           int(iBytes / 1024 / 1024));

    float **d_src = (float **)malloc(sizeof(float) * ngpus);
    float **d_rcv = (float **)malloc(sizeof(float) * ngpus);
    float **h_src = (float **)malloc(sizeof(float) * ngpus);
    hipStream_t *stream = (hipStream_t *)malloc(sizeof(hipStream_t) * ngpus);

    // Create CUDA event handles
    hipEvent_t start, stop;
    CHECK(hipSetDevice(0));
    CHECK(hipEventCreate(&start));
    CHECK(hipEventCreate(&stop));

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(hipSetDevice(i));
        CHECK(hipMalloc(&d_src[i], iBytes));
        CHECK(hipMalloc(&d_rcv[i], iBytes));
        CHECK(hipHostMalloc((void **) &h_src[i], iBytes));

        CHECK(hipStreamCreate(&stream[i]));
    }

    for (int i = 0; i < ngpus; i++)
    {
        initialData(h_src[i], iSize);
    }

    // unidirectional gmem copy
    CHECK(hipSetDevice(0));
    CHECK(hipEventRecord(start, 0));

    for (int i = 0; i < 100; i++)
    {
        if (i % 2 == 0)
        {
            CHECK(hipMemcpy(d_src[1], d_src[0], iBytes,
                        hipMemcpyDeviceToDevice));
        }
        else
        {
            CHECK(hipMemcpy(d_src[0], d_src[1], iBytes,
                        hipMemcpyDeviceToDevice));
        }
    }

    CHECK(hipSetDevice(0));
    CHECK(hipEventRecord(stop, 0));
    CHECK(hipEventSynchronize(stop));

    float elapsed_time_ms;
    CHECK(hipEventElapsedTime(&elapsed_time_ms, start, stop ));

    elapsed_time_ms /= 100.0f;
    printf("Ping-pong unidirectional hipMemcpy:\t\t %8.2f ms ",
           elapsed_time_ms);
    printf("performance: %8.2f GB/s\n",
            (float)iBytes / (elapsed_time_ms * 1e6f));

    //  bidirectional asynchronous gmem copy
    CHECK(hipEventRecord(start, 0));

    for (int i = 0; i < 100; i++)
    {
        CHECK(hipMemcpyAsync(d_src[1], d_src[0], iBytes,
                    hipMemcpyDeviceToDevice, stream[0]));
        CHECK(hipMemcpyAsync(d_rcv[0], d_rcv[1], iBytes,
                    hipMemcpyDeviceToDevice, stream[1]));
    }

    CHECK(hipSetDevice(0));
    CHECK(hipEventRecord(stop, 0));
    CHECK(hipEventSynchronize(stop));

    elapsed_time_ms = 0.0f;
    CHECK(hipEventElapsedTime(&elapsed_time_ms, start, stop ));

    elapsed_time_ms /= 100.0f;
    printf("Ping-pong bidirectional hipMemcpyAsync:\t %8.2fms ",
           elapsed_time_ms);
    printf("performance: %8.2f GB/s\n",
           (float) 2.0f * iBytes / (elapsed_time_ms * 1e6f) );

    disableP2P(ngpus);

    // free
    CHECK(hipSetDevice(0));
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(hipSetDevice(i));
        CHECK(hipFree(d_src[i]));
        CHECK(hipFree(d_rcv[i]));
        CHECK(hipStreamDestroy(stream[i]));
        CHECK(hipDeviceReset());
    }

    exit(EXIT_SUCCESS);
}

