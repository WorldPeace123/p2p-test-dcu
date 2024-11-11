

#include <bits/stdc++.h>
#include <hip/hip_runtime.h>
#include <omp.h>
#define size 1024*24
#define AllBytes size *size
using namespace std;

/**
 *
size: 40960
size_a: 1.34218e+10
12.5 GB
Data cpy time: 1807.14 ms
Bandwidth: 6.91699 GB/s

size: 20480
size_a: 3.35544e+09
3.125 GB
Data cpy time: 316.839 ms
Bandwidth: 9.86307 GB/s

size: 10240
size_a: 8.38861e+08
0.78125 GB
Data cpy time: 89.2606 ms
HtoD Bandwidth: 8.75246 GB/s

size: 8192
size_a: 5.36871e+08
0.5 GB
Data cpy time: 90.0406 ms
Bandwidth: 5.55305 GB/s

size: 5120
size_a: 2.09715e+08
0.195312 GB
Data cpy time: 27.1153 ms
Bandwidth: 7.20304 GB/s

size: 3072
size_a: 7.54975e+07
0.0703125 GB
Data cpy time: 8.52505 ms
Bandwidth: 8.24775 GB/s

size: 1024
size_a: 8.38861e+06
0.0078125 GB
Data cpy time: 2.76203 ms
HtoD Bandwidth: 2.82854 GB/s
 *
 *
 *
 */
__global__ void dcu_add(double *d_a)
{
    unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < AllBytes)
    {
        d_a[i] = 666.0;
    }
}
int main()
{

    double *a = (double *)malloc(sizeof(double) * AllBytes);

    double *a_res = (double *)malloc(sizeof(double) * AllBytes);

    double *d_a;
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            a[i * j + j] = 9.9;
        }
    }

    double GB = pow(2, 30);
    double size_a = sizeof(double) * AllBytes;
    cout << "size: " << size << endl;
    cout << "size_a: " << size_a << endl;
    double GB_ALL = size_a / GB;
    cout << GB_ALL << " GB\n";

    // p2p 传输
#ifdef p2p
    int ngpu = 4;
    int access = 0;
    hipGetDeviceCount(&ngpu);
    for (int i = 0; i < ngpu; i++)
    {
        for (int j = 0; j < ngpu; j++)
        {
            if (i != j)
            {
                hipSetDevice(i);
                hipDeviceCanAccessPeer(&access, i, j);
                if (!access)
                {
                    printf(" p2p access from source %d to destination %d is not ok\n", i, j);
                }
                else
                {
                    hipDeviceEnablePeerAccess(j, 0);
                    // cout << " Peer access from " << i << "th device to " << j << "th device can be enabled" << endl;
                }
            }
        }
    }
    // Malloc
    hipStream_t stream[ngpu];
    for (int id = 0; id < ngpu; id++)
    {
        hipSetDevice(id);
        hipGetDevice(&id);
        hipMalloc((void **)&d_a, sizeof(double) * AllBytes);
        hipStreamCreate(&stream[id]);
        cout<<"Malloc id : "<<id<<endl;
    }
    hipSetDevice(0);
     hipMemcpy(d_a, a, sizeof(double) * AllBytes, hipMemcpyHostToDevice);

    for (int id = 0; id < ngpu-1; id++)
    {
        hipSetDevice(id);
        hipEvent_t start, stop;
        hipEventCreate(&start); // 创建开始事件
        hipEventCreate(&stop);  // 创建结束事件

        hipEventRecord(start, 0); // 记录开始时间
       
        //int des_id=id++ error
        int des_id=id+1;
        hipMemcpyPeerAsync(d_a,des_id , d_a, id, sizeof(double) * AllBytes, stream[id]);
        hipDeviceSynchronize();
        hipEventRecord(stop, 0);   // 记录结束时间
        hipEventSynchronize(stop); // 确保所有操作完成

        float elapsedTime;
        hipEventElapsedTime(&elapsedTime, start, stop); // 计算时间差
        // cout << "p2p time: " << elapsedTime << " ms" << endl;
        cout << "p2p DCU Bandwidth: " << GB_ALL / (elapsedTime / 1000.0) << " GB/s" << endl;
       
        //res check
    //     if (id == 2)
    //     {
    //         hipSetDevice(id);
    //             dim3 blocksize(256, 1);
    // dim3 gridsize(AllBytes / 256 + 1, 1);
    // dcu_add<<<gridsize, blocksize>>>(d_a);
    //         hipMemcpy(a_res, d_a, sizeof(double) * AllBytes, hipMemcpyDeviceToHost);
    //         hipDeviceSynchronize();

    //         for (int i = 0; i < size; i++)
    //         {
    //             for (int j = 0; j < size; j++)
    //             {
    //                 if (a_res[i * size + j] == 666)
    //                 {
    //                     cout << a_res[i * size + j] << " ";
    //                 }
    //             }
    //             cout << endl;
    //         }
    //     }
    }

#endif

// hip传输
#ifdef cpy
    hipMalloc((void **)&d_a, sizeof(double) * AllBytes);
    hipEvent_t start, stop;
    hipEventCreate(&start); // 创建开始事件
    hipEventCreate(&stop);  // 创建结束事件

    hipEventRecord(start, 0); // 记录开始时间

    hipMemcpy(d_a, a, sizeof(double) * AllBytes, hipMemcpyHostToDevice);

    hipEventRecord(stop, 0);   // 记录结束时间
    hipEventSynchronize(stop); // 确保所有操作完成

    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, start, stop); // 计算时间差

    // cout << "HosttoDevice time: " << elapsedTime << " ms" << endl;
    cout << "D_to_host    cpy Bandwidth: " << GB_ALL / (elapsedTime / 1000.0) << " GB/s" << endl;
    dim3 blocksize(256, 1);
    dim3 gridsize(AllBytes / 256 + 1, 1);
    dcu_add<<<gridsize, blocksize>>>(d_a);

    hipEvent_t start2, stop2;
    hipEventCreate(&start2); // 创建开始事件
    hipEventCreate(&stop2);  // 创建结束事件

    hipEventRecord(start2, 0); // 记录开始时间
    hipMemcpy(a_res, d_a, sizeof(double) * AllBytes, hipMemcpyDeviceToHost);
    hipEventRecord(stop2, 0);   // 记录结束时间
    hipEventSynchronize(stop2); // 确保所有操作完成

    float elapsedTime2;
    hipEventElapsedTime(&elapsedTime2, start2, stop2); // 计算时间差

    // cout << "DevicetoHost time: " << elapsedTime2 << " ms" << endl;
    cout << "H_to_Device cpy Bandwidth: " << GB_ALL / (elapsedTime2 / 1000.0) << " GB/s" << endl;
#endif

#ifdef DEBUG
    int num = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            if (a_res[i * size + j] == 666)
            {

                num++;
                cout << a_res[i * size + j] << " ";
            }
        }
        cout << endl;
    }
    cout << "All 666 " << num << endl;
#endif

    return 0;
}
