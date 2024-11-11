#include<bits/stdc++.h>
#include<hip/hip_runtime.h>
#include "config.h"
typedef double test[DATAYSIZE][DATAXSIZE];
using namespace std;
__global__ void dcu_add(test *d_a,int x)
{
    unsigned i=threadIdx.x+blockIdx.x*blockDim.x;
    if(i<x)
    {
        for(int j=0;j<x;j++)
        {
            (*d_a)[i][j]=666.0;
        }
           
    }
}
int main()
{

    test *a,*a_res;
    test *d_a;
     const size_t size_all=DATAYSIZE*DATAXSIZE*sizeof(double);
    a = (test *)malloc(sizeof(double));
    a_res=(test *)malloc(sizeof(double));
   
    for(int i=0;i<DATAYSIZE;i++)
    {
        for(int j=0;j<DATAXSIZE;j++)
        {
            (*a)[i][j]=9.9f;
        }
    }

   
    double GB=1024*1024*8;
    double size_a=sizeof(*a);
    cout<<size_a/GB<<" GB\n";

//mpi 传输

//hip传输
    hipMalloc((void **)&d_a,sizeof(double));
    hipMemcpy(d_a,a,size_all,hipMemcpyHostToDevice);
    dim3 blocksize(512,2);
    dim3 gridsize(1,1);
    dcu_add<<<blocksize,gridsize>>>(d_a,DATAXSIZE);
    hipMemcpy(a_res,d_a,size_all,hipMemcpyDeviceToHost);
     for(int i=0;i<DATAYSIZE;i++)
    {
        for(int j=0;j<DATAXSIZE;j++)
        {
            cout<<(*a_res)[i][j]<<" ";
        }
        cout<<endl;
    }
    return 0;
}

