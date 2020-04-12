/*
** Author(s)      :  Yehia Arafa (yarafa@nmsu.edu) and Ammar ElWazir (ammarwa@nmsu.edu)
** 
** File           :  pipeline.cu  
** 
** Description    :  Host (CPU) code to call each device (GPU) microbenchmark and compute
**                   thier energy using Multi-Threaded Synchronized Monitoring (MTSM) - NVML
** 
** Paper          :  Y. Arafa et al., "Verified Instruction-Level Energy Consumption 
**                                     Measurement for NVIDIA GPUs," CF'20               
*/

#include <stdio.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nvml.h>
#include <signal.h>
#include <pthread.h>
#include <iostream>     
#include <algorithm>    
#include <vector>
#include <functional>
#include <numeric>
#include "device_functions.cu"

volatile float t;
volatile sig_atomic_t startFlag = 0;
volatile sig_atomic_t endFlag = 0;

nvmlDevice_t device;

void monitor_power(nvmlDevice_t device, std::vector<float> *powerArray)
{
    nvmlReturn_t result;
    unsigned int device_count;

    result = nvmlDeviceGetCount(&device_count);
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        goto Error;
    }

    unsigned int power;

    result = nvmlDeviceGetPowerUsage(device, &power);

    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support power measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get power for device %i: %s\n", 0, nvmlErrorString(result));
        goto Error;
    }
    if(startFlag == 1 && endFlag == 0){
    	(*powerArray).push_back(power/1000);
    	//printf("Power: %d mW\n", power); /* For Debugging */
    	//printf("%d\n", power); /* For Debugging */
    }
    return;


Error:
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));

    exit(1);
}

void shutdown_nvml()
{
    nvmlReturn_t result;
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
}

void usage() {
    printf("Usage: binary <monitoring period (ms)>\n");
}

volatile sig_atomic_t flag = 0;

void end_monitoring(int sig) {
    flag = 1;
}

void * runMonitor(void *){
    std::vector<float> powerArray;
    while(flag==0){
        monitor_power(device, &powerArray);
        usleep(1/(50*1000000));
    }
    int size = powerArray.size();
    float sum = std::accumulate(powerArray.begin(), powerArray.end(), 0);
    float power_result = sum/size;
    float result = ((t*sum)/(size*1000));
    printf("Average Power: %f W\n", power_result);
    printf("Energy: %f J\n", result);
    return 0;
}

int main(int argc, const char* argv[]){
    //********************** Starting NVML Part **********************
    nvmlReturn_t result;
    unsigned int device_count;

    // First initialize NVML library
    result = nvmlInit();
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        printf("Press ENTER to continue...\n");
        getchar();
        return 1;
    }


    result = nvmlDeviceGetCount(&device_count);
    if(NVML_SUCCESS != result)
    {
        printf("Failed to query device count: %s\n", nvmlErrorString(result));
        shutdown_nvml();
        exit(1);
    }

    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to get handle for device %i: %s\n", 0, nvmlErrorString(result));
        exit(1);
    }

    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to get name of device %i: %s\n", 0, nvmlErrorString(result));
        exit(1);
    }

    signal(SIGINT, end_monitoring);
    //********************** Finished NVML Part **********************

    //********************** NVML POWER START **********************
    pthread_t thread_id;
    if(pthread_create(&thread_id, NULL, runMonitor, NULL)){
        printf("Thread Init Error\n");
    }
    //********************** NVML POWER END **********************

    int n=10;

    /* Host variable Declaration */
    int *c;
   
    /* Device variable Declaration */
    int  *d_c;
   
    /* Allocation of Host Variables */
    c = (int *)malloc(n * sizeof(int));
   
    /* Allocation of Device Variables */ 
    cudaMalloc((void **)&d_c, n *sizeof(int));
    
    dim3 Db = dim3(1);
    dim3 Dg = dim3(1);    

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time;

    startFlag = 1;
    //******* Starting Time Recording ***************
    cudaEventRecord(start, 0);
    cudaProfilerStart();

    if(strcmp(argv[1],"Ovhd")==0){ Ovhd<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Add")==0){ Add<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Abs")==0){ Abs<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Bfind")==0){ Bfind<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Clz")==0){ Clz<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Cnot")==0){ Cnot<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Copysign")==0){ Copysign<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"DFAdd")==0){ DFAdd<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"DFDiv")==0){ DFDiv<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Div")==0){ Div<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"DivU")==0){ DivU<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Ex2")==0){ Ex2<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"FastSqrt")==0){ FastSqrt<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"FDiv")==0){ FDiv<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"HFAdd")==0){ HFAdd<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Lg2")==0){ Lg2<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"MAdd_cc")==0){ MAdd_cc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"MMad_cc")==0){ MMad_cc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"MSubc")==0){ MSubc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Mul")==0){ Mul<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Mul24")==0){ Mul24<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Mul64Hi")==0){ Mul64Hi<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Popc")==0){ Popc<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Rcp")==0){ Rcp<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Rem")==0){ Rem<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"RemU")==0){ RemU<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Rsqrt")==0){ Rsqrt<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Sad")==0){ Sad<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Sin")==0){ Sin<<<Db, Dg>>>(d_c); }
    else if(strcmp(argv[1],"Sqrt")==0){ Sqrt<<<Db, Dg>>>(d_c); }
    else { printf("Wrong Instruction\n"); exit(0); }

    //====================== Kernel Start =========================
    
    //====================== Kernel End =========================

    cudaProfilerStop();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    //******* Finished Time Recording ***************
    endFlag = 1;

    //********************** NVML POWER START **********************
    flag = 1;
    
    //********************** NVML POWER END **********************

    cudaEventElapsedTime(&time, start, stop);
    cudaDeviceSynchronize();
    cudaMemcpy(c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    t = time;
    printf("GPU Elapsed Time = %f ms\n",t);

    pthread_join(thread_id, NULL);
    shutdown_nvml();
  
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n");

    /* Free Device Memory */
    cudaFree(d_c);

    /* Free Host Memory */
    free(c);
    
    return 0;
}
