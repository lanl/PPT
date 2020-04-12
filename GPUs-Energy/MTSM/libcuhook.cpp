/*
** Author(s)      :  Ammar ElWazir (ammarwa@nmsu.edu) 
** 
** File           :  libcuhook.cpp  
** 
** Description    :  Multi-Threaded Synchronized Monitoring (MTSM) configured as 
**                   a shared library to be hooked with any kernel at runtime 
** 
** Paper          :  Y. Arafa et al., "Verified Instruction-Level Energy Consumption 
**                                     Measurement for NVIDIA GPUs," CF'20 
** 
** Notes          :  The CUDA Injection Methods is Provided by the Open-Source Nvidia Samples                                   
*/



#define __USE_GNU
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
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
#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cassert>
#include <map>
#include <list>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>
#include "libcuhook.h"

#define FILLED 0 
#define Ready 1 
#define NotReady -1 

#include <unistd.h>
#include <pthread.h>

#define SAMPLE_PERIOD_MS 50

/*
** Read the power metric from the device &
** save the results in a vector to be used later for energy calculation
*/
static volatile int testComplete = 0;
unsigned int device_count;
volatile float t;
volatile sig_atomic_t startFlag = 0;
volatile sig_atomic_t endFlag = 0;

nvmlDevice_t device;

void monitor_power(nvmlDevice_t device, std::vector<float> *powerArray)
{
    nvmlReturn_t result;

    unsigned int power;

    result = nvmlDeviceGetPowerUsage(device, &power);

    if (NVML_ERROR_NOT_SUPPORTED == result)
        printf("This does not support power measurement\n");
    else if (NVML_SUCCESS != result)
    {
        printf("Failed to get power for device %i: %s\n", 0, nvmlErrorString(result));
        exit(1);
    }
    if(startFlag == 1 && endFlag == 0){
        (*powerArray).push_back(power/1000);
        //printf("Power: %d mw\n", power); /* For Debugging */
        //printf("%d\n", power); /* For Debugging */
    }

}

/*
** This method is called to shutdown NVML
*/
void shutdown_nvml()
{
    nvmlReturn_t result;
    result = nvmlShutdown();
    if (NVML_SUCCESS != result)
        printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
}

volatile sig_atomic_t flag = 0;

/*
** End monitoring by setting the volatile atomic flag to zero as a signal to stop
*/
void end_monitoring(int sig) {
    flag = 1;
}

/*
** Keep monitoring power and caluclate the final energy
*/
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


nvmlReturn_t result;
cudaEvent_t start, stop;
int n=10;
pthread_t thread_id;

void initProfiling(){
       
    result = nvmlInit(); // NVML library initialization
    if (NVML_SUCCESS != result)
    { 
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        printf("Press ENTER to continue...\n");
        getchar();
        exit(1);
    }

    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (NVML_SUCCESS != result)
    { 
        printf("INIT: Failed to get handle for device %i: %s\n", 0, nvmlErrorString(result));
        exit(1);
    }

    signal(SIGINT, end_monitoring);

    //Start NMVL
    if(pthread_create(&thread_id, NULL, runMonitor, NULL)){
        printf("Thread Init Error\n");
    }
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

/*
** This Method is used to start the profiling by setting the startFlag 
** volatile variable to one
*/
void startProfiling(){
    if(startFlag == 0){
    	startFlag = 1;
    	//Starting Time Recording
    	cudaEventRecord(start, 0);
    	cudaProfilerStart();
    }
    //===================== Kernel Start ========================
}

/*
** This Method is used to stop the profiling and gather the test results
** by changing the volatile variables
*/
void stopProfiling(){
    if(endFlag == 0){
    	float time;

    	//====================== Kernel End =========================

    	cudaProfilerStop();
    	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);

    	//================= Finished Time Recording =================

    	endFlag = 1;

    	//==================== NVML POWER START =====================

    	flag = 1;
    
    	//==================== NVML POWER END =======================

    	cudaEventElapsedTime(&time, start, stop);
    	cudaDeviceSynchronize();
    	t = time;
    	printf("GPU Elapsed Time = %f ms\n",t);

    	pthread_join(thread_id, NULL);
    	shutdown_nvml();
  
    	cudaEventDestroy(start);
    	cudaEventDestroy(stop);
    }
}

/*
** constructor method called before the main function
*/
static __attribute__((constructor)) void init_method(void)
{
	initProfiling();
}

//================================CUDA Injection Tool Provided by Nvidia Samples===========================================

extern "C" { void* __libc_dlsym (void *map, const char *name); }
extern "C" { void* __libc_dlopen_mode (const char* name, int mode); }


#define STRINGIFY(x) #x
#define CUDA_SYMBOL_STRING(x) STRINGIFY(x)

typedef void* (*fnDlsym)(void*, const char*);

static void* real_dlsym(void *handle, const char* symbol)
{
    static fnDlsym internal_dlsym = (fnDlsym)__libc_dlsym(__libc_dlopen_mode("libdl.so.2", RTLD_LAZY), "dlsym");
    return (*internal_dlsym)(handle, symbol);
}

struct cuHookInfo
{
    void        *handle;
    void        *preHooks[CU_HOOK_SYMBOLS];
    void        *postHooks[CU_HOOK_SYMBOLS];

    // Debugging/Stats Info
    int         bDebugEnabled;
    int         hookedFunctionCalls[CU_HOOK_SYMBOLS];

    cuHookInfo()
    {
        const char* envHookDebug;

        // Check environment for CU_HOOK_DEBUG to facilitate debugging
        envHookDebug = getenv("CU_HOOK_DEBUG");
        if (envHookDebug && envHookDebug[0] == '1') {
            bDebugEnabled = 1;
            fprintf(stderr, "* %6d >> CUDA HOOK Library loaded.\n", getpid());
        }
    }

    ~cuHookInfo()
    {
        if (bDebugEnabled) {
            pid_t pid = getpid();
            // You can gather statistics, timings, etc.
            fprintf(stderr, "* %6d >> CUDA HOOK Library Unloaded - Statistics:\n", pid);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuMemAlloc), hookedFunctionCalls[CU_HOOK_MEM_ALLOC]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuMemFree), hookedFunctionCalls[CU_HOOK_MEM_FREE]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuCtxGetCurrent), hookedFunctionCalls[CU_HOOK_CTX_GET_CURRENT]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuCtxSetCurrent), hookedFunctionCalls[CU_HOOK_CTX_SET_CURRENT]);
            fprintf(stderr, "* %6d >> %20s ... %d\n", pid,
                    CUDA_SYMBOL_STRING(cuCtxDestroy), hookedFunctionCalls[CU_HOOK_CTX_DESTROY]);
        }
        if (handle) {
            dlclose(handle);
        }
    }

};

static struct cuHookInfo cuhl;


void cuHookRegisterCallback(HookSymbols symbol, HookTypes type, void* callback)
{
    if (type == PRE_CALL_HOOK) {
        cuhl.preHooks[symbol] = callback;
    }
    else if (type == POST_CALL_HOOK) {
        cuhl.postHooks[symbol] = callback;
    }
}

/*
** Interposed Functions
*/
void* dlsym(void *handle, const char *symbol)
{
    // Early out if not a CUDA driver symbol
    if (strncmp(symbol, "cu", 2) != 0) {
        return (real_dlsym(handle, symbol));
    }

    if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemAlloc)) == 0) {
	return (void*)(&cuMemAlloc);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuMemFree)) == 0) {
        return (void*)(&cuMemFree);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxGetCurrent)) == 0) {
	return (void*)(&cuCtxGetCurrent);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxSetCurrent)) == 0) {
	return (void*)(&cuCtxSetCurrent);
    }
    else if (strcmp(symbol, CUDA_SYMBOL_STRING(cuCtxDestroy)) == 0) {
        return (void*)(&cuCtxDestroy);
    }
    return (real_dlsym(handle, symbol));
}

#define CU_HOOK_GENERATE_INTERCEPT(hooksymbol, funcname, params, ...)   			\
    CUresult CUDAAPI funcname params                                    			\
    {                                                                   			\
        static void* real_func = (void*)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname)); 	\
        CUresult result = CUDA_SUCCESS;                                 			\
                                                                        			\
        if (cuhl.bDebugEnabled) {                                       			\
            cuhl.hookedFunctionCalls[hooksymbol]++;                     			\
        }                                                               			\
        if (cuhl.preHooks[hooksymbol]) {                                			\
            ((CUresult CUDAAPI (*)params)cuhl.preHooks[hooksymbol])(__VA_ARGS__);               \
        }											\
        result = ((CUresult CUDAAPI (*)params)real_func)(__VA_ARGS__);                          \
        if (cuhl.postHooks[hooksymbol] && result == CUDA_SUCCESS) {                             \
            ((CUresult CUDAAPI (*)params)cuhl.postHooks[hooksymbol])(__VA_ARGS__);              \
        }                                                                                       \
	startProfiling();									\
	return (result);                                                                       	\
    }

#define CU_HOOK_GENERATE_INTERCEPT1(hooksymbol, funcname, params, ...)   			\
    CUresult CUDAAPI funcname params                                    			\
    {                                                                   			\
	stopProfiling();									\
        static void* real_func = (void*)real_dlsym(RTLD_NEXT, CUDA_SYMBOL_STRING(funcname)); 	\
        CUresult result = CUDA_SUCCESS;                                 			\
                                                                        			\
        if (cuhl.bDebugEnabled) {                                       			\
            cuhl.hookedFunctionCalls[hooksymbol]++;                     			\
        }                                                               			\
        if (cuhl.preHooks[hooksymbol]) {                                			\
            ((CUresult CUDAAPI (*)params)cuhl.preHooks[hooksymbol])(__VA_ARGS__);               \
        }											\
        result = ((CUresult CUDAAPI (*)params)real_func)(__VA_ARGS__);                          \
        if (cuhl.postHooks[hooksymbol] && result == CUDA_SUCCESS) {                             \
            ((CUresult CUDAAPI (*)params)cuhl.postHooks[hooksymbol])(__VA_ARGS__);              \
        }                                                                                       \
	return (result);                                                                       	\
    }


CU_HOOK_GENERATE_INTERCEPT(CU_HOOK_MEM_ALLOC, cuMemAlloc, (CUdeviceptr *dptr, size_t bytesize), dptr, bytesize)
CU_HOOK_GENERATE_INTERCEPT1(CU_HOOK_MEM_FREE, cuMemFree, (CUdeviceptr dptr), dptr)
