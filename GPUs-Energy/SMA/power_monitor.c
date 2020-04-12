#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nvml.h>
#include <signal.h>
#include <time.h>

#define GPU_NAME ""

void monitor_power(nvmlDevice_t device)
{
    nvmlReturn_t result;
    unsigned int device_count, i;

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
        printf("Failed to get power for device %i: %s\n", i, nvmlErrorString(result));
        goto Error;
    }
	time_t ltime; /* calendar time */
    ltime=time(NULL); /* get current cal time */
    printf("Power: %fW at %s\n", (power/1000.00), asctime( localtime(&ltime) ));
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

int main(int argc, char** argv) 
{   
    float sleep_useconds = 1.0/(atof(argv[1]))*1000000.00;

    nvmlReturn_t result;
    unsigned int device_count, i;

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

    printf("Found %d device%s\n\n", device_count, device_count != 1 ? "s" : "");


    for (i = 0; i < device_count; i++)
    {
        nvmlDevice_t device;
        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
	nvmlPciInfo_t pci;
        nvmlComputeMode_t compute_mode;
	unsigned int power;
        
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get handle for device %i: %s\n", i, nvmlErrorString(result));
            exit(1);
        }

        result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);
        if (NVML_SUCCESS != result)
        { 
            printf("Failed to get name of device %i: %s\n", 0, nvmlErrorString(result));
            exit(1);
        }

	signal(SIGINT, end_monitoring);

	do 
    	{
        	if (flag) {
            		break;
        	}
        	monitor_power(device);
        	usleep(sleep_useconds);
    	} while (1);
    }

    shutdown_nvml();
    
    return 0;
}
