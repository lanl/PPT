## Sampling Monitoring Approach (SMA)

NVML API to query the power usage of the GPU device and provide an instantaneous power measurement depending on a sampling frequency

### Usage

* Configure the path of your cuda toolkit (nvcc) in the Makefile

* To compile and run:
```
make power_monitor 
./power_monitor <sampling frequency>
```
