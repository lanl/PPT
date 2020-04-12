## Multi-Threaded Synchronized Monitoring (MTSM)

MTSM provided in this repo uses NVML API to compute the energy consumption of any GPU kernel targeting NVIDIA GPUs. The code is first compiled as a shared library and then injected (preloaded) at runtime using LD_PRELOAD to any executable file that has a GPU kernel. *libcuhook.cpp* automatically synchronize the start and end of profiling with the start and end of the kernel at runtime.

### Usage

* Compile the benchmark kernel using nvcc or llvm compiler (let's say it generates ***a.out***)

* Configure the path of your cuda toolkit (nvcc) in the Makefile

* To hook and run MTSM with the kernel's binary you have two options:
    ```
    make
    LD_PRELOAD=./libcuhook.so <./a.out>
    ```
    or
    ```
     make app=<a.out absolute PATH> run
    ```
    
