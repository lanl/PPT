## PTX Instructions Microbenchmarks

The CPU host file (*pipeline.cu*) launches each GPU instruction's microbenchmark while using Multi-Threaded Synchronized Monitoring (*MTSM*) technique to compute its energy consumption. MTSM is implemented inside the host code where the start and the end of profiling is called when the kernel launches and finishes execution respectively. 

### Usage

* Configure the path of  Cuda toolkit (*nvcc*) in the Makefile

* Configure the *ARCH_CC* variable in the Makefile depending on the target NVIDIA GPU architecture.   
  - i.g., ***Volta TITAN V*** has a 70 SM arch generation and a 70 compute capabilty. Thus, ARCH_CC =70,  
  - i.g., ***Volta TITAN RTX*** as a 75 SM arch generation and a 75 compute capabilty. Thus, ARCH_CC =75

* To compile and run:

    **1. Run the optimized version (*-O0*):**

    ```
    make type=opt run
    ```

    **2. Run the non-optimized version (*-O3*):**

    ```
    make type=nonOpt run
    ```
    
* The results are saved in the output folder 


### Computing Instructions Energy

1. Subtract the overhead (*Ovhd file*) energy consumption from each instruction's kernel energy result

2. Divide each instruction's kernel result by (**5x10^6**) for the *opt* version and  (**20x10^6**) for the *nonOpt* version
