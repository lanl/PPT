The ***tracer_tool*** is used to extract the memory and SASS traces. This repo use and extend NVBit (NVidia Binary Instrumentation Tool) which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. Licence and agreement of NVBIT is found in the origianal [NVBIT repo](https://github.com/NVlabs/NVBit) (“This software contains source code provided by NVIDIA
    Corporation”)

NVBIT does not require application source code, any pre-compiled GPU application should work regardless of which compiler (or version) has been used (i.e. nvcc, pgicc, etc).

## Dependecies

  * A GPU with SM compute capability: >= 3.5 && <= 8.6
  * Host CPU: x86_64, ppc64le, arm64
  * OS: Linux
  * GCC version: >= 5.3.0
  * CUDA version: >= 8.0 && <= 11
  * CUDA driver version: <= 450.36
  * nvcc version for tool compilation >= 10.2

## Building the tool
  
  * Setup **ARCH** variable in the Makefile
  * run make clean; make

## Extracting the traces
  
  Assuming you have a precompiled application in *"/home/yarafa/2mm"* and the tool is built in *"/home/tracer_tool"* path. You can run the following command to get the traces
  
  ```
  LD_PRELOAD=~/PPT-GPU/tracing_tool/tracer.so ./2mm.out
  ```
  
  The above command outputs two folders ***memory_traces*** and ***sass_traces*** each has the applications kernel traces. Setup the **MAX_KERNELS** variable in ***tracer.cu*** to define the limit on the number of kernels you want to instrument in the application 
  


