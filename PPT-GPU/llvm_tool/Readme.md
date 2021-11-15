# CUDA Flux: A Lightweight Instruction Profiler for CUDA Applications

CUDA Flux is a profiler for GPU applications which reports the basic block executions frequencies of compute kernels

# Dependencies

* LLVM:  
  CUDA Flux is tested and developed with llvm 11.0

  ```
  git clone --branch release/11.x https://github.com/llvm/llvm-project.git
  ```

  CMake config:
  ```
  cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_INSTALL_PREFIX=/opt/llvm-11.0 \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_DOXYGEN=OFF -DLLVM_BUILD_DOCS=OFF -GNinja \
  -DLLVM_INSTALL_BINUTILS_SYMLINKS=ON -DBUILD_SHARED_LIBS=ON \
  ../llvm
  ```

* re2c lexer generator - http://re2c.org/ (make sure to check your package manager first)
* CUDA SDK >= 9.0  
* Python (python3 preferred) with the yaml package installed
* environment-modules (optional, but recommended)

# Recommended Dependencies As tested on:

* CMake 3.20
* GCC 7.3
* Cuda 10.2
* LLVM 11.1

## Install

```
# Make sure LLVM and CUDA are in your paths
git clone https://github.com/UniHD-CEG/cuda-flux.git
cd cuda-flux && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/cuda-flux .. # change install dir if you wish
make install
```

## Usage

Make sure the bin folder of your cuda-flux installation is in your path.
Better: use environment-module to load cuda-flux.
```
module load /opt/cuda-flux/module/cuda_flux # LLVM and CUDA need to be loaded first!
```

Compile your CUDA application:

```
clang_cf++ --cuda-gpu-arch=sm_35 -std=c++11 -lcudart test/saxpy.cu -o saxpy`
```

Execute the binary like usual: `./saxpy`


Output:  
If any kernel was executed there is a bbc.txt file. Each kernel execution
results in a line with the following information:
* kernel name
* gridDim{X,Y,Z}
* blockDim{X,Y,Z}
* shared Memory size
* Profiling Mode (full, CTA, warp)
* List of executions counters for each block  

The PTX instructions for all kernels are stored in the `PTX_Analysis.yml`
file. The order of the Basic Blocks corresponds with the order of the 
counters in the bbc.txt file.

Profiling mode is controlled by the environment variable `MEKONG_PROFILINGMODE`. If is is not set or set to 0 all threads are profiled, which is the recommended setting. `export MEKONG_PROFILINGMODE=1` will only profile one CTA and `export MEKONG_PROFILINGMODE=2` will profile only one warp.
