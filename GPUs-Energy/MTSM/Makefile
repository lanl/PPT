CUDA_ROOT = /software/cuda-10.1

NVML_ROOT ?= $(CUDA_ROOT)
NVML_INC ?= $(CUDA_ROOT)/include
NVML_LIB ?= $(CUDA_ROOT)/lib64/stubs

NVCC = $(CUDA_ROOT)/bin/nvcc
NVCFLAGS = -ccbin g++ -shared -m64 --compiler-options '-fPIC' 
GENCODE = -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
CUDALIBS = -L$(CUDA_ROOT)/lib64i -L$(NVML_LIB) -L$(CUDA_ROOT)/extras/CUPTI/lib64 -lcuda -lcudart -lnvidia-ml -lstdc++ -lm  -ldl -I$(NVML_INC) -I$(CUDA_ROOT)/extras/CUPTI/include

all: libcuhook.so

libcuhook.so: libcuhook.o
	$(NVCC) $(NVCFLAGS) $(GENCODE) $(CUDALIBS) -o libcuhook.so libcuhook.o

libcuhook.o: libcuhook.cpp 
	$(NVCC) $(NVCFLAGS) $(GENCODE) $(CUDALIBS) -ldl -o libcuhook.o -c libcuhook.cpp

run: all
	@rm -rf *.log
	@echo "========================================================" >> output.log
	@echo "Results for Running ".$(app) >> output.log
	@echo "========================================================" >> output.log
	@LD_PRELOAD=./libcuhook.so $(app) >> output.log

clean:
	rm -rf *.o *.so *.log
