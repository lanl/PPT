CUDA_ROOT = /software/cuda-10.1

NVML_ROOT ?= $(CUDA_ROOT)
NVML_INC ?= $(CUDA_ROOT)/include
NVML_LIB ?= $(CUDA_ROOT)/lib64/stubs

ARCH_CC = 50
ARCH = arch=compute_$(ARCH_CC),code=sm_$(ARCH_CC)

NVCC = $(CUDA_ROOT)/bin/nvcc
NVCFLAGS = -L$(NVML_INC) -ccbin=$(CC)
CUDALIBS = -L$(NVML_LIB) -L$(CUDA_ROOT)/lib64 -lcuda -lcudart -lnvidia-ml -lstdc++
NONOPT := -Xptxas -O0

nonOpt: pipeline.cu $(UTILOBJS)
	@echo "Compiling Non-Optimized Version"
	@$(NVCC) $(NVCFLAGS) $(INCLUDE) --keep -gencode $(ARCH) pipeline.cu $(UTILOBJS) $(NONOPT) $(CUDALIBS)
	@$(NVCC) $(NVCFLAGS) $(INCLUDE) -dryrun -gencode $(ARCH) pipeline.cu $(UTILOBJS) $(NONOPT) $(CUDALIBS) --keep 2>dryrun.out 
	@rm pipeline.ptx
	@cp nonOpt_pipeline_temp.ptx pipeline.ptx
	@sed -i "s/^..//g" dryrun.out
	@tail -n 9 dryrun.out >dryrun.out.temp 
	@mv dryrun.out.temp dryrun.out
	@chmod +x dryrun.out
	@./dryrun.out

opt: pipeline.cu $(UTILOBJS)
	@echo "Compiling Optimized Version"
	@$(NVCC) $(NVCFLAGS) $(INCLUDE) --keep -gencode $(ARCH) pipeline.cu $(UTILOBJS) $(CUDALIBS)
	@$(NVCC) $(NVCFLAGS) $(INCLUDE) -dryrun -gencode $(ARCH) pipeline.cu $(UTILOBJS) $(CUDALIBS) --keep 2>dryrun.out 
	@rm pipeline.ptx
	@cp opt_pipeline_temp.ptx pipeline.ptx
	@sed -i "s/^..//g" dryrun.out
	@tail -n 9 dryrun.out >dryrun.out.temp 
	@mv dryrun.out.temp dryrun.out
	@chmod +x dryrun.out
	@./dryrun.out

run: $(type)
	@mkdir -p output
	@./run.sh

clean:
	rm -f a_dlink* pipeline.cpp* pipeline.cubin* pipeline.cudafe* pipeline.fatbin* pipeline.module_id pipeline.ptx
	rm -f a.out dryrun*
	rm -rf output/*

