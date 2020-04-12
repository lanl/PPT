CUDA_Root ?= "/software/cuda-10.1"

NVML_LIB += $(CUDA_Root)/lib/
NVML_LIB_L := $(addprefix -L , $(NVML_LIB))

CFLAGS  := -I $(CUDA_Root)/include -I $(CUDA_Root)/include
LDFLAGS := -lnvidia-ml $(NVML_LIB_L)

power_monitor: power_monitor.o
	$(CC) $< $(CFLAGS) $(LDFLAGS) -o $@
clean:
	-rm -f power_monitor
	-rm -rf power_monitor.o
