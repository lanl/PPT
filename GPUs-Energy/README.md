# Verified Instruction-Level Energy Consumption Measurement for NVIDIA GPUs

1. Energy consumption measurement for the different instructions that can execute in modern NVIDIA GPUs

2. Accurate measurement of any GPU kernel's energy consumption using **Multi-Threaded Synchronized Monitoring (MTSM)**

## Hardware Dependencies

* A GPU device with cc=30+ (Kepler architecture)

## Software Dependencies

* Cuda v.10.1 or later
* GCC v.7 or later
* make

## In this Repository

* **Instructions_Microbenchmarks:** PTX microbenchmarks to stress the GPU and compute the energy consumption of each instruction using MTSM technique

* **SMA:** Query the GPU's onboard power sensors to read the instantaneous power usage of the GPU device using NVML - Sampling Monitoring Approach running in the background

* **MTSM:** Compute the energy consumption of any GPU Kernel using Multi-Threaded Synchronized Monitoring (MTSM)


## Paper

* [CF'20] Verified Instruction-Level Energy Consumption Measurement for NVIDIA GPUs

* If you find this code useful in your research, please consider citing as:

```
@inproceedings{Arafa2020GPUEnergy,
  author = {Y. {Arafa} and A. {ElWazir} and A. {ElKanishy} and Y. {Aly} and A. {Elsayed} and A. {Badawy} 
  and G. {Chennupati} and S. {Eidenbenz} and N. {Santhi}},
  title = {Verified Instruction-Level Energy Consumption Measurement for NVIDIA GPUs},
  year = {2020},
  booktitle = {Proceedings of the 17th ACM International Conference on Computing Frontiers},
  series = {CF ’20},
  url = {https://doi.org/10.1145/3387902.3392613},
  doi = {10.1145/3387902.3392613},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA}
}
```
