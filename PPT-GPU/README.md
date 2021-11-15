# PPT-GPU: Performance Prediction Toolkit for GPUs


PPT-GPU is a scalable and flexible framework to predict the performance of GPUs running general purpose workloads. PPT-GPU can use the virtual (PTX) or the native (SASS) ISAs without sacrificing accuracy, ease of use, or portability. The tool is currently focused on NVIDIA GPUs. We plan to extend our approach to model other vendors' GPUs such as AMD and Intel.


### Papers

- For more information, check out the [SC' 21](https://doi.org/10.1145/3458817.3476221) paper ***(Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs)***.

    If you find this a helpful tool in your research, please consider citing as:

    ```
    @inproceedings{Arafa2021PPT-GPU,
      author = {Y. {Arafa} and A. {Badawy} and A. {ElWazir} and A. {Barai} and A. {Eker} and G. {Chennupati} and N. {Santhi} and S. {Eidenbenz}},
      title = {Hybrid, Scalable, Trace-Driven Performance Modeling of GPGPUs},
      year = {2021},
      booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
      series = {SC '21}
    }
    ```

- The memory model is descibed in the [ICS' 20](https://doi.org/10.1145/3392717.3392761) paper.
     ```
    @inproceedings{Arafa2020PPT-GPU-MEM,
      author = {Y. {Arafa} and A. {Badawy} and G. {Chennupati} and A. {Barai} and N. {Santhi} and S. {Eidenbenz}},
      title = {Fast, Accurate, and Scalable Memory Modeling of GPGPUs Using Reuse Profiles},
      year = {2020},
      booktitle = {Proceedings of the 34th ACM International Conference on Supercomputing},
      series = {ICS '20}
    }
    ```

## Dependencies

### Simulation

* Linux OS
* python v3.x   
  * scipy package (pip install scipy)
  * greenlet package (conda install -c anaconda greenlet)
  * joblib package (conda install -c anaconda joblib)
* GCC > v5.x tested with 7.3.1 and 9 on centos 8
* make
* glibc
* MPICH v.3.2.3 (if you plan to use the PDES engine to run multiple kernels in parallel)

### Extracting the traces   

* A GPU device with compute capability = 3.5 or later
* Software dependencies for extracting the memory traces and the SASS instructions traces are in the ***tracing_tool*** directory
* Software dependencies for extracting the PTX instructions traces are in the ***llvm_tool*** directory

#### see *dependecies* for the packages and versions tested on


## Steps for running  

Running simulation is straightforward. Here are the steps: 

1. **Configure & Update MPI path**
    * In *simian.py*, update ***defaultMpichLibName*** with the ibmpich.so
    * If you dont want to use MPI, update **useMPI** to False instead of True for the Simian engine paramater inside *ppt.py* file (line 278, **simianEngine** variable)

2. **Extract the traces of the application**
    * Go to ***tracing_tool*** folder and follow the instructions in the Readme file to build the tracing tool files
    * The ***tracing_tool*** extracts the application memory trace (automatically output a folder named ***memory_traces***) and the application SASS trace (automatically output a folder named ***sass_traces***). It also outputs a configuration file named **app_config.py** that has all information about the application kernels
    * For example, to get the traces for a certain application you have to call the tracer.so file that was built from the ***tracing_tool*** before running the application:   
     
      ```
      LD_PRELOAD=~/PPT-GPU/tracing_tool/tracer.so ./2mm.out
      ```
    
    * You can also extract the PTX traces using the ***llvm_tool*** for the PTX option.
    
      * Go to ***llvm_tool*** folder follow the instructions in the Readme file to build the llvm_tool 
      * (1) you need to recompile the application using llvm and clang++ compiler option, (2) execute and run the application normally 
      * There will be a ***PTX_traces*** directory that has per kernel ".ptx" traces just like the ***sass_traces***
  
 
3. **Build the Reuse Distance tool**
   * Go to ***reuse_distance_tool*** and follow the instructions in the Readme file to build the code

4. **Modeling the correct GPU configurations**  

    The ***hardware*** folder has an example of multiple hardware configurations. You can choose to model these or define your own in a new file. You can also define the ISA latencies numbers, and the compute capability configurations inside ***hardware/ISA*** and ***hardware/compute_capability***, respectively 

5. **Running the simulations**   
  
  * For Parallel kernels execution, make sure to set the right libmpich.so library path in *defaultMpichLibName* variable inside *simian.py* file**
  
  * TO RUN: 
      ```
    python ppt.py --app <application path> --sass <or --ptx> (for patx/sass instructions traces)
    --config <target GPU hardware configuration file> --granularity 2 
    ```
    
    For example, running 2mm application on TITANV with sass traces. Assuming that 2mm path is *"/home/test/Workloads/2mm"*
    ```
    python ppt.py --app /home/test/Workloads/2mm/ --sass --config TITANV --granularity 2 
    ```
    
    The above command will run all kernels sequentially, to run all kernel in parallel using the PDES engine:     
    ```
    mpirun -n 2 python ppt.py --app /home/test/Workloads/2mm/ --sass --config TITANV --granularity 2 
    ```
    
    To choose specific kernels only, (let's say in PTX traces): 

    ```
    mpirun -n 1 python ppt.py --app /home/test/Workloads/2mm/ --ptx --config TITANV --granularity 2 --kernel 1
    ```
    
    
    **Kernels are ordered in the *app_config.py* file. Please refer to the file to know the information of kernels and the orders**   
  

6. **Reading the output**

  The performance results are found inside each application file path. Outputs are per kernel.  
  

## Worklaods

You can find various GPU benchmarks that can be used in your research in the following repo: https://github.com/NMSU-PEARL/GPGPUs-Workloads 



<br />
<br />
<br />

## Classification

PPT-GPU is part of the original PPT (https://github.com/lanl/PPT) and is Unclassified and contains no Unclassified Controlled Nuclear Information. It abides with the following computer code from Los Alamos National Laboratory

  * Code Name: Performance Prediction Toolkit, C17098
  * Export Control Review Information: DOC-U.S. Department of Commerce, EAR99
  * B&R Code: YN0100000

## License

&copy 2017. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
