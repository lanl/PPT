## PPT-GPU

Scalable GPU Performance Modeling

### Files: 

* **_configs-->arch_latencies_config:_** The latencies for various ALU and memory operations for the target architecture/generation are defined in this file. These latencies is needed for getting the application's tasklist from the corresponding the PTX file.

* **_configs-->gpu_config:_** The modeled GPU configurations are defined this file.

* **_PTXParser.py:_** This script takes the application's PTX as input and output the it's corresponding tasklist 

* **_gpu_app.py:_** This contains the main application to run on the modeled GPU

* **_hardware-->accelerators.py:_** This contain the main algorithm for computing the runtime and performance

### The steps of getting any application's performance:

1. **_Extract the PTX file from CUDA source file_**   
    
    A real hardware (GPU) is not needed only a nvcc CUDA compiler installed locally for CUDA source files. This is usaually done by the following command: 
    > nvcc --ptx [ compiler options ] app.cuda
    
2. **_Get the application's tasklist for a given application's kernel_**   
    
    - The application's tasklist rely on the architecture/generation ALU and memory latencies which is defined in arch_latencies_config. The latencies for Kepler, Maxwell, and Pascal are already defined there. To add any other architectue add a new class in there and call it when using the PTXParser.      
    - We get a tasklist per kernel in the application, you have to pass the target kernel name while running 
    - For each loop in the target kernel add the iteration counts seprated by comas in the the order they presented in the source code.   
    
     > python PTXParser.py [ PTX File ] [ Target Kernel Name ] [ Loop Iteration Counts ] [ Target GPU Generation ]  
     
   For instance if you want to get the tasklist of gaussian application which you will run on a GPU from Kepler architecture.   

    > python PTXParser.py benchmarks/Rodinia/gaussian/gaussian.ptx _Z4Fan2PfS_S_iii 1 Kepler

3. **_Run the application on the target GPU_**
 
    
    - You must choose a class from gpu_config for your target GPU configuration. The configurations for K40m GPU from Kepler architecture and TitanX GPU from Maxwell from architecture are already defined there. To add any other GPU, add a new class in there and update the applications's handler in gpu_app.py file.
    
    - Define your function which has applications's tasklist and couple of other paramaters such as number of register, allocated static shared memory, L2 hit rates and update the application's handler with this function.
    
    After adding everything and updating the handler, runing the application is simple as just runing the gpu_app.py 
    
    > python gpu_app.py
    
    - The output of the prediction can be seen in a file named **perf.0.out** 
 
### Notes: 

* _In order to have a high accuracy within **90%** of the real device, you have to add an accurate taget GPU configuration and per application's loop iteration count, number of registers, static shared memeory, and L2 hit rates._

* _**If you have used PPT-GPU in your research, please cite:**_

Y. Arafa, A. A. Badawy, G. Chennupati, N. Santhi and S. Eidenbenz, "PPT-GPU: Scalable GPU Performance Modeling," in IEEE Computer Architecture Letters, vol. 18, no. 1, pp. 55-58, 1 Jan.-June 2019.
doi: 10.1109/LCA.2019.2904497, URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8665984&isnumber=8610345


