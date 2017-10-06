"""
 Test for GPU performance prediction.
 Original Date: January 29, 2015
*********** Performance Prediction Toolkit PPT *********

File: GPUTest.py
Description: 
 2015-11-06: Included into PPT repository
 Contains benchmark type application models for GPU tests.
 Used for the ValueTools 2015 paper
"""

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../../simian/simian-master/SimianPie')
path.append('../../hardware')

#import simian
from simian import Simian 
import clusters
import nodes

from copy import deepcopy
import math

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

########################
# 0. Initialization stuff 

simName, startTime, endTime, minDelay = "GPUTest", 0.0, 1000000.0, 0.000001
simianEngine = Simian(simName, startTime, endTime, minDelay)



########################
            
def GPUTest_naive(this, arg, *args):
    node = this.entity
    core = node.cores[0]
    # This is an example of a naive Jacobi implementation on the GPU
    # This example does not take advantage of the local memory and transfers data between every kernel
    # Computations are run over a 4096*4096 matrix with 1000 iterations before convergence
    # GPU computations use blocks of 256 threads, only the outermost loop is gridified
    nb_iter=1000
    inner_loop_size = 4096
    GPU_tasklist = [['GLOB_MEM_ACCESS', 1],['L2_ACCESS', 1],['fALU', 1],['GLOB_MEM_ACCESS',1],['L2_ACCESS', 1],['fALU', 3],['GLOB_MEM_ACCESS',3],['fALU',1],['SFU', 2],['GLOB_MEM_ACCESS',1]]
    GPU_tasklist2 = [['GLOB_MEM_ACCESS',1],['L2_ACCESS',1]]
    CPU_tasklist = [['DEVICE_ALLOC', 0, 4096*4096*4],['DEVICE_TRANSFER', 0, 4096*4096*4],['KERNEL_CALL', 0, GPU_tasklist, 128, 32*4094,10],['DEVICE_SYNC', 0],['DEVICE_ALLOC', 0, 4096*4096*4],['DEVICE_TRANSFER', 0, 4096*4096*4],\
                    ['KERNEL_CALL', 0, GPU_tasklist2, 128, 32*4094,10],['DEVICE_SYNC', 0],['DEVICE_TRANSFER', 0, 4096*4096*4],['DEVICE_TRANSFER', 0, 4096*4096*4],['DEVICE_ALLOC', 0, -4096*4096*4],['DEVICE_ALLOC', 0, -4096*4096*4]]
    
    # Compute time for a single iteration
    (time_iter,stats) = core.time_compute(CPU_tasklist, simianEngine.now, True)
    time = time_iter*nb_iter
        
    this.sleep(time)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    
def GPUTest_optimized(this, arg, *args):
    node = this.entity
    core = node.cores[0]
    # This is an example of an optimized Jacobi implementation on the GPU
    # Computations are run over a 4096*4096 matrix with 1000 iterations before convergence
    # GPU computations use blocks of 256 threads, only the outermost loop is gridified
    nb_iter=1000
    inner_loop_size = 4096
    GPU_tasklist = [['GLOB_MEM_ACCESS', 1],['L2_ACCESS', 1],['fALU', 1],['GLOB_MEM_ACCESS',1],['L2_ACCESS', 1],['fALU', 3],['GLOB_MEM_ACCESS',3],['fALU',1],['SFU', 2],['GLOB_MEM_ACCESS',1]]
    GPU_tasklist2 = [['GLOB_MEM_ACCESS',1],['L2_ACCESS',1]]
    CPU_tasklist1 = [['DEVICE_ALLOC', 0, 4096*4096*4],['DEVICE_ALLOC', 0, 4096*4096*4],['DEVICE_TRANSFER', 0, 4096*4096*4]]
    CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, 128, 32*4094,10],['DEVICE_SYNC', 0],\
                    ['KERNEL_CALL', 0, GPU_tasklist2, 128, 32*4094,10],['DEVICE_SYNC', 0]]
    CPU_tasklist3 = [['DEVICE_TRANSFER', 0, 4096*4096*4],['DEVICE_TRANSFER', 0, 4096*4096*4],['DEVICE_ALLOC', 0, -4096*4096*4],['DEVICE_ALLOC', 0, -4096*4096*4]]
    
    # Compute time for a single iteration
    (time_init,stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
    this.sleep(time_init)
    (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
    time = time_iter*nb_iter
    this.sleep(time)
    (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True) 
    this.sleep(time_finalize)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    
def simpleDaxpy(this, arg, *args):
    regcount=10
    node = this.entity
    core = node.cores[0]
    # This is a simple example of a Daxpy implementation on the GPU
    nb_iter=10000
    vector_size = 2048*14*128
    blocksize = 256 
    gridsize = int((vector_size+blocksize-1)/blocksize)
    GPU_tasklist  = [['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU'], ['iALU', 3], ['diALU', 1], ['iALU', 3], ['diALU', 5, 6], ['GLOB_MEM_ACCESS', 7], ['diALU', 2], ['diALU', 9, 6], ['GLOB_MEM_ACCESS', 10], ['dfALU', 8, 0, 11], ['GLOB_MEM_ACCESS', 12]]
    CPU_tasklist1 = [['DEVICE_ALLOC', 0, vector_size*8],['DEVICE_ALLOC', 0, vector_size*8],['DEVICE_TRANSFER', 0, vector_size*8],['DEVICE_TRANSFER', 0, vector_size*8]]
    CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize,regcount],['DEVICE_SYNC', 0]]
    CPU_tasklist3 = [['DEVICE_TRANSFER', 0, vector_size*8],['DEVICE_ALLOC', 0, -vector_size*8],['DEVICE_ALLOC', 0, -vector_size*8]]
    
    # Compute time for a single iteration
    (time_init,stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
    this.sleep(time_init)
    (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
    print "Time for a single iteration = ", time_iter
    time = time_iter*nb_iter
    this.sleep(time)
    (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True) 
    this.sleep(time_finalize)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    
def ComputeBound(this, arg, *args):
    regcount=8
    node = this.entity
    core = node.cores[0]
    # This is a simple example of a compute bound kernel on the GPU
    nb_iter=1000
    vector_size = 2048*14*128
    blocksize = 256
    gridsize = int((vector_size+blocksize-1)/blocksize)
    GPU_tasklist  = [['L1_ACCESS'], ['L1_ACCESS'], ['diALU', 0], ['iALU'], ['iALU', 3], ['diALU', 2, 4], ['GLOB_MEM_ACCESS', 5], ['iALU', 1], ['fALU', 7, 6, 7], ['fALU', 6, 8], ['fALU', 7, 9, 7], ['fALU', 9, 10], ['fALU', 7, 11, 7], ['fALU', 11, 12], ['fALU', 7, 13, 7], ['fALU', 13, 14], ['fALU', 7, 15, 7], ['fALU', 15, 16], ['fALU', 7, 17, 7], ['fALU', 17, 18], ['fALU', 7, 19, 7], ['fALU', 19, 20], ['fALU', 7, 21, 7], ['fALU', 21, 22], ['fALU', 7, 23, 7], ['fALU', 23, 24], ['fALU', 7, 25, 7], ['fALU', 25, 26], ['fALU', 7, 27, 7], ['fALU', 27, 28], ['fALU', 7, 29, 7], ['fALU', 29, 30], ['fALU', 7, 31, 7], ['fALU', 31, 32], ['fALU', 7, 33, 7], ['fALU', 33, 34], ['fALU', 7, 35, 7], ['fALU', 35, 36], ['fALU', 7, 37, 7], ['fALU', 37, 38], ['fALU', 7, 39, 7], ['fALU', 39, 40], ['fALU', 7, 41, 7], ['fALU', 41, 42], ['fALU', 7, 43, 7], ['fALU', 43, 44], ['fALU', 7, 45, 7], ['fALU', 45, 46], ['fALU', 7, 47, 7], ['fALU', 47, 48], ['fALU', 7, 49, 7], ['fALU', 49, 50], ['fALU', 7, 51, 7], ['fALU', 51, 52], ['fALU', 7, 53, 7], ['fALU', 53, 54], ['fALU', 7, 55, 7], ['fALU', 55, 56], ['fALU', 7, 57, 7], ['fALU', 57, 58], ['fALU', 7, 59, 7], ['fALU', 59, 60], ['fALU', 7, 61, 7], ['fALU', 61, 62], ['fALU', 7, 63, 7], ['fALU', 63, 64], ['fALU', 7, 65, 7], ['fALU', 65, 66], ['fALU', 7, 67, 7], ['fALU', 67, 68], ['fALU', 7, 69, 7], ['fALU', 69, 70], ['fALU', 7, 71, 7], ['fALU', 71, 72], ['fALU', 7, 73, 7], ['fALU', 73, 74], ['fALU', 7, 75, 7], ['fALU', 75, 76], ['fALU', 7, 77, 7], ['fALU', 77, 78], ['fALU', 7, 79, 7], ['fALU', 79, 80], ['fALU', 7, 81, 7], ['fALU', 81, 82], ['fALU', 7, 83, 7], ['fALU', 83, 84], ['fALU', 7, 85, 7], ['fALU', 85, 86], ['GLOB_MEM_ACCESS', 87]]

    CPU_tasklist1 = [['DEVICE_ALLOC', 0, vector_size*4],['DEVICE_TRANSFER', 0, vector_size*4]]
    CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize,regcount],['DEVICE_SYNC', 0]]
    CPU_tasklist3 = [['DEVICE_TRANSFER', 0, vector_size*4],['DEVICE_ALLOC', 0, -vector_size*4]]
    
    # Compute time for a single iteration
    (time_init,stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
    this.sleep(time_init)
    (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
    print "Time for a single iteration = ", time_iter
    time = time_iter*nb_iter
    this.sleep(time)
    (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True) 
    this.sleep(time_finalize)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")

def SGEMM(this, arg, *args):
    regcount=8
    node = this.entity
    core = node.cores[0]
    # This is a simple example of a compute bound kernel on the GPU
    nb_iter=1000
    vector_size = 16384
    blocksize = 16*4
    gridsize = int((vector_size/blocksize)*(4*vector_size/blocksize))
    print("Block size is:", blocksize)
    print("Grid size is:", gridsize)
    GPU_tasklist  = [['iALU'], ['iALU'], ['iALU'], ['L1_ACCESS'], ['iALU', 1], ['iALU', 2, 4], ['L1_ACCESS'], ['iALU', 6, 0], ['iALU', 5, 7], ['iALU', 8], ['iALU', 8], ['diALU', 3, 10], ['GLOB_MEM_ACCESS'], ['iALU', 2], ['iALU', 0], ['iALU', 0], ['diALU', 13, 15], ['diALU', 16], ['diALU', 17], ['L1_ACCESS', 12], ['THREAD_SYNC'], ['iALU'], ['iALU', 21, 0], ['iALU', 22, 2], ['iALU'], ['iALU', 23, 24], ['L1_ACCESS'], ['iALU', 25], ['iALU', 25], ['diALU', 26, 28], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['fALU', 31, 30], ['L1_ACCESS'], ['fALU', 33, 30], ['L1_ACCESS'], ['fALU', 35, 30], ['L1_ACCESS'], ['fALU', 37, 30], ['L1_ACCESS'], ['fALU', 39, 30], ['L1_ACCESS'], ['fALU', 41, 30], ['L1_ACCESS'], ['fALU', 43, 30], ['L1_ACCESS'], ['fALU', 45, 30], ['L1_ACCESS'], ['fALU', 47, 30], ['L1_ACCESS'], ['fALU', 49, 30], ['L1_ACCESS'], ['fALU', 51, 30], ['L1_ACCESS'], ['fALU', 53, 30], ['L1_ACCESS'], ['fALU', 55, 30], ['L1_ACCESS'], ['fALU', 57, 30], ['L1_ACCESS'], ['fALU', 59, 30], ['L1_ACCESS'], ['fALU', 61, 30], ['L1_ACCESS'], ['iALU', 63, 25], ['iALU', 64], ['iALU', 64], ['diALU', 26, 66], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['fALU', 69, 68, 32], ['L1_ACCESS'], ['fALU', 71, 68, 34], ['L1_ACCESS'], ['fALU', 73, 68, 36], ['L1_ACCESS'], ['fALU', 75, 68, 38], ['L1_ACCESS'], ['fALU', 77, 68, 40], ['L1_ACCESS'], ['fALU', 79, 68, 42], ['L1_ACCESS'], ['fALU', 81, 68, 44], ['L1_ACCESS'], ['fALU', 83, 68, 46], ['L1_ACCESS'], ['fALU', 85, 68, 48], ['L1_ACCESS'], ['fALU', 87, 68, 50], ['L1_ACCESS'], ['fALU', 89, 68, 52], ['L1_ACCESS'], ['fALU', 91, 68, 54], ['L1_ACCESS'], ['fALU', 93, 68, 56], ['L1_ACCESS'], ['fALU', 95, 68, 58], ['L1_ACCESS'], ['fALU', 97, 68, 60], ['L1_ACCESS'], ['fALU', 99, 68, 62], ['iALU', 63], ['iALU', 25, 101], ['iALU', 102], ['iALU', 102], ['diALU', 26, 104], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['fALU', 107, 106, 70], ['L1_ACCESS'], ['fALU', 109, 106, 72], ['L1_ACCESS'], ['fALU', 111, 106, 74], ['L1_ACCESS'], ['fALU', 113, 106, 76], ['L1_ACCESS'], ['fALU', 115, 106, 78], ['L1_ACCESS'], ['fALU', 117, 106, 80], ['L1_ACCESS'], ['fALU', 119, 106, 82], ['L1_ACCESS'], ['fALU', 121, 106, 84], ['L1_ACCESS'], ['fALU', 123, 106, 86], ['L1_ACCESS'], ['fALU', 125, 106, 88], ['L1_ACCESS'], ['fALU', 127, 106, 90], ['L1_ACCESS'], ['fALU', 129, 106, 92], ['L1_ACCESS'], ['fALU', 131, 106, 94], ['L1_ACCESS'], ['fALU', 133, 106, 96], ['L1_ACCESS'], ['fALU', 135, 106, 98], ['L1_ACCESS'], ['fALU', 137, 106, 100], ['iALU', 63], ['iALU', 25, 139], ['iALU', 140], ['iALU', 140], ['diALU', 26, 142], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['fALU', 145, 144, 108], ['L1_ACCESS'], ['fALU', 147, 144, 110], ['L1_ACCESS'], ['fALU', 149, 144, 112], ['L1_ACCESS'], ['fALU', 151, 144, 114], ['L1_ACCESS'], ['fALU', 153, 144, 116], ['L1_ACCESS'], ['fALU', 155, 144, 118], ['L1_ACCESS'], ['fALU', 157, 144, 120], ['L1_ACCESS'], ['fALU', 159, 144, 122], ['L1_ACCESS'], ['fALU', 161, 144, 124], ['L1_ACCESS'], ['fALU', 163, 144, 126], ['L1_ACCESS'], ['fALU', 165, 144, 128], ['L1_ACCESS'], ['fALU', 167, 144, 130], ['L1_ACCESS'], ['fALU', 169, 144, 132], ['L1_ACCESS'], ['fALU', 171, 144, 134], ['L1_ACCESS'], ['fALU', 173, 144, 136], ['L1_ACCESS'], ['fALU', 175, 144, 138], ['THREAD_SYNC'], ['L1_ACCESS'], ['iALU', 178, 1], ['iALU', 179], ['iALU', 180, 25], ['L1_ACCESS'], ['iALU', 181], ['iALU', 181], ['diALU', 182, 184], ['L1_ACCESS'], ['L1_ACCESS'], ['fALU', 146, 187], ['GLOB_MEM_ACCESS'], ['fALU', 189, 186, 188], ['GLOB_MEM_ACCESS', 190], ['iALU', 178, 181], ['iALU', 192], ['iALU', 192], ['diALU', 182, 194], ['fALU', 148, 187], ['GLOB_MEM_ACCESS'], ['fALU', 197, 186, 196], ['GLOB_MEM_ACCESS', 198], ['iALU', 178], ['iALU', 181, 200], ['iALU', 201], ['iALU', 201], ['diALU', 182, 203], ['fALU', 150, 187], ['GLOB_MEM_ACCESS'], ['fALU', 206, 186, 205], ['GLOB_MEM_ACCESS', 207], ['iALU', 178], ['iALU', 181, 209], ['iALU', 210], ['iALU', 210], ['diALU', 182, 212], ['fALU', 152, 187], ['GLOB_MEM_ACCESS'], ['fALU', 215, 186, 214], ['GLOB_MEM_ACCESS', 216], ['iALU', 178], ['iALU', 181, 218], ['iALU', 219], ['iALU', 219], ['diALU', 182, 221], ['fALU', 154, 187], ['GLOB_MEM_ACCESS'], ['fALU', 224, 186, 223], ['GLOB_MEM_ACCESS', 225], ['iALU', 178], ['iALU', 181, 227], ['iALU', 228], ['iALU', 228], ['diALU', 182, 230], ['fALU', 156, 187], ['GLOB_MEM_ACCESS'], ['fALU', 233, 186, 232], ['GLOB_MEM_ACCESS', 234], ['iALU', 178], ['iALU', 181, 236], ['iALU', 237], ['iALU', 237], ['diALU', 182, 239], ['fALU', 158, 187], ['GLOB_MEM_ACCESS'], ['fALU', 242, 186, 241], ['GLOB_MEM_ACCESS', 243], ['iALU', 178], ['iALU', 181, 245], ['iALU', 246], ['iALU', 246], ['diALU', 182, 248], ['fALU', 160, 187], ['GLOB_MEM_ACCESS'], ['fALU', 251, 186, 250], ['GLOB_MEM_ACCESS', 252], ['iALU', 178], ['iALU', 181, 254], ['iALU', 255], ['iALU', 255], ['diALU', 182, 257], ['fALU', 162, 187], ['GLOB_MEM_ACCESS'], ['fALU', 260, 186, 259], ['GLOB_MEM_ACCESS', 261], ['iALU', 178], ['iALU', 181, 263], ['iALU', 264], ['iALU', 264], ['diALU', 182, 266], ['fALU', 164, 187], ['GLOB_MEM_ACCESS'], ['fALU', 269, 186, 268], ['GLOB_MEM_ACCESS', 270], ['iALU', 178], ['iALU', 181, 272], ['iALU', 273], ['iALU', 273], ['diALU', 182, 275], ['fALU', 166, 187], ['GLOB_MEM_ACCESS'], ['fALU', 278, 186, 277], ['GLOB_MEM_ACCESS', 279], ['iALU', 178], ['iALU', 181, 281], ['iALU', 282], ['iALU', 282], ['diALU', 182, 284], ['fALU', 168, 187], ['GLOB_MEM_ACCESS'], ['fALU', 287, 186, 286], ['GLOB_MEM_ACCESS', 288], ['iALU', 178], ['iALU', 181, 290], ['iALU', 291], ['iALU', 291], ['diALU', 182, 293], ['fALU', 170, 187], ['GLOB_MEM_ACCESS'], ['fALU', 296, 186, 295], ['GLOB_MEM_ACCESS', 297], ['iALU', 178], ['iALU', 181, 299], ['iALU', 300], ['iALU', 300], ['diALU', 182, 302], ['fALU', 172, 187], ['GLOB_MEM_ACCESS'], ['fALU', 305, 186, 304], ['GLOB_MEM_ACCESS', 306], ['iALU', 178], ['iALU', 181, 308], ['iALU', 309], ['iALU', 309], ['diALU', 182, 311], ['fALU', 174, 187], ['GLOB_MEM_ACCESS'], ['fALU', 314, 186, 313], ['GLOB_MEM_ACCESS', 315], ['iALU', 178], ['iALU', 181, 317], ['iALU', 318], ['iALU', 318], ['diALU', 182, 320], ['fALU', 176, 187], ['GLOB_MEM_ACCESS'], ['fALU', 323, 186, 322], ['GLOB_MEM_ACCESS', 324]]
    
    CPU_tasklist1 = [['DEVICE_ALLOC', 0, vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*4]]
    CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize,regcount],['DEVICE_SYNC', 0]]
    CPU_tasklist3 = [['DEVICE_TRANSFER', 0, vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*4]]
    
    # Compute time for a single iteration
    (time_init,stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
    this.sleep(time_init)
    print "Time for initialization = ", time_init
    (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
    print "Time for a single iteration = ", time_iter
    time = time_iter*nb_iter
    this.sleep(time)
    (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
    print "Time for finalization = ", time_finalize 
    this.sleep(time_finalize)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    
def STENCIL(this, arg, *args):
    regcount=18
    node = this.entity
    core = node.cores[0]
    # This is a simple example of a compute bound kernel on the GPU
    nb_iter=1000
    vector_size = 756
    blocksize = vector_size
    gridsize = int((vector_size-2)*(vector_size-2))
    print("Block size is:", blocksize)
    print("Grid size is:", gridsize)
    GPU_tasklist  = [['iALU'], ['iALU', 0], ['iALU'], ['iALU', 2], ['L1_ACCESS'], ['iALU', 3, 4], ['iALU'], ['iALU', 5, 6], ['iALU', 7], ['L1_ACCESS'], ['iALU', 8, 9], ['iALU', 0, 10], ['iALU', 11], ['iALU', 11], ['L1_ACCESS'], ['diALU', 13, 14], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['fALU', 16, 17], ['L1_ACCESS'], ['GLOB_MEM_ACCESS'], ['GLOB_MEM_ACCESS'], ['iALU', 7, 9], ['iALU', 0, 22], ['iALU', 23], ['iALU', 23], ['diALU', 14, 25], ['GLOB_MEM_ACCESS'], ['iALU', 7], ['iALU', 9, 28], ['iALU', 0, 29], ['iALU', 30], ['iALU', 30], ['diALU', 14, 32], ['GLOB_MEM_ACCESS'], ['iALU', 4, 2], ['iALU', 6, 35], ['iALU', 36], ['iALU', 9, 37], ['iALU', 0, 38], ['iALU', 39], ['iALU', 39], ['diALU', 14, 41], ['GLOB_MEM_ACCESS'], ['iALU', 2], ['iALU', 4, 44], ['iALU', 6, 45], ['iALU', 46], ['iALU', 9, 47], ['iALU', 0, 48], ['iALU', 49], ['iALU', 49], ['diALU', 14, 51], ['GLOB_MEM_ACCESS'], ['fALU', 43, 53], ['fALU', 34, 54], ['fALU', 27, 55], ['fALU', 21, 56], ['fALU', 20, 57], ['fALU', 19, 58], ['fALU', 59, 18], ['L1_ACCESS'], ['diALU', 61, 13], ['GLOB_MEM_ACCESS', 60]]
    CPU_tasklist1 = [['DEVICE_ALLOC', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4]]
    CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize,regcount],['DEVICE_SYNC', 0]]
    CPU_tasklist3 = [['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*vector_size*4]]
    
    # Compute time for a single iteration
    (time_init,stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
    this.sleep(time_init)
    print "Time for initialization = ", time_init
    (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
    print "Time for a single iteration = ", time_iter
    time = time_iter*nb_iter
    this.sleep(time)
    (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
    print "Time for finalization = ", time_finalize 
    this.sleep(time_finalize)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    
    
def OPT_STENCIL(this, arg, *args):
    regcount=25
    node = this.entity
    core = node.cores[0]
    # This is a simple example of a compute bound kernel on the GPU
    nb_iter=1000
    vector_size = 1024
    tx = 32
    ty = 4
    blocksize = tx*ty
    gridsize = int(((vector_size+tx*2-1)/(tx*2))*((vector_size)/ty))
    print("Block size is:", blocksize)
    print("Grid size is:", gridsize)
    GPU_tasklist  = [['iALU'], ['iALU'], ['iALU', 0, 1], ['iALU', 2], ['iALU'], ['iALU', 3, 4], ['iALU', 5], ['iALU', 5], ['diALU', 7], ['L1_ACCESS'], ['iALU', 4, 1], ['iALU', 10, 3], ['iALU', 11], ['iALU', 11], ['diALU', 13], ['L1_ACCESS'], ['THREAD_SYNC'], ['iALU'], ['iALU', 17, 1], ['iALU'], ['iALU'], ['iALU', 20, 19], ['iALU', 18], ['iALU', 21, 0], ['iALU', 22, 4], ['L1_ACCESS'], ['iALU', 25, 23], ['L1_ACCESS'], ['iALU', 27, 24], ['iALU', 26], ['iALU', 28], ['iALU', 29, 30], ['iALU', 31], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU', 34, 23], ['iALU', 24, 35], ['iALU', 36], ['iALU', 36], ['diALU', 33, 38], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['iALU', 41, 23], ['iALU', 34, 42], ['iALU', 24, 43], ['iALU', 44], ['iALU', 44], ['diALU', 33, 46], ['GLOB_MEM_ACCESS'], ['L1_ACCESS', 48], ['iALU', 24, 1], ['L1_ACCESS'], ['iALU', 51, 50], ['iALU', 52], ['iALU', 29, 53], ['iALU', 54], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU', 57, 23], ['iALU', 50, 58], ['iALU', 59], ['iALU', 59], ['diALU', 56, 61], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['iALU', 64, 23], ['iALU', 57, 65], ['iALU', 50, 66], ['iALU', 67], ['iALU', 67], ['diALU', 56, 69], ['GLOB_MEM_ACCESS'], ['L1_ACCESS', 71], ['THREAD_SYNC'], ['L1_ACCESS'], ['iALU', 74], ['iALU', 75], ['iALU', 19], ['iALU', 77, 0], ['iALU', 78], ['iALU', 0], ['iALU', 80], ['iALU', 1], ['iALU', 82], ['iALU', 10, 83], ['iALU', 84], ['iALU', 4], ['iALU', 86], ['L1_ACCESS'], ['iALU', 88], ['iALU', 64], ['iALU', 57], ['iALU', 90, 23], ['iALU', 23], ['iALU', 91, 24], ['iALU', 24], ['iALU', 92], ['iALU', 93], ['iALU', 91, 50], ['iALU', 94], ['iALU', 95], ['iALU', 98], ['iALU', 97, 100], ['iALU', 97, 101], ['iALU', 99, 102], ['iALU', 96, 103], ['iALU', 96, 104]]
    GPU_loop = [['iALU'], ['L1_ACCESS'], ['iALU'], ['iALU', 2], ['iALU', 3], ['iALU', 4], ['iALU', 5], ['iALU', 6], ['iALU', 6], ['diALU', 1, 8], ['GLOB_MEM_ACCESS'], ['iALU'], ['iALU'], ['iALU', 12], ['L1_ACCESS'], ['iALU', 13], ['iALU', 15], ['iALU', 16], ['iALU', 17], ['iALU', 17], ['diALU', 14, 19], ['GLOB_MEM_ACCESS'], ['iALU'], ['iALU', 22], ['iALU', 22], ['diALU', 24], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU', 13], ['iALU', 28], ['iALU', 29], ['iALU', 30], ['iALU', 30], ['diALU', 27, 32], ['GLOB_MEM_ACCESS'], ['iALU'], ['iALU', 35], ['iALU', 35], ['diALU', 37], ['L1_ACCESS'], ['iALU', 13], ['iALU', 40], ['iALU', 41], ['iALU', 41], ['L1_ACCESS'], ['diALU', 44, 43], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['fALU', 48, 49], ['L1_ACCESS'], ['L1_ACCESS'], ['fALU', 10], ['fALU', 26, 53], ['fALU', 39, 54], ['fALU', 52, 55], ['fALU', 47, 56], ['fALU', 51, 57], ['fALU', 58, 50], ['L1_ACCESS'], ['diALU', 60, 43], ['GLOB_MEM_ACCESS', 59], ['iALU'], ['L1_ACCESS'], ['iALU'], ['iALU', 65], ['iALU', 66], ['iALU', 67], ['iALU', 68], ['iALU', 69], ['iALU', 69], ['diALU', 64, 71], ['GLOB_MEM_ACCESS'], ['iALU'], ['iALU'], ['iALU', 75], ['L1_ACCESS'], ['iALU', 76], ['iALU', 78], ['iALU', 79], ['iALU', 80], ['iALU', 80], ['diALU', 77, 82], ['GLOB_MEM_ACCESS'], ['iALU'], ['iALU', 85], ['iALU', 85], ['diALU', 87], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU', 76], ['iALU', 91], ['iALU', 92], ['iALU', 93], ['iALU', 93], ['diALU', 90, 95], ['GLOB_MEM_ACCESS'], ['iALU'], ['iALU', 98], ['iALU', 98], ['diALU', 100], ['L1_ACCESS'], ['iALU', 76], ['iALU', 103], ['iALU', 104], ['iALU', 104], ['L1_ACCESS'], ['diALU', 107, 106], ['GLOB_MEM_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['fALU', 111, 112], ['L1_ACCESS'], ['L1_ACCESS'], ['fALU', 73], ['fALU', 89, 116], ['fALU', 102, 117], ['fALU', 110, 118], ['fALU', 115, 119], ['fALU', 114, 120], ['fALU', 121, 113], ['L1_ACCESS'], ['diALU', 123, 106], ['GLOB_MEM_ACCESS', 122], ['THREAD_SYNC'], ['L1_ACCESS'], ['L1_ACCESS', 10], ['L1_ACCESS'], ['L1_ACCESS', 73], ['THREAD_SYNC'], ['iALU'], ['iALU', 132]]
    last_inst = len(GPU_tasklist)
    for inst in GPU_loop:
        for i in range(1,len(inst)):
            inst[i]+=last_inst
    for j in xrange(vector_size-1):
        for inst in GPU_loop:
            GPU_tasklist.append(list(inst))
            for i in range(1,len(inst)):
                inst[i]+=len(GPU_loop)
    CPU_tasklist1 = [['DEVICE_ALLOC', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4]]
    CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize,regcount],['DEVICE_SYNC', 0]]
    CPU_tasklist3 = [['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_TRANSFER', 0, vector_size*vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*vector_size*4],
                     ['DEVICE_ALLOC', 0, -vector_size*vector_size*vector_size*4]]
    
    # Compute time for a single iteration
    (time_init,stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
    this.sleep(time_init)
    print "Time for initialization = ", time_init
    (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
    print "Time for a single iteration = ", time_iter
    time = time_iter*nb_iter
    this.sleep(time)
    (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
    print "Time for finalization = ", time_finalize 
    this.sleep(time_finalize)
    this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                     " computations completed on core id "+str(0)+"; execution time: "+\
                     str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    
def GPUTest_Handler(self, msg, *args):
    self.createProcess("STENCIL", STENCIL) 
    self.startProcess("STENCIL", self)   

################################
# "MAIN"
################################



# 1. Choose and instantiate the Cluster that we want to simulate 

cluster = clusters.Titan(simianEngine)
#cluster = clusters.HalfTrinity(simianEngine)

# 2. Create a GPUtest Service on the node
simianEngine.attachService(nodes.Node, "GPUTest_Handler" , GPUTest_Handler)

simianEngine.schedService(0, "GPUTest_Handler", None, "Node", 0)
    
# 3. Run simx
simianEngine.run()
simianEngine.exit()

