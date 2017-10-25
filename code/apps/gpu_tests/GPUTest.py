"""
 Test for GPU performance prediction.
 Original Date: January 29, 2015
*********** Performance Prediction Toolkit PPT *********

File: GPUTest.py
Description: 
 2015-11-06: Included into PPT repository
 Contains benchmark type application models for GPU tests.
 Used for the ValueTools 2015 paper

 2016-12-15: MR added LU_APP test
"""

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
from sys import path
path.append('../..')
from ppt import *

#import simian
from simian import Simian 
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


def LU_APP(this, arg, *args):
    node = this.entity
    core = node.cores[0]
    # This attempts to simulate the call to MAGMA's dgetrf(s)_gpu from a proxy app for cwbjsim.
    # For now we assume a single iteration (Nit = 1) over a single spatial cell (Nc = 1),
    # in other words, we are solving one single linear system. We are also making the simplifying
    # assumption that the bulk of the GPU work is in the call to dgemm_fermi.cu's kernel,
    # ignoring all other work.
    ng = 1 # number of groups
    ns = 18 # order of SquareCL sn quadrature (even)
    na = ns*ns # number of quadrature angles
    nv = 3 # number of finite element vertices (assume triangle)
    N = ng*na*nv # linear system size
    print "N = ", N
    Np = ((N + 31)/32)*32 # closest multiple of 32 >= N 
    print "Np = ", Np
    A_dev_size = N*Np 
    b_dev_size = N
    if ns == 6:
      nb_iter = 2 # number of calls to dgemm_fermi's kernel (hard-wired for ns = 6)
      blocksize = 256 # hard-wired for ns = 6
      gridsize = 1 # hard-wired for ns = 6
      print("Block size is:", blocksize)
      print("Grid size is:", gridsize)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 8:
      nb_iter_1 = 4 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 2 # number of calls to dgemm_fermi's kernel for gridsize_2
      blocksize = 256 
      gridsize_1 = 1
      gridsize_2 = 2 
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 10:
      nb_iter_1 = 6 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 6
      gridsize_6 = 12
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized 
    elif ns == 12:
      nb_iter_1 = 8 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 2 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 12
      gridsize_8 = 20
      gridsize_9 = 30
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 14:
      nb_iter_1 = 11 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 1 # number of calls to dgemm_fermi's kernel for gridsize_15
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 12
      gridsize_11 = 20
      gridsize_12 = 30
      gridsize_13 = 42
      gridsize_14 = 56
      gridsize_15 = 72
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 16:
      nb_iter_1 = 13 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 2 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 2 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 1 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 1 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 1 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 10
      gridsize_11 = 11
      gridsize_12 = 12
      gridsize_13 = 20
      gridsize_14 = 30
      gridsize_15 = 42
      gridsize_16 = 56
      gridsize_17 = 72
      gridsize_18 = 90
      gridsize_19 = 110
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 18:
      nb_iter_1 = 17 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 2 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 2 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 2 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 3 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 2 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 2 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 1 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 1 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 1 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 1 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 1 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 1 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 10
      gridsize_11 = 11
      gridsize_12 = 12
      gridsize_13 = 13
      gridsize_14 = 14
      gridsize_15 = 15
      gridsize_16 = 20
      gridsize_17 = 30
      gridsize_18 = 42
      gridsize_19 = 56
      gridsize_20 = 72
      gridsize_21 = 90
      gridsize_22 = 110
      gridsize_23 = 132
      gridsize_24 = 156
      gridsize_25 = 182
      gridsize_26 = 210
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 20:
      nb_iter_1 = 20 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 2 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 2 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 2 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 3 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 2 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 2 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 2 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 2 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 2 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 1 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 1 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 1 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 10
      gridsize_11 = 11
      gridsize_12 = 12
      gridsize_13 = 13
      gridsize_14 = 14
      gridsize_15 = 15
      gridsize_16 = 16
      gridsize_17 = 17
      gridsize_18 = 18
      gridsize_19 = 20
      gridsize_20 = 30
      gridsize_21 = 42
      gridsize_22 = 56
      gridsize_23 = 72
      gridsize_24 = 90
      gridsize_25 = 110
      gridsize_26 = 132
      gridsize_27 = 156
      gridsize_28 = 182
      gridsize_29 = 210
      gridsize_30 = 240
      gridsize_31 = 272
      gridsize_32 = 306
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 22:
      nb_iter_1 = 24 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 2 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 2 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 2 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 3 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 2 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 2 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 2 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 2 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 2 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 2 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 2 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 3 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 2 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 1 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 10
      gridsize_11 = 11
      gridsize_12 = 12
      gridsize_13 = 13
      gridsize_14 = 14
      gridsize_15 = 15
      gridsize_16 = 16
      gridsize_17 = 17
      gridsize_18 = 18
      gridsize_19 = 19
      gridsize_20 = 20
      gridsize_21 = 21
      gridsize_22 = 22
      gridsize_23 = 30
      gridsize_24 = 42
      gridsize_25 = 56
      gridsize_26 = 72
      gridsize_27 = 90
      gridsize_28 = 110
      gridsize_29 = 132
      gridsize_30 = 156
      gridsize_31 = 182
      gridsize_32 = 210
      gridsize_33 = 240
      gridsize_34 = 272
      gridsize_35 = 306
      gridsize_36 = 342
      gridsize_37 = 380
      gridsize_38 = 420
      gridsize_39 = 462
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 24:
      nb_iter_1 = 28 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 2 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 2 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 2 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 3 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 2 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 2 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 2 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 2 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 2 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 2 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 2 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 3 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 2 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 2 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 2 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 2 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 2 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 10
      gridsize_11 = 11
      gridsize_12 = 12
      gridsize_13 = 13
      gridsize_14 = 14
      gridsize_15 = 15
      gridsize_16 = 16
      gridsize_17 = 17
      gridsize_18 = 18
      gridsize_19 = 19
      gridsize_20 = 20
      gridsize_21 = 21
      gridsize_22 = 22
      gridsize_23 = 23
      gridsize_24 = 24
      gridsize_25 = 25
      gridsize_26 = 26
      gridsize_27 = 30
      gridsize_28 = 42
      gridsize_29 = 56
      gridsize_30 = 72
      gridsize_31 = 90
      gridsize_32 = 110
      gridsize_33 = 132
      gridsize_34 = 156
      gridsize_35 = 182
      gridsize_36 = 210
      gridsize_37 = 240
      gridsize_38 = 272
      gridsize_39 = 306
      gridsize_40 = 342
      gridsize_41 = 380
      gridsize_42 = 420
      gridsize_43 = 462
      gridsize_44 = 506
      gridsize_45 = 552
      gridsize_46 = 600
      gridsize_47 = 650
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)    
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 26:
      nb_iter_1 = 33 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 3 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 2 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 2 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 2 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 3 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 2 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 2 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 2 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 2 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 2 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 3 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 2 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 2 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 2 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 2 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 2 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 2 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 2 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 3 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 2 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 2 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 2 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 2 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 2 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 2 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 2 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 2 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 2 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 3 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 2
      gridsize_3 = 3
      gridsize_4 = 4
      gridsize_5 = 5
      gridsize_6 = 6
      gridsize_7 = 7
      gridsize_8 = 8
      gridsize_9 = 9
      gridsize_10 = 10
      gridsize_11 = 11
      gridsize_12 = 12
      gridsize_13 = 13
      gridsize_14 = 14
      gridsize_15 = 15
      gridsize_16 = 16
      gridsize_17 = 17
      gridsize_18 = 18
      gridsize_19 = 19
      gridsize_20 = 20
      gridsize_21 = 21
      gridsize_22 = 22
      gridsize_23 = 23
      gridsize_24 = 24
      gridsize_25 = 25
      gridsize_26 = 26
      gridsize_27 = 27
      gridsize_28 = 28
      gridsize_29 = 29
      gridsize_30 = 30
      gridsize_31 = 31
      gridsize_32 = 42
      gridsize_33 = 56
      gridsize_34 = 72
      gridsize_35 = 90
      gridsize_36 = 110
      gridsize_37 = 132
      gridsize_38 = 156
      gridsize_39 = 182
      gridsize_40 = 210
      gridsize_41 = 240
      gridsize_42 = 272
      gridsize_43 = 306
      gridsize_44 = 342
      gridsize_45 = 380
      gridsize_46 = 420
      gridsize_47 = 462
      gridsize_48 = 506
      gridsize_49 = 552
      gridsize_50 = 600
      gridsize_51 = 650
      gridsize_52 = 702
      gridsize_53 = 756
      gridsize_54 = 812
      gridsize_55 = 870
      gridsize_56 = 930
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 28:
      nb_iter_1 = 2 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 11 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 1 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 2 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 2 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 11 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 1 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 11 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 1 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 1 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 1 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      nb_iter_57 = 1 # number of calls to dgemm_fermi's kernel for gridsize_57
      nb_iter_58 = 1 # number of calls to dgemm_fermi's kernel for gridsize_58
      nb_iter_59 = 1 # number of calls to dgemm_fermi's kernel for gridsize_59
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 3
      gridsize_3 = 4
      gridsize_4 = 7
      gridsize_5 = 10
      gridsize_6 = 13
      gridsize_7 = 16
      gridsize_8 = 19
      gridsize_9 = 22
      gridsize_10 = 25
      gridsize_11 = 28
      gridsize_12 = 31
      gridsize_13 = 1
      gridsize_14 = 2
      gridsize_15 = 3
      gridsize_16 = 4
      gridsize_17 = 6
      gridsize_18 = 7
      gridsize_19 = 8
      gridsize_20 = 10
      gridsize_21 = 13
      gridsize_22 = 14
      gridsize_23 = 16
      gridsize_24 = 19
      gridsize_25 = 20
      gridsize_26 = 22
      gridsize_27 = 25
      gridsize_28 = 26
      gridsize_29 = 28
      gridsize_30 = 31
      gridsize_31 = 32
      gridsize_32 = 38
      gridsize_33 = 44
      gridsize_34 = 50
      gridsize_35 = 56
      gridsize_36 = 62
      gridsize_37 = 1
      gridsize_38 = 4
      gridsize_39 = 12
      gridsize_40 = 21
      gridsize_41 = 28
      gridsize_42 = 30
      gridsize_43 = 39
      gridsize_44 = 48
      gridsize_45 = 57
      gridsize_46 = 66
      gridsize_47 = 70
      gridsize_48 = 75
      gridsize_49 = 84
      gridsize_50 = 93
      gridsize_51 = 102
      gridsize_52 = 130
      gridsize_53 = 208
      gridsize_54 = 304
      gridsize_55 = 418
      gridsize_56 = 550
      gridsize_57 = 700
      gridsize_58 = 868
      gridsize_59 = 1054
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      print("Grid size 57 is:", gridsize_57)
      print("Grid size 58 is:", gridsize_58)
      print("Grid size 59 is:", gridsize_59)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 30:
      nb_iter_1 = 2 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 13 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 1 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 2 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 2 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 13 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 13 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 1 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 1 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 1 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      nb_iter_57 = 1 # number of calls to dgemm_fermi's kernel for gridsize_57
      nb_iter_58 = 1 # number of calls to dgemm_fermi's kernel for gridsize_58
      nb_iter_59 = 1 # number of calls to dgemm_fermi's kernel for gridsize_59
      nb_iter_60 = 1 # number of calls to dgemm_fermi's kernel for gridsize_60
      nb_iter_61 = 1 # number of calls to dgemm_fermi's kernel for gridsize_61
      nb_iter_62 = 1 # number of calls to dgemm_fermi's kernel for gridsize_62
      nb_iter_63 = 1 # number of calls to dgemm_fermi's kernel for gridsize_63
      nb_iter_64 = 1 # number of calls to dgemm_fermi's kernel for gridsize_64
      nb_iter_65 = 1 # number of calls to dgemm_fermi's kernel for gridsize_65
      nb_iter_66 = 1 # number of calls to dgemm_fermi's kernel for gridsize_66
      nb_iter_67 = 1 # number of calls to dgemm_fermi's kernel for gridsize_67
      nb_iter_68 = 1 # number of calls to dgemm_fermi's kernel for gridsize_68
      nb_iter_69 = 1 # number of calls to dgemm_fermi's kernel for gridsize_69
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 3
      gridsize_3 = 4
      gridsize_4 = 7
      gridsize_5 = 10
      gridsize_6 = 13
      gridsize_7 = 16
      gridsize_8 = 19
      gridsize_9 = 22
      gridsize_10 = 25
      gridsize_11 = 28
      gridsize_12 = 31
      gridsize_13 = 34
      gridsize_14 = 37
      gridsize_15 = 1
      gridsize_16 = 2
      gridsize_17 = 3
      gridsize_18 = 4
      gridsize_19 = 6
      gridsize_20 = 7
      gridsize_21 = 8
      gridsize_22 = 10
      gridsize_23 = 13
      gridsize_24 = 14
      gridsize_25 = 16
      gridsize_26 = 19
      gridsize_27 = 20
      gridsize_28 = 22
      gridsize_29 = 25
      gridsize_30 = 26
      gridsize_31 = 28
      gridsize_32 = 31
      gridsize_33 = 32
      gridsize_34 = 34
      gridsize_35 = 37
      gridsize_36 = 38
      gridsize_37 = 44
      gridsize_38 = 50
      gridsize_39 = 56
      gridsize_40 = 62
      gridsize_41 = 68
      gridsize_42 = 74
      gridsize_43 = 1
      gridsize_44 = 4
      gridsize_45 = 12
      gridsize_46 = 21
      gridsize_47 = 28
      gridsize_48 = 30
      gridsize_49 = 39
      gridsize_50 = 48
      gridsize_51 = 57
      gridsize_52 = 66
      gridsize_53 = 70
      gridsize_54 = 75
      gridsize_55 = 84
      gridsize_56 = 93
      gridsize_57 = 102
      gridsize_58 = 111
      gridsize_59 = 120
      gridsize_60 = 130
      gridsize_61 = 208
      gridsize_62 = 304
      gridsize_63 = 418
      gridsize_64 = 550
      gridsize_65 = 700
      gridsize_66 = 868
      gridsize_67 = 1054
      gridsize_68 = 1258
      gridsize_69 = 1480
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      print("Grid size 57 is:", gridsize_57)
      print("Grid size 58 is:", gridsize_58)
      print("Grid size 59 is:", gridsize_59)
      print("Grid size 60 is:", gridsize_60)
      print("Grid size 61 is:", gridsize_61)
      print("Grid size 62 is:", gridsize_62)
      print("Grid size 63 is:", gridsize_63)
      print("Grid size 64 is:", gridsize_64)
      print("Grid size 65 is:", gridsize_65)
      print("Grid size 66 is:", gridsize_66)
      print("Grid size 67 is:", gridsize_67)
      print("Grid size 68 is:", gridsize_68)
      print("Grid size 69 is:", gridsize_69)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized       
    elif ns == 32:
      nb_iter_1 = 16 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 1 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 1 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 16 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 17 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 1 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 2 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 2 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 1 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 2 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 1 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 2 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 2 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 2 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 2 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 2 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 2 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      nb_iter_57 = 1 # number of calls to dgemm_fermi's kernel for gridsize_57
      nb_iter_58 = 1 # number of calls to dgemm_fermi's kernel for gridsize_58
      nb_iter_59 = 1 # number of calls to dgemm_fermi's kernel for gridsize_59
      nb_iter_60 = 1 # number of calls to dgemm_fermi's kernel for gridsize_60
      nb_iter_61 = 1 # number of calls to dgemm_fermi's kernel for gridsize_61
      blocksize = 256
      gridsize_1 = 3
      gridsize_2 = 6
      gridsize_3 = 9
      gridsize_4 = 12
      gridsize_5 = 15
      gridsize_6 = 18
      gridsize_7 = 21
      gridsize_8 = 24
      gridsize_9 = 27
      gridsize_10 = 30
      gridsize_11 = 33
      gridsize_12 = 36
      gridsize_13 = 39
      gridsize_14 = 42
      gridsize_15 = 3
      gridsize_16 = 6
      gridsize_17 = 9
      gridsize_18 = 12
      gridsize_19 = 15
      gridsize_20 = 18
      gridsize_21 = 21
      gridsize_22 = 24
      gridsize_23 = 27
      gridsize_24 = 30
      gridsize_25 = 33
      gridsize_26 = 36
      gridsize_27 = 39
      gridsize_28 = 42
      gridsize_29 = 48
      gridsize_30 = 54
      gridsize_31 = 60
      gridsize_32 = 66
      gridsize_33 = 72
      gridsize_34 = 78
      gridsize_35 = 84
      gridsize_36 = 9
      gridsize_37 = 18
      gridsize_38 = 27
      gridsize_39 = 36
      gridsize_40 = 45
      gridsize_41 = 54
      gridsize_42 = 63
      gridsize_43 = 72
      gridsize_44 = 81
      gridsize_45 = 90
      gridsize_46 = 99
      gridsize_47 = 108
      gridsize_48 = 117
      gridsize_49 = 126
      gridsize_50 = 135
      gridsize_51 = 180
      gridsize_52 = 270
      gridsize_53 = 378
      gridsize_54 = 504
      gridsize_55 = 648
      gridsize_56 = 810
      gridsize_57 = 990
      gridsize_58 = 1188
      gridsize_59 = 1404
      gridsize_60 = 1638
      gridsize_61 = 1890
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      print("Grid size 57 is:", gridsize_57)
      print("Grid size 58 is:", gridsize_58)
      print("Grid size 59 is:", gridsize_59)
      print("Grid size 60 is:", gridsize_60)
      print("Grid size 61 is:", gridsize_61)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 34:
      nb_iter_1 = 2 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 17 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 1 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 1 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 1 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 1 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 2 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 2 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 17 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 17 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 1 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      nb_iter_57 = 1 # number of calls to dgemm_fermi's kernel for gridsize_57
      nb_iter_58 = 1 # number of calls to dgemm_fermi's kernel for gridsize_58
      nb_iter_59 = 1 # number of calls to dgemm_fermi's kernel for gridsize_59
      nb_iter_60 = 1 # number of calls to dgemm_fermi's kernel for gridsize_60
      nb_iter_61 = 1 # number of calls to dgemm_fermi's kernel for gridsize_61
      nb_iter_62 = 1 # number of calls to dgemm_fermi's kernel for gridsize_62
      nb_iter_63 = 1 # number of calls to dgemm_fermi's kernel for gridsize_63
      nb_iter_64 = 1 # number of calls to dgemm_fermi's kernel for gridsize_64
      nb_iter_65 = 1 # number of calls to dgemm_fermi's kernel for gridsize_65
      nb_iter_66 = 1 # number of calls to dgemm_fermi's kernel for gridsize_66
      nb_iter_67 = 1 # number of calls to dgemm_fermi's kernel for gridsize_67
      nb_iter_68 = 1 # number of calls to dgemm_fermi's kernel for gridsize_68
      nb_iter_69 = 1 # number of calls to dgemm_fermi's kernel for gridsize_69
      nb_iter_70 = 1 # number of calls to dgemm_fermi's kernel for gridsize_70
      nb_iter_71 = 1 # number of calls to dgemm_fermi's kernel for gridsize_71
      nb_iter_72 = 1 # number of calls to dgemm_fermi's kernel for gridsize_72
      nb_iter_73 = 1 # number of calls to dgemm_fermi's kernel for gridsize_73
      nb_iter_74 = 1 # number of calls to dgemm_fermi's kernel for gridsize_74
      nb_iter_75 = 1 # number of calls to dgemm_fermi's kernel for gridsize_75
      nb_iter_76 = 1 # number of calls to dgemm_fermi's kernel for gridsize_76
      nb_iter_77 = 1 # number of calls to dgemm_fermi's kernel for gridsize_77
      nb_iter_78 = 1 # number of calls to dgemm_fermi's kernel for gridsize_78
      nb_iter_79 = 1 # number of calls to dgemm_fermi's kernel for gridsize_79
      nb_iter_80 = 1 # number of calls to dgemm_fermi's kernel for gridsize_80
      nb_iter_81 = 1 # number of calls to dgemm_fermi's kernel for gridsize_81
      nb_iter_82 = 1 # number of calls to dgemm_fermi's kernel for gridsize_82
      nb_iter_83 = 1 # number of calls to dgemm_fermi's kernel for gridsize_83
      nb_iter_84 = 1 # number of calls to dgemm_fermi's kernel for gridsize_84
      nb_iter_85 = 1 # number of calls to dgemm_fermi's kernel for gridsize_85
      nb_iter_86 = 1 # number of calls to dgemm_fermi's kernel for gridsize_86
      nb_iter_87 = 1 # number of calls to dgemm_fermi's kernel for gridsize_87
      nb_iter_88 = 1 # number of calls to dgemm_fermi's kernel for gridsize_88
      nb_iter_89 = 1 # number of calls to dgemm_fermi's kernel for gridsize_89
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 3
      gridsize_3 = 4
      gridsize_4 = 7
      gridsize_5 = 10
      gridsize_6 = 13
      gridsize_7 = 16
      gridsize_8 = 19
      gridsize_9 = 22
      gridsize_10 = 25
      gridsize_11 = 28
      gridsize_12 = 31
      gridsize_13 = 34
      gridsize_14 = 37
      gridsize_15 = 40
      gridsize_16 = 43
      gridsize_17 = 46
      gridsize_18 = 49
      gridsize_19 = 1
      gridsize_20 = 2
      gridsize_21 = 3
      gridsize_22 = 4
      gridsize_23 = 6
      gridsize_24 = 7
      gridsize_25 = 8
      gridsize_26 = 10
      gridsize_27 = 13
      gridsize_28 = 14
      gridsize_29 = 16
      gridsize_30 = 19
      gridsize_31 = 20
      gridsize_32 = 22
      gridsize_33 = 25
      gridsize_34 = 26
      gridsize_35 = 28
      gridsize_36 = 31
      gridsize_37 = 32
      gridsize_38 = 34
      gridsize_39 = 37
      gridsize_40 = 38
      gridsize_41 = 40
      gridsize_42 = 43
      gridsize_43 = 44
      gridsize_44 = 46
      gridsize_45 = 49
      gridsize_46 = 50
      gridsize_47 = 56
      gridsize_48 = 62
      gridsize_49 = 68
      gridsize_50 = 74
      gridsize_51 = 80
      gridsize_52 = 86
      gridsize_53 = 92
      gridsize_54 = 98
      gridsize_55 = 1
      gridsize_56 = 4
      gridsize_57 = 12
      gridsize_58 = 21
      gridsize_59 = 28
      gridsize_60 = 30
      gridsize_61 = 39
      gridsize_62 = 48 
      gridsize_63 = 57
      gridsize_64 = 66
      gridsize_65 = 70
      gridsize_66 = 75
      gridsize_67 = 84
      gridsize_68 = 93
      gridsize_69 = 102
      gridsize_70 = 111
      gridsize_71 = 120
      gridsize_72 = 129
      gridsize_73 = 130
      gridsize_74 = 138
      gridsize_75 = 147
      gridsize_76 = 156
      gridsize_77 = 208
      gridsize_78 = 304
      gridsize_79 = 418
      gridsize_80 = 550
      gridsize_81 = 700
      gridsize_82 = 868
      gridsize_83 = 1054
      gridsize_84 = 1258
      gridsize_85 = 1480
      gridsize_86 = 1720
      gridsize_87 = 1978
      gridsize_88 = 2254
      gridsize_89 = 2548
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      print("Grid size 57 is:", gridsize_57)
      print("Grid size 58 is:", gridsize_58)
      print("Grid size 59 is:", gridsize_59)
      print("Grid size 60 is:", gridsize_60)
      print("Grid size 61 is:", gridsize_61)
      print("Grid size 62 is:", gridsize_62)
      print("Grid size 63 is:", gridsize_63)
      print("Grid size 64 is:", gridsize_64)
      print("Grid size 65 is:", gridsize_65)
      print("Grid size 66 is:", gridsize_66)
      print("Grid size 67 is:", gridsize_67)
      print("Grid size 68 is:", gridsize_68)
      print("Grid size 69 is:", gridsize_69)
      print("Grid size 70 is:", gridsize_70)
      print("Grid size 71 is:", gridsize_71)
      print("Grid size 72 is:", gridsize_72)
      print("Grid size 73 is:", gridsize_73)
      print("Grid size 74 is:", gridsize_74)
      print("Grid size 75 is:", gridsize_75)
      print("Grid size 76 is:", gridsize_76)
      print("Grid size 77 is:", gridsize_77)
      print("Grid size 78 is:", gridsize_78)
      print("Grid size 79 is:", gridsize_79)
      print("Grid size 80 is:", gridsize_80)
      print("Grid size 81 is:", gridsize_81)
      print("Grid size 82 is:", gridsize_82)
      print("Grid size 83 is:", gridsize_83)
      print("Grid size 84 is:", gridsize_84)
      print("Grid size 85 is:", gridsize_85)
      print("Grid size 86 is:", gridsize_86)
      print("Grid size 87 is:", gridsize_87)
      print("Grid size 88 is:", gridsize_88)
      print("Grid size 89 is:", gridsize_89)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 36:
      nb_iter_1 = 2 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 19 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 1 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 1 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 1 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 1 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 1 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 2 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 2 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 19 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 1 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 19 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 1 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      nb_iter_57 = 1 # number of calls to dgemm_fermi's kernel for gridsize_57
      nb_iter_58 = 1 # number of calls to dgemm_fermi's kernel for gridsize_58
      nb_iter_59 = 1 # number of calls to dgemm_fermi's kernel for gridsize_59
      nb_iter_60 = 1 # number of calls to dgemm_fermi's kernel for gridsize_60
      nb_iter_61 = 1 # number of calls to dgemm_fermi's kernel for gridsize_61
      nb_iter_62 = 1 # number of calls to dgemm_fermi's kernel for gridsize_62
      nb_iter_63 = 1 # number of calls to dgemm_fermi's kernel for gridsize_63
      nb_iter_64 = 1 # number of calls to dgemm_fermi's kernel for gridsize_64
      nb_iter_65 = 1 # number of calls to dgemm_fermi's kernel for gridsize_65
      nb_iter_66 = 1 # number of calls to dgemm_fermi's kernel for gridsize_66
      nb_iter_67 = 1 # number of calls to dgemm_fermi's kernel for gridsize_67
      nb_iter_68 = 1 # number of calls to dgemm_fermi's kernel for gridsize_68
      nb_iter_69 = 1 # number of calls to dgemm_fermi's kernel for gridsize_69
      nb_iter_70 = 1 # number of calls to dgemm_fermi's kernel for gridsize_70
      nb_iter_71 = 1 # number of calls to dgemm_fermi's kernel for gridsize_71
      nb_iter_72 = 1 # number of calls to dgemm_fermi's kernel for gridsize_72
      nb_iter_73 = 1 # number of calls to dgemm_fermi's kernel for gridsize_73
      nb_iter_74 = 1 # number of calls to dgemm_fermi's kernel for gridsize_74
      nb_iter_75 = 1 # number of calls to dgemm_fermi's kernel for gridsize_75
      nb_iter_76 = 1 # number of calls to dgemm_fermi's kernel for gridsize_76
      nb_iter_77 = 1 # number of calls to dgemm_fermi's kernel for gridsize_77
      nb_iter_78 = 1 # number of calls to dgemm_fermi's kernel for gridsize_78
      nb_iter_79 = 1 # number of calls to dgemm_fermi's kernel for gridsize_79
      nb_iter_80 = 1 # number of calls to dgemm_fermi's kernel for gridsize_80
      nb_iter_81 = 1 # number of calls to dgemm_fermi's kernel for gridsize_81
      nb_iter_82 = 1 # number of calls to dgemm_fermi's kernel for gridsize_82
      nb_iter_83 = 1 # number of calls to dgemm_fermi's kernel for gridsize_83
      nb_iter_84 = 1 # number of calls to dgemm_fermi's kernel for gridsize_84
      nb_iter_85 = 1 # number of calls to dgemm_fermi's kernel for gridsize_85
      nb_iter_86 = 1 # number of calls to dgemm_fermi's kernel for gridsize_86
      nb_iter_87 = 1 # number of calls to dgemm_fermi's kernel for gridsize_87
      nb_iter_88 = 1 # number of calls to dgemm_fermi's kernel for gridsize_88
      nb_iter_89 = 1 # number of calls to dgemm_fermi's kernel for gridsize_89
      nb_iter_90 = 1 # number of calls to dgemm_fermi's kernel for gridsize_90
      nb_iter_91 = 1 # number of calls to dgemm_fermi's kernel for gridsize_91
      nb_iter_92 = 1 # number of calls to dgemm_fermi's kernel for gridsize_92
      nb_iter_93 = 1 # number of calls to dgemm_fermi's kernel for gridsize_93
      nb_iter_94 = 1 # number of calls to dgemm_fermi's kernel for gridsize_94
      nb_iter_95 = 1 # number of calls to dgemm_fermi's kernel for gridsize_95
      nb_iter_96 = 1 # number of calls to dgemm_fermi's kernel for gridsize_96
      nb_iter_97 = 1 # number of calls to dgemm_fermi's kernel for gridsize_97
      nb_iter_98 = 1 # number of calls to dgemm_fermi's kernel for gridsize_98
      nb_iter_99 = 1 # number of calls to dgemm_fermi's kernel for gridsize_99
      blocksize = 256
      gridsize_1 = 1
      gridsize_2 = 3
      gridsize_3 = 4
      gridsize_4 = 7
      gridsize_5 = 10
      gridsize_6 = 13
      gridsize_7 = 16
      gridsize_8 = 19
      gridsize_9 = 22
      gridsize_10 = 25
      gridsize_11 = 28
      gridsize_12 = 31
      gridsize_13 = 34
      gridsize_14 = 37
      gridsize_15 = 40
      gridsize_16 = 43
      gridsize_17 = 46
      gridsize_18 = 49
      gridsize_19 = 52
      gridsize_20 = 55
      gridsize_21 = 1
      gridsize_22 = 2
      gridsize_23 = 3
      gridsize_24 = 4
      gridsize_25 = 6
      gridsize_26 = 7
      gridsize_27 = 8
      gridsize_28 = 10
      gridsize_29 = 13
      gridsize_30 = 14
      gridsize_31 = 16
      gridsize_32 = 19
      gridsize_33 = 20
      gridsize_34 = 22
      gridsize_35 = 25
      gridsize_36 = 26
      gridsize_37 = 28
      gridsize_38 = 31
      gridsize_39 = 32
      gridsize_40 = 34
      gridsize_41 = 37
      gridsize_42 = 38
      gridsize_43 = 40
      gridsize_44 = 43
      gridsize_45 = 44
      gridsize_46 = 46
      gridsize_47 = 49
      gridsize_48 = 50
      gridsize_49 = 52
      gridsize_50 = 55
      gridsize_51 = 56
      gridsize_52 = 62
      gridsize_53 = 68
      gridsize_54 = 74
      gridsize_55 = 80
      gridsize_56 = 86
      gridsize_57 = 92
      gridsize_58 = 98
      gridsize_59 = 104
      gridsize_60 = 110
      gridsize_61 = 1
      gridsize_62 = 4
      gridsize_63 = 12
      gridsize_64 = 21
      gridsize_65 = 28
      gridsize_66 = 30
      gridsize_67 = 39
      gridsize_68 = 48
      gridsize_69 = 57
      gridsize_70 = 66
      gridsize_71 = 70
      gridsize_72 = 75
      gridsize_73 = 84
      gridsize_74 = 93
      gridsize_75 = 102
      gridsize_76 = 111
      gridsize_77 = 120
      gridsize_78 = 129
      gridsize_79 = 130
      gridsize_80 = 138
      gridsize_81 = 147
      gridsize_82 = 156
      gridsize_83 = 165
      gridsize_84 = 174
      gridsize_85 = 208
      gridsize_86 = 304
      gridsize_87 = 418
      gridsize_88 = 550
      gridsize_89 = 700
      gridsize_90 = 868
      gridsize_91 = 1054
      gridsize_92 = 1258
      gridsize_93 = 1480
      gridsize_94 = 1720
      gridsize_95 = 1978
      gridsize_96 = 2254
      gridsize_97 = 2548
      gridsize_98 = 2860
      gridsize_99 = 3190
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      print("Grid size 57 is:", gridsize_57)
      print("Grid size 58 is:", gridsize_58)
      print("Grid size 59 is:", gridsize_59)
      print("Grid size 60 is:", gridsize_60)
      print("Grid size 61 is:", gridsize_61)
      print("Grid size 62 is:", gridsize_62)
      print("Grid size 63 is:", gridsize_63)
      print("Grid size 64 is:", gridsize_64)
      print("Grid size 65 is:", gridsize_65)
      print("Grid size 66 is:", gridsize_66)
      print("Grid size 67 is:", gridsize_67)
      print("Grid size 68 is:", gridsize_68)
      print("Grid size 69 is:", gridsize_69)
      print("Grid size 70 is:", gridsize_70)
      print("Grid size 71 is:", gridsize_71)
      print("Grid size 72 is:", gridsize_72)
      print("Grid size 73 is:", gridsize_73)
      print("Grid size 74 is:", gridsize_74)
      print("Grid size 75 is:", gridsize_75)
      print("Grid size 76 is:", gridsize_76)
      print("Grid size 77 is:", gridsize_77)
      print("Grid size 78 is:", gridsize_78)
      print("Grid size 79 is:", gridsize_79)
      print("Grid size 80 is:", gridsize_80)
      print("Grid size 81 is:", gridsize_81)
      print("Grid size 82 is:", gridsize_82)
      print("Grid size 83 is:", gridsize_83)
      print("Grid size 84 is:", gridsize_84)
      print("Grid size 85 is:", gridsize_85)
      print("Grid size 86 is:", gridsize_86)
      print("Grid size 87 is:", gridsize_87)
      print("Grid size 88 is:", gridsize_88)
      print("Grid size 89 is:", gridsize_89)
      print("Grid size 90 is:", gridsize_90)
      print("Grid size 91 is:", gridsize_91)
      print("Grid size 92 is:", gridsize_92)
      print("Grid size 93 is:", gridsize_93)
      print("Grid size 94 is:", gridsize_94)
      print("Grid size 95 is:", gridsize_95)
      print("Grid size 96 is:", gridsize_96)
      print("Grid size 97 is:", gridsize_97)
      print("Grid size 98 is:", gridsize_98)
      print("Grid size 99 is:", gridsize_99)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized
    elif ns == 38:
      nb_iter_1 = 2 # number of calls to dgemm_fermi's kernel for gridsize_1
      nb_iter_2 = 21 # number of calls to dgemm_fermi's kernel for gridsize_2
      nb_iter_3 = 1 # number of calls to dgemm_fermi's kernel for gridsize_3
      nb_iter_4 = 1 # number of calls to dgemm_fermi's kernel for gridsize_4
      nb_iter_5 = 1 # number of calls to dgemm_fermi's kernel for gridsize_5
      nb_iter_6 = 1 # number of calls to dgemm_fermi's kernel for gridsize_6
      nb_iter_7 = 1 # number of calls to dgemm_fermi's kernel for gridsize_7
      nb_iter_8 = 1 # number of calls to dgemm_fermi's kernel for gridsize_8
      nb_iter_9 = 1 # number of calls to dgemm_fermi's kernel for gridsize_9
      nb_iter_10 = 1 # number of calls to dgemm_fermi's kernel for gridsize_10
      nb_iter_11 = 1 # number of calls to dgemm_fermi's kernel for gridsize_11
      nb_iter_12 = 1 # number of calls to dgemm_fermi's kernel for gridsize_12
      nb_iter_13 = 1 # number of calls to dgemm_fermi's kernel for gridsize_13
      nb_iter_14 = 1 # number of calls to dgemm_fermi's kernel for gridsize_14
      nb_iter_15 = 1 # number of calls to dgemm_fermi's kernel for gridsize_15
      nb_iter_16 = 1 # number of calls to dgemm_fermi's kernel for gridsize_16
      nb_iter_17 = 1 # number of calls to dgemm_fermi's kernel for gridsize_17
      nb_iter_18 = 1 # number of calls to dgemm_fermi's kernel for gridsize_18
      nb_iter_19 = 1 # number of calls to dgemm_fermi's kernel for gridsize_19
      nb_iter_20 = 1 # number of calls to dgemm_fermi's kernel for gridsize_20
      nb_iter_21 = 1 # number of calls to dgemm_fermi's kernel for gridsize_21
      nb_iter_22 = 1 # number of calls to dgemm_fermi's kernel for gridsize_22
      nb_iter_23 = 2 # number of calls to dgemm_fermi's kernel for gridsize_23
      nb_iter_24 = 21 # number of calls to dgemm_fermi's kernel for gridsize_24
      nb_iter_25 = 2 # number of calls to dgemm_fermi's kernel for gridsize_25
      nb_iter_26 = 1 # number of calls to dgemm_fermi's kernel for gridsize_26
      nb_iter_27 = 21 # number of calls to dgemm_fermi's kernel for gridsize_27
      nb_iter_28 = 1 # number of calls to dgemm_fermi's kernel for gridsize_28
      nb_iter_29 = 1 # number of calls to dgemm_fermi's kernel for gridsize_29
      nb_iter_30 = 1 # number of calls to dgemm_fermi's kernel for gridsize_30
      nb_iter_31 = 1 # number of calls to dgemm_fermi's kernel for gridsize_31
      nb_iter_32 = 1 # number of calls to dgemm_fermi's kernel for gridsize_32
      nb_iter_33 = 1 # number of calls to dgemm_fermi's kernel for gridsize_33
      nb_iter_34 = 1 # number of calls to dgemm_fermi's kernel for gridsize_34
      nb_iter_35 = 1 # number of calls to dgemm_fermi's kernel for gridsize_35
      nb_iter_36 = 1 # number of calls to dgemm_fermi's kernel for gridsize_36
      nb_iter_37 = 1 # number of calls to dgemm_fermi's kernel for gridsize_37
      nb_iter_38 = 1 # number of calls to dgemm_fermi's kernel for gridsize_38
      nb_iter_39 = 1 # number of calls to dgemm_fermi's kernel for gridsize_39
      nb_iter_40 = 1 # number of calls to dgemm_fermi's kernel for gridsize_40
      nb_iter_41 = 1 # number of calls to dgemm_fermi's kernel for gridsize_41
      nb_iter_42 = 1 # number of calls to dgemm_fermi's kernel for gridsize_42
      nb_iter_43 = 1 # number of calls to dgemm_fermi's kernel for gridsize_43
      nb_iter_44 = 1 # number of calls to dgemm_fermi's kernel for gridsize_44
      nb_iter_45 = 1 # number of calls to dgemm_fermi's kernel for gridsize_45
      nb_iter_46 = 1 # number of calls to dgemm_fermi's kernel for gridsize_46
      nb_iter_47 = 1 # number of calls to dgemm_fermi's kernel for gridsize_47
      nb_iter_48 = 1 # number of calls to dgemm_fermi's kernel for gridsize_48
      nb_iter_49 = 1 # number of calls to dgemm_fermi's kernel for gridsize_49
      nb_iter_50 = 1 # number of calls to dgemm_fermi's kernel for gridsize_50
      nb_iter_51 = 1 # number of calls to dgemm_fermi's kernel for gridsize_51
      nb_iter_52 = 1 # number of calls to dgemm_fermi's kernel for gridsize_52
      nb_iter_53 = 1 # number of calls to dgemm_fermi's kernel for gridsize_53
      nb_iter_54 = 1 # number of calls to dgemm_fermi's kernel for gridsize_54
      nb_iter_55 = 1 # number of calls to dgemm_fermi's kernel for gridsize_55
      nb_iter_56 = 1 # number of calls to dgemm_fermi's kernel for gridsize_56
      nb_iter_57 = 1 # number of calls to dgemm_fermi's kernel for gridsize_57
      nb_iter_58 = 1 # number of calls to dgemm_fermi's kernel for gridsize_58
      nb_iter_59 = 1 # number of calls to dgemm_fermi's kernel for gridsize_59
      nb_iter_60 = 1 # number of calls to dgemm_fermi's kernel for gridsize_60
      nb_iter_61 = 1 # number of calls to dgemm_fermi's kernel for gridsize_61
      nb_iter_62 = 1 # number of calls to dgemm_fermi's kernel for gridsize_62
      nb_iter_63 = 1 # number of calls to dgemm_fermi's kernel for gridsize_63
      nb_iter_64 = 1 # number of calls to dgemm_fermi's kernel for gridsize_64
      nb_iter_65 = 1 # number of calls to dgemm_fermi's kernel for gridsize_65
      nb_iter_66 = 1 # number of calls to dgemm_fermi's kernel for gridsize_66
      nb_iter_67 = 1 # number of calls to dgemm_fermi's kernel for gridsize_67
      nb_iter_68 = 1 # number of calls to dgemm_fermi's kernel for gridsize_68
      nb_iter_69 = 1 # number of calls to dgemm_fermi's kernel for gridsize_69
      nb_iter_70 = 1 # number of calls to dgemm_fermi's kernel for gridsize_70
      nb_iter_71 = 1 # number of calls to dgemm_fermi's kernel for gridsize_71
      nb_iter_72 = 1 # number of calls to dgemm_fermi's kernel for gridsize_72
      nb_iter_73 = 1 # number of calls to dgemm_fermi's kernel for gridsize_73
      nb_iter_74 = 1 # number of calls to dgemm_fermi's kernel for gridsize_74
      nb_iter_75 = 1 # number of calls to dgemm_fermi's kernel for gridsize_75
      nb_iter_76 = 1 # number of calls to dgemm_fermi's kernel for gridsize_76
      nb_iter_77 = 1 # number of calls to dgemm_fermi's kernel for gridsize_77
      nb_iter_78 = 1 # number of calls to dgemm_fermi's kernel for gridsize_78
      nb_iter_79 = 1 # number of calls to dgemm_fermi's kernel for gridsize_79
      nb_iter_80 = 1 # number of calls to dgemm_fermi's kernel for gridsize_80
      nb_iter_81 = 1 # number of calls to dgemm_fermi's kernel for gridsize_81
      nb_iter_82 = 1 # number of calls to dgemm_fermi's kernel for gridsize_82
      nb_iter_83 = 1 # number of calls to dgemm_fermi's kernel for gridsize_83
      nb_iter_84 = 1 # number of calls to dgemm_fermi's kernel for gridsize_84
      nb_iter_85 = 1 # number of calls to dgemm_fermi's kernel for gridsize_85
      nb_iter_86 = 1 # number of calls to dgemm_fermi's kernel for gridsize_86
      nb_iter_87 = 1 # number of calls to dgemm_fermi's kernel for gridsize_87
      nb_iter_88 = 1 # number of calls to dgemm_fermi's kernel for gridsize_88
      nb_iter_89 = 1 # number of calls to dgemm_fermi's kernel for gridsize_89
      nb_iter_90 = 1 # number of calls to dgemm_fermi's kernel for gridsize_90
      nb_iter_91 = 1 # number of calls to dgemm_fermi's kernel for gridsize_91
      nb_iter_92 = 1 # number of calls to dgemm_fermi's kernel for gridsize_92
      nb_iter_93 = 1 # number of calls to dgemm_fermi's kernel for gridsize_93
      nb_iter_94 = 1 # number of calls to dgemm_fermi's kernel for gridsize_94
      nb_iter_95 = 1 # number of calls to dgemm_fermi's kernel for gridsize_95
      nb_iter_96 = 1 # number of calls to dgemm_fermi's kernel for gridsize_96
      nb_iter_97 = 1 # number of calls to dgemm_fermi's kernel for gridsize_97
      nb_iter_98 = 1 # number of calls to dgemm_fermi's kernel for gridsize_98
      nb_iter_99 = 1 # number of calls to dgemm_fermi's kernel for gridsize_99
      nb_iter_100 = 1 # number of calls to dgemm_fermi's kernel for gridsize_100
      nb_iter_101 = 1 # number of calls to dgemm_fermi's kernel for gridsize_101
      nb_iter_102 = 1 # number of calls to dgemm_fermi's kernel for gridsize_102
      nb_iter_103 = 1 # number of calls to dgemm_fermi's kernel for gridsize_103
      nb_iter_104 = 1 # number of calls to dgemm_fermi's kernel for gridsize_104
      nb_iter_105 = 1 # number of calls to dgemm_fermi's kernel for gridsize_105
      nb_iter_106 = 1 # number of calls to dgemm_fermi's kernel for gridsize_106
      nb_iter_107 = 1 # number of calls to dgemm_fermi's kernel for gridsize_107
      nb_iter_108 = 1 # number of calls to dgemm_fermi's kernel for gridsize_108
      nb_iter_109 = 1 # number of calls to dgemm_fermi's kernel for gridsize_109
      blocksize = 256
      gridsize_1 = 2
      gridsize_2 = 3
      gridsize_3 = 5
      gridsize_4 = 8
      gridsize_5 = 11
      gridsize_6 = 14
      gridsize_7 = 17
      gridsize_8 = 20
      gridsize_9 = 23
      gridsize_10 = 26
      gridsize_11 = 29
      gridsize_12 = 32
      gridsize_13 = 35
      gridsize_14 = 38
      gridsize_15 = 41
      gridsize_16 = 44
      gridsize_17 = 47
      gridsize_18 = 50
      gridsize_19 = 53
      gridsize_20 = 56
      gridsize_21 = 59
      gridsize_22 = 62
      gridsize_23 = 2
      gridsize_24 = 3
      gridsize_25 = 4
      gridsize_26 = 5
      gridsize_27 = 6
      gridsize_28 = 8
      gridsize_29 = 10
      gridsize_30 = 11
      gridsize_31 = 14
      gridsize_32 = 16
      gridsize_33 = 17
      gridsize_34 = 20
      gridsize_35 = 22
      gridsize_36 = 23
      gridsize_37 = 26
      gridsize_38 = 28
      gridsize_39 = 29
      gridsize_40 = 32
      gridsize_41 = 34
      gridsize_42 = 35
      gridsize_43 = 38
      gridsize_44 = 40
      gridsize_45 = 41
      gridsize_46 = 44
      gridsize_47 = 46
      gridsize_48 = 47
      gridsize_49 = 50
      gridsize_50 = 52
      gridsize_51 = 53
      gridsize_52 = 56
      gridsize_53 = 58
      gridsize_54 = 59
      gridsize_55 = 62
      gridsize_56 = 64
      gridsize_57 = 70
      gridsize_58 = 76
      gridsize_59 = 82
      gridsize_60 = 88
      gridsize_61 = 94
      gridsize_62 = 100
      gridsize_63 = 106
      gridsize_64 = 112
      gridsize_65 = 118
      gridsize_66 = 124
      gridsize_67 = 4
      gridsize_68 = 10
      gridsize_69 = 15
      gridsize_70 = 24
      gridsize_71 = 33
      gridsize_72 = 40
      gridsize_73 = 42
      gridsize_74 = 51
      gridsize_75 = 60
      gridsize_76 = 69
      gridsize_77 = 78
      gridsize_78 = 87
      gridsize_79 = 88
      gridsize_80 = 96
      gridsize_81 = 105
      gridsize_82 = 114
      gridsize_83 = 123
      gridsize_84 = 132
      gridsize_85 = 141
      gridsize_86 = 150
      gridsize_87 = 154
      gridsize_88 = 159
      gridsize_89 = 168
      gridsize_90 = 177
      gridsize_91 = 186
      gridsize_92 = 195
      gridsize_93 = 238
      gridsize_94 = 340
      gridsize_95 = 460
      gridsize_96 = 598
      gridsize_97 = 754
      gridsize_98 = 928
      gridsize_99 = 1120
      gridsize_100 = 1330
      gridsize_101 = 1558
      gridsize_102 = 1804
      gridsize_103 = 2068
      gridsize_104 = 2350
      gridsize_105 = 2650
      gridsize_106 = 2968
      gridsize_107 = 3304
      gridsize_108 = 3658
      gridsize_109 = 4030
      print("Block size is:", blocksize)
      print("Grid size 1 is:", gridsize_1)
      print("Grid size 2 is:", gridsize_2)
      print("Grid size 3 is:", gridsize_3)
      print("Grid size 4 is:", gridsize_4)
      print("Grid size 5 is:", gridsize_5)
      print("Grid size 6 is:", gridsize_6)
      print("Grid size 7 is:", gridsize_7)
      print("Grid size 8 is:", gridsize_8)
      print("Grid size 9 is:", gridsize_9)
      print("Grid size 10 is:", gridsize_10)
      print("Grid size 11 is:", gridsize_11)
      print("Grid size 12 is:", gridsize_12)
      print("Grid size 13 is:", gridsize_13)
      print("Grid size 14 is:", gridsize_14)
      print("Grid size 15 is:", gridsize_15)
      print("Grid size 16 is:", gridsize_16)
      print("Grid size 17 is:", gridsize_17)
      print("Grid size 18 is:", gridsize_18)
      print("Grid size 19 is:", gridsize_19)
      print("Grid size 20 is:", gridsize_20)
      print("Grid size 21 is:", gridsize_21)
      print("Grid size 22 is:", gridsize_22)
      print("Grid size 23 is:", gridsize_23)
      print("Grid size 24 is:", gridsize_24)
      print("Grid size 25 is:", gridsize_25)
      print("Grid size 26 is:", gridsize_26)
      print("Grid size 27 is:", gridsize_27)
      print("Grid size 28 is:", gridsize_28)
      print("Grid size 29 is:", gridsize_29)
      print("Grid size 30 is:", gridsize_30)
      print("Grid size 31 is:", gridsize_31)
      print("Grid size 32 is:", gridsize_32)
      print("Grid size 33 is:", gridsize_33)
      print("Grid size 34 is:", gridsize_34)
      print("Grid size 35 is:", gridsize_35)
      print("Grid size 36 is:", gridsize_36)
      print("Grid size 37 is:", gridsize_37)
      print("Grid size 38 is:", gridsize_38)
      print("Grid size 39 is:", gridsize_39)
      print("Grid size 40 is:", gridsize_40)
      print("Grid size 41 is:", gridsize_41)
      print("Grid size 42 is:", gridsize_42)
      print("Grid size 43 is:", gridsize_43)
      print("Grid size 44 is:", gridsize_44)
      print("Grid size 45 is:", gridsize_45)
      print("Grid size 46 is:", gridsize_46)
      print("Grid size 47 is:", gridsize_47)
      print("Grid size 48 is:", gridsize_48)
      print("Grid size 49 is:", gridsize_49)
      print("Grid size 50 is:", gridsize_50)
      print("Grid size 51 is:", gridsize_51)
      print("Grid size 52 is:", gridsize_52)
      print("Grid size 53 is:", gridsize_53)
      print("Grid size 54 is:", gridsize_54)
      print("Grid size 55 is:", gridsize_55)
      print("Grid size 56 is:", gridsize_56)
      print("Grid size 57 is:", gridsize_57)
      print("Grid size 58 is:", gridsize_58)
      print("Grid size 59 is:", gridsize_59)
      print("Grid size 60 is:", gridsize_60)
      print("Grid size 61 is:", gridsize_61)
      print("Grid size 62 is:", gridsize_62)
      print("Grid size 63 is:", gridsize_63)
      print("Grid size 64 is:", gridsize_64)
      print("Grid size 65 is:", gridsize_65)
      print("Grid size 66 is:", gridsize_66)
      print("Grid size 67 is:", gridsize_67)
      print("Grid size 68 is:", gridsize_68)
      print("Grid size 69 is:", gridsize_69)
      print("Grid size 70 is:", gridsize_70)
      print("Grid size 71 is:", gridsize_71)
      print("Grid size 72 is:", gridsize_72)
      print("Grid size 73 is:", gridsize_73)
      print("Grid size 74 is:", gridsize_74)
      print("Grid size 75 is:", gridsize_75)
      print("Grid size 76 is:", gridsize_76)
      print("Grid size 77 is:", gridsize_77)
      print("Grid size 78 is:", gridsize_78)
      print("Grid size 79 is:", gridsize_79)
      print("Grid size 80 is:", gridsize_80)
      print("Grid size 81 is:", gridsize_81)
      print("Grid size 82 is:", gridsize_82)
      print("Grid size 83 is:", gridsize_83)
      print("Grid size 84 is:", gridsize_84)
      print("Grid size 85 is:", gridsize_85)
      print("Grid size 86 is:", gridsize_86)
      print("Grid size 87 is:", gridsize_87)
      print("Grid size 88 is:", gridsize_88)
      print("Grid size 89 is:", gridsize_89)
      print("Grid size 90 is:", gridsize_90)
      print("Grid size 91 is:", gridsize_91)
      print("Grid size 92 is:", gridsize_92)
      print("Grid size 93 is:", gridsize_93)
      print("Grid size 94 is:", gridsize_94)
      print("Grid size 95 is:", gridsize_95)
      print("Grid size 96 is:", gridsize_96)
      print("Grid size 97 is:", gridsize_97)
      print("Grid size 98 is:", gridsize_98)
      print("Grid size 99 is:", gridsize_99)
      print("Grid size 100 is:", gridsize_100)
      print("Grid size 101 is:", gridsize_101)
      print("Grid size 102 is:", gridsize_102)
      print("Grid size 103 is:", gridsize_103)
      print("Grid size 104 is:", gridsize_104)
      print("Grid size 105 is:", gridsize_105)
      print("Grid size 106 is:", gridsize_106)
      print("Grid size 107 is:", gridsize_107)
      print("Grid size 108 is:", gridsize_108)
      print("Grid size 109 is:", gridsize_109)
      regcount = 1 #MR: value suggested by GS since MAGMA's routines are quite surely optimized 
    else:
      print "Warning: unsupported ns value!"
      sys.exit(1)
    GPU_tasklist = [['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU'], ['iALU'], ['iALU', 11], ['iALU', 12], ['iALU', 13], ['iALU', 12, 14], ['iALU', 15], ['iALU', 12, 16], ['iALU', 15], ['iALU', 18, 4], ['iALU', 18], ['iALU', 20, 10], ['iALU', 18], ['iALU', 19, 17], ['iALU', 23, 8], ['iALU'], ['iALU', 24, 25], ['L2_ACCESS'], ['iALU', 17], ['diALU', 28], ['iALU', 18], ['diALU', 29, 30], ['L1_ACCESS', 27], ['iALU', 26], ['L2_ACCESS'], ['diALU', 28, 30], ['diALU', 35], ['L1_ACCESS', 34], ['iALU', 26], ['L2_ACCESS'], ['L1_ACCESS', 39], ['iALU', 26], ['L2_ACCESS'], ['L1_ACCESS', 42], ['iALU', 21, 5], ['iALU', 44, 17], ['iALU', 45, 9], ['L2_ACCESS'], ['diALU', 28], ['iALU', 20], ['diALU', 48, 49], ['L1_ACCESS', 47], ['iALU', 46, 5], ['L2_ACCESS'], ['diALU', 28, 49], ['diALU', 54], ['L1_ACCESS', 53], ['iALU', 5], ['iALU', 57, 46], ['L2_ACCESS'], ['L1_ACCESS', 59], ['iALU', 5, 46], ['L2_ACCESS'], ['L1_ACCESS', 62], ['THREAD_SYNC'], ['iALU', 3], ['diALU', 22], ['diALU', 66], ['iALU', 65], ['iALU', 4], ['iALU', 26, 69], ['iALU', 46], ['iALU', 5], ['iALU', 5], ['iALU'], ['iALU', 71, 74], ['iALU', 69, 70], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 75], ['iALU', 5, 75], ['L2_ACCESS'], ['iALU', 5, 85], ['iALU', 73, 85], ['L2_ACCESS'], ['iALU', 73, 88], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 94, 93], ['L1_ACCESS'], ['dfALU', 94, 96], ['L1_ACCESS'], ['dfALU', 94, 98], ['L1_ACCESS'], ['dfALU', 94, 100], ['L1_ACCESS'], ['dfALU', 102, 93], ['dfALU', 102, 96], ['dfALU', 102, 98], ['dfALU', 102, 100], ['L1_ACCESS'], ['dfALU', 107, 93], ['dfALU', 107, 96], ['dfALU', 107, 98], ['dfALU', 107, 100], ['L1_ACCESS'], ['dfALU', 112, 93], ['dfALU', 112, 96], ['dfALU', 112, 98], ['dfALU', 112, 100], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 118, 117, 95], ['L1_ACCESS'], ['dfALU', 118, 120, 97], ['L1_ACCESS'], ['dfALU', 118, 122, 99], ['L1_ACCESS'], ['dfALU', 118, 124, 101], ['L1_ACCESS'], ['dfALU', 126, 117, 103], ['dfALU', 126, 120, 104], ['dfALU', 126, 122, 105], ['dfALU', 126, 124, 106], ['L1_ACCESS'], ['dfALU', 131, 117, 108], ['dfALU', 131, 120, 109], ['dfALU', 131, 122, 110], ['dfALU', 131, 124, 111], ['L1_ACCESS'], ['dfALU', 136, 117, 113], ['dfALU', 136, 120, 114], ['dfALU', 136, 122, 115], ['dfALU', 136, 124, 116], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 142, 141, 119], ['L1_ACCESS'], ['dfALU', 142, 144, 121], ['L1_ACCESS'], ['dfALU', 142, 146, 123], ['L1_ACCESS'], ['dfALU', 142, 148, 125], ['L1_ACCESS'], ['dfALU', 150, 141, 127], ['dfALU', 150, 144, 128], ['dfALU', 150, 146, 129], ['dfALU', 150, 148, 130], ['L1_ACCESS'], ['dfALU', 155, 141, 132], ['dfALU', 155, 144, 133], ['dfALU', 155, 146, 134], ['dfALU', 155, 148, 135], ['L1_ACCESS'], ['dfALU', 160, 141, 137], ['dfALU', 160, 144, 138], ['dfALU', 160, 146, 139], ['dfALU', 160, 148, 140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 166, 165, 143], ['L1_ACCESS'], ['dfALU', 166, 168, 145], ['L1_ACCESS'], ['dfALU', 166, 170, 147], ['L1_ACCESS'], ['dfALU', 166, 172, 149], ['L1_ACCESS'], ['dfALU', 174, 165, 151], ['dfALU', 174, 168, 152], ['dfALU', 174, 170, 153], ['dfALU', 174, 172, 154], ['L1_ACCESS'], ['dfALU', 179, 165, 156], ['dfALU', 179, 168, 157], ['dfALU', 179, 170, 158], ['dfALU', 179, 172, 159], ['L1_ACCESS'], ['dfALU', 184, 165, 161], ['dfALU', 184, 168, 162], ['dfALU', 184, 170, 163], ['dfALU', 184, 172, 164], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 190, 189, 167], ['L1_ACCESS'], ['dfALU', 190, 192, 169], ['L1_ACCESS'], ['dfALU', 190, 194, 171], ['L1_ACCESS'], ['dfALU', 190, 196, 173], ['L1_ACCESS'], ['dfALU', 198, 189, 175], ['dfALU', 198, 192, 176], ['dfALU', 198, 194, 177], ['dfALU', 198, 196, 178], ['L1_ACCESS'], ['dfALU', 203, 189, 180], ['dfALU', 203, 192, 181], ['dfALU', 203, 194, 182], ['dfALU', 203, 196, 183], ['L1_ACCESS'], ['dfALU', 208, 189, 185], ['dfALU', 208, 192, 186], ['dfALU', 208, 194, 187], ['dfALU', 208, 196, 188], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 214, 213, 191], ['L1_ACCESS'], ['dfALU', 214, 216, 193], ['L1_ACCESS'], ['dfALU', 214, 218, 195], ['L1_ACCESS'], ['dfALU', 214, 220, 197], ['L1_ACCESS'], ['dfALU', 222, 213, 199], ['dfALU', 222, 216, 200], ['dfALU', 222, 218, 201], ['dfALU', 222, 220, 202], ['L1_ACCESS'], ['dfALU', 227, 213, 204], ['dfALU', 227, 216, 205], ['dfALU', 227, 218, 206], ['dfALU', 227, 220, 207], ['L1_ACCESS'], ['dfALU', 232, 213, 209], ['dfALU', 232, 216, 210], ['dfALU', 232, 218, 211], ['dfALU', 232, 220, 212], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 238, 237, 215], ['L1_ACCESS'], ['dfALU', 238, 240, 217], ['L1_ACCESS'], ['dfALU', 238, 242, 219], ['L1_ACCESS'], ['dfALU', 238, 244, 221], ['L1_ACCESS'], ['dfALU', 246, 237, 223], ['dfALU', 246, 240, 224], ['dfALU', 246, 242, 225], ['dfALU', 246, 244, 226], ['L1_ACCESS'], ['dfALU', 251, 237, 228], ['dfALU', 251, 240, 229], ['dfALU', 251, 242, 230], ['dfALU', 251, 244, 231], ['L1_ACCESS'], ['dfALU', 256, 237, 233], ['dfALU', 256, 240, 234], ['dfALU', 256, 242, 235], ['dfALU', 256, 244, 236], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 262, 261, 239], ['L1_ACCESS'], ['dfALU', 262, 264, 241], ['L1_ACCESS'], ['dfALU', 262, 266, 243], ['L1_ACCESS'], ['dfALU', 262, 268, 245], ['L1_ACCESS'], ['dfALU', 270, 261, 247], ['dfALU', 270, 264, 248], ['dfALU', 270, 266, 249], ['dfALU', 270, 268, 250], ['L1_ACCESS'], ['dfALU', 275, 261, 252], ['dfALU', 275, 264, 253], ['dfALU', 275, 266, 254], ['dfALU', 275, 268, 255], ['L1_ACCESS'], ['dfALU', 280, 261, 257], ['dfALU', 280, 264, 258], ['dfALU', 280, 266, 259], ['dfALU', 280, 268, 260], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 286, 285, 263], ['L1_ACCESS'], ['dfALU', 286, 288, 265], ['L1_ACCESS'], ['dfALU', 286, 290, 267], ['L1_ACCESS'], ['dfALU', 286, 292, 269], ['L1_ACCESS'], ['dfALU', 294, 285, 271], ['dfALU', 294, 288, 272], ['dfALU', 294, 290, 273], ['dfALU', 294, 292, 274], ['L1_ACCESS'], ['dfALU', 299, 285, 276], ['dfALU', 299, 288, 277], ['dfALU', 299, 290, 278], ['dfALU', 299, 292, 279], ['L1_ACCESS'], ['dfALU', 304, 285, 281], ['dfALU', 304, 288, 282], ['dfALU', 304, 290, 283], ['dfALU', 304, 292, 284], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 310, 309, 287], ['L1_ACCESS'], ['dfALU', 310, 312, 289], ['L1_ACCESS'], ['dfALU', 310, 314, 291], ['L1_ACCESS'], ['dfALU', 310, 316, 293], ['L1_ACCESS'], ['dfALU', 318, 309, 295], ['dfALU', 318, 312, 296], ['dfALU', 318, 314, 297], ['dfALU', 318, 316, 298], ['L1_ACCESS'], ['dfALU', 323, 309, 300], ['dfALU', 323, 312, 301], ['dfALU', 323, 314, 302], ['dfALU', 323, 316, 303], ['L1_ACCESS'], ['dfALU', 328, 309, 305], ['dfALU', 328, 312, 306], ['dfALU', 328, 314, 307], ['dfALU', 328, 316, 308], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 334, 333, 311], ['L1_ACCESS'], ['dfALU', 334, 336, 313], ['L1_ACCESS'], ['dfALU', 334, 338, 315], ['L1_ACCESS'], ['dfALU', 334, 340, 317], ['L1_ACCESS'], ['dfALU', 342, 333, 319], ['dfALU', 342, 336, 320], ['dfALU', 342, 338, 321], ['dfALU', 342, 340, 322], ['L1_ACCESS'], ['dfALU', 347, 333, 324], ['dfALU', 347, 336, 325], ['dfALU', 347, 338, 326], ['dfALU', 347, 340, 327], ['L1_ACCESS'], ['dfALU', 352, 333, 329], ['dfALU', 352, 336, 330], ['dfALU', 352, 338, 331], ['dfALU', 352, 340, 332], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 358, 357, 335], ['L1_ACCESS'], ['dfALU', 358, 360, 337], ['L1_ACCESS'], ['dfALU', 358, 362, 339], ['L1_ACCESS'], ['dfALU', 358, 364, 341], ['L1_ACCESS'], ['dfALU', 366, 357, 343], ['dfALU', 366, 360, 344], ['dfALU', 366, 362, 345], ['dfALU', 366, 364, 346], ['L1_ACCESS'], ['dfALU', 371, 357, 348], ['dfALU', 371, 360, 349], ['dfALU', 371, 362, 350], ['dfALU', 371, 364, 351], ['L1_ACCESS'], ['dfALU', 376, 357, 353], ['dfALU', 376, 360, 354], ['dfALU', 376, 362, 355], ['dfALU', 376, 364, 356], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 382, 381, 359], ['L1_ACCESS'], ['dfALU', 382, 384, 361], ['L1_ACCESS'], ['dfALU', 382, 386, 363], ['L1_ACCESS'], ['dfALU', 382, 388, 365], ['L1_ACCESS'], ['dfALU', 390, 381, 367], ['dfALU', 390, 384, 368], ['dfALU', 390, 386, 369], ['dfALU', 390, 388, 370], ['L1_ACCESS'], ['dfALU', 395, 381, 372], ['dfALU', 395, 384, 373], ['dfALU', 395, 386, 374], ['dfALU', 395, 388, 375], ['L1_ACCESS'], ['dfALU', 400, 381, 377], ['dfALU', 400, 384, 378], ['dfALU', 400, 386, 379], ['dfALU', 400, 388, 380], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 406, 405, 383], ['L1_ACCESS'], ['dfALU', 406, 408, 385], ['L1_ACCESS'], ['dfALU', 406, 410, 387], ['L1_ACCESS'], ['dfALU', 406, 412, 389], ['L1_ACCESS'], ['dfALU', 414, 405, 391], ['dfALU', 414, 408, 392], ['dfALU', 414, 410, 393], ['dfALU', 414, 412, 394], ['L1_ACCESS'], ['dfALU', 419, 405, 396], ['dfALU', 419, 408, 397], ['dfALU', 419, 410, 398], ['dfALU', 419, 412, 399], ['L1_ACCESS'], ['dfALU', 424, 405, 401], ['dfALU', 424, 408, 402], ['dfALU', 424, 410, 403], ['dfALU', 424, 412, 404], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 430, 429, 407], ['L1_ACCESS'], ['dfALU', 430, 432, 409], ['L1_ACCESS'], ['dfALU', 430, 434, 411], ['L1_ACCESS'], ['dfALU', 430, 436, 413], ['L1_ACCESS'], ['dfALU', 438, 429, 415], ['dfALU', 438, 432, 416], ['dfALU', 438, 434, 417], ['dfALU', 438, 436, 418], ['L1_ACCESS'], ['dfALU', 443, 429, 420], ['dfALU', 443, 432, 421], ['dfALU', 443, 434, 422], ['dfALU', 443, 436, 423], ['L1_ACCESS'], ['dfALU', 448, 429, 425], ['dfALU', 448, 432, 426], ['dfALU', 448, 434, 427], ['dfALU', 448, 436, 428], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 454, 453, 431], ['L1_ACCESS'], ['dfALU', 454, 456, 433], ['L1_ACCESS'], ['dfALU', 454, 458, 435], ['L1_ACCESS'], ['dfALU', 454, 460, 437], ['L1_ACCESS'], ['dfALU', 462, 453, 439], ['dfALU', 462, 456, 440], ['dfALU', 462, 458, 441], ['dfALU', 462, 460, 442], ['L1_ACCESS'], ['dfALU', 467, 453, 444], ['dfALU', 467, 456, 445], ['dfALU', 467, 458, 446], ['dfALU', 467, 460, 447], ['L1_ACCESS'], ['dfALU', 472, 453, 449], ['dfALU', 472, 456, 450], ['dfALU', 472, 458, 451], ['dfALU', 472, 460, 452], ['THREAD_SYNC'], ['L1_ACCESS', 77], ['L1_ACCESS', 79], ['L1_ACCESS', 81], ['L1_ACCESS', 83], ['L1_ACCESS', 84], ['L1_ACCESS', 87], ['L1_ACCESS', 90], ['L1_ACCESS', 92], ['THREAD_SYNC'], ['iALU'], ['iALU', 487, 65], ['iALU'], ['iALU', 489], ['iALU', 71, 490], ['iALU', 69, 489, 70], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 491], ['iALU', 5, 491], ['L2_ACCESS'], ['iALU', 5, 501], ['iALU', 73, 501], ['L2_ACCESS'], ['iALU', 73, 504], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 510, 509, 455], ['L1_ACCESS'], ['dfALU', 510, 512, 457], ['L1_ACCESS'], ['dfALU', 510, 514, 459], ['L1_ACCESS'], ['dfALU', 510, 516, 461], ['L1_ACCESS'], ['dfALU', 518, 509, 463], ['dfALU', 518, 512, 464], ['dfALU', 518, 514, 465], ['dfALU', 518, 516, 466], ['L1_ACCESS'], ['dfALU', 523, 509, 468], ['dfALU', 523, 512, 469], ['dfALU', 523, 514, 470], ['dfALU', 523, 516, 471], ['L1_ACCESS'], ['dfALU', 528, 509, 473], ['dfALU', 528, 512, 474], ['dfALU', 528, 514, 475], ['dfALU', 528, 516, 476], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 534, 533, 511], ['L1_ACCESS'], ['dfALU', 534, 536, 513], ['L1_ACCESS'], ['dfALU', 534, 538, 515], ['L1_ACCESS'], ['dfALU', 534, 540, 517], ['L1_ACCESS'], ['dfALU', 542, 533, 519], ['dfALU', 542, 536, 520], ['dfALU', 542, 538, 521], ['dfALU', 542, 540, 522], ['L1_ACCESS'], ['dfALU', 547, 533, 524], ['dfALU', 547, 536, 525], ['dfALU', 547, 538, 526], ['dfALU', 547, 540, 527], ['L1_ACCESS'], ['dfALU', 552, 533, 529], ['dfALU', 552, 536, 530], ['dfALU', 552, 538, 531], ['dfALU', 552, 540, 532], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 558, 557, 535], ['L1_ACCESS'], ['dfALU', 558, 560, 537], ['L1_ACCESS'], ['dfALU', 558, 562, 539], ['L1_ACCESS'], ['dfALU', 558, 564, 541], ['L1_ACCESS'], ['dfALU', 566, 557, 543], ['dfALU', 566, 560, 544], ['dfALU', 566, 562, 545], ['dfALU', 566, 564, 546], ['L1_ACCESS'], ['dfALU', 571, 557, 548], ['dfALU', 571, 560, 549], ['dfALU', 571, 562, 550], ['dfALU', 571, 564, 551], ['L1_ACCESS'], ['dfALU', 576, 557, 553], ['dfALU', 576, 560, 554], ['dfALU', 576, 562, 555], ['dfALU', 576, 564, 556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 582, 581, 559], ['L1_ACCESS'], ['dfALU', 582, 584, 561], ['L1_ACCESS'], ['dfALU', 582, 586, 563], ['L1_ACCESS'], ['dfALU', 582, 588, 565], ['L1_ACCESS'], ['dfALU', 590, 581, 567], ['dfALU', 590, 584, 568], ['dfALU', 590, 586, 569], ['dfALU', 590, 588, 570], ['L1_ACCESS'], ['dfALU', 595, 581, 572], ['dfALU', 595, 584, 573], ['dfALU', 595, 586, 574], ['dfALU', 595, 588, 575], ['L1_ACCESS'], ['dfALU', 600, 581, 577], ['dfALU', 600, 584, 578], ['dfALU', 600, 586, 579], ['dfALU', 600, 588, 580], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 606, 605, 583], ['L1_ACCESS'], ['dfALU', 606, 608, 585], ['L1_ACCESS'], ['dfALU', 606, 610, 587], ['L1_ACCESS'], ['dfALU', 606, 612, 589], ['L1_ACCESS'], ['dfALU', 614, 605, 591], ['dfALU', 614, 608, 592], ['dfALU', 614, 610, 593], ['dfALU', 614, 612, 594], ['L1_ACCESS'], ['dfALU', 619, 605, 596], ['dfALU', 619, 608, 597], ['dfALU', 619, 610, 598], ['dfALU', 619, 612, 599], ['L1_ACCESS'], ['dfALU', 624, 605, 601], ['dfALU', 624, 608, 602], ['dfALU', 624, 610, 603], ['dfALU', 624, 612, 604], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 630, 629, 607], ['L1_ACCESS'], ['dfALU', 630, 632, 609], ['L1_ACCESS'], ['dfALU', 630, 634, 611], ['L1_ACCESS'], ['dfALU', 630, 636, 613], ['L1_ACCESS'], ['dfALU', 638, 629, 615], ['dfALU', 638, 632, 616], ['dfALU', 638, 634, 617], ['dfALU', 638, 636, 618], ['L1_ACCESS'], ['dfALU', 643, 629, 620], ['dfALU', 643, 632, 621], ['dfALU', 643, 634, 622], ['dfALU', 643, 636, 623], ['L1_ACCESS'], ['dfALU', 648, 629, 625], ['dfALU', 648, 632, 626], ['dfALU', 648, 634, 627], ['dfALU', 648, 636, 628], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 654, 653, 631], ['L1_ACCESS'], ['dfALU', 654, 656, 633], ['L1_ACCESS'], ['dfALU', 654, 658, 635], ['L1_ACCESS'], ['dfALU', 654, 660, 637], ['L1_ACCESS'], ['dfALU', 662, 653, 639], ['dfALU', 662, 656, 640], ['dfALU', 662, 658, 641], ['dfALU', 662, 660, 642], ['L1_ACCESS'], ['dfALU', 667, 653, 644], ['dfALU', 667, 656, 645], ['dfALU', 667, 658, 646], ['dfALU', 667, 660, 647], ['L1_ACCESS'], ['dfALU', 672, 653, 649], ['dfALU', 672, 656, 650], ['dfALU', 672, 658, 651], ['dfALU', 672, 660, 652], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 678, 677, 655], ['L1_ACCESS'], ['dfALU', 678, 680, 657], ['L1_ACCESS'], ['dfALU', 678, 682, 659], ['L1_ACCESS'], ['dfALU', 678, 684, 661], ['L1_ACCESS'], ['dfALU', 686, 677, 663], ['dfALU', 686, 680, 664], ['dfALU', 686, 682, 665], ['dfALU', 686, 684, 666], ['L1_ACCESS'], ['dfALU', 691, 677, 668], ['dfALU', 691, 680, 669], ['dfALU', 691, 682, 670], ['dfALU', 691, 684, 671], ['L1_ACCESS'], ['dfALU', 696, 677, 673], ['dfALU', 696, 680, 674], ['dfALU', 696, 682, 675], ['dfALU', 696, 684, 676], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 702, 701, 679], ['L1_ACCESS'], ['dfALU', 702, 704, 681], ['L1_ACCESS'], ['dfALU', 702, 706, 683], ['L1_ACCESS'], ['dfALU', 702, 708, 685], ['L1_ACCESS'], ['dfALU', 710, 701, 687], ['dfALU', 710, 704, 688], ['dfALU', 710, 706, 689], ['dfALU', 710, 708, 690], ['L1_ACCESS'], ['dfALU', 715, 701, 692], ['dfALU', 715, 704, 693], ['dfALU', 715, 706, 694], ['dfALU', 715, 708, 695], ['L1_ACCESS'], ['dfALU', 720, 701, 697], ['dfALU', 720, 704, 698], ['dfALU', 720, 706, 699], ['dfALU', 720, 708, 700], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 726, 725, 703], ['L1_ACCESS'], ['dfALU', 726, 728, 705], ['L1_ACCESS'], ['dfALU', 726, 730, 707], ['L1_ACCESS'], ['dfALU', 726, 732, 709], ['L1_ACCESS'], ['dfALU', 734, 725, 711], ['dfALU', 734, 728, 712], ['dfALU', 734, 730, 713], ['dfALU', 734, 732, 714], ['L1_ACCESS'], ['dfALU', 739, 725, 716], ['dfALU', 739, 728, 717], ['dfALU', 739, 730, 718], ['dfALU', 739, 732, 719], ['L1_ACCESS'], ['dfALU', 744, 725, 721], ['dfALU', 744, 728, 722], ['dfALU', 744, 730, 723], ['dfALU', 744, 732, 724], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 750, 749, 727], ['L1_ACCESS'], ['dfALU', 750, 752, 729], ['L1_ACCESS'], ['dfALU', 750, 754, 731], ['L1_ACCESS'], ['dfALU', 750, 756, 733], ['L1_ACCESS'], ['dfALU', 758, 749, 735], ['dfALU', 758, 752, 736], ['dfALU', 758, 754, 737], ['dfALU', 758, 756, 738], ['L1_ACCESS'], ['dfALU', 763, 749, 740], ['dfALU', 763, 752, 741], ['dfALU', 763, 754, 742], ['dfALU', 763, 756, 743], ['L1_ACCESS'], ['dfALU', 768, 749, 745], ['dfALU', 768, 752, 746], ['dfALU', 768, 754, 747], ['dfALU', 768, 756, 748], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 774, 773, 751], ['L1_ACCESS'], ['dfALU', 774, 776, 753], ['L1_ACCESS'], ['dfALU', 774, 778, 755], ['L1_ACCESS'], ['dfALU', 774, 780, 757], ['L1_ACCESS'], ['dfALU', 782, 773, 759], ['dfALU', 782, 776, 760], ['dfALU', 782, 778, 761], ['dfALU', 782, 780, 762], ['L1_ACCESS'], ['dfALU', 787, 773, 764], ['dfALU', 787, 776, 765], ['dfALU', 787, 778, 766], ['dfALU', 787, 780, 767], ['L1_ACCESS'], ['dfALU', 792, 773, 769], ['dfALU', 792, 776, 770], ['dfALU', 792, 778, 771], ['dfALU', 792, 780, 772], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 798, 797, 775], ['L1_ACCESS'], ['dfALU', 798, 800, 777], ['L1_ACCESS'], ['dfALU', 798, 802, 779], ['L1_ACCESS'], ['dfALU', 798, 804, 781], ['L1_ACCESS'], ['dfALU', 806, 797, 783], ['dfALU', 806, 800, 784], ['dfALU', 806, 802, 785], ['dfALU', 806, 804, 786], ['L1_ACCESS'], ['dfALU', 811, 797, 788], ['dfALU', 811, 800, 789], ['dfALU', 811, 802, 790], ['dfALU', 811, 804, 791], ['L1_ACCESS'], ['dfALU', 816, 797, 793], ['dfALU', 816, 800, 794], ['dfALU', 816, 802, 795], ['dfALU', 816, 804, 796], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 822, 821, 799], ['L1_ACCESS'], ['dfALU', 822, 824, 801], ['L1_ACCESS'], ['dfALU', 822, 826, 803], ['L1_ACCESS'], ['dfALU', 822, 828, 805], ['L1_ACCESS'], ['dfALU', 830, 821, 807], ['dfALU', 830, 824, 808], ['dfALU', 830, 826, 809], ['dfALU', 830, 828, 810], ['L1_ACCESS'], ['dfALU', 835, 821, 812], ['dfALU', 835, 824, 813], ['dfALU', 835, 826, 814], ['dfALU', 835, 828, 815], ['L1_ACCESS'], ['dfALU', 840, 821, 817], ['dfALU', 840, 824, 818], ['dfALU', 840, 826, 819], ['dfALU', 840, 828, 820], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 846, 845, 823], ['L1_ACCESS'], ['dfALU', 846, 848, 825], ['L1_ACCESS'], ['dfALU', 846, 850, 827], ['L1_ACCESS'], ['dfALU', 846, 852, 829], ['L1_ACCESS'], ['dfALU', 854, 845, 831], ['dfALU', 854, 848, 832], ['dfALU', 854, 850, 833], ['dfALU', 854, 852, 834], ['L1_ACCESS'], ['dfALU', 859, 845, 836], ['dfALU', 859, 848, 837], ['dfALU', 859, 850, 838], ['dfALU', 859, 852, 839], ['L1_ACCESS'], ['dfALU', 864, 845, 841], ['dfALU', 864, 848, 842], ['dfALU', 864, 850, 843], ['dfALU', 864, 852, 844], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 870, 869, 847], ['L1_ACCESS'], ['dfALU', 870, 872, 849], ['L1_ACCESS'], ['dfALU', 870, 874, 851], ['L1_ACCESS'], ['dfALU', 870, 876, 853], ['L1_ACCESS'], ['dfALU', 878, 869, 855], ['dfALU', 878, 872, 856], ['dfALU', 878, 874, 857], ['dfALU', 878, 876, 858], ['L1_ACCESS'], ['dfALU', 883, 869, 860], ['dfALU', 883, 872, 861], ['dfALU', 883, 874, 862], ['dfALU', 883, 876, 863], ['L1_ACCESS'], ['dfALU', 888, 869, 865], ['dfALU', 888, 872, 866], ['dfALU', 888, 874, 867], ['dfALU', 888, 876, 868], ['THREAD_SYNC'], ['L1_ACCESS', 493], ['L1_ACCESS', 495], ['L1_ACCESS', 497], ['L1_ACCESS', 499], ['L1_ACCESS', 500], ['L1_ACCESS', 503], ['L1_ACCESS', 506], ['L1_ACCESS', 508], ['THREAD_SYNC'], ['iALU', 487], ['iALU', 903, 65], ['iALU', 489], ['iALU', 905], ['iALU', 71, 906], ['iALU', 69, 905, 70], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 907], ['iALU', 5, 907], ['L2_ACCESS'], ['iALU', 5, 917], ['iALU', 73, 917], ['L2_ACCESS'], ['iALU', 73, 920], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 926, 925, 871], ['L1_ACCESS'], ['dfALU', 926, 928, 873], ['L1_ACCESS'], ['dfALU', 926, 930, 875], ['L1_ACCESS'], ['dfALU', 926, 932, 877], ['L1_ACCESS'], ['dfALU', 934, 925, 879], ['dfALU', 934, 928, 880], ['dfALU', 934, 930, 881], ['dfALU', 934, 932, 882], ['L1_ACCESS'], ['dfALU', 939, 925, 884], ['dfALU', 939, 928, 885], ['dfALU', 939, 930, 886], ['dfALU', 939, 932, 887], ['L1_ACCESS'], ['dfALU', 944, 925, 889], ['dfALU', 944, 928, 890], ['dfALU', 944, 930, 891], ['dfALU', 944, 932, 892], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 950, 949, 927], ['L1_ACCESS'], ['dfALU', 950, 952, 929], ['L1_ACCESS'], ['dfALU', 950, 954, 931], ['L1_ACCESS'], ['dfALU', 950, 956, 933], ['L1_ACCESS'], ['dfALU', 958, 949, 935], ['dfALU', 958, 952, 936], ['dfALU', 958, 954, 937], ['dfALU', 958, 956, 938], ['L1_ACCESS'], ['dfALU', 963, 949, 940], ['dfALU', 963, 952, 941], ['dfALU', 963, 954, 942], ['dfALU', 963, 956, 943], ['L1_ACCESS'], ['dfALU', 968, 949, 945], ['dfALU', 968, 952, 946], ['dfALU', 968, 954, 947], ['dfALU', 968, 956, 948], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 974, 973, 951], ['L1_ACCESS'], ['dfALU', 974, 976, 953], ['L1_ACCESS'], ['dfALU', 974, 978, 955], ['L1_ACCESS'], ['dfALU', 974, 980, 957], ['L1_ACCESS'], ['dfALU', 982, 973, 959], ['dfALU', 982, 976, 960], ['dfALU', 982, 978, 961], ['dfALU', 982, 980, 962], ['L1_ACCESS'], ['dfALU', 987, 973, 964], ['dfALU', 987, 976, 965], ['dfALU', 987, 978, 966], ['dfALU', 987, 980, 967], ['L1_ACCESS'], ['dfALU', 992, 973, 969], ['dfALU', 992, 976, 970], ['dfALU', 992, 978, 971], ['dfALU', 992, 980, 972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 998, 997, 975], ['L1_ACCESS'], ['dfALU', 998, 1000, 977], ['L1_ACCESS'], ['dfALU', 998, 1002, 979], ['L1_ACCESS'], ['dfALU', 998, 1004, 981], ['L1_ACCESS'], ['dfALU', 1006, 997, 983], ['dfALU', 1006, 1000, 984], ['dfALU', 1006, 1002, 985], ['dfALU', 1006, 1004, 986], ['L1_ACCESS'], ['dfALU', 1011, 997, 988], ['dfALU', 1011, 1000, 989], ['dfALU', 1011, 1002, 990], ['dfALU', 1011, 1004, 991], ['L1_ACCESS'], ['dfALU', 1016, 997, 993], ['dfALU', 1016, 1000, 994], ['dfALU', 1016, 1002, 995], ['dfALU', 1016, 1004, 996], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1022, 1021, 999], ['L1_ACCESS'], ['dfALU', 1022, 1024, 1001], ['L1_ACCESS'], ['dfALU', 1022, 1026, 1003], ['L1_ACCESS'], ['dfALU', 1022, 1028, 1005], ['L1_ACCESS'], ['dfALU', 1030, 1021, 1007], ['dfALU', 1030, 1024, 1008], ['dfALU', 1030, 1026, 1009], ['dfALU', 1030, 1028, 1010], ['L1_ACCESS'], ['dfALU', 1035, 1021, 1012], ['dfALU', 1035, 1024, 1013], ['dfALU', 1035, 1026, 1014], ['dfALU', 1035, 1028, 1015], ['L1_ACCESS'], ['dfALU', 1040, 1021, 1017], ['dfALU', 1040, 1024, 1018], ['dfALU', 1040, 1026, 1019], ['dfALU', 1040, 1028, 1020], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1046, 1045, 1023], ['L1_ACCESS'], ['dfALU', 1046, 1048, 1025], ['L1_ACCESS'], ['dfALU', 1046, 1050, 1027], ['L1_ACCESS'], ['dfALU', 1046, 1052, 1029], ['L1_ACCESS'], ['dfALU', 1054, 1045, 1031], ['dfALU', 1054, 1048, 1032], ['dfALU', 1054, 1050, 1033], ['dfALU', 1054, 1052, 1034], ['L1_ACCESS'], ['dfALU', 1059, 1045, 1036], ['dfALU', 1059, 1048, 1037], ['dfALU', 1059, 1050, 1038], ['dfALU', 1059, 1052, 1039], ['L1_ACCESS'], ['dfALU', 1064, 1045, 1041], ['dfALU', 1064, 1048, 1042], ['dfALU', 1064, 1050, 1043], ['dfALU', 1064, 1052, 1044], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1070, 1069, 1047], ['L1_ACCESS'], ['dfALU', 1070, 1072, 1049], ['L1_ACCESS'], ['dfALU', 1070, 1074, 1051], ['L1_ACCESS'], ['dfALU', 1070, 1076, 1053], ['L1_ACCESS'], ['dfALU', 1078, 1069, 1055], ['dfALU', 1078, 1072, 1056], ['dfALU', 1078, 1074, 1057], ['dfALU', 1078, 1076, 1058], ['L1_ACCESS'], ['dfALU', 1083, 1069, 1060], ['dfALU', 1083, 1072, 1061], ['dfALU', 1083, 1074, 1062], ['dfALU', 1083, 1076, 1063], ['L1_ACCESS'], ['dfALU', 1088, 1069, 1065], ['dfALU', 1088, 1072, 1066], ['dfALU', 1088, 1074, 1067], ['dfALU', 1088, 1076, 1068], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1094, 1093, 1071], ['L1_ACCESS'], ['dfALU', 1094, 1096, 1073], ['L1_ACCESS'], ['dfALU', 1094, 1098, 1075], ['L1_ACCESS'], ['dfALU', 1094, 1100, 1077], ['L1_ACCESS'], ['dfALU', 1102, 1093, 1079], ['dfALU', 1102, 1096, 1080], ['dfALU', 1102, 1098, 1081], ['dfALU', 1102, 1100, 1082], ['L1_ACCESS'], ['dfALU', 1107, 1093, 1084], ['dfALU', 1107, 1096, 1085], ['dfALU', 1107, 1098, 1086], ['dfALU', 1107, 1100, 1087], ['L1_ACCESS'], ['dfALU', 1112, 1093, 1089], ['dfALU', 1112, 1096, 1090], ['dfALU', 1112, 1098, 1091], ['dfALU', 1112, 1100, 1092], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1118, 1117, 1095], ['L1_ACCESS'], ['dfALU', 1118, 1120, 1097], ['L1_ACCESS'], ['dfALU', 1118, 1122, 1099], ['L1_ACCESS'], ['dfALU', 1118, 1124, 1101], ['L1_ACCESS'], ['dfALU', 1126, 1117, 1103], ['dfALU', 1126, 1120, 1104], ['dfALU', 1126, 1122, 1105], ['dfALU', 1126, 1124, 1106], ['L1_ACCESS'], ['dfALU', 1131, 1117, 1108], ['dfALU', 1131, 1120, 1109], ['dfALU', 1131, 1122, 1110], ['dfALU', 1131, 1124, 1111], ['L1_ACCESS'], ['dfALU', 1136, 1117, 1113], ['dfALU', 1136, 1120, 1114], ['dfALU', 1136, 1122, 1115], ['dfALU', 1136, 1124, 1116], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1142, 1141, 1119], ['L1_ACCESS'], ['dfALU', 1142, 1144, 1121], ['L1_ACCESS'], ['dfALU', 1142, 1146, 1123], ['L1_ACCESS'], ['dfALU', 1142, 1148, 1125], ['L1_ACCESS'], ['dfALU', 1150, 1141, 1127], ['dfALU', 1150, 1144, 1128], ['dfALU', 1150, 1146, 1129], ['dfALU', 1150, 1148, 1130], ['L1_ACCESS'], ['dfALU', 1155, 1141, 1132], ['dfALU', 1155, 1144, 1133], ['dfALU', 1155, 1146, 1134], ['dfALU', 1155, 1148, 1135], ['L1_ACCESS'], ['dfALU', 1160, 1141, 1137], ['dfALU', 1160, 1144, 1138], ['dfALU', 1160, 1146, 1139], ['dfALU', 1160, 1148, 1140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1166, 1165, 1143], ['L1_ACCESS'], ['dfALU', 1166, 1168, 1145], ['L1_ACCESS'], ['dfALU', 1166, 1170, 1147], ['L1_ACCESS'], ['dfALU', 1166, 1172, 1149], ['L1_ACCESS'], ['dfALU', 1174, 1165, 1151], ['dfALU', 1174, 1168, 1152], ['dfALU', 1174, 1170, 1153], ['dfALU', 1174, 1172, 1154], ['L1_ACCESS'], ['dfALU', 1179, 1165, 1156], ['dfALU', 1179, 1168, 1157], ['dfALU', 1179, 1170, 1158], ['dfALU', 1179, 1172, 1159], ['L1_ACCESS'], ['dfALU', 1184, 1165, 1161], ['dfALU', 1184, 1168, 1162], ['dfALU', 1184, 1170, 1163], ['dfALU', 1184, 1172, 1164], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1190, 1189, 1167], ['L1_ACCESS'], ['dfALU', 1190, 1192, 1169], ['L1_ACCESS'], ['dfALU', 1190, 1194, 1171], ['L1_ACCESS'], ['dfALU', 1190, 1196, 1173], ['L1_ACCESS'], ['dfALU', 1198, 1189, 1175], ['dfALU', 1198, 1192, 1176], ['dfALU', 1198, 1194, 1177], ['dfALU', 1198, 1196, 1178], ['L1_ACCESS'], ['dfALU', 1203, 1189, 1180], ['dfALU', 1203, 1192, 1181], ['dfALU', 1203, 1194, 1182], ['dfALU', 1203, 1196, 1183], ['L1_ACCESS'], ['dfALU', 1208, 1189, 1185], ['dfALU', 1208, 1192, 1186], ['dfALU', 1208, 1194, 1187], ['dfALU', 1208, 1196, 1188], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1214, 1213, 1191], ['L1_ACCESS'], ['dfALU', 1214, 1216, 1193], ['L1_ACCESS'], ['dfALU', 1214, 1218, 1195], ['L1_ACCESS'], ['dfALU', 1214, 1220, 1197], ['L1_ACCESS'], ['dfALU', 1222, 1213, 1199], ['dfALU', 1222, 1216, 1200], ['dfALU', 1222, 1218, 1201], ['dfALU', 1222, 1220, 1202], ['L1_ACCESS'], ['dfALU', 1227, 1213, 1204], ['dfALU', 1227, 1216, 1205], ['dfALU', 1227, 1218, 1206], ['dfALU', 1227, 1220, 1207], ['L1_ACCESS'], ['dfALU', 1232, 1213, 1209], ['dfALU', 1232, 1216, 1210], ['dfALU', 1232, 1218, 1211], ['dfALU', 1232, 1220, 1212], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1238, 1237, 1215], ['L1_ACCESS'], ['dfALU', 1238, 1240, 1217], ['L1_ACCESS'], ['dfALU', 1238, 1242, 1219], ['L1_ACCESS'], ['dfALU', 1238, 1244, 1221], ['L1_ACCESS'], ['dfALU', 1246, 1237, 1223], ['dfALU', 1246, 1240, 1224], ['dfALU', 1246, 1242, 1225], ['dfALU', 1246, 1244, 1226], ['L1_ACCESS'], ['dfALU', 1251, 1237, 1228], ['dfALU', 1251, 1240, 1229], ['dfALU', 1251, 1242, 1230], ['dfALU', 1251, 1244, 1231], ['L1_ACCESS'], ['dfALU', 1256, 1237, 1233], ['dfALU', 1256, 1240, 1234], ['dfALU', 1256, 1242, 1235], ['dfALU', 1256, 1244, 1236], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1262, 1261, 1239], ['L1_ACCESS'], ['dfALU', 1262, 1264, 1241], ['L1_ACCESS'], ['dfALU', 1262, 1266, 1243], ['L1_ACCESS'], ['dfALU', 1262, 1268, 1245], ['L1_ACCESS'], ['dfALU', 1270, 1261, 1247], ['dfALU', 1270, 1264, 1248], ['dfALU', 1270, 1266, 1249], ['dfALU', 1270, 1268, 1250], ['L1_ACCESS'], ['dfALU', 1275, 1261, 1252], ['dfALU', 1275, 1264, 1253], ['dfALU', 1275, 1266, 1254], ['dfALU', 1275, 1268, 1255], ['L1_ACCESS'], ['dfALU', 1280, 1261, 1257], ['dfALU', 1280, 1264, 1258], ['dfALU', 1280, 1266, 1259], ['dfALU', 1280, 1268, 1260], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1286, 1285, 1263], ['L1_ACCESS'], ['dfALU', 1286, 1288, 1265], ['L1_ACCESS'], ['dfALU', 1286, 1290, 1267], ['L1_ACCESS'], ['dfALU', 1286, 1292, 1269], ['L1_ACCESS'], ['dfALU', 1294, 1285, 1271], ['dfALU', 1294, 1288, 1272], ['dfALU', 1294, 1290, 1273], ['dfALU', 1294, 1292, 1274], ['L1_ACCESS'], ['dfALU', 1299, 1285, 1276], ['dfALU', 1299, 1288, 1277], ['dfALU', 1299, 1290, 1278], ['dfALU', 1299, 1292, 1279], ['L1_ACCESS'], ['dfALU', 1304, 1285, 1281], ['dfALU', 1304, 1288, 1282], ['dfALU', 1304, 1290, 1283], ['dfALU', 1304, 1292, 1284], ['THREAD_SYNC'], ['L1_ACCESS', 909], ['L1_ACCESS', 911], ['L1_ACCESS', 913], ['L1_ACCESS', 915], ['L1_ACCESS', 916], ['L1_ACCESS', 919], ['L1_ACCESS', 922], ['L1_ACCESS', 924], ['THREAD_SYNC'], ['iALU', 903], ['iALU', 1319, 65], ['iALU', 905], ['iALU'], ['L1_ACCESS'], ['iALU'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 1326, 1325, 1287], ['L1_ACCESS'], ['dfALU', 1326, 1328, 1289], ['L1_ACCESS'], ['dfALU', 1326, 1330, 1291], ['L1_ACCESS'], ['dfALU', 1326, 1332, 1293], ['L1_ACCESS'], ['dfALU', 1334, 1325, 1295], ['dfALU', 1334, 1328, 1296], ['dfALU', 1334, 1330, 1297], ['dfALU', 1334, 1332, 1298], ['L1_ACCESS'], ['dfALU', 1339, 1325, 1300], ['dfALU', 1339, 1328, 1301], ['dfALU', 1339, 1330, 1302], ['dfALU', 1339, 1332, 1303], ['L1_ACCESS'], ['dfALU', 1344, 1325, 1305], ['dfALU', 1344, 1328, 1306], ['dfALU', 1344, 1330, 1307], ['dfALU', 1344, 1332, 1308], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1350, 1349, 1327], ['L1_ACCESS'], ['dfALU', 1350, 1352, 1329], ['L1_ACCESS'], ['dfALU', 1350, 1354, 1331], ['L1_ACCESS'], ['dfALU', 1350, 1356, 1333], ['L1_ACCESS'], ['dfALU', 1358, 1349, 1335], ['dfALU', 1358, 1352, 1336], ['dfALU', 1358, 1354, 1337], ['dfALU', 1358, 1356, 1338], ['L1_ACCESS'], ['dfALU', 1363, 1349, 1340], ['dfALU', 1363, 1352, 1341], ['dfALU', 1363, 1354, 1342], ['dfALU', 1363, 1356, 1343], ['L1_ACCESS'], ['dfALU', 1368, 1349, 1345], ['dfALU', 1368, 1352, 1346], ['dfALU', 1368, 1354, 1347], ['dfALU', 1368, 1356, 1348], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1374, 1373, 1351], ['L1_ACCESS'], ['dfALU', 1374, 1376, 1353], ['L1_ACCESS'], ['dfALU', 1374, 1378, 1355], ['L1_ACCESS'], ['dfALU', 1374, 1380, 1357], ['L1_ACCESS'], ['dfALU', 1382, 1373, 1359], ['dfALU', 1382, 1376, 1360], ['dfALU', 1382, 1378, 1361], ['dfALU', 1382, 1380, 1362], ['L1_ACCESS'], ['dfALU', 1387, 1373, 1364], ['dfALU', 1387, 1376, 1365], ['dfALU', 1387, 1378, 1366], ['dfALU', 1387, 1380, 1367], ['L1_ACCESS'], ['dfALU', 1392, 1373, 1369], ['dfALU', 1392, 1376, 1370], ['dfALU', 1392, 1378, 1371], ['dfALU', 1392, 1380, 1372], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1398, 1397, 1375], ['L1_ACCESS'], ['dfALU', 1398, 1400, 1377], ['L1_ACCESS'], ['dfALU', 1398, 1402, 1379], ['L1_ACCESS'], ['dfALU', 1398, 1404, 1381], ['L1_ACCESS'], ['dfALU', 1406, 1397, 1383], ['dfALU', 1406, 1400, 1384], ['dfALU', 1406, 1402, 1385], ['dfALU', 1406, 1404, 1386], ['L1_ACCESS'], ['dfALU', 1411, 1397, 1388], ['dfALU', 1411, 1400, 1389], ['dfALU', 1411, 1402, 1390], ['dfALU', 1411, 1404, 1391], ['L1_ACCESS'], ['dfALU', 1416, 1397, 1393], ['dfALU', 1416, 1400, 1394], ['dfALU', 1416, 1402, 1395], ['dfALU', 1416, 1404, 1396], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1422, 1421, 1399], ['L1_ACCESS'], ['dfALU', 1422, 1424, 1401], ['L1_ACCESS'], ['dfALU', 1422, 1426, 1403], ['L1_ACCESS'], ['dfALU', 1422, 1428, 1405], ['L1_ACCESS'], ['dfALU', 1430, 1421, 1407], ['dfALU', 1430, 1424, 1408], ['dfALU', 1430, 1426, 1409], ['dfALU', 1430, 1428, 1410], ['L1_ACCESS'], ['dfALU', 1435, 1421, 1412], ['dfALU', 1435, 1424, 1413], ['dfALU', 1435, 1426, 1414], ['dfALU', 1435, 1428, 1415], ['L1_ACCESS'], ['dfALU', 1440, 1421, 1417], ['dfALU', 1440, 1424, 1418], ['dfALU', 1440, 1426, 1419], ['dfALU', 1440, 1428, 1420], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1446, 1445, 1423], ['L1_ACCESS'], ['dfALU', 1446, 1448, 1425], ['L1_ACCESS'], ['dfALU', 1446, 1450, 1427], ['L1_ACCESS'], ['dfALU', 1446, 1452, 1429], ['L1_ACCESS'], ['dfALU', 1454, 1445, 1431], ['dfALU', 1454, 1448, 1432], ['dfALU', 1454, 1450, 1433], ['dfALU', 1454, 1452, 1434], ['L1_ACCESS'], ['dfALU', 1459, 1445, 1436], ['dfALU', 1459, 1448, 1437], ['dfALU', 1459, 1450, 1438], ['dfALU', 1459, 1452, 1439], ['L1_ACCESS'], ['dfALU', 1464, 1445, 1441], ['dfALU', 1464, 1448, 1442], ['dfALU', 1464, 1450, 1443], ['dfALU', 1464, 1452, 1444], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1470, 1469, 1447], ['L1_ACCESS'], ['dfALU', 1470, 1472, 1449], ['L1_ACCESS'], ['dfALU', 1470, 1474, 1451], ['L1_ACCESS'], ['dfALU', 1470, 1476, 1453], ['L1_ACCESS'], ['dfALU', 1478, 1469, 1455], ['dfALU', 1478, 1472, 1456], ['dfALU', 1478, 1474, 1457], ['dfALU', 1478, 1476, 1458], ['L1_ACCESS'], ['dfALU', 1483, 1469, 1460], ['dfALU', 1483, 1472, 1461], ['dfALU', 1483, 1474, 1462], ['dfALU', 1483, 1476, 1463], ['L1_ACCESS'], ['dfALU', 1488, 1469, 1465], ['dfALU', 1488, 1472, 1466], ['dfALU', 1488, 1474, 1467], ['dfALU', 1488, 1476, 1468], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1494, 1493, 1471], ['L1_ACCESS'], ['dfALU', 1494, 1496, 1473], ['L1_ACCESS'], ['dfALU', 1494, 1498, 1475], ['L1_ACCESS'], ['dfALU', 1494, 1500, 1477], ['L1_ACCESS'], ['dfALU', 1502, 1493, 1479], ['dfALU', 1502, 1496, 1480], ['dfALU', 1502, 1498, 1481], ['dfALU', 1502, 1500, 1482], ['L1_ACCESS'], ['dfALU', 1507, 1493, 1484], ['dfALU', 1507, 1496, 1485], ['dfALU', 1507, 1498, 1486], ['dfALU', 1507, 1500, 1487], ['L1_ACCESS'], ['dfALU', 1512, 1493, 1489], ['dfALU', 1512, 1496, 1490], ['dfALU', 1512, 1498, 1491], ['dfALU', 1512, 1500, 1492], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1518, 1517, 1495], ['L1_ACCESS'], ['dfALU', 1518, 1520, 1497], ['L1_ACCESS'], ['dfALU', 1518, 1522, 1499], ['L1_ACCESS'], ['dfALU', 1518, 1524, 1501], ['L1_ACCESS'], ['dfALU', 1526, 1517, 1503], ['dfALU', 1526, 1520, 1504], ['dfALU', 1526, 1522, 1505], ['dfALU', 1526, 1524, 1506], ['L1_ACCESS'], ['dfALU', 1531, 1517, 1508], ['dfALU', 1531, 1520, 1509], ['dfALU', 1531, 1522, 1510], ['dfALU', 1531, 1524, 1511], ['L1_ACCESS'], ['dfALU', 1536, 1517, 1513], ['dfALU', 1536, 1520, 1514], ['dfALU', 1536, 1522, 1515], ['dfALU', 1536, 1524, 1516], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1542, 1541, 1519], ['L1_ACCESS'], ['dfALU', 1542, 1544, 1521], ['L1_ACCESS'], ['dfALU', 1542, 1546, 1523], ['L1_ACCESS'], ['dfALU', 1542, 1548, 1525], ['L1_ACCESS'], ['dfALU', 1550, 1541, 1527], ['dfALU', 1550, 1544, 1528], ['dfALU', 1550, 1546, 1529], ['dfALU', 1550, 1548, 1530], ['L1_ACCESS'], ['dfALU', 1555, 1541, 1532], ['dfALU', 1555, 1544, 1533], ['dfALU', 1555, 1546, 1534], ['dfALU', 1555, 1548, 1535], ['L1_ACCESS'], ['dfALU', 1560, 1541, 1537], ['dfALU', 1560, 1544, 1538], ['dfALU', 1560, 1546, 1539], ['dfALU', 1560, 1548, 1540], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1566, 1565, 1543], ['L1_ACCESS'], ['dfALU', 1566, 1568, 1545], ['L1_ACCESS'], ['dfALU', 1566, 1570, 1547], ['L1_ACCESS'], ['dfALU', 1566, 1572, 1549], ['L1_ACCESS'], ['dfALU', 1574, 1565, 1551], ['dfALU', 1574, 1568, 1552], ['dfALU', 1574, 1570, 1553], ['dfALU', 1574, 1572, 1554], ['L1_ACCESS'], ['dfALU', 1579, 1565, 1556], ['dfALU', 1579, 1568, 1557], ['dfALU', 1579, 1570, 1558], ['dfALU', 1579, 1572, 1559], ['L1_ACCESS'], ['dfALU', 1584, 1565, 1561], ['dfALU', 1584, 1568, 1562], ['dfALU', 1584, 1570, 1563], ['dfALU', 1584, 1572, 1564], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1590, 1589, 1567], ['L1_ACCESS'], ['dfALU', 1590, 1592, 1569], ['L1_ACCESS'], ['dfALU', 1590, 1594, 1571], ['L1_ACCESS'], ['dfALU', 1590, 1596, 1573], ['L1_ACCESS'], ['dfALU', 1598, 1589, 1575], ['dfALU', 1598, 1592, 1576], ['dfALU', 1598, 1594, 1577], ['dfALU', 1598, 1596, 1578], ['L1_ACCESS'], ['dfALU', 1603, 1589, 1580], ['dfALU', 1603, 1592, 1581], ['dfALU', 1603, 1594, 1582], ['dfALU', 1603, 1596, 1583], ['L1_ACCESS'], ['dfALU', 1608, 1589, 1585], ['dfALU', 1608, 1592, 1586], ['dfALU', 1608, 1594, 1587], ['dfALU', 1608, 1596, 1588], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1614, 1613, 1591], ['L1_ACCESS'], ['dfALU', 1614, 1616, 1593], ['L1_ACCESS'], ['dfALU', 1614, 1618, 1595], ['L1_ACCESS'], ['dfALU', 1614, 1620, 1597], ['L1_ACCESS'], ['dfALU', 1622, 1613, 1599], ['dfALU', 1622, 1616, 1600], ['dfALU', 1622, 1618, 1601], ['dfALU', 1622, 1620, 1602], ['L1_ACCESS'], ['dfALU', 1627, 1613, 1604], ['dfALU', 1627, 1616, 1605], ['dfALU', 1627, 1618, 1606], ['dfALU', 1627, 1620, 1607], ['L1_ACCESS'], ['dfALU', 1632, 1613, 1609], ['dfALU', 1632, 1616, 1610], ['dfALU', 1632, 1618, 1611], ['dfALU', 1632, 1620, 1612], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1638, 1637, 1615], ['L1_ACCESS'], ['dfALU', 1638, 1640, 1617], ['L1_ACCESS'], ['dfALU', 1638, 1642, 1619], ['L1_ACCESS'], ['dfALU', 1638, 1644, 1621], ['L1_ACCESS'], ['dfALU', 1646, 1637, 1623], ['dfALU', 1646, 1640, 1624], ['dfALU', 1646, 1642, 1625], ['dfALU', 1646, 1644, 1626], ['L1_ACCESS'], ['dfALU', 1651, 1637, 1628], ['dfALU', 1651, 1640, 1629], ['dfALU', 1651, 1642, 1630], ['dfALU', 1651, 1644, 1631], ['L1_ACCESS'], ['dfALU', 1656, 1637, 1633], ['dfALU', 1656, 1640, 1634], ['dfALU', 1656, 1642, 1635], ['dfALU', 1656, 1644, 1636], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1662, 1661, 1639], ['L1_ACCESS'], ['dfALU', 1662, 1664, 1641], ['L1_ACCESS'], ['dfALU', 1662, 1666, 1643], ['L1_ACCESS'], ['dfALU', 1662, 1668, 1645], ['L1_ACCESS'], ['dfALU', 1670, 1661, 1647], ['dfALU', 1670, 1664, 1648], ['dfALU', 1670, 1666, 1649], ['dfALU', 1670, 1668, 1650], ['L1_ACCESS'], ['dfALU', 1675, 1661, 1652], ['dfALU', 1675, 1664, 1653], ['dfALU', 1675, 1666, 1654], ['dfALU', 1675, 1668, 1655], ['L1_ACCESS'], ['dfALU', 1680, 1661, 1657], ['dfALU', 1680, 1664, 1658], ['dfALU', 1680, 1666, 1659], ['dfALU', 1680, 1668, 1660], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1686, 1685, 1663], ['L1_ACCESS'], ['dfALU', 1686, 1688, 1665], ['L1_ACCESS'], ['dfALU', 1686, 1690, 1667], ['L1_ACCESS'], ['dfALU', 1686, 1692, 1669], ['L1_ACCESS'], ['dfALU', 1694, 1685, 1671], ['dfALU', 1694, 1688, 1672], ['dfALU', 1694, 1690, 1673], ['dfALU', 1694, 1692, 1674], ['L1_ACCESS'], ['dfALU', 1699, 1685, 1676], ['dfALU', 1699, 1688, 1677], ['dfALU', 1699, 1690, 1678], ['dfALU', 1699, 1692, 1679], ['L1_ACCESS'], ['dfALU', 1704, 1685, 1681], ['dfALU', 1704, 1688, 1682], ['dfALU', 1704, 1690, 1683], ['dfALU', 1704, 1692, 1684], ['iALU', 18, 1324], ['iALU', 1709, 1323], ['iALU', 1710, 17], ['iALU', 1711, 1322], ['iALU', 1709, 2], ['iALU', 17, 1322], ['iALU', 1714, 1], ['iALU', 1715, 1713], ['diALU', 0], ['iALU', 1712], ['diALU', 1717, 1718], ['GLOB_MEM_ACCESS', 1719], ['dfALU', 1720, 7], ['dfALU', 1687, 6, 1721], ['GLOB_MEM_ACCESS', 1722], ['iALU', 1714], ['iALU', 1724, 1], ['iALU', 1725, 1713], ['diALU', 0], ['iALU', 1712], ['diALU', 1727, 1728], ['GLOB_MEM_ACCESS'], ['dfALU', 1730, 7], ['dfALU', 1695, 6, 1731], ['GLOB_MEM_ACCESS', 1732], ['iALU', 1714], ['iALU', 1734, 1], ['iALU', 1735, 1713], ['diALU', 0], ['iALU', 1712], ['diALU', 1737, 1738], ['GLOB_MEM_ACCESS'], ['dfALU', 1740, 7], ['dfALU', 1700, 6, 1741], ['GLOB_MEM_ACCESS', 1742], ['iALU', 1714], ['iALU', 1744, 1], ['iALU', 1745, 1713], ['diALU', 0], ['iALU', 1712], ['diALU', 1747, 1748], ['GLOB_MEM_ACCESS'], ['dfALU', 1750, 7], ['dfALU', 1705, 6, 1751], ['GLOB_MEM_ACCESS', 1752], ['L1_ACCESS'], ['iALU', 1754], ['iALU', 1755], ['iALU', 1712], ['diALU', 1757, 1756], ['diALU', 0], ['diALU', 1758], ['diALU', 1759, 1760], ['iALU', 1709], ['iALU', 1762, 2], ['iALU', 1715, 1763], ['GLOB_MEM_ACCESS', 1761], ['dfALU', 1765, 7], ['dfALU', 1689, 6, 1766], ['GLOB_MEM_ACCESS', 1767], ['iALU', 1725, 1763], ['GLOB_MEM_ACCESS'], ['dfALU', 1770, 7], ['dfALU', 1696, 6, 1771], ['GLOB_MEM_ACCESS', 1772], ['iALU', 1735, 1763], ['GLOB_MEM_ACCESS'], ['dfALU', 1775, 7], ['dfALU', 1701, 6, 1776], ['GLOB_MEM_ACCESS', 1777], ['iALU', 1745, 1763], ['GLOB_MEM_ACCESS'], ['dfALU', 1780, 7], ['dfALU', 1706, 6, 1781], ['GLOB_MEM_ACCESS', 1782], ['iALU', 1755], ['iALU', 1712], ['diALU', 1785, 1784], ['diALU', 1786], ['diALU', 1759, 1787], ['iALU', 1709], ['iALU', 1789, 2], ['iALU', 1715, 1790], ['GLOB_MEM_ACCESS', 1788], ['dfALU', 1792, 7], ['dfALU', 1691, 6, 1793], ['GLOB_MEM_ACCESS', 1794], ['iALU', 1725, 1790], ['GLOB_MEM_ACCESS'], ['dfALU', 1797, 7], ['dfALU', 1697, 6, 1798], ['GLOB_MEM_ACCESS', 1799], ['iALU', 1735, 1790], ['GLOB_MEM_ACCESS'], ['dfALU', 1802, 7], ['dfALU', 1702, 6, 1803], ['GLOB_MEM_ACCESS', 1804], ['iALU', 1745, 1790], ['GLOB_MEM_ACCESS'], ['dfALU', 1807, 7], ['dfALU', 1707, 6, 1808], ['GLOB_MEM_ACCESS', 1809], ['iALU', 1755], ['iALU', 1712], ['diALU', 1811, 1812], ['diALU', 1813], ['diALU', 1759, 1814], ['iALU', 1709], ['iALU', 1816, 2], ['iALU', 1715, 1817], ['GLOB_MEM_ACCESS', 1815], ['dfALU', 1819, 7], ['dfALU', 1693, 6, 1820], ['GLOB_MEM_ACCESS', 1821], ['iALU', 1725, 1817], ['GLOB_MEM_ACCESS'], ['dfALU', 1824, 7], ['dfALU', 1698, 6, 1825], ['GLOB_MEM_ACCESS', 1826], ['iALU', 1735, 1817], ['GLOB_MEM_ACCESS'], ['dfALU', 1829, 7], ['dfALU', 1703, 6, 1830], ['GLOB_MEM_ACCESS', 1831], ['iALU', 1745, 1817], ['GLOB_MEM_ACCESS'], ['dfALU', 1834, 7], ['dfALU', 1708, 6, 1835], ['GLOB_MEM_ACCESS', 1836]]

    GPU_tasklist2 = [['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU'], ['iALU'], ['iALU', 11], ['iALU', 12], ['iALU', 13], ['iALU', 12, 14], ['iALU', 15], ['iALU', 12, 16], ['iALU', 15], ['iALU', 18, 4], ['iALU', 18], ['iALU', 20, 10], ['iALU', 18], ['iALU', 19, 17], ['iALU', 23, 8], ['iALU'], ['iALU', 24, 25], ['L2_ACCESS'], ['iALU', 17], ['diALU', 28], ['iALU', 18], ['diALU', 29, 30], ['L1_ACCESS', 27], ['iALU', 26], ['L2_ACCESS'], ['diALU', 28, 30], ['diALU', 35], ['L1_ACCESS', 34], ['iALU', 26], ['L2_ACCESS'], ['L1_ACCESS', 39], ['iALU', 26], ['L2_ACCESS'], ['L1_ACCESS', 42], ['iALU', 21, 5], ['iALU', 44, 17], ['iALU', 45, 9], ['L2_ACCESS'], ['diALU', 28], ['iALU', 20], ['diALU', 48, 49], ['L1_ACCESS', 47], ['iALU', 46, 5], ['L2_ACCESS'], ['diALU', 28, 49], ['diALU', 54], ['L1_ACCESS', 53], ['iALU', 5], ['iALU', 57, 46], ['L2_ACCESS'], ['L1_ACCESS', 59], ['iALU', 5, 46], ['L2_ACCESS'], ['L1_ACCESS', 62], ['THREAD_SYNC'], ['iALU', 3], ['diALU', 22], ['diALU', 66], ['iALU', 65], ['iALU', 4], ['iALU', 26, 69], ['iALU', 46], ['iALU', 5], ['iALU', 5], ['iALU'], ['iALU', 71, 74], ['iALU', 69, 70], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 75], ['iALU', 5, 75], ['L2_ACCESS'], ['iALU', 5, 85], ['iALU', 73, 85], ['L2_ACCESS'], ['iALU', 73, 88], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 94, 93], ['L1_ACCESS'], ['dfALU', 94, 96], ['L1_ACCESS'], ['dfALU', 94, 98], ['L1_ACCESS'], ['dfALU', 94, 100], ['L1_ACCESS'], ['dfALU', 102, 93], ['dfALU', 102, 96], ['dfALU', 102, 98], ['dfALU', 102, 100], ['L1_ACCESS'], ['dfALU', 107, 93], ['dfALU', 107, 96], ['dfALU', 107, 98], ['dfALU', 107, 100], ['L1_ACCESS'], ['dfALU', 112, 93], ['dfALU', 112, 96], ['dfALU', 112, 98], ['dfALU', 112, 100], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 118, 117, 95], ['L1_ACCESS'], ['dfALU', 118, 120, 97], ['L1_ACCESS'], ['dfALU', 118, 122, 99], ['L1_ACCESS'], ['dfALU', 118, 124, 101], ['L1_ACCESS'], ['dfALU', 126, 117, 103], ['dfALU', 126, 120, 104], ['dfALU', 126, 122, 105], ['dfALU', 126, 124, 106], ['L1_ACCESS'], ['dfALU', 131, 117, 108], ['dfALU', 131, 120, 109], ['dfALU', 131, 122, 110], ['dfALU', 131, 124, 111], ['L1_ACCESS'], ['dfALU', 136, 117, 113], ['dfALU', 136, 120, 114], ['dfALU', 136, 122, 115], ['dfALU', 136, 124, 116], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 142, 141, 119], ['L1_ACCESS'], ['dfALU', 142, 144, 121], ['L1_ACCESS'], ['dfALU', 142, 146, 123], ['L1_ACCESS'], ['dfALU', 142, 148, 125], ['L1_ACCESS'], ['dfALU', 150, 141, 127], ['dfALU', 150, 144, 128], ['dfALU', 150, 146, 129], ['dfALU', 150, 148, 130], ['L1_ACCESS'], ['dfALU', 155, 141, 132], ['dfALU', 155, 144, 133], ['dfALU', 155, 146, 134], ['dfALU', 155, 148, 135], ['L1_ACCESS'], ['dfALU', 160, 141, 137], ['dfALU', 160, 144, 138], ['dfALU', 160, 146, 139], ['dfALU', 160, 148, 140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 166, 165, 143], ['L1_ACCESS'], ['dfALU', 166, 168, 145], ['L1_ACCESS'], ['dfALU', 166, 170, 147], ['L1_ACCESS'], ['dfALU', 166, 172, 149], ['L1_ACCESS'], ['dfALU', 174, 165, 151], ['dfALU', 174, 168, 152], ['dfALU', 174, 170, 153], ['dfALU', 174, 172, 154], ['L1_ACCESS'], ['dfALU', 179, 165, 156], ['dfALU', 179, 168, 157], ['dfALU', 179, 170, 158], ['dfALU', 179, 172, 159], ['L1_ACCESS'], ['dfALU', 184, 165, 161], ['dfALU', 184, 168, 162], ['dfALU', 184, 170, 163], ['dfALU', 184, 172, 164], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 190, 189, 167], ['L1_ACCESS'], ['dfALU', 190, 192, 169], ['L1_ACCESS'], ['dfALU', 190, 194, 171], ['L1_ACCESS'], ['dfALU', 190, 196, 173], ['L1_ACCESS'], ['dfALU', 198, 189, 175], ['dfALU', 198, 192, 176], ['dfALU', 198, 194, 177], ['dfALU', 198, 196, 178], ['L1_ACCESS'], ['dfALU', 203, 189, 180], ['dfALU', 203, 192, 181], ['dfALU', 203, 194, 182], ['dfALU', 203, 196, 183], ['L1_ACCESS'], ['dfALU', 208, 189, 185], ['dfALU', 208, 192, 186], ['dfALU', 208, 194, 187], ['dfALU', 208, 196, 188], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 214, 213, 191], ['L1_ACCESS'], ['dfALU', 214, 216, 193], ['L1_ACCESS'], ['dfALU', 214, 218, 195], ['L1_ACCESS'], ['dfALU', 214, 220, 197], ['L1_ACCESS'], ['dfALU', 222, 213, 199], ['dfALU', 222, 216, 200], ['dfALU', 222, 218, 201], ['dfALU', 222, 220, 202], ['L1_ACCESS'], ['dfALU', 227, 213, 204], ['dfALU', 227, 216, 205], ['dfALU', 227, 218, 206], ['dfALU', 227, 220, 207], ['L1_ACCESS'], ['dfALU', 232, 213, 209], ['dfALU', 232, 216, 210], ['dfALU', 232, 218, 211], ['dfALU', 232, 220, 212], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 238, 237, 215], ['L1_ACCESS'], ['dfALU', 238, 240, 217], ['L1_ACCESS'], ['dfALU', 238, 242, 219], ['L1_ACCESS'], ['dfALU', 238, 244, 221], ['L1_ACCESS'], ['dfALU', 246, 237, 223], ['dfALU', 246, 240, 224], ['dfALU', 246, 242, 225], ['dfALU', 246, 244, 226], ['L1_ACCESS'], ['dfALU', 251, 237, 228], ['dfALU', 251, 240, 229], ['dfALU', 251, 242, 230], ['dfALU', 251, 244, 231], ['L1_ACCESS'], ['dfALU', 256, 237, 233], ['dfALU', 256, 240, 234], ['dfALU', 256, 242, 235], ['dfALU', 256, 244, 236], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 262, 261, 239], ['L1_ACCESS'], ['dfALU', 262, 264, 241], ['L1_ACCESS'], ['dfALU', 262, 266, 243], ['L1_ACCESS'], ['dfALU', 262, 268, 245], ['L1_ACCESS'], ['dfALU', 270, 261, 247], ['dfALU', 270, 264, 248], ['dfALU', 270, 266, 249], ['dfALU', 270, 268, 250], ['L1_ACCESS'], ['dfALU', 275, 261, 252], ['dfALU', 275, 264, 253], ['dfALU', 275, 266, 254], ['dfALU', 275, 268, 255], ['L1_ACCESS'], ['dfALU', 280, 261, 257], ['dfALU', 280, 264, 258], ['dfALU', 280, 266, 259], ['dfALU', 280, 268, 260], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 286, 285, 263], ['L1_ACCESS'], ['dfALU', 286, 288, 265], ['L1_ACCESS'], ['dfALU', 286, 290, 267], ['L1_ACCESS'], ['dfALU', 286, 292, 269], ['L1_ACCESS'], ['dfALU', 294, 285, 271], ['dfALU', 294, 288, 272], ['dfALU', 294, 290, 273], ['dfALU', 294, 292, 274], ['L1_ACCESS'], ['dfALU', 299, 285, 276], ['dfALU', 299, 288, 277], ['dfALU', 299, 290, 278], ['dfALU', 299, 292, 279], ['L1_ACCESS'], ['dfALU', 304, 285, 281], ['dfALU', 304, 288, 282], ['dfALU', 304, 290, 283], ['dfALU', 304, 292, 284], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 310, 309, 287], ['L1_ACCESS'], ['dfALU', 310, 312, 289], ['L1_ACCESS'], ['dfALU', 310, 314, 291], ['L1_ACCESS'], ['dfALU', 310, 316, 293], ['L1_ACCESS'], ['dfALU', 318, 309, 295], ['dfALU', 318, 312, 296], ['dfALU', 318, 314, 297], ['dfALU', 318, 316, 298], ['L1_ACCESS'], ['dfALU', 323, 309, 300], ['dfALU', 323, 312, 301], ['dfALU', 323, 314, 302], ['dfALU', 323, 316, 303], ['L1_ACCESS'], ['dfALU', 328, 309, 305], ['dfALU', 328, 312, 306], ['dfALU', 328, 314, 307], ['dfALU', 328, 316, 308], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 334, 333, 311], ['L1_ACCESS'], ['dfALU', 334, 336, 313], ['L1_ACCESS'], ['dfALU', 334, 338, 315], ['L1_ACCESS'], ['dfALU', 334, 340, 317], ['L1_ACCESS'], ['dfALU', 342, 333, 319], ['dfALU', 342, 336, 320], ['dfALU', 342, 338, 321], ['dfALU', 342, 340, 322], ['L1_ACCESS'], ['dfALU', 347, 333, 324], ['dfALU', 347, 336, 325], ['dfALU', 347, 338, 326], ['dfALU', 347, 340, 327], ['L1_ACCESS'], ['dfALU', 352, 333, 329], ['dfALU', 352, 336, 330], ['dfALU', 352, 338, 331], ['dfALU', 352, 340, 332], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 358, 357, 335], ['L1_ACCESS'], ['dfALU', 358, 360, 337], ['L1_ACCESS'], ['dfALU', 358, 362, 339], ['L1_ACCESS'], ['dfALU', 358, 364, 341], ['L1_ACCESS'], ['dfALU', 366, 357, 343], ['dfALU', 366, 360, 344], ['dfALU', 366, 362, 345], ['dfALU', 366, 364, 346], ['L1_ACCESS'], ['dfALU', 371, 357, 348], ['dfALU', 371, 360, 349], ['dfALU', 371, 362, 350], ['dfALU', 371, 364, 351], ['L1_ACCESS'], ['dfALU', 376, 357, 353], ['dfALU', 376, 360, 354], ['dfALU', 376, 362, 355], ['dfALU', 376, 364, 356], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 382, 381, 359], ['L1_ACCESS'], ['dfALU', 382, 384, 361], ['L1_ACCESS'], ['dfALU', 382, 386, 363], ['L1_ACCESS'], ['dfALU', 382, 388, 365], ['L1_ACCESS'], ['dfALU', 390, 381, 367], ['dfALU', 390, 384, 368], ['dfALU', 390, 386, 369], ['dfALU', 390, 388, 370], ['L1_ACCESS'], ['dfALU', 395, 381, 372], ['dfALU', 395, 384, 373], ['dfALU', 395, 386, 374], ['dfALU', 395, 388, 375], ['L1_ACCESS'], ['dfALU', 400, 381, 377], ['dfALU', 400, 384, 378], ['dfALU', 400, 386, 379], ['dfALU', 400, 388, 380], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 406, 405, 383], ['L1_ACCESS'], ['dfALU', 406, 408, 385], ['L1_ACCESS'], ['dfALU', 406, 410, 387], ['L1_ACCESS'], ['dfALU', 406, 412, 389], ['L1_ACCESS'], ['dfALU', 414, 405, 391], ['dfALU', 414, 408, 392], ['dfALU', 414, 410, 393], ['dfALU', 414, 412, 394], ['L1_ACCESS'], ['dfALU', 419, 405, 396], ['dfALU', 419, 408, 397], ['dfALU', 419, 410, 398], ['dfALU', 419, 412, 399], ['L1_ACCESS'], ['dfALU', 424, 405, 401], ['dfALU', 424, 408, 402], ['dfALU', 424, 410, 403], ['dfALU', 424, 412, 404], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 430, 429, 407], ['L1_ACCESS'], ['dfALU', 430, 432, 409], ['L1_ACCESS'], ['dfALU', 430, 434, 411], ['L1_ACCESS'], ['dfALU', 430, 436, 413], ['L1_ACCESS'], ['dfALU', 438, 429, 415], ['dfALU', 438, 432, 416], ['dfALU', 438, 434, 417], ['dfALU', 438, 436, 418], ['L1_ACCESS'], ['dfALU', 443, 429, 420], ['dfALU', 443, 432, 421], ['dfALU', 443, 434, 422], ['dfALU', 443, 436, 423], ['L1_ACCESS'], ['dfALU', 448, 429, 425], ['dfALU', 448, 432, 426], ['dfALU', 448, 434, 427], ['dfALU', 448, 436, 428], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 454, 453, 431], ['L1_ACCESS'], ['dfALU', 454, 456, 433], ['L1_ACCESS'], ['dfALU', 454, 458, 435], ['L1_ACCESS'], ['dfALU', 454, 460, 437], ['L1_ACCESS'], ['dfALU', 462, 453, 439], ['dfALU', 462, 456, 440], ['dfALU', 462, 458, 441], ['dfALU', 462, 460, 442], ['L1_ACCESS'], ['dfALU', 467, 453, 444], ['dfALU', 467, 456, 445], ['dfALU', 467, 458, 446], ['dfALU', 467, 460, 447], ['L1_ACCESS'], ['dfALU', 472, 453, 449], ['dfALU', 472, 456, 450], ['dfALU', 472, 458, 451], ['dfALU', 472, 460, 452], ['THREAD_SYNC'], ['L1_ACCESS', 77], ['L1_ACCESS', 79], ['L1_ACCESS', 81], ['L1_ACCESS', 83], ['L1_ACCESS', 84], ['L1_ACCESS', 87], ['L1_ACCESS', 90], ['L1_ACCESS', 92], ['THREAD_SYNC'], ['iALU'], ['iALU', 487, 65], ['iALU'], ['iALU', 489], ['iALU', 71, 490], ['iALU', 69, 489, 70], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 491], ['iALU', 5, 491], ['L2_ACCESS'], ['iALU', 5, 501], ['iALU', 73, 501], ['L2_ACCESS'], ['iALU', 73, 504], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 510, 509, 455], ['L1_ACCESS'], ['dfALU', 510, 512, 457], ['L1_ACCESS'], ['dfALU', 510, 514, 459], ['L1_ACCESS'], ['dfALU', 510, 516, 461], ['L1_ACCESS'], ['dfALU', 518, 509, 463], ['dfALU', 518, 512, 464], ['dfALU', 518, 514, 465], ['dfALU', 518, 516, 466], ['L1_ACCESS'], ['dfALU', 523, 509, 468], ['dfALU', 523, 512, 469], ['dfALU', 523, 514, 470], ['dfALU', 523, 516, 471], ['L1_ACCESS'], ['dfALU', 528, 509, 473], ['dfALU', 528, 512, 474], ['dfALU', 528, 514, 475], ['dfALU', 528, 516, 476], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 534, 533, 511], ['L1_ACCESS'], ['dfALU', 534, 536, 513], ['L1_ACCESS'], ['dfALU', 534, 538, 515], ['L1_ACCESS'], ['dfALU', 534, 540, 517], ['L1_ACCESS'], ['dfALU', 542, 533, 519], ['dfALU', 542, 536, 520], ['dfALU', 542, 538, 521], ['dfALU', 542, 540, 522], ['L1_ACCESS'], ['dfALU', 547, 533, 524], ['dfALU', 547, 536, 525], ['dfALU', 547, 538, 526], ['dfALU', 547, 540, 527], ['L1_ACCESS'], ['dfALU', 552, 533, 529], ['dfALU', 552, 536, 530], ['dfALU', 552, 538, 531], ['dfALU', 552, 540, 532], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 558, 557, 535], ['L1_ACCESS'], ['dfALU', 558, 560, 537], ['L1_ACCESS'], ['dfALU', 558, 562, 539], ['L1_ACCESS'], ['dfALU', 558, 564, 541], ['L1_ACCESS'], ['dfALU', 566, 557, 543], ['dfALU', 566, 560, 544], ['dfALU', 566, 562, 545], ['dfALU', 566, 564, 546], ['L1_ACCESS'], ['dfALU', 571, 557, 548], ['dfALU', 571, 560, 549], ['dfALU', 571, 562, 550], ['dfALU', 571, 564, 551], ['L1_ACCESS'], ['dfALU', 576, 557, 553], ['dfALU', 576, 560, 554], ['dfALU', 576, 562, 555], ['dfALU', 576, 564, 556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 582, 581, 559], ['L1_ACCESS'], ['dfALU', 582, 584, 561], ['L1_ACCESS'], ['dfALU', 582, 586, 563], ['L1_ACCESS'], ['dfALU', 582, 588, 565], ['L1_ACCESS'], ['dfALU', 590, 581, 567], ['dfALU', 590, 584, 568], ['dfALU', 590, 586, 569], ['dfALU', 590, 588, 570], ['L1_ACCESS'], ['dfALU', 595, 581, 572], ['dfALU', 595, 584, 573], ['dfALU', 595, 586, 574], ['dfALU', 595, 588, 575], ['L1_ACCESS'], ['dfALU', 600, 581, 577], ['dfALU', 600, 584, 578], ['dfALU', 600, 586, 579], ['dfALU', 600, 588, 580], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 606, 605, 583], ['L1_ACCESS'], ['dfALU', 606, 608, 585], ['L1_ACCESS'], ['dfALU', 606, 610, 587], ['L1_ACCESS'], ['dfALU', 606, 612, 589], ['L1_ACCESS'], ['dfALU', 614, 605, 591], ['dfALU', 614, 608, 592], ['dfALU', 614, 610, 593], ['dfALU', 614, 612, 594], ['L1_ACCESS'], ['dfALU', 619, 605, 596], ['dfALU', 619, 608, 597], ['dfALU', 619, 610, 598], ['dfALU', 619, 612, 599], ['L1_ACCESS'], ['dfALU', 624, 605, 601], ['dfALU', 624, 608, 602], ['dfALU', 624, 610, 603], ['dfALU', 624, 612, 604], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 630, 629, 607], ['L1_ACCESS'], ['dfALU', 630, 632, 609], ['L1_ACCESS'], ['dfALU', 630, 634, 611], ['L1_ACCESS'], ['dfALU', 630, 636, 613], ['L1_ACCESS'], ['dfALU', 638, 629, 615], ['dfALU', 638, 632, 616], ['dfALU', 638, 634, 617], ['dfALU', 638, 636, 618], ['L1_ACCESS'], ['dfALU', 643, 629, 620], ['dfALU', 643, 632, 621], ['dfALU', 643, 634, 622], ['dfALU', 643, 636, 623], ['L1_ACCESS'], ['dfALU', 648, 629, 625], ['dfALU', 648, 632, 626], ['dfALU', 648, 634, 627], ['dfALU', 648, 636, 628], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 654, 653, 631], ['L1_ACCESS'], ['dfALU', 654, 656, 633], ['L1_ACCESS'], ['dfALU', 654, 658, 635], ['L1_ACCESS'], ['dfALU', 654, 660, 637], ['L1_ACCESS'], ['dfALU', 662, 653, 639], ['dfALU', 662, 656, 640], ['dfALU', 662, 658, 641], ['dfALU', 662, 660, 642], ['L1_ACCESS'], ['dfALU', 667, 653, 644], ['dfALU', 667, 656, 645], ['dfALU', 667, 658, 646], ['dfALU', 667, 660, 647], ['L1_ACCESS'], ['dfALU', 672, 653, 649], ['dfALU', 672, 656, 650], ['dfALU', 672, 658, 651], ['dfALU', 672, 660, 652], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 678, 677, 655], ['L1_ACCESS'], ['dfALU', 678, 680, 657], ['L1_ACCESS'], ['dfALU', 678, 682, 659], ['L1_ACCESS'], ['dfALU', 678, 684, 661], ['L1_ACCESS'], ['dfALU', 686, 677, 663], ['dfALU', 686, 680, 664], ['dfALU', 686, 682, 665], ['dfALU', 686, 684, 666], ['L1_ACCESS'], ['dfALU', 691, 677, 668], ['dfALU', 691, 680, 669], ['dfALU', 691, 682, 670], ['dfALU', 691, 684, 671], ['L1_ACCESS'], ['dfALU', 696, 677, 673], ['dfALU', 696, 680, 674], ['dfALU', 696, 682, 675], ['dfALU', 696, 684, 676], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 702, 701, 679], ['L1_ACCESS'], ['dfALU', 702, 704, 681], ['L1_ACCESS'], ['dfALU', 702, 706, 683], ['L1_ACCESS'], ['dfALU', 702, 708, 685], ['L1_ACCESS'], ['dfALU', 710, 701, 687], ['dfALU', 710, 704, 688], ['dfALU', 710, 706, 689], ['dfALU', 710, 708, 690], ['L1_ACCESS'], ['dfALU', 715, 701, 692], ['dfALU', 715, 704, 693], ['dfALU', 715, 706, 694], ['dfALU', 715, 708, 695], ['L1_ACCESS'], ['dfALU', 720, 701, 697], ['dfALU', 720, 704, 698], ['dfALU', 720, 706, 699], ['dfALU', 720, 708, 700], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 726, 725, 703], ['L1_ACCESS'], ['dfALU', 726, 728, 705], ['L1_ACCESS'], ['dfALU', 726, 730, 707], ['L1_ACCESS'], ['dfALU', 726, 732, 709], ['L1_ACCESS'], ['dfALU', 734, 725, 711], ['dfALU', 734, 728, 712], ['dfALU', 734, 730, 713], ['dfALU', 734, 732, 714], ['L1_ACCESS'], ['dfALU', 739, 725, 716], ['dfALU', 739, 728, 717], ['dfALU', 739, 730, 718], ['dfALU', 739, 732, 719], ['L1_ACCESS'], ['dfALU', 744, 725, 721], ['dfALU', 744, 728, 722], ['dfALU', 744, 730, 723], ['dfALU', 744, 732, 724], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 750, 749, 727], ['L1_ACCESS'], ['dfALU', 750, 752, 729], ['L1_ACCESS'], ['dfALU', 750, 754, 731], ['L1_ACCESS'], ['dfALU', 750, 756, 733], ['L1_ACCESS'], ['dfALU', 758, 749, 735], ['dfALU', 758, 752, 736], ['dfALU', 758, 754, 737], ['dfALU', 758, 756, 738], ['L1_ACCESS'], ['dfALU', 763, 749, 740], ['dfALU', 763, 752, 741], ['dfALU', 763, 754, 742], ['dfALU', 763, 756, 743], ['L1_ACCESS'], ['dfALU', 768, 749, 745], ['dfALU', 768, 752, 746], ['dfALU', 768, 754, 747], ['dfALU', 768, 756, 748], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 774, 773, 751], ['L1_ACCESS'], ['dfALU', 774, 776, 753], ['L1_ACCESS'], ['dfALU', 774, 778, 755], ['L1_ACCESS'], ['dfALU', 774, 780, 757], ['L1_ACCESS'], ['dfALU', 782, 773, 759], ['dfALU', 782, 776, 760], ['dfALU', 782, 778, 761], ['dfALU', 782, 780, 762], ['L1_ACCESS'], ['dfALU', 787, 773, 764], ['dfALU', 787, 776, 765], ['dfALU', 787, 778, 766], ['dfALU', 787, 780, 767], ['L1_ACCESS'], ['dfALU', 792, 773, 769], ['dfALU', 792, 776, 770], ['dfALU', 792, 778, 771], ['dfALU', 792, 780, 772], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 798, 797, 775], ['L1_ACCESS'], ['dfALU', 798, 800, 777], ['L1_ACCESS'], ['dfALU', 798, 802, 779], ['L1_ACCESS'], ['dfALU', 798, 804, 781], ['L1_ACCESS'], ['dfALU', 806, 797, 783], ['dfALU', 806, 800, 784], ['dfALU', 806, 802, 785], ['dfALU', 806, 804, 786], ['L1_ACCESS'], ['dfALU', 811, 797, 788], ['dfALU', 811, 800, 789], ['dfALU', 811, 802, 790], ['dfALU', 811, 804, 791], ['L1_ACCESS'], ['dfALU', 816, 797, 793], ['dfALU', 816, 800, 794], ['dfALU', 816, 802, 795], ['dfALU', 816, 804, 796], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 822, 821, 799], ['L1_ACCESS'], ['dfALU', 822, 824, 801], ['L1_ACCESS'], ['dfALU', 822, 826, 803], ['L1_ACCESS'], ['dfALU', 822, 828, 805], ['L1_ACCESS'], ['dfALU', 830, 821, 807], ['dfALU', 830, 824, 808], ['dfALU', 830, 826, 809], ['dfALU', 830, 828, 810], ['L1_ACCESS'], ['dfALU', 835, 821, 812], ['dfALU', 835, 824, 813], ['dfALU', 835, 826, 814], ['dfALU', 835, 828, 815], ['L1_ACCESS'], ['dfALU', 840, 821, 817], ['dfALU', 840, 824, 818], ['dfALU', 840, 826, 819], ['dfALU', 840, 828, 820], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 846, 845, 823], ['L1_ACCESS'], ['dfALU', 846, 848, 825], ['L1_ACCESS'], ['dfALU', 846, 850, 827], ['L1_ACCESS'], ['dfALU', 846, 852, 829], ['L1_ACCESS'], ['dfALU', 854, 845, 831], ['dfALU', 854, 848, 832], ['dfALU', 854, 850, 833], ['dfALU', 854, 852, 834], ['L1_ACCESS'], ['dfALU', 859, 845, 836], ['dfALU', 859, 848, 837], ['dfALU', 859, 850, 838], ['dfALU', 859, 852, 839], ['L1_ACCESS'], ['dfALU', 864, 845, 841], ['dfALU', 864, 848, 842], ['dfALU', 864, 850, 843], ['dfALU', 864, 852, 844], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 870, 869, 847], ['L1_ACCESS'], ['dfALU', 870, 872, 849], ['L1_ACCESS'], ['dfALU', 870, 874, 851], ['L1_ACCESS'], ['dfALU', 870, 876, 853], ['L1_ACCESS'], ['dfALU', 878, 869, 855], ['dfALU', 878, 872, 856], ['dfALU', 878, 874, 857], ['dfALU', 878, 876, 858], ['L1_ACCESS'], ['dfALU', 883, 869, 860], ['dfALU', 883, 872, 861], ['dfALU', 883, 874, 862], ['dfALU', 883, 876, 863], ['L1_ACCESS'], ['dfALU', 888, 869, 865], ['dfALU', 888, 872, 866], ['dfALU', 888, 874, 867], ['dfALU', 888, 876, 868], ['THREAD_SYNC'], ['L1_ACCESS', 493], ['L1_ACCESS', 495], ['L1_ACCESS', 497], ['L1_ACCESS', 499], ['L1_ACCESS', 500], ['L1_ACCESS', 503], ['L1_ACCESS', 506], ['L1_ACCESS', 508], ['THREAD_SYNC'], ['iALU', 487], ['iALU', 903, 65], ['iALU', 489], ['iALU', 905], ['iALU', 71, 906], ['iALU', 69, 905, 70], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 907], ['iALU', 5, 907], ['L2_ACCESS'], ['iALU', 5, 917], ['iALU', 73, 917], ['L2_ACCESS'], ['iALU', 73, 920], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 926, 925, 871], ['L1_ACCESS'], ['dfALU', 926, 928, 873], ['L1_ACCESS'], ['dfALU', 926, 930, 875], ['L1_ACCESS'], ['dfALU', 926, 932, 877], ['L1_ACCESS'], ['dfALU', 934, 925, 879], ['dfALU', 934, 928, 880], ['dfALU', 934, 930, 881], ['dfALU', 934, 932, 882], ['L1_ACCESS'], ['dfALU', 939, 925, 884], ['dfALU', 939, 928, 885], ['dfALU', 939, 930, 886], ['dfALU', 939, 932, 887], ['L1_ACCESS'], ['dfALU', 944, 925, 889], ['dfALU', 944, 928, 890], ['dfALU', 944, 930, 891], ['dfALU', 944, 932, 892], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 950, 949, 927], ['L1_ACCESS'], ['dfALU', 950, 952, 929], ['L1_ACCESS'], ['dfALU', 950, 954, 931], ['L1_ACCESS'], ['dfALU', 950, 956, 933], ['L1_ACCESS'], ['dfALU', 958, 949, 935], ['dfALU', 958, 952, 936], ['dfALU', 958, 954, 937], ['dfALU', 958, 956, 938], ['L1_ACCESS'], ['dfALU', 963, 949, 940], ['dfALU', 963, 952, 941], ['dfALU', 963, 954, 942], ['dfALU', 963, 956, 943], ['L1_ACCESS'], ['dfALU', 968, 949, 945], ['dfALU', 968, 952, 946], ['dfALU', 968, 954, 947], ['dfALU', 968, 956, 948], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 974, 973, 951], ['L1_ACCESS'], ['dfALU', 974, 976, 953], ['L1_ACCESS'], ['dfALU', 974, 978, 955], ['L1_ACCESS'], ['dfALU', 974, 980, 957], ['L1_ACCESS'], ['dfALU', 982, 973, 959], ['dfALU', 982, 976, 960], ['dfALU', 982, 978, 961], ['dfALU', 982, 980, 962], ['L1_ACCESS'], ['dfALU', 987, 973, 964], ['dfALU', 987, 976, 965], ['dfALU', 987, 978, 966], ['dfALU', 987, 980, 967], ['L1_ACCESS'], ['dfALU', 992, 973, 969], ['dfALU', 992, 976, 970], ['dfALU', 992, 978, 971], ['dfALU', 992, 980, 972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 998, 997, 975], ['L1_ACCESS'], ['dfALU', 998, 1000, 977], ['L1_ACCESS'], ['dfALU', 998, 1002, 979], ['L1_ACCESS'], ['dfALU', 998, 1004, 981], ['L1_ACCESS'], ['dfALU', 1006, 997, 983], ['dfALU', 1006, 1000, 984], ['dfALU', 1006, 1002, 985], ['dfALU', 1006, 1004, 986], ['L1_ACCESS'], ['dfALU', 1011, 997, 988], ['dfALU', 1011, 1000, 989], ['dfALU', 1011, 1002, 990], ['dfALU', 1011, 1004, 991], ['L1_ACCESS'], ['dfALU', 1016, 997, 993], ['dfALU', 1016, 1000, 994], ['dfALU', 1016, 1002, 995], ['dfALU', 1016, 1004, 996], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1022, 1021, 999], ['L1_ACCESS'], ['dfALU', 1022, 1024, 1001], ['L1_ACCESS'], ['dfALU', 1022, 1026, 1003], ['L1_ACCESS'], ['dfALU', 1022, 1028, 1005], ['L1_ACCESS'], ['dfALU', 1030, 1021, 1007], ['dfALU', 1030, 1024, 1008], ['dfALU', 1030, 1026, 1009], ['dfALU', 1030, 1028, 1010], ['L1_ACCESS'], ['dfALU', 1035, 1021, 1012], ['dfALU', 1035, 1024, 1013], ['dfALU', 1035, 1026, 1014], ['dfALU', 1035, 1028, 1015], ['L1_ACCESS'], ['dfALU', 1040, 1021, 1017], ['dfALU', 1040, 1024, 1018], ['dfALU', 1040, 1026, 1019], ['dfALU', 1040, 1028, 1020], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1046, 1045, 1023], ['L1_ACCESS'], ['dfALU', 1046, 1048, 1025], ['L1_ACCESS'], ['dfALU', 1046, 1050, 1027], ['L1_ACCESS'], ['dfALU', 1046, 1052, 1029], ['L1_ACCESS'], ['dfALU', 1054, 1045, 1031], ['dfALU', 1054, 1048, 1032], ['dfALU', 1054, 1050, 1033], ['dfALU', 1054, 1052, 1034], ['L1_ACCESS'], ['dfALU', 1059, 1045, 1036], ['dfALU', 1059, 1048, 1037], ['dfALU', 1059, 1050, 1038], ['dfALU', 1059, 1052, 1039], ['L1_ACCESS'], ['dfALU', 1064, 1045, 1041], ['dfALU', 1064, 1048, 1042], ['dfALU', 1064, 1050, 1043], ['dfALU', 1064, 1052, 1044], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1070, 1069, 1047], ['L1_ACCESS'], ['dfALU', 1070, 1072, 1049], ['L1_ACCESS'], ['dfALU', 1070, 1074, 1051], ['L1_ACCESS'], ['dfALU', 1070, 1076, 1053], ['L1_ACCESS'], ['dfALU', 1078, 1069, 1055], ['dfALU', 1078, 1072, 1056], ['dfALU', 1078, 1074, 1057], ['dfALU', 1078, 1076, 1058], ['L1_ACCESS'], ['dfALU', 1083, 1069, 1060], ['dfALU', 1083, 1072, 1061], ['dfALU', 1083, 1074, 1062], ['dfALU', 1083, 1076, 1063], ['L1_ACCESS'], ['dfALU', 1088, 1069, 1065], ['dfALU', 1088, 1072, 1066], ['dfALU', 1088, 1074, 1067], ['dfALU', 1088, 1076, 1068], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1094, 1093, 1071], ['L1_ACCESS'], ['dfALU', 1094, 1096, 1073], ['L1_ACCESS'], ['dfALU', 1094, 1098, 1075], ['L1_ACCESS'], ['dfALU', 1094, 1100, 1077], ['L1_ACCESS'], ['dfALU', 1102, 1093, 1079], ['dfALU', 1102, 1096, 1080], ['dfALU', 1102, 1098, 1081], ['dfALU', 1102, 1100, 1082], ['L1_ACCESS'], ['dfALU', 1107, 1093, 1084], ['dfALU', 1107, 1096, 1085], ['dfALU', 1107, 1098, 1086], ['dfALU', 1107, 1100, 1087], ['L1_ACCESS'], ['dfALU', 1112, 1093, 1089], ['dfALU', 1112, 1096, 1090], ['dfALU', 1112, 1098, 1091], ['dfALU', 1112, 1100, 1092], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1118, 1117, 1095], ['L1_ACCESS'], ['dfALU', 1118, 1120, 1097], ['L1_ACCESS'], ['dfALU', 1118, 1122, 1099], ['L1_ACCESS'], ['dfALU', 1118, 1124, 1101], ['L1_ACCESS'], ['dfALU', 1126, 1117, 1103], ['dfALU', 1126, 1120, 1104], ['dfALU', 1126, 1122, 1105], ['dfALU', 1126, 1124, 1106], ['L1_ACCESS'], ['dfALU', 1131, 1117, 1108], ['dfALU', 1131, 1120, 1109], ['dfALU', 1131, 1122, 1110], ['dfALU', 1131, 1124, 1111], ['L1_ACCESS'], ['dfALU', 1136, 1117, 1113], ['dfALU', 1136, 1120, 1114], ['dfALU', 1136, 1122, 1115], ['dfALU', 1136, 1124, 1116], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1142, 1141, 1119], ['L1_ACCESS'], ['dfALU', 1142, 1144, 1121], ['L1_ACCESS'], ['dfALU', 1142, 1146, 1123], ['L1_ACCESS'], ['dfALU', 1142, 1148, 1125], ['L1_ACCESS'], ['dfALU', 1150, 1141, 1127], ['dfALU', 1150, 1144, 1128], ['dfALU', 1150, 1146, 1129], ['dfALU', 1150, 1148, 1130], ['L1_ACCESS'], ['dfALU', 1155, 1141, 1132], ['dfALU', 1155, 1144, 1133], ['dfALU', 1155, 1146, 1134], ['dfALU', 1155, 1148, 1135], ['L1_ACCESS'], ['dfALU', 1160, 1141, 1137], ['dfALU', 1160, 1144, 1138], ['dfALU', 1160, 1146, 1139], ['dfALU', 1160, 1148, 1140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1166, 1165, 1143], ['L1_ACCESS'], ['dfALU', 1166, 1168, 1145], ['L1_ACCESS'], ['dfALU', 1166, 1170, 1147], ['L1_ACCESS'], ['dfALU', 1166, 1172, 1149], ['L1_ACCESS'], ['dfALU', 1174, 1165, 1151], ['dfALU', 1174, 1168, 1152], ['dfALU', 1174, 1170, 1153], ['dfALU', 1174, 1172, 1154], ['L1_ACCESS'], ['dfALU', 1179, 1165, 1156], ['dfALU', 1179, 1168, 1157], ['dfALU', 1179, 1170, 1158], ['dfALU', 1179, 1172, 1159], ['L1_ACCESS'], ['dfALU', 1184, 1165, 1161], ['dfALU', 1184, 1168, 1162], ['dfALU', 1184, 1170, 1163], ['dfALU', 1184, 1172, 1164], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1190, 1189, 1167], ['L1_ACCESS'], ['dfALU', 1190, 1192, 1169], ['L1_ACCESS'], ['dfALU', 1190, 1194, 1171], ['L1_ACCESS'], ['dfALU', 1190, 1196, 1173], ['L1_ACCESS'], ['dfALU', 1198, 1189, 1175], ['dfALU', 1198, 1192, 1176], ['dfALU', 1198, 1194, 1177], ['dfALU', 1198, 1196, 1178], ['L1_ACCESS'], ['dfALU', 1203, 1189, 1180], ['dfALU', 1203, 1192, 1181], ['dfALU', 1203, 1194, 1182], ['dfALU', 1203, 1196, 1183], ['L1_ACCESS'], ['dfALU', 1208, 1189, 1185], ['dfALU', 1208, 1192, 1186], ['dfALU', 1208, 1194, 1187], ['dfALU', 1208, 1196, 1188], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1214, 1213, 1191], ['L1_ACCESS'], ['dfALU', 1214, 1216, 1193], ['L1_ACCESS'], ['dfALU', 1214, 1218, 1195], ['L1_ACCESS'], ['dfALU', 1214, 1220, 1197], ['L1_ACCESS'], ['dfALU', 1222, 1213, 1199], ['dfALU', 1222, 1216, 1200], ['dfALU', 1222, 1218, 1201], ['dfALU', 1222, 1220, 1202], ['L1_ACCESS'], ['dfALU', 1227, 1213, 1204], ['dfALU', 1227, 1216, 1205], ['dfALU', 1227, 1218, 1206], ['dfALU', 1227, 1220, 1207], ['L1_ACCESS'], ['dfALU', 1232, 1213, 1209], ['dfALU', 1232, 1216, 1210], ['dfALU', 1232, 1218, 1211], ['dfALU', 1232, 1220, 1212], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1238, 1237, 1215], ['L1_ACCESS'], ['dfALU', 1238, 1240, 1217], ['L1_ACCESS'], ['dfALU', 1238, 1242, 1219], ['L1_ACCESS'], ['dfALU', 1238, 1244, 1221], ['L1_ACCESS'], ['dfALU', 1246, 1237, 1223], ['dfALU', 1246, 1240, 1224], ['dfALU', 1246, 1242, 1225], ['dfALU', 1246, 1244, 1226], ['L1_ACCESS'], ['dfALU', 1251, 1237, 1228], ['dfALU', 1251, 1240, 1229], ['dfALU', 1251, 1242, 1230], ['dfALU', 1251, 1244, 1231], ['L1_ACCESS'], ['dfALU', 1256, 1237, 1233], ['dfALU', 1256, 1240, 1234], ['dfALU', 1256, 1242, 1235], ['dfALU', 1256, 1244, 1236], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1262, 1261, 1239], ['L1_ACCESS'], ['dfALU', 1262, 1264, 1241], ['L1_ACCESS'], ['dfALU', 1262, 1266, 1243], ['L1_ACCESS'], ['dfALU', 1262, 1268, 1245], ['L1_ACCESS'], ['dfALU', 1270, 1261, 1247], ['dfALU', 1270, 1264, 1248], ['dfALU', 1270, 1266, 1249], ['dfALU', 1270, 1268, 1250], ['L1_ACCESS'], ['dfALU', 1275, 1261, 1252], ['dfALU', 1275, 1264, 1253], ['dfALU', 1275, 1266, 1254], ['dfALU', 1275, 1268, 1255], ['L1_ACCESS'], ['dfALU', 1280, 1261, 1257], ['dfALU', 1280, 1264, 1258], ['dfALU', 1280, 1266, 1259], ['dfALU', 1280, 1268, 1260], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1286, 1285, 1263], ['L1_ACCESS'], ['dfALU', 1286, 1288, 1265], ['L1_ACCESS'], ['dfALU', 1286, 1290, 1267], ['L1_ACCESS'], ['dfALU', 1286, 1292, 1269], ['L1_ACCESS'], ['dfALU', 1294, 1285, 1271], ['dfALU', 1294, 1288, 1272], ['dfALU', 1294, 1290, 1273], ['dfALU', 1294, 1292, 1274], ['L1_ACCESS'], ['dfALU', 1299, 1285, 1276], ['dfALU', 1299, 1288, 1277], ['dfALU', 1299, 1290, 1278], ['dfALU', 1299, 1292, 1279], ['L1_ACCESS'], ['dfALU', 1304, 1285, 1281], ['dfALU', 1304, 1288, 1282], ['dfALU', 1304, 1290, 1283], ['dfALU', 1304, 1292, 1284], ['THREAD_SYNC'], ['L1_ACCESS', 909], ['L1_ACCESS', 911], ['L1_ACCESS', 913], ['L1_ACCESS', 915], ['L1_ACCESS', 916], ['L1_ACCESS', 919], ['L1_ACCESS', 922], ['L1_ACCESS', 924], ['THREAD_SYNC'], ['iALU', 903], ['iALU', 1319, 65], ['iALU', 905], ['iALU', 1321], ['iALU', 71, 1322], ['iALU', 69, 1321, 70], ['L2_ACCESS'], ['iALU', 1324], ['L2_ACCESS'], ['iALU', 1324], ['L2_ACCESS'], ['iALU', 1324], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 1323], ['iALU', 5, 1323], ['L2_ACCESS'], ['iALU', 5, 1333], ['iALU', 73, 1333], ['L2_ACCESS'], ['iALU', 73, 1336], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 1342, 1341, 1287], ['L1_ACCESS'], ['dfALU', 1342, 1344, 1289], ['L1_ACCESS'], ['dfALU', 1342, 1346, 1291], ['L1_ACCESS'], ['dfALU', 1342, 1348, 1293], ['L1_ACCESS'], ['dfALU', 1350, 1341, 1295], ['dfALU', 1350, 1344, 1296], ['dfALU', 1350, 1346, 1297], ['dfALU', 1350, 1348, 1298], ['L1_ACCESS'], ['dfALU', 1355, 1341, 1300], ['dfALU', 1355, 1344, 1301], ['dfALU', 1355, 1346, 1302], ['dfALU', 1355, 1348, 1303], ['L1_ACCESS'], ['dfALU', 1360, 1341, 1305], ['dfALU', 1360, 1344, 1306], ['dfALU', 1360, 1346, 1307], ['dfALU', 1360, 1348, 1308], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1366, 1365, 1343], ['L1_ACCESS'], ['dfALU', 1366, 1368, 1345], ['L1_ACCESS'], ['dfALU', 1366, 1370, 1347], ['L1_ACCESS'], ['dfALU', 1366, 1372, 1349], ['L1_ACCESS'], ['dfALU', 1374, 1365, 1351], ['dfALU', 1374, 1368, 1352], ['dfALU', 1374, 1370, 1353], ['dfALU', 1374, 1372, 1354], ['L1_ACCESS'], ['dfALU', 1379, 1365, 1356], ['dfALU', 1379, 1368, 1357], ['dfALU', 1379, 1370, 1358], ['dfALU', 1379, 1372, 1359], ['L1_ACCESS'], ['dfALU', 1384, 1365, 1361], ['dfALU', 1384, 1368, 1362], ['dfALU', 1384, 1370, 1363], ['dfALU', 1384, 1372, 1364], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1390, 1389, 1367], ['L1_ACCESS'], ['dfALU', 1390, 1392, 1369], ['L1_ACCESS'], ['dfALU', 1390, 1394, 1371], ['L1_ACCESS'], ['dfALU', 1390, 1396, 1373], ['L1_ACCESS'], ['dfALU', 1398, 1389, 1375], ['dfALU', 1398, 1392, 1376], ['dfALU', 1398, 1394, 1377], ['dfALU', 1398, 1396, 1378], ['L1_ACCESS'], ['dfALU', 1403, 1389, 1380], ['dfALU', 1403, 1392, 1381], ['dfALU', 1403, 1394, 1382], ['dfALU', 1403, 1396, 1383], ['L1_ACCESS'], ['dfALU', 1408, 1389, 1385], ['dfALU', 1408, 1392, 1386], ['dfALU', 1408, 1394, 1387], ['dfALU', 1408, 1396, 1388], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1414, 1413, 1391], ['L1_ACCESS'], ['dfALU', 1414, 1416, 1393], ['L1_ACCESS'], ['dfALU', 1414, 1418, 1395], ['L1_ACCESS'], ['dfALU', 1414, 1420, 1397], ['L1_ACCESS'], ['dfALU', 1422, 1413, 1399], ['dfALU', 1422, 1416, 1400], ['dfALU', 1422, 1418, 1401], ['dfALU', 1422, 1420, 1402], ['L1_ACCESS'], ['dfALU', 1427, 1413, 1404], ['dfALU', 1427, 1416, 1405], ['dfALU', 1427, 1418, 1406], ['dfALU', 1427, 1420, 1407], ['L1_ACCESS'], ['dfALU', 1432, 1413, 1409], ['dfALU', 1432, 1416, 1410], ['dfALU', 1432, 1418, 1411], ['dfALU', 1432, 1420, 1412], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1438, 1437, 1415], ['L1_ACCESS'], ['dfALU', 1438, 1440, 1417], ['L1_ACCESS'], ['dfALU', 1438, 1442, 1419], ['L1_ACCESS'], ['dfALU', 1438, 1444, 1421], ['L1_ACCESS'], ['dfALU', 1446, 1437, 1423], ['dfALU', 1446, 1440, 1424], ['dfALU', 1446, 1442, 1425], ['dfALU', 1446, 1444, 1426], ['L1_ACCESS'], ['dfALU', 1451, 1437, 1428], ['dfALU', 1451, 1440, 1429], ['dfALU', 1451, 1442, 1430], ['dfALU', 1451, 1444, 1431], ['L1_ACCESS'], ['dfALU', 1456, 1437, 1433], ['dfALU', 1456, 1440, 1434], ['dfALU', 1456, 1442, 1435], ['dfALU', 1456, 1444, 1436], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1462, 1461, 1439], ['L1_ACCESS'], ['dfALU', 1462, 1464, 1441], ['L1_ACCESS'], ['dfALU', 1462, 1466, 1443], ['L1_ACCESS'], ['dfALU', 1462, 1468, 1445], ['L1_ACCESS'], ['dfALU', 1470, 1461, 1447], ['dfALU', 1470, 1464, 1448], ['dfALU', 1470, 1466, 1449], ['dfALU', 1470, 1468, 1450], ['L1_ACCESS'], ['dfALU', 1475, 1461, 1452], ['dfALU', 1475, 1464, 1453], ['dfALU', 1475, 1466, 1454], ['dfALU', 1475, 1468, 1455], ['L1_ACCESS'], ['dfALU', 1480, 1461, 1457], ['dfALU', 1480, 1464, 1458], ['dfALU', 1480, 1466, 1459], ['dfALU', 1480, 1468, 1460], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1486, 1485, 1463], ['L1_ACCESS'], ['dfALU', 1486, 1488, 1465], ['L1_ACCESS'], ['dfALU', 1486, 1490, 1467], ['L1_ACCESS'], ['dfALU', 1486, 1492, 1469], ['L1_ACCESS'], ['dfALU', 1494, 1485, 1471], ['dfALU', 1494, 1488, 1472], ['dfALU', 1494, 1490, 1473], ['dfALU', 1494, 1492, 1474], ['L1_ACCESS'], ['dfALU', 1499, 1485, 1476], ['dfALU', 1499, 1488, 1477], ['dfALU', 1499, 1490, 1478], ['dfALU', 1499, 1492, 1479], ['L1_ACCESS'], ['dfALU', 1504, 1485, 1481], ['dfALU', 1504, 1488, 1482], ['dfALU', 1504, 1490, 1483], ['dfALU', 1504, 1492, 1484], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1510, 1509, 1487], ['L1_ACCESS'], ['dfALU', 1510, 1512, 1489], ['L1_ACCESS'], ['dfALU', 1510, 1514, 1491], ['L1_ACCESS'], ['dfALU', 1510, 1516, 1493], ['L1_ACCESS'], ['dfALU', 1518, 1509, 1495], ['dfALU', 1518, 1512, 1496], ['dfALU', 1518, 1514, 1497], ['dfALU', 1518, 1516, 1498], ['L1_ACCESS'], ['dfALU', 1523, 1509, 1500], ['dfALU', 1523, 1512, 1501], ['dfALU', 1523, 1514, 1502], ['dfALU', 1523, 1516, 1503], ['L1_ACCESS'], ['dfALU', 1528, 1509, 1505], ['dfALU', 1528, 1512, 1506], ['dfALU', 1528, 1514, 1507], ['dfALU', 1528, 1516, 1508], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1534, 1533, 1511], ['L1_ACCESS'], ['dfALU', 1534, 1536, 1513], ['L1_ACCESS'], ['dfALU', 1534, 1538, 1515], ['L1_ACCESS'], ['dfALU', 1534, 1540, 1517], ['L1_ACCESS'], ['dfALU', 1542, 1533, 1519], ['dfALU', 1542, 1536, 1520], ['dfALU', 1542, 1538, 1521], ['dfALU', 1542, 1540, 1522], ['L1_ACCESS'], ['dfALU', 1547, 1533, 1524], ['dfALU', 1547, 1536, 1525], ['dfALU', 1547, 1538, 1526], ['dfALU', 1547, 1540, 1527], ['L1_ACCESS'], ['dfALU', 1552, 1533, 1529], ['dfALU', 1552, 1536, 1530], ['dfALU', 1552, 1538, 1531], ['dfALU', 1552, 1540, 1532], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1558, 1557, 1535], ['L1_ACCESS'], ['dfALU', 1558, 1560, 1537], ['L1_ACCESS'], ['dfALU', 1558, 1562, 1539], ['L1_ACCESS'], ['dfALU', 1558, 1564, 1541], ['L1_ACCESS'], ['dfALU', 1566, 1557, 1543], ['dfALU', 1566, 1560, 1544], ['dfALU', 1566, 1562, 1545], ['dfALU', 1566, 1564, 1546], ['L1_ACCESS'], ['dfALU', 1571, 1557, 1548], ['dfALU', 1571, 1560, 1549], ['dfALU', 1571, 1562, 1550], ['dfALU', 1571, 1564, 1551], ['L1_ACCESS'], ['dfALU', 1576, 1557, 1553], ['dfALU', 1576, 1560, 1554], ['dfALU', 1576, 1562, 1555], ['dfALU', 1576, 1564, 1556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1582, 1581, 1559], ['L1_ACCESS'], ['dfALU', 1582, 1584, 1561], ['L1_ACCESS'], ['dfALU', 1582, 1586, 1563], ['L1_ACCESS'], ['dfALU', 1582, 1588, 1565], ['L1_ACCESS'], ['dfALU', 1590, 1581, 1567], ['dfALU', 1590, 1584, 1568], ['dfALU', 1590, 1586, 1569], ['dfALU', 1590, 1588, 1570], ['L1_ACCESS'], ['dfALU', 1595, 1581, 1572], ['dfALU', 1595, 1584, 1573], ['dfALU', 1595, 1586, 1574], ['dfALU', 1595, 1588, 1575], ['L1_ACCESS'], ['dfALU', 1600, 1581, 1577], ['dfALU', 1600, 1584, 1578], ['dfALU', 1600, 1586, 1579], ['dfALU', 1600, 1588, 1580], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1606, 1605, 1583], ['L1_ACCESS'], ['dfALU', 1606, 1608, 1585], ['L1_ACCESS'], ['dfALU', 1606, 1610, 1587], ['L1_ACCESS'], ['dfALU', 1606, 1612, 1589], ['L1_ACCESS'], ['dfALU', 1614, 1605, 1591], ['dfALU', 1614, 1608, 1592], ['dfALU', 1614, 1610, 1593], ['dfALU', 1614, 1612, 1594], ['L1_ACCESS'], ['dfALU', 1619, 1605, 1596], ['dfALU', 1619, 1608, 1597], ['dfALU', 1619, 1610, 1598], ['dfALU', 1619, 1612, 1599], ['L1_ACCESS'], ['dfALU', 1624, 1605, 1601], ['dfALU', 1624, 1608, 1602], ['dfALU', 1624, 1610, 1603], ['dfALU', 1624, 1612, 1604], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1630, 1629, 1607], ['L1_ACCESS'], ['dfALU', 1630, 1632, 1609], ['L1_ACCESS'], ['dfALU', 1630, 1634, 1611], ['L1_ACCESS'], ['dfALU', 1630, 1636, 1613], ['L1_ACCESS'], ['dfALU', 1638, 1629, 1615], ['dfALU', 1638, 1632, 1616], ['dfALU', 1638, 1634, 1617], ['dfALU', 1638, 1636, 1618], ['L1_ACCESS'], ['dfALU', 1643, 1629, 1620], ['dfALU', 1643, 1632, 1621], ['dfALU', 1643, 1634, 1622], ['dfALU', 1643, 1636, 1623], ['L1_ACCESS'], ['dfALU', 1648, 1629, 1625], ['dfALU', 1648, 1632, 1626], ['dfALU', 1648, 1634, 1627], ['dfALU', 1648, 1636, 1628], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1654, 1653, 1631], ['L1_ACCESS'], ['dfALU', 1654, 1656, 1633], ['L1_ACCESS'], ['dfALU', 1654, 1658, 1635], ['L1_ACCESS'], ['dfALU', 1654, 1660, 1637], ['L1_ACCESS'], ['dfALU', 1662, 1653, 1639], ['dfALU', 1662, 1656, 1640], ['dfALU', 1662, 1658, 1641], ['dfALU', 1662, 1660, 1642], ['L1_ACCESS'], ['dfALU', 1667, 1653, 1644], ['dfALU', 1667, 1656, 1645], ['dfALU', 1667, 1658, 1646], ['dfALU', 1667, 1660, 1647], ['L1_ACCESS'], ['dfALU', 1672, 1653, 1649], ['dfALU', 1672, 1656, 1650], ['dfALU', 1672, 1658, 1651], ['dfALU', 1672, 1660, 1652], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1678, 1677, 1655], ['L1_ACCESS'], ['dfALU', 1678, 1680, 1657], ['L1_ACCESS'], ['dfALU', 1678, 1682, 1659], ['L1_ACCESS'], ['dfALU', 1678, 1684, 1661], ['L1_ACCESS'], ['dfALU', 1686, 1677, 1663], ['dfALU', 1686, 1680, 1664], ['dfALU', 1686, 1682, 1665], ['dfALU', 1686, 1684, 1666], ['L1_ACCESS'], ['dfALU', 1691, 1677, 1668], ['dfALU', 1691, 1680, 1669], ['dfALU', 1691, 1682, 1670], ['dfALU', 1691, 1684, 1671], ['L1_ACCESS'], ['dfALU', 1696, 1677, 1673], ['dfALU', 1696, 1680, 1674], ['dfALU', 1696, 1682, 1675], ['dfALU', 1696, 1684, 1676], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1702, 1701, 1679], ['L1_ACCESS'], ['dfALU', 1702, 1704, 1681], ['L1_ACCESS'], ['dfALU', 1702, 1706, 1683], ['L1_ACCESS'], ['dfALU', 1702, 1708, 1685], ['L1_ACCESS'], ['dfALU', 1710, 1701, 1687], ['dfALU', 1710, 1704, 1688], ['dfALU', 1710, 1706, 1689], ['dfALU', 1710, 1708, 1690], ['L1_ACCESS'], ['dfALU', 1715, 1701, 1692], ['dfALU', 1715, 1704, 1693], ['dfALU', 1715, 1706, 1694], ['dfALU', 1715, 1708, 1695], ['L1_ACCESS'], ['dfALU', 1720, 1701, 1697], ['dfALU', 1720, 1704, 1698], ['dfALU', 1720, 1706, 1699], ['dfALU', 1720, 1708, 1700], ['THREAD_SYNC'], ['L1_ACCESS', 1325], ['L1_ACCESS', 1327], ['L1_ACCESS', 1329], ['L1_ACCESS', 1331], ['L1_ACCESS', 1332], ['L1_ACCESS', 1335], ['L1_ACCESS', 1338], ['L1_ACCESS', 1340], ['THREAD_SYNC'], ['iALU', 1319], ['iALU', 1735, 65], ['iALU', 1321], ['iALU', 1737], ['iALU', 71, 1738], ['iALU', 69, 1737, 70], ['L2_ACCESS'], ['iALU', 1740], ['L2_ACCESS'], ['iALU', 1740], ['L2_ACCESS'], ['iALU', 1740], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 1739], ['iALU', 5, 1739], ['L2_ACCESS'], ['iALU', 5, 1749], ['iALU', 73, 1749], ['L2_ACCESS'], ['iALU', 73, 1752], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 1758, 1757, 1703], ['L1_ACCESS'], ['dfALU', 1758, 1760, 1705], ['L1_ACCESS'], ['dfALU', 1758, 1762, 1707], ['L1_ACCESS'], ['dfALU', 1758, 1764, 1709], ['L1_ACCESS'], ['dfALU', 1766, 1757, 1711], ['dfALU', 1766, 1760, 1712], ['dfALU', 1766, 1762, 1713], ['dfALU', 1766, 1764, 1714], ['L1_ACCESS'], ['dfALU', 1771, 1757, 1716], ['dfALU', 1771, 1760, 1717], ['dfALU', 1771, 1762, 1718], ['dfALU', 1771, 1764, 1719], ['L1_ACCESS'], ['dfALU', 1776, 1757, 1721], ['dfALU', 1776, 1760, 1722], ['dfALU', 1776, 1762, 1723], ['dfALU', 1776, 1764, 1724], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1782, 1781, 1759], ['L1_ACCESS'], ['dfALU', 1782, 1784, 1761], ['L1_ACCESS'], ['dfALU', 1782, 1786, 1763], ['L1_ACCESS'], ['dfALU', 1782, 1788, 1765], ['L1_ACCESS'], ['dfALU', 1790, 1781, 1767], ['dfALU', 1790, 1784, 1768], ['dfALU', 1790, 1786, 1769], ['dfALU', 1790, 1788, 1770], ['L1_ACCESS'], ['dfALU', 1795, 1781, 1772], ['dfALU', 1795, 1784, 1773], ['dfALU', 1795, 1786, 1774], ['dfALU', 1795, 1788, 1775], ['L1_ACCESS'], ['dfALU', 1800, 1781, 1777], ['dfALU', 1800, 1784, 1778], ['dfALU', 1800, 1786, 1779], ['dfALU', 1800, 1788, 1780], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1806, 1805, 1783], ['L1_ACCESS'], ['dfALU', 1806, 1808, 1785], ['L1_ACCESS'], ['dfALU', 1806, 1810, 1787], ['L1_ACCESS'], ['dfALU', 1806, 1812, 1789], ['L1_ACCESS'], ['dfALU', 1814, 1805, 1791], ['dfALU', 1814, 1808, 1792], ['dfALU', 1814, 1810, 1793], ['dfALU', 1814, 1812, 1794], ['L1_ACCESS'], ['dfALU', 1819, 1805, 1796], ['dfALU', 1819, 1808, 1797], ['dfALU', 1819, 1810, 1798], ['dfALU', 1819, 1812, 1799], ['L1_ACCESS'], ['dfALU', 1824, 1805, 1801], ['dfALU', 1824, 1808, 1802], ['dfALU', 1824, 1810, 1803], ['dfALU', 1824, 1812, 1804], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1830, 1829, 1807], ['L1_ACCESS'], ['dfALU', 1830, 1832, 1809], ['L1_ACCESS'], ['dfALU', 1830, 1834, 1811], ['L1_ACCESS'], ['dfALU', 1830, 1836, 1813], ['L1_ACCESS'], ['dfALU', 1838, 1829, 1815], ['dfALU', 1838, 1832, 1816], ['dfALU', 1838, 1834, 1817], ['dfALU', 1838, 1836, 1818], ['L1_ACCESS'], ['dfALU', 1843, 1829, 1820], ['dfALU', 1843, 1832, 1821], ['dfALU', 1843, 1834, 1822], ['dfALU', 1843, 1836, 1823], ['L1_ACCESS'], ['dfALU', 1848, 1829, 1825], ['dfALU', 1848, 1832, 1826], ['dfALU', 1848, 1834, 1827], ['dfALU', 1848, 1836, 1828], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1854, 1853, 1831], ['L1_ACCESS'], ['dfALU', 1854, 1856, 1833], ['L1_ACCESS'], ['dfALU', 1854, 1858, 1835], ['L1_ACCESS'], ['dfALU', 1854, 1860, 1837], ['L1_ACCESS'], ['dfALU', 1862, 1853, 1839], ['dfALU', 1862, 1856, 1840], ['dfALU', 1862, 1858, 1841], ['dfALU', 1862, 1860, 1842], ['L1_ACCESS'], ['dfALU', 1867, 1853, 1844], ['dfALU', 1867, 1856, 1845], ['dfALU', 1867, 1858, 1846], ['dfALU', 1867, 1860, 1847], ['L1_ACCESS'], ['dfALU', 1872, 1853, 1849], ['dfALU', 1872, 1856, 1850], ['dfALU', 1872, 1858, 1851], ['dfALU', 1872, 1860, 1852], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1878, 1877, 1855], ['L1_ACCESS'], ['dfALU', 1878, 1880, 1857], ['L1_ACCESS'], ['dfALU', 1878, 1882, 1859], ['L1_ACCESS'], ['dfALU', 1878, 1884, 1861], ['L1_ACCESS'], ['dfALU', 1886, 1877, 1863], ['dfALU', 1886, 1880, 1864], ['dfALU', 1886, 1882, 1865], ['dfALU', 1886, 1884, 1866], ['L1_ACCESS'], ['dfALU', 1891, 1877, 1868], ['dfALU', 1891, 1880, 1869], ['dfALU', 1891, 1882, 1870], ['dfALU', 1891, 1884, 1871], ['L1_ACCESS'], ['dfALU', 1896, 1877, 1873], ['dfALU', 1896, 1880, 1874], ['dfALU', 1896, 1882, 1875], ['dfALU', 1896, 1884, 1876], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1902, 1901, 1879], ['L1_ACCESS'], ['dfALU', 1902, 1904, 1881], ['L1_ACCESS'], ['dfALU', 1902, 1906, 1883], ['L1_ACCESS'], ['dfALU', 1902, 1908, 1885], ['L1_ACCESS'], ['dfALU', 1910, 1901, 1887], ['dfALU', 1910, 1904, 1888], ['dfALU', 1910, 1906, 1889], ['dfALU', 1910, 1908, 1890], ['L1_ACCESS'], ['dfALU', 1915, 1901, 1892], ['dfALU', 1915, 1904, 1893], ['dfALU', 1915, 1906, 1894], ['dfALU', 1915, 1908, 1895], ['L1_ACCESS'], ['dfALU', 1920, 1901, 1897], ['dfALU', 1920, 1904, 1898], ['dfALU', 1920, 1906, 1899], ['dfALU', 1920, 1908, 1900], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1926, 1925, 1903], ['L1_ACCESS'], ['dfALU', 1926, 1928, 1905], ['L1_ACCESS'], ['dfALU', 1926, 1930, 1907], ['L1_ACCESS'], ['dfALU', 1926, 1932, 1909], ['L1_ACCESS'], ['dfALU', 1934, 1925, 1911], ['dfALU', 1934, 1928, 1912], ['dfALU', 1934, 1930, 1913], ['dfALU', 1934, 1932, 1914], ['L1_ACCESS'], ['dfALU', 1939, 1925, 1916], ['dfALU', 1939, 1928, 1917], ['dfALU', 1939, 1930, 1918], ['dfALU', 1939, 1932, 1919], ['L1_ACCESS'], ['dfALU', 1944, 1925, 1921], ['dfALU', 1944, 1928, 1922], ['dfALU', 1944, 1930, 1923], ['dfALU', 1944, 1932, 1924], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1950, 1949, 1927], ['L1_ACCESS'], ['dfALU', 1950, 1952, 1929], ['L1_ACCESS'], ['dfALU', 1950, 1954, 1931], ['L1_ACCESS'], ['dfALU', 1950, 1956, 1933], ['L1_ACCESS'], ['dfALU', 1958, 1949, 1935], ['dfALU', 1958, 1952, 1936], ['dfALU', 1958, 1954, 1937], ['dfALU', 1958, 1956, 1938], ['L1_ACCESS'], ['dfALU', 1963, 1949, 1940], ['dfALU', 1963, 1952, 1941], ['dfALU', 1963, 1954, 1942], ['dfALU', 1963, 1956, 1943], ['L1_ACCESS'], ['dfALU', 1968, 1949, 1945], ['dfALU', 1968, 1952, 1946], ['dfALU', 1968, 1954, 1947], ['dfALU', 1968, 1956, 1948], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1974, 1973, 1951], ['L1_ACCESS'], ['dfALU', 1974, 1976, 1953], ['L1_ACCESS'], ['dfALU', 1974, 1978, 1955], ['L1_ACCESS'], ['dfALU', 1974, 1980, 1957], ['L1_ACCESS'], ['dfALU', 1982, 1973, 1959], ['dfALU', 1982, 1976, 1960], ['dfALU', 1982, 1978, 1961], ['dfALU', 1982, 1980, 1962], ['L1_ACCESS'], ['dfALU', 1987, 1973, 1964], ['dfALU', 1987, 1976, 1965], ['dfALU', 1987, 1978, 1966], ['dfALU', 1987, 1980, 1967], ['L1_ACCESS'], ['dfALU', 1992, 1973, 1969], ['dfALU', 1992, 1976, 1970], ['dfALU', 1992, 1978, 1971], ['dfALU', 1992, 1980, 1972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1998, 1997, 1975], ['L1_ACCESS'], ['dfALU', 1998, 2000, 1977], ['L1_ACCESS'], ['dfALU', 1998, 2002, 1979], ['L1_ACCESS'], ['dfALU', 1998, 2004, 1981], ['L1_ACCESS'], ['dfALU', 2006, 1997, 1983], ['dfALU', 2006, 2000, 1984], ['dfALU', 2006, 2002, 1985], ['dfALU', 2006, 2004, 1986], ['L1_ACCESS'], ['dfALU', 2011, 1997, 1988], ['dfALU', 2011, 2000, 1989], ['dfALU', 2011, 2002, 1990], ['dfALU', 2011, 2004, 1991], ['L1_ACCESS'], ['dfALU', 2016, 1997, 1993], ['dfALU', 2016, 2000, 1994], ['dfALU', 2016, 2002, 1995], ['dfALU', 2016, 2004, 1996], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2022, 2021, 1999], ['L1_ACCESS'], ['dfALU', 2022, 2024, 2001], ['L1_ACCESS'], ['dfALU', 2022, 2026, 2003], ['L1_ACCESS'], ['dfALU', 2022, 2028, 2005], ['L1_ACCESS'], ['dfALU', 2030, 2021, 2007], ['dfALU', 2030, 2024, 2008], ['dfALU', 2030, 2026, 2009], ['dfALU', 2030, 2028, 2010], ['L1_ACCESS'], ['dfALU', 2035, 2021, 2012], ['dfALU', 2035, 2024, 2013], ['dfALU', 2035, 2026, 2014], ['dfALU', 2035, 2028, 2015], ['L1_ACCESS'], ['dfALU', 2040, 2021, 2017], ['dfALU', 2040, 2024, 2018], ['dfALU', 2040, 2026, 2019], ['dfALU', 2040, 2028, 2020], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2046, 2045, 2023], ['L1_ACCESS'], ['dfALU', 2046, 2048, 2025], ['L1_ACCESS'], ['dfALU', 2046, 2050, 2027], ['L1_ACCESS'], ['dfALU', 2046, 2052, 2029], ['L1_ACCESS'], ['dfALU', 2054, 2045, 2031], ['dfALU', 2054, 2048, 2032], ['dfALU', 2054, 2050, 2033], ['dfALU', 2054, 2052, 2034], ['L1_ACCESS'], ['dfALU', 2059, 2045, 2036], ['dfALU', 2059, 2048, 2037], ['dfALU', 2059, 2050, 2038], ['dfALU', 2059, 2052, 2039], ['L1_ACCESS'], ['dfALU', 2064, 2045, 2041], ['dfALU', 2064, 2048, 2042], ['dfALU', 2064, 2050, 2043], ['dfALU', 2064, 2052, 2044], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2070, 2069, 2047], ['L1_ACCESS'], ['dfALU', 2070, 2072, 2049], ['L1_ACCESS'], ['dfALU', 2070, 2074, 2051], ['L1_ACCESS'], ['dfALU', 2070, 2076, 2053], ['L1_ACCESS'], ['dfALU', 2078, 2069, 2055], ['dfALU', 2078, 2072, 2056], ['dfALU', 2078, 2074, 2057], ['dfALU', 2078, 2076, 2058], ['L1_ACCESS'], ['dfALU', 2083, 2069, 2060], ['dfALU', 2083, 2072, 2061], ['dfALU', 2083, 2074, 2062], ['dfALU', 2083, 2076, 2063], ['L1_ACCESS'], ['dfALU', 2088, 2069, 2065], ['dfALU', 2088, 2072, 2066], ['dfALU', 2088, 2074, 2067], ['dfALU', 2088, 2076, 2068], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2094, 2093, 2071], ['L1_ACCESS'], ['dfALU', 2094, 2096, 2073], ['L1_ACCESS'], ['dfALU', 2094, 2098, 2075], ['L1_ACCESS'], ['dfALU', 2094, 2100, 2077], ['L1_ACCESS'], ['dfALU', 2102, 2093, 2079], ['dfALU', 2102, 2096, 2080], ['dfALU', 2102, 2098, 2081], ['dfALU', 2102, 2100, 2082], ['L1_ACCESS'], ['dfALU', 2107, 2093, 2084], ['dfALU', 2107, 2096, 2085], ['dfALU', 2107, 2098, 2086], ['dfALU', 2107, 2100, 2087], ['L1_ACCESS'], ['dfALU', 2112, 2093, 2089], ['dfALU', 2112, 2096, 2090], ['dfALU', 2112, 2098, 2091], ['dfALU', 2112, 2100, 2092], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2118, 2117, 2095], ['L1_ACCESS'], ['dfALU', 2118, 2120, 2097], ['L1_ACCESS'], ['dfALU', 2118, 2122, 2099], ['L1_ACCESS'], ['dfALU', 2118, 2124, 2101], ['L1_ACCESS'], ['dfALU', 2126, 2117, 2103], ['dfALU', 2126, 2120, 2104], ['dfALU', 2126, 2122, 2105], ['dfALU', 2126, 2124, 2106], ['L1_ACCESS'], ['dfALU', 2131, 2117, 2108], ['dfALU', 2131, 2120, 2109], ['dfALU', 2131, 2122, 2110], ['dfALU', 2131, 2124, 2111], ['L1_ACCESS'], ['dfALU', 2136, 2117, 2113], ['dfALU', 2136, 2120, 2114], ['dfALU', 2136, 2122, 2115], ['dfALU', 2136, 2124, 2116], ['THREAD_SYNC'], ['L1_ACCESS', 1741], ['L1_ACCESS', 1743], ['L1_ACCESS', 1745], ['L1_ACCESS', 1747], ['L1_ACCESS', 1748], ['L1_ACCESS', 1751], ['L1_ACCESS', 1754], ['L1_ACCESS', 1756], ['THREAD_SYNC'], ['iALU', 1735], ['iALU', 2151, 65], ['iALU', 1737], ['iALU', 2153], ['iALU', 71, 2154], ['iALU', 69, 2153, 70], ['L2_ACCESS'], ['iALU', 2156], ['L2_ACCESS'], ['iALU', 2156], ['L2_ACCESS'], ['iALU', 2156], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 2155], ['iALU', 5, 2155], ['L2_ACCESS'], ['iALU', 5, 2165], ['iALU', 73, 2165], ['L2_ACCESS'], ['iALU', 73, 2168], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 2174, 2173, 2119], ['L1_ACCESS'], ['dfALU', 2174, 2176, 2121], ['L1_ACCESS'], ['dfALU', 2174, 2178, 2123], ['L1_ACCESS'], ['dfALU', 2174, 2180, 2125], ['L1_ACCESS'], ['dfALU', 2182, 2173, 2127], ['dfALU', 2182, 2176, 2128], ['dfALU', 2182, 2178, 2129], ['dfALU', 2182, 2180, 2130], ['L1_ACCESS'], ['dfALU', 2187, 2173, 2132], ['dfALU', 2187, 2176, 2133], ['dfALU', 2187, 2178, 2134], ['dfALU', 2187, 2180, 2135], ['L1_ACCESS'], ['dfALU', 2192, 2173, 2137], ['dfALU', 2192, 2176, 2138], ['dfALU', 2192, 2178, 2139], ['dfALU', 2192, 2180, 2140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2198, 2197, 2175], ['L1_ACCESS'], ['dfALU', 2198, 2200, 2177], ['L1_ACCESS'], ['dfALU', 2198, 2202, 2179], ['L1_ACCESS'], ['dfALU', 2198, 2204, 2181], ['L1_ACCESS'], ['dfALU', 2206, 2197, 2183], ['dfALU', 2206, 2200, 2184], ['dfALU', 2206, 2202, 2185], ['dfALU', 2206, 2204, 2186], ['L1_ACCESS'], ['dfALU', 2211, 2197, 2188], ['dfALU', 2211, 2200, 2189], ['dfALU', 2211, 2202, 2190], ['dfALU', 2211, 2204, 2191], ['L1_ACCESS'], ['dfALU', 2216, 2197, 2193], ['dfALU', 2216, 2200, 2194], ['dfALU', 2216, 2202, 2195], ['dfALU', 2216, 2204, 2196], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2222, 2221, 2199], ['L1_ACCESS'], ['dfALU', 2222, 2224, 2201], ['L1_ACCESS'], ['dfALU', 2222, 2226, 2203], ['L1_ACCESS'], ['dfALU', 2222, 2228, 2205], ['L1_ACCESS'], ['dfALU', 2230, 2221, 2207], ['dfALU', 2230, 2224, 2208], ['dfALU', 2230, 2226, 2209], ['dfALU', 2230, 2228, 2210], ['L1_ACCESS'], ['dfALU', 2235, 2221, 2212], ['dfALU', 2235, 2224, 2213], ['dfALU', 2235, 2226, 2214], ['dfALU', 2235, 2228, 2215], ['L1_ACCESS'], ['dfALU', 2240, 2221, 2217], ['dfALU', 2240, 2224, 2218], ['dfALU', 2240, 2226, 2219], ['dfALU', 2240, 2228, 2220], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2246, 2245, 2223], ['L1_ACCESS'], ['dfALU', 2246, 2248, 2225], ['L1_ACCESS'], ['dfALU', 2246, 2250, 2227], ['L1_ACCESS'], ['dfALU', 2246, 2252, 2229], ['L1_ACCESS'], ['dfALU', 2254, 2245, 2231], ['dfALU', 2254, 2248, 2232], ['dfALU', 2254, 2250, 2233], ['dfALU', 2254, 2252, 2234], ['L1_ACCESS'], ['dfALU', 2259, 2245, 2236], ['dfALU', 2259, 2248, 2237], ['dfALU', 2259, 2250, 2238], ['dfALU', 2259, 2252, 2239], ['L1_ACCESS'], ['dfALU', 2264, 2245, 2241], ['dfALU', 2264, 2248, 2242], ['dfALU', 2264, 2250, 2243], ['dfALU', 2264, 2252, 2244], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2270, 2269, 2247], ['L1_ACCESS'], ['dfALU', 2270, 2272, 2249], ['L1_ACCESS'], ['dfALU', 2270, 2274, 2251], ['L1_ACCESS'], ['dfALU', 2270, 2276, 2253], ['L1_ACCESS'], ['dfALU', 2278, 2269, 2255], ['dfALU', 2278, 2272, 2256], ['dfALU', 2278, 2274, 2257], ['dfALU', 2278, 2276, 2258], ['L1_ACCESS'], ['dfALU', 2283, 2269, 2260], ['dfALU', 2283, 2272, 2261], ['dfALU', 2283, 2274, 2262], ['dfALU', 2283, 2276, 2263], ['L1_ACCESS'], ['dfALU', 2288, 2269, 2265], ['dfALU', 2288, 2272, 2266], ['dfALU', 2288, 2274, 2267], ['dfALU', 2288, 2276, 2268], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2294, 2293, 2271], ['L1_ACCESS'], ['dfALU', 2294, 2296, 2273], ['L1_ACCESS'], ['dfALU', 2294, 2298, 2275], ['L1_ACCESS'], ['dfALU', 2294, 2300, 2277], ['L1_ACCESS'], ['dfALU', 2302, 2293, 2279], ['dfALU', 2302, 2296, 2280], ['dfALU', 2302, 2298, 2281], ['dfALU', 2302, 2300, 2282], ['L1_ACCESS'], ['dfALU', 2307, 2293, 2284], ['dfALU', 2307, 2296, 2285], ['dfALU', 2307, 2298, 2286], ['dfALU', 2307, 2300, 2287], ['L1_ACCESS'], ['dfALU', 2312, 2293, 2289], ['dfALU', 2312, 2296, 2290], ['dfALU', 2312, 2298, 2291], ['dfALU', 2312, 2300, 2292], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2318, 2317, 2295], ['L1_ACCESS'], ['dfALU', 2318, 2320, 2297], ['L1_ACCESS'], ['dfALU', 2318, 2322, 2299], ['L1_ACCESS'], ['dfALU', 2318, 2324, 2301], ['L1_ACCESS'], ['dfALU', 2326, 2317, 2303], ['dfALU', 2326, 2320, 2304], ['dfALU', 2326, 2322, 2305], ['dfALU', 2326, 2324, 2306], ['L1_ACCESS'], ['dfALU', 2331, 2317, 2308], ['dfALU', 2331, 2320, 2309], ['dfALU', 2331, 2322, 2310], ['dfALU', 2331, 2324, 2311], ['L1_ACCESS'], ['dfALU', 2336, 2317, 2313], ['dfALU', 2336, 2320, 2314], ['dfALU', 2336, 2322, 2315], ['dfALU', 2336, 2324, 2316], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2342, 2341, 2319], ['L1_ACCESS'], ['dfALU', 2342, 2344, 2321], ['L1_ACCESS'], ['dfALU', 2342, 2346, 2323], ['L1_ACCESS'], ['dfALU', 2342, 2348, 2325], ['L1_ACCESS'], ['dfALU', 2350, 2341, 2327], ['dfALU', 2350, 2344, 2328], ['dfALU', 2350, 2346, 2329], ['dfALU', 2350, 2348, 2330], ['L1_ACCESS'], ['dfALU', 2355, 2341, 2332], ['dfALU', 2355, 2344, 2333], ['dfALU', 2355, 2346, 2334], ['dfALU', 2355, 2348, 2335], ['L1_ACCESS'], ['dfALU', 2360, 2341, 2337], ['dfALU', 2360, 2344, 2338], ['dfALU', 2360, 2346, 2339], ['dfALU', 2360, 2348, 2340], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2366, 2365, 2343], ['L1_ACCESS'], ['dfALU', 2366, 2368, 2345], ['L1_ACCESS'], ['dfALU', 2366, 2370, 2347], ['L1_ACCESS'], ['dfALU', 2366, 2372, 2349], ['L1_ACCESS'], ['dfALU', 2374, 2365, 2351], ['dfALU', 2374, 2368, 2352], ['dfALU', 2374, 2370, 2353], ['dfALU', 2374, 2372, 2354], ['L1_ACCESS'], ['dfALU', 2379, 2365, 2356], ['dfALU', 2379, 2368, 2357], ['dfALU', 2379, 2370, 2358], ['dfALU', 2379, 2372, 2359], ['L1_ACCESS'], ['dfALU', 2384, 2365, 2361], ['dfALU', 2384, 2368, 2362], ['dfALU', 2384, 2370, 2363], ['dfALU', 2384, 2372, 2364], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2390, 2389, 2367], ['L1_ACCESS'], ['dfALU', 2390, 2392, 2369], ['L1_ACCESS'], ['dfALU', 2390, 2394, 2371], ['L1_ACCESS'], ['dfALU', 2390, 2396, 2373], ['L1_ACCESS'], ['dfALU', 2398, 2389, 2375], ['dfALU', 2398, 2392, 2376], ['dfALU', 2398, 2394, 2377], ['dfALU', 2398, 2396, 2378], ['L1_ACCESS'], ['dfALU', 2403, 2389, 2380], ['dfALU', 2403, 2392, 2381], ['dfALU', 2403, 2394, 2382], ['dfALU', 2403, 2396, 2383], ['L1_ACCESS'], ['dfALU', 2408, 2389, 2385], ['dfALU', 2408, 2392, 2386], ['dfALU', 2408, 2394, 2387], ['dfALU', 2408, 2396, 2388], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2414, 2413, 2391], ['L1_ACCESS'], ['dfALU', 2414, 2416, 2393], ['L1_ACCESS'], ['dfALU', 2414, 2418, 2395], ['L1_ACCESS'], ['dfALU', 2414, 2420, 2397], ['L1_ACCESS'], ['dfALU', 2422, 2413, 2399], ['dfALU', 2422, 2416, 2400], ['dfALU', 2422, 2418, 2401], ['dfALU', 2422, 2420, 2402], ['L1_ACCESS'], ['dfALU', 2427, 2413, 2404], ['dfALU', 2427, 2416, 2405], ['dfALU', 2427, 2418, 2406], ['dfALU', 2427, 2420, 2407], ['L1_ACCESS'], ['dfALU', 2432, 2413, 2409], ['dfALU', 2432, 2416, 2410], ['dfALU', 2432, 2418, 2411], ['dfALU', 2432, 2420, 2412], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2438, 2437, 2415], ['L1_ACCESS'], ['dfALU', 2438, 2440, 2417], ['L1_ACCESS'], ['dfALU', 2438, 2442, 2419], ['L1_ACCESS'], ['dfALU', 2438, 2444, 2421], ['L1_ACCESS'], ['dfALU', 2446, 2437, 2423], ['dfALU', 2446, 2440, 2424], ['dfALU', 2446, 2442, 2425], ['dfALU', 2446, 2444, 2426], ['L1_ACCESS'], ['dfALU', 2451, 2437, 2428], ['dfALU', 2451, 2440, 2429], ['dfALU', 2451, 2442, 2430], ['dfALU', 2451, 2444, 2431], ['L1_ACCESS'], ['dfALU', 2456, 2437, 2433], ['dfALU', 2456, 2440, 2434], ['dfALU', 2456, 2442, 2435], ['dfALU', 2456, 2444, 2436], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2462, 2461, 2439], ['L1_ACCESS'], ['dfALU', 2462, 2464, 2441], ['L1_ACCESS'], ['dfALU', 2462, 2466, 2443], ['L1_ACCESS'], ['dfALU', 2462, 2468, 2445], ['L1_ACCESS'], ['dfALU', 2470, 2461, 2447], ['dfALU', 2470, 2464, 2448], ['dfALU', 2470, 2466, 2449], ['dfALU', 2470, 2468, 2450], ['L1_ACCESS'], ['dfALU', 2475, 2461, 2452], ['dfALU', 2475, 2464, 2453], ['dfALU', 2475, 2466, 2454], ['dfALU', 2475, 2468, 2455], ['L1_ACCESS'], ['dfALU', 2480, 2461, 2457], ['dfALU', 2480, 2464, 2458], ['dfALU', 2480, 2466, 2459], ['dfALU', 2480, 2468, 2460], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2486, 2485, 2463], ['L1_ACCESS'], ['dfALU', 2486, 2488, 2465], ['L1_ACCESS'], ['dfALU', 2486, 2490, 2467], ['L1_ACCESS'], ['dfALU', 2486, 2492, 2469], ['L1_ACCESS'], ['dfALU', 2494, 2485, 2471], ['dfALU', 2494, 2488, 2472], ['dfALU', 2494, 2490, 2473], ['dfALU', 2494, 2492, 2474], ['L1_ACCESS'], ['dfALU', 2499, 2485, 2476], ['dfALU', 2499, 2488, 2477], ['dfALU', 2499, 2490, 2478], ['dfALU', 2499, 2492, 2479], ['L1_ACCESS'], ['dfALU', 2504, 2485, 2481], ['dfALU', 2504, 2488, 2482], ['dfALU', 2504, 2490, 2483], ['dfALU', 2504, 2492, 2484], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2510, 2509, 2487], ['L1_ACCESS'], ['dfALU', 2510, 2512, 2489], ['L1_ACCESS'], ['dfALU', 2510, 2514, 2491], ['L1_ACCESS'], ['dfALU', 2510, 2516, 2493], ['L1_ACCESS'], ['dfALU', 2518, 2509, 2495], ['dfALU', 2518, 2512, 2496], ['dfALU', 2518, 2514, 2497], ['dfALU', 2518, 2516, 2498], ['L1_ACCESS'], ['dfALU', 2523, 2509, 2500], ['dfALU', 2523, 2512, 2501], ['dfALU', 2523, 2514, 2502], ['dfALU', 2523, 2516, 2503], ['L1_ACCESS'], ['dfALU', 2528, 2509, 2505], ['dfALU', 2528, 2512, 2506], ['dfALU', 2528, 2514, 2507], ['dfALU', 2528, 2516, 2508], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2534, 2533, 2511], ['L1_ACCESS'], ['dfALU', 2534, 2536, 2513], ['L1_ACCESS'], ['dfALU', 2534, 2538, 2515], ['L1_ACCESS'], ['dfALU', 2534, 2540, 2517], ['L1_ACCESS'], ['dfALU', 2542, 2533, 2519], ['dfALU', 2542, 2536, 2520], ['dfALU', 2542, 2538, 2521], ['dfALU', 2542, 2540, 2522], ['L1_ACCESS'], ['dfALU', 2547, 2533, 2524], ['dfALU', 2547, 2536, 2525], ['dfALU', 2547, 2538, 2526], ['dfALU', 2547, 2540, 2527], ['L1_ACCESS'], ['dfALU', 2552, 2533, 2529], ['dfALU', 2552, 2536, 2530], ['dfALU', 2552, 2538, 2531], ['dfALU', 2552, 2540, 2532], ['THREAD_SYNC'], ['L1_ACCESS', 2157], ['L1_ACCESS', 2159], ['L1_ACCESS', 2161], ['L1_ACCESS', 2163], ['L1_ACCESS', 2164], ['L1_ACCESS', 2167], ['L1_ACCESS', 2170], ['L1_ACCESS', 2172], ['THREAD_SYNC'], ['iALU', 2151], ['iALU', 2567, 65], ['iALU', 2153], ['iALU', 2569], ['iALU', 71, 2570], ['iALU', 69, 2569, 70], ['L2_ACCESS'], ['iALU', 2572], ['L2_ACCESS'], ['iALU', 2572], ['L2_ACCESS'], ['iALU', 2572], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 2571], ['iALU', 5, 2571], ['L2_ACCESS'], ['iALU', 5, 2581], ['iALU', 73, 2581], ['L2_ACCESS'], ['iALU', 73, 2584], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 2590, 2589, 2535], ['L1_ACCESS'], ['dfALU', 2590, 2592, 2537], ['L1_ACCESS'], ['dfALU', 2590, 2594, 2539], ['L1_ACCESS'], ['dfALU', 2590, 2596, 2541], ['L1_ACCESS'], ['dfALU', 2598, 2589, 2543], ['dfALU', 2598, 2592, 2544], ['dfALU', 2598, 2594, 2545], ['dfALU', 2598, 2596, 2546], ['L1_ACCESS'], ['dfALU', 2603, 2589, 2548], ['dfALU', 2603, 2592, 2549], ['dfALU', 2603, 2594, 2550], ['dfALU', 2603, 2596, 2551], ['L1_ACCESS'], ['dfALU', 2608, 2589, 2553], ['dfALU', 2608, 2592, 2554], ['dfALU', 2608, 2594, 2555], ['dfALU', 2608, 2596, 2556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2614, 2613, 2591], ['L1_ACCESS'], ['dfALU', 2614, 2616, 2593], ['L1_ACCESS'], ['dfALU', 2614, 2618, 2595], ['L1_ACCESS'], ['dfALU', 2614, 2620, 2597], ['L1_ACCESS'], ['dfALU', 2622, 2613, 2599], ['dfALU', 2622, 2616, 2600], ['dfALU', 2622, 2618, 2601], ['dfALU', 2622, 2620, 2602], ['L1_ACCESS'], ['dfALU', 2627, 2613, 2604], ['dfALU', 2627, 2616, 2605], ['dfALU', 2627, 2618, 2606], ['dfALU', 2627, 2620, 2607], ['L1_ACCESS'], ['dfALU', 2632, 2613, 2609], ['dfALU', 2632, 2616, 2610], ['dfALU', 2632, 2618, 2611], ['dfALU', 2632, 2620, 2612], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2638, 2637, 2615], ['L1_ACCESS'], ['dfALU', 2638, 2640, 2617], ['L1_ACCESS'], ['dfALU', 2638, 2642, 2619], ['L1_ACCESS'], ['dfALU', 2638, 2644, 2621], ['L1_ACCESS'], ['dfALU', 2646, 2637, 2623], ['dfALU', 2646, 2640, 2624], ['dfALU', 2646, 2642, 2625], ['dfALU', 2646, 2644, 2626], ['L1_ACCESS'], ['dfALU', 2651, 2637, 2628], ['dfALU', 2651, 2640, 2629], ['dfALU', 2651, 2642, 2630], ['dfALU', 2651, 2644, 2631], ['L1_ACCESS'], ['dfALU', 2656, 2637, 2633], ['dfALU', 2656, 2640, 2634], ['dfALU', 2656, 2642, 2635], ['dfALU', 2656, 2644, 2636], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2662, 2661, 2639], ['L1_ACCESS'], ['dfALU', 2662, 2664, 2641], ['L1_ACCESS'], ['dfALU', 2662, 2666, 2643], ['L1_ACCESS'], ['dfALU', 2662, 2668, 2645], ['L1_ACCESS'], ['dfALU', 2670, 2661, 2647], ['dfALU', 2670, 2664, 2648], ['dfALU', 2670, 2666, 2649], ['dfALU', 2670, 2668, 2650], ['L1_ACCESS'], ['dfALU', 2675, 2661, 2652], ['dfALU', 2675, 2664, 2653], ['dfALU', 2675, 2666, 2654], ['dfALU', 2675, 2668, 2655], ['L1_ACCESS'], ['dfALU', 2680, 2661, 2657], ['dfALU', 2680, 2664, 2658], ['dfALU', 2680, 2666, 2659], ['dfALU', 2680, 2668, 2660], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2686, 2685, 2663], ['L1_ACCESS'], ['dfALU', 2686, 2688, 2665], ['L1_ACCESS'], ['dfALU', 2686, 2690, 2667], ['L1_ACCESS'], ['dfALU', 2686, 2692, 2669], ['L1_ACCESS'], ['dfALU', 2694, 2685, 2671], ['dfALU', 2694, 2688, 2672], ['dfALU', 2694, 2690, 2673], ['dfALU', 2694, 2692, 2674], ['L1_ACCESS'], ['dfALU', 2699, 2685, 2676], ['dfALU', 2699, 2688, 2677], ['dfALU', 2699, 2690, 2678], ['dfALU', 2699, 2692, 2679], ['L1_ACCESS'], ['dfALU', 2704, 2685, 2681], ['dfALU', 2704, 2688, 2682], ['dfALU', 2704, 2690, 2683], ['dfALU', 2704, 2692, 2684], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2710, 2709, 2687], ['L1_ACCESS'], ['dfALU', 2710, 2712, 2689], ['L1_ACCESS'], ['dfALU', 2710, 2714, 2691], ['L1_ACCESS'], ['dfALU', 2710, 2716, 2693], ['L1_ACCESS'], ['dfALU', 2718, 2709, 2695], ['dfALU', 2718, 2712, 2696], ['dfALU', 2718, 2714, 2697], ['dfALU', 2718, 2716, 2698], ['L1_ACCESS'], ['dfALU', 2723, 2709, 2700], ['dfALU', 2723, 2712, 2701], ['dfALU', 2723, 2714, 2702], ['dfALU', 2723, 2716, 2703], ['L1_ACCESS'], ['dfALU', 2728, 2709, 2705], ['dfALU', 2728, 2712, 2706], ['dfALU', 2728, 2714, 2707], ['dfALU', 2728, 2716, 2708], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2734, 2733, 2711], ['L1_ACCESS'], ['dfALU', 2734, 2736, 2713], ['L1_ACCESS'], ['dfALU', 2734, 2738, 2715], ['L1_ACCESS'], ['dfALU', 2734, 2740, 2717], ['L1_ACCESS'], ['dfALU', 2742, 2733, 2719], ['dfALU', 2742, 2736, 2720], ['dfALU', 2742, 2738, 2721], ['dfALU', 2742, 2740, 2722], ['L1_ACCESS'], ['dfALU', 2747, 2733, 2724], ['dfALU', 2747, 2736, 2725], ['dfALU', 2747, 2738, 2726], ['dfALU', 2747, 2740, 2727], ['L1_ACCESS'], ['dfALU', 2752, 2733, 2729], ['dfALU', 2752, 2736, 2730], ['dfALU', 2752, 2738, 2731], ['dfALU', 2752, 2740, 2732], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2758, 2757, 2735], ['L1_ACCESS'], ['dfALU', 2758, 2760, 2737], ['L1_ACCESS'], ['dfALU', 2758, 2762, 2739], ['L1_ACCESS'], ['dfALU', 2758, 2764, 2741], ['L1_ACCESS'], ['dfALU', 2766, 2757, 2743], ['dfALU', 2766, 2760, 2744], ['dfALU', 2766, 2762, 2745], ['dfALU', 2766, 2764, 2746], ['L1_ACCESS'], ['dfALU', 2771, 2757, 2748], ['dfALU', 2771, 2760, 2749], ['dfALU', 2771, 2762, 2750], ['dfALU', 2771, 2764, 2751], ['L1_ACCESS'], ['dfALU', 2776, 2757, 2753], ['dfALU', 2776, 2760, 2754], ['dfALU', 2776, 2762, 2755], ['dfALU', 2776, 2764, 2756], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2782, 2781, 2759], ['L1_ACCESS'], ['dfALU', 2782, 2784, 2761], ['L1_ACCESS'], ['dfALU', 2782, 2786, 2763], ['L1_ACCESS'], ['dfALU', 2782, 2788, 2765], ['L1_ACCESS'], ['dfALU', 2790, 2781, 2767], ['dfALU', 2790, 2784, 2768], ['dfALU', 2790, 2786, 2769], ['dfALU', 2790, 2788, 2770], ['L1_ACCESS'], ['dfALU', 2795, 2781, 2772], ['dfALU', 2795, 2784, 2773], ['dfALU', 2795, 2786, 2774], ['dfALU', 2795, 2788, 2775], ['L1_ACCESS'], ['dfALU', 2800, 2781, 2777], ['dfALU', 2800, 2784, 2778], ['dfALU', 2800, 2786, 2779], ['dfALU', 2800, 2788, 2780], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2806, 2805, 2783], ['L1_ACCESS'], ['dfALU', 2806, 2808, 2785], ['L1_ACCESS'], ['dfALU', 2806, 2810, 2787], ['L1_ACCESS'], ['dfALU', 2806, 2812, 2789], ['L1_ACCESS'], ['dfALU', 2814, 2805, 2791], ['dfALU', 2814, 2808, 2792], ['dfALU', 2814, 2810, 2793], ['dfALU', 2814, 2812, 2794], ['L1_ACCESS'], ['dfALU', 2819, 2805, 2796], ['dfALU', 2819, 2808, 2797], ['dfALU', 2819, 2810, 2798], ['dfALU', 2819, 2812, 2799], ['L1_ACCESS'], ['dfALU', 2824, 2805, 2801], ['dfALU', 2824, 2808, 2802], ['dfALU', 2824, 2810, 2803], ['dfALU', 2824, 2812, 2804], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2830, 2829, 2807], ['L1_ACCESS'], ['dfALU', 2830, 2832, 2809], ['L1_ACCESS'], ['dfALU', 2830, 2834, 2811], ['L1_ACCESS'], ['dfALU', 2830, 2836, 2813], ['L1_ACCESS'], ['dfALU', 2838, 2829, 2815], ['dfALU', 2838, 2832, 2816], ['dfALU', 2838, 2834, 2817], ['dfALU', 2838, 2836, 2818], ['L1_ACCESS'], ['dfALU', 2843, 2829, 2820], ['dfALU', 2843, 2832, 2821], ['dfALU', 2843, 2834, 2822], ['dfALU', 2843, 2836, 2823], ['L1_ACCESS'], ['dfALU', 2848, 2829, 2825], ['dfALU', 2848, 2832, 2826], ['dfALU', 2848, 2834, 2827], ['dfALU', 2848, 2836, 2828], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2854, 2853, 2831], ['L1_ACCESS'], ['dfALU', 2854, 2856, 2833], ['L1_ACCESS'], ['dfALU', 2854, 2858, 2835], ['L1_ACCESS'], ['dfALU', 2854, 2860, 2837], ['L1_ACCESS'], ['dfALU', 2862, 2853, 2839], ['dfALU', 2862, 2856, 2840], ['dfALU', 2862, 2858, 2841], ['dfALU', 2862, 2860, 2842], ['L1_ACCESS'], ['dfALU', 2867, 2853, 2844], ['dfALU', 2867, 2856, 2845], ['dfALU', 2867, 2858, 2846], ['dfALU', 2867, 2860, 2847], ['L1_ACCESS'], ['dfALU', 2872, 2853, 2849], ['dfALU', 2872, 2856, 2850], ['dfALU', 2872, 2858, 2851], ['dfALU', 2872, 2860, 2852], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2878, 2877, 2855], ['L1_ACCESS'], ['dfALU', 2878, 2880, 2857], ['L1_ACCESS'], ['dfALU', 2878, 2882, 2859], ['L1_ACCESS'], ['dfALU', 2878, 2884, 2861], ['L1_ACCESS'], ['dfALU', 2886, 2877, 2863], ['dfALU', 2886, 2880, 2864], ['dfALU', 2886, 2882, 2865], ['dfALU', 2886, 2884, 2866], ['L1_ACCESS'], ['dfALU', 2891, 2877, 2868], ['dfALU', 2891, 2880, 2869], ['dfALU', 2891, 2882, 2870], ['dfALU', 2891, 2884, 2871], ['L1_ACCESS'], ['dfALU', 2896, 2877, 2873], ['dfALU', 2896, 2880, 2874], ['dfALU', 2896, 2882, 2875], ['dfALU', 2896, 2884, 2876], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2902, 2901, 2879], ['L1_ACCESS'], ['dfALU', 2902, 2904, 2881], ['L1_ACCESS'], ['dfALU', 2902, 2906, 2883], ['L1_ACCESS'], ['dfALU', 2902, 2908, 2885], ['L1_ACCESS'], ['dfALU', 2910, 2901, 2887], ['dfALU', 2910, 2904, 2888], ['dfALU', 2910, 2906, 2889], ['dfALU', 2910, 2908, 2890], ['L1_ACCESS'], ['dfALU', 2915, 2901, 2892], ['dfALU', 2915, 2904, 2893], ['dfALU', 2915, 2906, 2894], ['dfALU', 2915, 2908, 2895], ['L1_ACCESS'], ['dfALU', 2920, 2901, 2897], ['dfALU', 2920, 2904, 2898], ['dfALU', 2920, 2906, 2899], ['dfALU', 2920, 2908, 2900], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2926, 2925, 2903], ['L1_ACCESS'], ['dfALU', 2926, 2928, 2905], ['L1_ACCESS'], ['dfALU', 2926, 2930, 2907], ['L1_ACCESS'], ['dfALU', 2926, 2932, 2909], ['L1_ACCESS'], ['dfALU', 2934, 2925, 2911], ['dfALU', 2934, 2928, 2912], ['dfALU', 2934, 2930, 2913], ['dfALU', 2934, 2932, 2914], ['L1_ACCESS'], ['dfALU', 2939, 2925, 2916], ['dfALU', 2939, 2928, 2917], ['dfALU', 2939, 2930, 2918], ['dfALU', 2939, 2932, 2919], ['L1_ACCESS'], ['dfALU', 2944, 2925, 2921], ['dfALU', 2944, 2928, 2922], ['dfALU', 2944, 2930, 2923], ['dfALU', 2944, 2932, 2924], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2950, 2949, 2927], ['L1_ACCESS'], ['dfALU', 2950, 2952, 2929], ['L1_ACCESS'], ['dfALU', 2950, 2954, 2931], ['L1_ACCESS'], ['dfALU', 2950, 2956, 2933], ['L1_ACCESS'], ['dfALU', 2958, 2949, 2935], ['dfALU', 2958, 2952, 2936], ['dfALU', 2958, 2954, 2937], ['dfALU', 2958, 2956, 2938], ['L1_ACCESS'], ['dfALU', 2963, 2949, 2940], ['dfALU', 2963, 2952, 2941], ['dfALU', 2963, 2954, 2942], ['dfALU', 2963, 2956, 2943], ['L1_ACCESS'], ['dfALU', 2968, 2949, 2945], ['dfALU', 2968, 2952, 2946], ['dfALU', 2968, 2954, 2947], ['dfALU', 2968, 2956, 2948], ['THREAD_SYNC'], ['L1_ACCESS', 2573], ['L1_ACCESS', 2575], ['L1_ACCESS', 2577], ['L1_ACCESS', 2579], ['L1_ACCESS', 2580], ['L1_ACCESS', 2583], ['L1_ACCESS', 2586], ['L1_ACCESS', 2588], ['THREAD_SYNC'], ['iALU', 2567], ['iALU', 2983, 65], ['iALU', 2569], ['iALU'], ['L1_ACCESS'], ['iALU'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 2990, 2989, 2951], ['L1_ACCESS'], ['dfALU', 2990, 2992, 2953], ['L1_ACCESS'], ['dfALU', 2990, 2994, 2955], ['L1_ACCESS'], ['dfALU', 2990, 2996, 2957], ['L1_ACCESS'], ['dfALU', 2998, 2989, 2959], ['dfALU', 2998, 2992, 2960], ['dfALU', 2998, 2994, 2961], ['dfALU', 2998, 2996, 2962], ['L1_ACCESS'], ['dfALU', 3003, 2989, 2964], ['dfALU', 3003, 2992, 2965], ['dfALU', 3003, 2994, 2966], ['dfALU', 3003, 2996, 2967], ['L1_ACCESS'], ['dfALU', 3008, 2989, 2969], ['dfALU', 3008, 2992, 2970], ['dfALU', 3008, 2994, 2971], ['dfALU', 3008, 2996, 2972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3014, 3013, 2991], ['L1_ACCESS'], ['dfALU', 3014, 3016, 2993], ['L1_ACCESS'], ['dfALU', 3014, 3018, 2995], ['L1_ACCESS'], ['dfALU', 3014, 3020, 2997], ['L1_ACCESS'], ['dfALU', 3022, 3013, 2999], ['dfALU', 3022, 3016, 3000], ['dfALU', 3022, 3018, 3001], ['dfALU', 3022, 3020, 3002], ['L1_ACCESS'], ['dfALU', 3027, 3013, 3004], ['dfALU', 3027, 3016, 3005], ['dfALU', 3027, 3018, 3006], ['dfALU', 3027, 3020, 3007], ['L1_ACCESS'], ['dfALU', 3032, 3013, 3009], ['dfALU', 3032, 3016, 3010], ['dfALU', 3032, 3018, 3011], ['dfALU', 3032, 3020, 3012], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3038, 3037, 3015], ['L1_ACCESS'], ['dfALU', 3038, 3040, 3017], ['L1_ACCESS'], ['dfALU', 3038, 3042, 3019], ['L1_ACCESS'], ['dfALU', 3038, 3044, 3021], ['L1_ACCESS'], ['dfALU', 3046, 3037, 3023], ['dfALU', 3046, 3040, 3024], ['dfALU', 3046, 3042, 3025], ['dfALU', 3046, 3044, 3026], ['L1_ACCESS'], ['dfALU', 3051, 3037, 3028], ['dfALU', 3051, 3040, 3029], ['dfALU', 3051, 3042, 3030], ['dfALU', 3051, 3044, 3031], ['L1_ACCESS'], ['dfALU', 3056, 3037, 3033], ['dfALU', 3056, 3040, 3034], ['dfALU', 3056, 3042, 3035], ['dfALU', 3056, 3044, 3036], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3062, 3061, 3039], ['L1_ACCESS'], ['dfALU', 3062, 3064, 3041], ['L1_ACCESS'], ['dfALU', 3062, 3066, 3043], ['L1_ACCESS'], ['dfALU', 3062, 3068, 3045], ['L1_ACCESS'], ['dfALU', 3070, 3061, 3047], ['dfALU', 3070, 3064, 3048], ['dfALU', 3070, 3066, 3049], ['dfALU', 3070, 3068, 3050], ['L1_ACCESS'], ['dfALU', 3075, 3061, 3052], ['dfALU', 3075, 3064, 3053], ['dfALU', 3075, 3066, 3054], ['dfALU', 3075, 3068, 3055], ['L1_ACCESS'], ['dfALU', 3080, 3061, 3057], ['dfALU', 3080, 3064, 3058], ['dfALU', 3080, 3066, 3059], ['dfALU', 3080, 3068, 3060], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3086, 3085, 3063], ['L1_ACCESS'], ['dfALU', 3086, 3088, 3065], ['L1_ACCESS'], ['dfALU', 3086, 3090, 3067], ['L1_ACCESS'], ['dfALU', 3086, 3092, 3069], ['L1_ACCESS'], ['dfALU', 3094, 3085, 3071], ['dfALU', 3094, 3088, 3072], ['dfALU', 3094, 3090, 3073], ['dfALU', 3094, 3092, 3074], ['L1_ACCESS'], ['dfALU', 3099, 3085, 3076], ['dfALU', 3099, 3088, 3077], ['dfALU', 3099, 3090, 3078], ['dfALU', 3099, 3092, 3079], ['L1_ACCESS'], ['dfALU', 3104, 3085, 3081], ['dfALU', 3104, 3088, 3082], ['dfALU', 3104, 3090, 3083], ['dfALU', 3104, 3092, 3084], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3110, 3109, 3087], ['L1_ACCESS'], ['dfALU', 3110, 3112, 3089], ['L1_ACCESS'], ['dfALU', 3110, 3114, 3091], ['L1_ACCESS'], ['dfALU', 3110, 3116, 3093], ['L1_ACCESS'], ['dfALU', 3118, 3109, 3095], ['dfALU', 3118, 3112, 3096], ['dfALU', 3118, 3114, 3097], ['dfALU', 3118, 3116, 3098], ['L1_ACCESS'], ['dfALU', 3123, 3109, 3100], ['dfALU', 3123, 3112, 3101], ['dfALU', 3123, 3114, 3102], ['dfALU', 3123, 3116, 3103], ['L1_ACCESS'], ['dfALU', 3128, 3109, 3105], ['dfALU', 3128, 3112, 3106], ['dfALU', 3128, 3114, 3107], ['dfALU', 3128, 3116, 3108], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3134, 3133, 3111], ['L1_ACCESS'], ['dfALU', 3134, 3136, 3113], ['L1_ACCESS'], ['dfALU', 3134, 3138, 3115], ['L1_ACCESS'], ['dfALU', 3134, 3140, 3117], ['L1_ACCESS'], ['dfALU', 3142, 3133, 3119], ['dfALU', 3142, 3136, 3120], ['dfALU', 3142, 3138, 3121], ['dfALU', 3142, 3140, 3122], ['L1_ACCESS'], ['dfALU', 3147, 3133, 3124], ['dfALU', 3147, 3136, 3125], ['dfALU', 3147, 3138, 3126], ['dfALU', 3147, 3140, 3127], ['L1_ACCESS'], ['dfALU', 3152, 3133, 3129], ['dfALU', 3152, 3136, 3130], ['dfALU', 3152, 3138, 3131], ['dfALU', 3152, 3140, 3132], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3158, 3157, 3135], ['L1_ACCESS'], ['dfALU', 3158, 3160, 3137], ['L1_ACCESS'], ['dfALU', 3158, 3162, 3139], ['L1_ACCESS'], ['dfALU', 3158, 3164, 3141], ['L1_ACCESS'], ['dfALU', 3166, 3157, 3143], ['dfALU', 3166, 3160, 3144], ['dfALU', 3166, 3162, 3145], ['dfALU', 3166, 3164, 3146], ['L1_ACCESS'], ['dfALU', 3171, 3157, 3148], ['dfALU', 3171, 3160, 3149], ['dfALU', 3171, 3162, 3150], ['dfALU', 3171, 3164, 3151], ['L1_ACCESS'], ['dfALU', 3176, 3157, 3153], ['dfALU', 3176, 3160, 3154], ['dfALU', 3176, 3162, 3155], ['dfALU', 3176, 3164, 3156], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3182, 3181, 3159], ['L1_ACCESS'], ['dfALU', 3182, 3184, 3161], ['L1_ACCESS'], ['dfALU', 3182, 3186, 3163], ['L1_ACCESS'], ['dfALU', 3182, 3188, 3165], ['L1_ACCESS'], ['dfALU', 3190, 3181, 3167], ['dfALU', 3190, 3184, 3168], ['dfALU', 3190, 3186, 3169], ['dfALU', 3190, 3188, 3170], ['L1_ACCESS'], ['dfALU', 3195, 3181, 3172], ['dfALU', 3195, 3184, 3173], ['dfALU', 3195, 3186, 3174], ['dfALU', 3195, 3188, 3175], ['L1_ACCESS'], ['dfALU', 3200, 3181, 3177], ['dfALU', 3200, 3184, 3178], ['dfALU', 3200, 3186, 3179], ['dfALU', 3200, 3188, 3180], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3206, 3205, 3183], ['L1_ACCESS'], ['dfALU', 3206, 3208, 3185], ['L1_ACCESS'], ['dfALU', 3206, 3210, 3187], ['L1_ACCESS'], ['dfALU', 3206, 3212, 3189], ['L1_ACCESS'], ['dfALU', 3214, 3205, 3191], ['dfALU', 3214, 3208, 3192], ['dfALU', 3214, 3210, 3193], ['dfALU', 3214, 3212, 3194], ['L1_ACCESS'], ['dfALU', 3219, 3205, 3196], ['dfALU', 3219, 3208, 3197], ['dfALU', 3219, 3210, 3198], ['dfALU', 3219, 3212, 3199], ['L1_ACCESS'], ['dfALU', 3224, 3205, 3201], ['dfALU', 3224, 3208, 3202], ['dfALU', 3224, 3210, 3203], ['dfALU', 3224, 3212, 3204], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3230, 3229, 3207], ['L1_ACCESS'], ['dfALU', 3230, 3232, 3209], ['L1_ACCESS'], ['dfALU', 3230, 3234, 3211], ['L1_ACCESS'], ['dfALU', 3230, 3236, 3213], ['L1_ACCESS'], ['dfALU', 3238, 3229, 3215], ['dfALU', 3238, 3232, 3216], ['dfALU', 3238, 3234, 3217], ['dfALU', 3238, 3236, 3218], ['L1_ACCESS'], ['dfALU', 3243, 3229, 3220], ['dfALU', 3243, 3232, 3221], ['dfALU', 3243, 3234, 3222], ['dfALU', 3243, 3236, 3223], ['L1_ACCESS'], ['dfALU', 3248, 3229, 3225], ['dfALU', 3248, 3232, 3226], ['dfALU', 3248, 3234, 3227], ['dfALU', 3248, 3236, 3228], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3254, 3253, 3231], ['L1_ACCESS'], ['dfALU', 3254, 3256, 3233], ['L1_ACCESS'], ['dfALU', 3254, 3258, 3235], ['L1_ACCESS'], ['dfALU', 3254, 3260, 3237], ['L1_ACCESS'], ['dfALU', 3262, 3253, 3239], ['dfALU', 3262, 3256, 3240], ['dfALU', 3262, 3258, 3241], ['dfALU', 3262, 3260, 3242], ['L1_ACCESS'], ['dfALU', 3267, 3253, 3244], ['dfALU', 3267, 3256, 3245], ['dfALU', 3267, 3258, 3246], ['dfALU', 3267, 3260, 3247], ['L1_ACCESS'], ['dfALU', 3272, 3253, 3249], ['dfALU', 3272, 3256, 3250], ['dfALU', 3272, 3258, 3251], ['dfALU', 3272, 3260, 3252], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3278, 3277, 3255], ['L1_ACCESS'], ['dfALU', 3278, 3280, 3257], ['L1_ACCESS'], ['dfALU', 3278, 3282, 3259], ['L1_ACCESS'], ['dfALU', 3278, 3284, 3261], ['L1_ACCESS'], ['dfALU', 3286, 3277, 3263], ['dfALU', 3286, 3280, 3264], ['dfALU', 3286, 3282, 3265], ['dfALU', 3286, 3284, 3266], ['L1_ACCESS'], ['dfALU', 3291, 3277, 3268], ['dfALU', 3291, 3280, 3269], ['dfALU', 3291, 3282, 3270], ['dfALU', 3291, 3284, 3271], ['L1_ACCESS'], ['dfALU', 3296, 3277, 3273], ['dfALU', 3296, 3280, 3274], ['dfALU', 3296, 3282, 3275], ['dfALU', 3296, 3284, 3276], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3302, 3301, 3279], ['L1_ACCESS'], ['dfALU', 3302, 3304, 3281], ['L1_ACCESS'], ['dfALU', 3302, 3306, 3283], ['L1_ACCESS'], ['dfALU', 3302, 3308, 3285], ['L1_ACCESS'], ['dfALU', 3310, 3301, 3287], ['dfALU', 3310, 3304, 3288], ['dfALU', 3310, 3306, 3289], ['dfALU', 3310, 3308, 3290], ['L1_ACCESS'], ['dfALU', 3315, 3301, 3292], ['dfALU', 3315, 3304, 3293], ['dfALU', 3315, 3306, 3294], ['dfALU', 3315, 3308, 3295], ['L1_ACCESS'], ['dfALU', 3320, 3301, 3297], ['dfALU', 3320, 3304, 3298], ['dfALU', 3320, 3306, 3299], ['dfALU', 3320, 3308, 3300], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3326, 3325, 3303], ['L1_ACCESS'], ['dfALU', 3326, 3328, 3305], ['L1_ACCESS'], ['dfALU', 3326, 3330, 3307], ['L1_ACCESS'], ['dfALU', 3326, 3332, 3309], ['L1_ACCESS'], ['dfALU', 3334, 3325, 3311], ['dfALU', 3334, 3328, 3312], ['dfALU', 3334, 3330, 3313], ['dfALU', 3334, 3332, 3314], ['L1_ACCESS'], ['dfALU', 3339, 3325, 3316], ['dfALU', 3339, 3328, 3317], ['dfALU', 3339, 3330, 3318], ['dfALU', 3339, 3332, 3319], ['L1_ACCESS'], ['dfALU', 3344, 3325, 3321], ['dfALU', 3344, 3328, 3322], ['dfALU', 3344, 3330, 3323], ['dfALU', 3344, 3332, 3324], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3350, 3349, 3327], ['L1_ACCESS'], ['dfALU', 3350, 3352, 3329], ['L1_ACCESS'], ['dfALU', 3350, 3354, 3331], ['L1_ACCESS'], ['dfALU', 3350, 3356, 3333], ['L1_ACCESS'], ['dfALU', 3358, 3349, 3335], ['dfALU', 3358, 3352, 3336], ['dfALU', 3358, 3354, 3337], ['dfALU', 3358, 3356, 3338], ['L1_ACCESS'], ['dfALU', 3363, 3349, 3340], ['dfALU', 3363, 3352, 3341], ['dfALU', 3363, 3354, 3342], ['dfALU', 3363, 3356, 3343], ['L1_ACCESS'], ['dfALU', 3368, 3349, 3345], ['dfALU', 3368, 3352, 3346], ['dfALU', 3368, 3354, 3347], ['dfALU', 3368, 3356, 3348], ['iALU', 18, 2988], ['iALU', 3373, 2987], ['iALU', 3374, 17], ['iALU', 3375, 2986], ['iALU', 3373, 2], ['iALU', 17, 2986], ['iALU', 3378, 1], ['iALU', 3379, 3377], ['diALU', 0], ['iALU', 3376], ['diALU', 3381, 3382], ['GLOB_MEM_ACCESS', 3383], ['dfALU', 3384, 7], ['dfALU', 3351, 6, 3385], ['GLOB_MEM_ACCESS', 3386], ['iALU', 3378], ['iALU', 3388, 1], ['iALU', 3389, 3377], ['diALU', 0], ['iALU', 3376], ['diALU', 3391, 3392], ['GLOB_MEM_ACCESS'], ['dfALU', 3394, 7], ['dfALU', 3359, 6, 3395], ['GLOB_MEM_ACCESS', 3396], ['iALU', 3378], ['iALU', 3398, 1], ['iALU', 3399, 3377], ['diALU', 0], ['iALU', 3376], ['diALU', 3401, 3402], ['GLOB_MEM_ACCESS'], ['dfALU', 3404, 7], ['dfALU', 3364, 6, 3405], ['GLOB_MEM_ACCESS', 3406], ['iALU', 3378], ['iALU', 3408, 1], ['iALU', 3409, 3377], ['diALU', 0], ['iALU', 3376], ['diALU', 3411, 3412], ['GLOB_MEM_ACCESS'], ['dfALU', 3414, 7], ['dfALU', 3369, 6, 3415], ['GLOB_MEM_ACCESS', 3416], ['L1_ACCESS'], ['iALU', 3418], ['iALU', 3419], ['iALU', 3376], ['diALU', 3421, 3420], ['diALU', 0], ['diALU', 3422], ['diALU', 3423, 3424], ['iALU', 3373], ['iALU', 3426, 2], ['iALU', 3379, 3427], ['GLOB_MEM_ACCESS', 3425], ['dfALU', 3429, 7], ['dfALU', 3353, 6, 3430], ['GLOB_MEM_ACCESS', 3431], ['iALU', 3389, 3427], ['GLOB_MEM_ACCESS'], ['dfALU', 3434, 7], ['dfALU', 3360, 6, 3435], ['GLOB_MEM_ACCESS', 3436], ['iALU', 3399, 3427], ['GLOB_MEM_ACCESS'], ['dfALU', 3439, 7], ['dfALU', 3365, 6, 3440], ['GLOB_MEM_ACCESS', 3441], ['iALU', 3409, 3427], ['GLOB_MEM_ACCESS'], ['dfALU', 3444, 7], ['dfALU', 3370, 6, 3445], ['GLOB_MEM_ACCESS', 3446], ['iALU', 3419], ['iALU', 3376], ['diALU', 3449, 3448], ['diALU', 3450], ['diALU', 3423, 3451], ['iALU', 3373], ['iALU', 3453, 2], ['iALU', 3379, 3454], ['GLOB_MEM_ACCESS', 3452], ['dfALU', 3456, 7], ['dfALU', 3355, 6, 3457], ['GLOB_MEM_ACCESS', 3458], ['iALU', 3389, 3454], ['GLOB_MEM_ACCESS'], ['dfALU', 3461, 7], ['dfALU', 3361, 6, 3462], ['GLOB_MEM_ACCESS', 3463], ['iALU', 3399, 3454], ['GLOB_MEM_ACCESS'], ['dfALU', 3466, 7], ['dfALU', 3366, 6, 3467], ['GLOB_MEM_ACCESS', 3468], ['iALU', 3409, 3454], ['GLOB_MEM_ACCESS'], ['dfALU', 3471, 7], ['dfALU', 3371, 6, 3472], ['GLOB_MEM_ACCESS', 3473], ['iALU', 3419], ['iALU', 3376], ['diALU', 3475, 3476], ['diALU', 3477], ['diALU', 3423, 3478], ['iALU', 3373], ['iALU', 3480, 2], ['iALU', 3379, 3481], ['GLOB_MEM_ACCESS', 3479], ['dfALU', 3483, 7], ['dfALU', 3357, 6, 3484], ['GLOB_MEM_ACCESS', 3485], ['iALU', 3389, 3481], ['GLOB_MEM_ACCESS'], ['dfALU', 3488, 7], ['dfALU', 3362, 6, 3489], ['GLOB_MEM_ACCESS', 3490], ['iALU', 3399, 3481], ['GLOB_MEM_ACCESS'], ['dfALU', 3493, 7], ['dfALU', 3367, 6, 3494], ['GLOB_MEM_ACCESS', 3495], ['iALU', 3409, 3481], ['GLOB_MEM_ACCESS'], ['dfALU', 3498, 7], ['dfALU', 3372, 6, 3499], ['GLOB_MEM_ACCESS', 3500]]

    GPU_tasklist3 = [['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['L1_ACCESS'], ['iALU'], ['iALU'], ['iALU', 11], ['iALU', 12], ['iALU', 13], ['iALU', 12, 14], ['iALU', 15], ['iALU', 12, 16], ['iALU', 15], ['iALU', 18, 4], ['iALU', 18], ['iALU', 20, 10], ['iALU', 18], ['iALU', 19, 17], ['iALU', 23, 8], ['iALU'], ['iALU', 24, 25], ['L2_ACCESS'], ['iALU', 17], ['diALU', 28], ['iALU', 18], ['diALU', 29, 30], ['L1_ACCESS', 27], ['iALU', 26], ['L2_ACCESS'], ['diALU', 28, 30], ['diALU', 35], ['L1_ACCESS', 34], ['iALU', 26], ['L2_ACCESS'], ['L1_ACCESS', 39], ['iALU', 26], ['L2_ACCESS'], ['L1_ACCESS', 42], ['iALU', 21, 5], ['iALU', 44, 17], ['iALU', 45, 9], ['L2_ACCESS'], ['diALU', 28], ['iALU', 20], ['diALU', 48, 49], ['L1_ACCESS', 47], ['iALU', 46, 5], ['L2_ACCESS'], ['diALU', 28, 49], ['diALU', 54], ['L1_ACCESS', 53], ['iALU', 5], ['iALU', 57, 46], ['L2_ACCESS'], ['L1_ACCESS', 59], ['iALU', 5, 46], ['L2_ACCESS'], ['L1_ACCESS', 62], ['THREAD_SYNC'], ['iALU', 3], ['diALU', 22], ['diALU', 66], ['iALU', 65], ['iALU', 4], ['iALU', 26, 69], ['iALU', 46], ['iALU', 5], ['iALU', 5], ['iALU'], ['iALU', 71, 74], ['iALU', 69, 70], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['iALU', 76], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 75], ['iALU', 5, 75], ['L2_ACCESS'], ['iALU', 5, 85], ['iALU', 73, 85], ['L2_ACCESS'], ['iALU', 73, 88], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 94, 93], ['L1_ACCESS'], ['dfALU', 94, 96], ['L1_ACCESS'], ['dfALU', 94, 98], ['L1_ACCESS'], ['dfALU', 94, 100], ['L1_ACCESS'], ['dfALU', 102, 93], ['dfALU', 102, 96], ['dfALU', 102, 98], ['dfALU', 102, 100], ['L1_ACCESS'], ['dfALU', 107, 93], ['dfALU', 107, 96], ['dfALU', 107, 98], ['dfALU', 107, 100], ['L1_ACCESS'], ['dfALU', 112, 93], ['dfALU', 112, 96], ['dfALU', 112, 98], ['dfALU', 112, 100], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 118, 117, 95], ['L1_ACCESS'], ['dfALU', 118, 120, 97], ['L1_ACCESS'], ['dfALU', 118, 122, 99], ['L1_ACCESS'], ['dfALU', 118, 124, 101], ['L1_ACCESS'], ['dfALU', 126, 117, 103], ['dfALU', 126, 120, 104], ['dfALU', 126, 122, 105], ['dfALU', 126, 124, 106], ['L1_ACCESS'], ['dfALU', 131, 117, 108], ['dfALU', 131, 120, 109], ['dfALU', 131, 122, 110], ['dfALU', 131, 124, 111], ['L1_ACCESS'], ['dfALU', 136, 117, 113], ['dfALU', 136, 120, 114], ['dfALU', 136, 122, 115], ['dfALU', 136, 124, 116], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 142, 141, 119], ['L1_ACCESS'], ['dfALU', 142, 144, 121], ['L1_ACCESS'], ['dfALU', 142, 146, 123], ['L1_ACCESS'], ['dfALU', 142, 148, 125], ['L1_ACCESS'], ['dfALU', 150, 141, 127], ['dfALU', 150, 144, 128], ['dfALU', 150, 146, 129], ['dfALU', 150, 148, 130], ['L1_ACCESS'], ['dfALU', 155, 141, 132], ['dfALU', 155, 144, 133], ['dfALU', 155, 146, 134], ['dfALU', 155, 148, 135], ['L1_ACCESS'], ['dfALU', 160, 141, 137], ['dfALU', 160, 144, 138], ['dfALU', 160, 146, 139], ['dfALU', 160, 148, 140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 166, 165, 143], ['L1_ACCESS'], ['dfALU', 166, 168, 145], ['L1_ACCESS'], ['dfALU', 166, 170, 147], ['L1_ACCESS'], ['dfALU', 166, 172, 149], ['L1_ACCESS'], ['dfALU', 174, 165, 151], ['dfALU', 174, 168, 152], ['dfALU', 174, 170, 153], ['dfALU', 174, 172, 154], ['L1_ACCESS'], ['dfALU', 179, 165, 156], ['dfALU', 179, 168, 157], ['dfALU', 179, 170, 158], ['dfALU', 179, 172, 159], ['L1_ACCESS'], ['dfALU', 184, 165, 161], ['dfALU', 184, 168, 162], ['dfALU', 184, 170, 163], ['dfALU', 184, 172, 164], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 190, 189, 167], ['L1_ACCESS'], ['dfALU', 190, 192, 169], ['L1_ACCESS'], ['dfALU', 190, 194, 171], ['L1_ACCESS'], ['dfALU', 190, 196, 173], ['L1_ACCESS'], ['dfALU', 198, 189, 175], ['dfALU', 198, 192, 176], ['dfALU', 198, 194, 177], ['dfALU', 198, 196, 178], ['L1_ACCESS'], ['dfALU', 203, 189, 180], ['dfALU', 203, 192, 181], ['dfALU', 203, 194, 182], ['dfALU', 203, 196, 183], ['L1_ACCESS'], ['dfALU', 208, 189, 185], ['dfALU', 208, 192, 186], ['dfALU', 208, 194, 187], ['dfALU', 208, 196, 188], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 214, 213, 191], ['L1_ACCESS'], ['dfALU', 214, 216, 193], ['L1_ACCESS'], ['dfALU', 214, 218, 195], ['L1_ACCESS'], ['dfALU', 214, 220, 197], ['L1_ACCESS'], ['dfALU', 222, 213, 199], ['dfALU', 222, 216, 200], ['dfALU', 222, 218, 201], ['dfALU', 222, 220, 202], ['L1_ACCESS'], ['dfALU', 227, 213, 204], ['dfALU', 227, 216, 205], ['dfALU', 227, 218, 206], ['dfALU', 227, 220, 207], ['L1_ACCESS'], ['dfALU', 232, 213, 209], ['dfALU', 232, 216, 210], ['dfALU', 232, 218, 211], ['dfALU', 232, 220, 212], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 238, 237, 215], ['L1_ACCESS'], ['dfALU', 238, 240, 217], ['L1_ACCESS'], ['dfALU', 238, 242, 219], ['L1_ACCESS'], ['dfALU', 238, 244, 221], ['L1_ACCESS'], ['dfALU', 246, 237, 223], ['dfALU', 246, 240, 224], ['dfALU', 246, 242, 225], ['dfALU', 246, 244, 226], ['L1_ACCESS'], ['dfALU', 251, 237, 228], ['dfALU', 251, 240, 229], ['dfALU', 251, 242, 230], ['dfALU', 251, 244, 231], ['L1_ACCESS'], ['dfALU', 256, 237, 233], ['dfALU', 256, 240, 234], ['dfALU', 256, 242, 235], ['dfALU', 256, 244, 236], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 262, 261, 239], ['L1_ACCESS'], ['dfALU', 262, 264, 241], ['L1_ACCESS'], ['dfALU', 262, 266, 243], ['L1_ACCESS'], ['dfALU', 262, 268, 245], ['L1_ACCESS'], ['dfALU', 270, 261, 247], ['dfALU', 270, 264, 248], ['dfALU', 270, 266, 249], ['dfALU', 270, 268, 250], ['L1_ACCESS'], ['dfALU', 275, 261, 252], ['dfALU', 275, 264, 253], ['dfALU', 275, 266, 254], ['dfALU', 275, 268, 255], ['L1_ACCESS'], ['dfALU', 280, 261, 257], ['dfALU', 280, 264, 258], ['dfALU', 280, 266, 259], ['dfALU', 280, 268, 260], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 286, 285, 263], ['L1_ACCESS'], ['dfALU', 286, 288, 265], ['L1_ACCESS'], ['dfALU', 286, 290, 267], ['L1_ACCESS'], ['dfALU', 286, 292, 269], ['L1_ACCESS'], ['dfALU', 294, 285, 271], ['dfALU', 294, 288, 272], ['dfALU', 294, 290, 273], ['dfALU', 294, 292, 274], ['L1_ACCESS'], ['dfALU', 299, 285, 276], ['dfALU', 299, 288, 277], ['dfALU', 299, 290, 278], ['dfALU', 299, 292, 279], ['L1_ACCESS'], ['dfALU', 304, 285, 281], ['dfALU', 304, 288, 282], ['dfALU', 304, 290, 283], ['dfALU', 304, 292, 284], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 310, 309, 287], ['L1_ACCESS'], ['dfALU', 310, 312, 289], ['L1_ACCESS'], ['dfALU', 310, 314, 291], ['L1_ACCESS'], ['dfALU', 310, 316, 293], ['L1_ACCESS'], ['dfALU', 318, 309, 295], ['dfALU', 318, 312, 296], ['dfALU', 318, 314, 297], ['dfALU', 318, 316, 298], ['L1_ACCESS'], ['dfALU', 323, 309, 300], ['dfALU', 323, 312, 301], ['dfALU', 323, 314, 302], ['dfALU', 323, 316, 303], ['L1_ACCESS'], ['dfALU', 328, 309, 305], ['dfALU', 328, 312, 306], ['dfALU', 328, 314, 307], ['dfALU', 328, 316, 308], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 334, 333, 311], ['L1_ACCESS'], ['dfALU', 334, 336, 313], ['L1_ACCESS'], ['dfALU', 334, 338, 315], ['L1_ACCESS'], ['dfALU', 334, 340, 317], ['L1_ACCESS'], ['dfALU', 342, 333, 319], ['dfALU', 342, 336, 320], ['dfALU', 342, 338, 321], ['dfALU', 342, 340, 322], ['L1_ACCESS'], ['dfALU', 347, 333, 324], ['dfALU', 347, 336, 325], ['dfALU', 347, 338, 326], ['dfALU', 347, 340, 327], ['L1_ACCESS'], ['dfALU', 352, 333, 329], ['dfALU', 352, 336, 330], ['dfALU', 352, 338, 331], ['dfALU', 352, 340, 332], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 358, 357, 335], ['L1_ACCESS'], ['dfALU', 358, 360, 337], ['L1_ACCESS'], ['dfALU', 358, 362, 339], ['L1_ACCESS'], ['dfALU', 358, 364, 341], ['L1_ACCESS'], ['dfALU', 366, 357, 343], ['dfALU', 366, 360, 344], ['dfALU', 366, 362, 345], ['dfALU', 366, 364, 346], ['L1_ACCESS'], ['dfALU', 371, 357, 348], ['dfALU', 371, 360, 349], ['dfALU', 371, 362, 350], ['dfALU', 371, 364, 351], ['L1_ACCESS'], ['dfALU', 376, 357, 353], ['dfALU', 376, 360, 354], ['dfALU', 376, 362, 355], ['dfALU', 376, 364, 356], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 382, 381, 359], ['L1_ACCESS'], ['dfALU', 382, 384, 361], ['L1_ACCESS'], ['dfALU', 382, 386, 363], ['L1_ACCESS'], ['dfALU', 382, 388, 365], ['L1_ACCESS'], ['dfALU', 390, 381, 367], ['dfALU', 390, 384, 368], ['dfALU', 390, 386, 369], ['dfALU', 390, 388, 370], ['L1_ACCESS'], ['dfALU', 395, 381, 372], ['dfALU', 395, 384, 373], ['dfALU', 395, 386, 374], ['dfALU', 395, 388, 375], ['L1_ACCESS'], ['dfALU', 400, 381, 377], ['dfALU', 400, 384, 378], ['dfALU', 400, 386, 379], ['dfALU', 400, 388, 380], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 406, 405, 383], ['L1_ACCESS'], ['dfALU', 406, 408, 385], ['L1_ACCESS'], ['dfALU', 406, 410, 387], ['L1_ACCESS'], ['dfALU', 406, 412, 389], ['L1_ACCESS'], ['dfALU', 414, 405, 391], ['dfALU', 414, 408, 392], ['dfALU', 414, 410, 393], ['dfALU', 414, 412, 394], ['L1_ACCESS'], ['dfALU', 419, 405, 396], ['dfALU', 419, 408, 397], ['dfALU', 419, 410, 398], ['dfALU', 419, 412, 399], ['L1_ACCESS'], ['dfALU', 424, 405, 401], ['dfALU', 424, 408, 402], ['dfALU', 424, 410, 403], ['dfALU', 424, 412, 404], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 430, 429, 407], ['L1_ACCESS'], ['dfALU', 430, 432, 409], ['L1_ACCESS'], ['dfALU', 430, 434, 411], ['L1_ACCESS'], ['dfALU', 430, 436, 413], ['L1_ACCESS'], ['dfALU', 438, 429, 415], ['dfALU', 438, 432, 416], ['dfALU', 438, 434, 417], ['dfALU', 438, 436, 418], ['L1_ACCESS'], ['dfALU', 443, 429, 420], ['dfALU', 443, 432, 421], ['dfALU', 443, 434, 422], ['dfALU', 443, 436, 423], ['L1_ACCESS'], ['dfALU', 448, 429, 425], ['dfALU', 448, 432, 426], ['dfALU', 448, 434, 427], ['dfALU', 448, 436, 428], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 454, 453, 431], ['L1_ACCESS'], ['dfALU', 454, 456, 433], ['L1_ACCESS'], ['dfALU', 454, 458, 435], ['L1_ACCESS'], ['dfALU', 454, 460, 437], ['L1_ACCESS'], ['dfALU', 462, 453, 439], ['dfALU', 462, 456, 440], ['dfALU', 462, 458, 441], ['dfALU', 462, 460, 442], ['L1_ACCESS'], ['dfALU', 467, 453, 444], ['dfALU', 467, 456, 445], ['dfALU', 467, 458, 446], ['dfALU', 467, 460, 447], ['L1_ACCESS'], ['dfALU', 472, 453, 449], ['dfALU', 472, 456, 450], ['dfALU', 472, 458, 451], ['dfALU', 472, 460, 452], ['THREAD_SYNC'], ['L1_ACCESS', 77], ['L1_ACCESS', 79], ['L1_ACCESS', 81], ['L1_ACCESS', 83], ['L1_ACCESS', 84], ['L1_ACCESS', 87], ['L1_ACCESS', 90], ['L1_ACCESS', 92], ['THREAD_SYNC'], ['iALU'], ['iALU', 487, 65], ['iALU'], ['iALU', 489], ['iALU', 71, 490], ['iALU', 69, 489, 70], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['iALU', 492], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 491], ['iALU', 5, 491], ['L2_ACCESS'], ['iALU', 5, 501], ['iALU', 73, 501], ['L2_ACCESS'], ['iALU', 73, 504], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 510, 509, 455], ['L1_ACCESS'], ['dfALU', 510, 512, 457], ['L1_ACCESS'], ['dfALU', 510, 514, 459], ['L1_ACCESS'], ['dfALU', 510, 516, 461], ['L1_ACCESS'], ['dfALU', 518, 509, 463], ['dfALU', 518, 512, 464], ['dfALU', 518, 514, 465], ['dfALU', 518, 516, 466], ['L1_ACCESS'], ['dfALU', 523, 509, 468], ['dfALU', 523, 512, 469], ['dfALU', 523, 514, 470], ['dfALU', 523, 516, 471], ['L1_ACCESS'], ['dfALU', 528, 509, 473], ['dfALU', 528, 512, 474], ['dfALU', 528, 514, 475], ['dfALU', 528, 516, 476], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 534, 533, 511], ['L1_ACCESS'], ['dfALU', 534, 536, 513], ['L1_ACCESS'], ['dfALU', 534, 538, 515], ['L1_ACCESS'], ['dfALU', 534, 540, 517], ['L1_ACCESS'], ['dfALU', 542, 533, 519], ['dfALU', 542, 536, 520], ['dfALU', 542, 538, 521], ['dfALU', 542, 540, 522], ['L1_ACCESS'], ['dfALU', 547, 533, 524], ['dfALU', 547, 536, 525], ['dfALU', 547, 538, 526], ['dfALU', 547, 540, 527], ['L1_ACCESS'], ['dfALU', 552, 533, 529], ['dfALU', 552, 536, 530], ['dfALU', 552, 538, 531], ['dfALU', 552, 540, 532], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 558, 557, 535], ['L1_ACCESS'], ['dfALU', 558, 560, 537], ['L1_ACCESS'], ['dfALU', 558, 562, 539], ['L1_ACCESS'], ['dfALU', 558, 564, 541], ['L1_ACCESS'], ['dfALU', 566, 557, 543], ['dfALU', 566, 560, 544], ['dfALU', 566, 562, 545], ['dfALU', 566, 564, 546], ['L1_ACCESS'], ['dfALU', 571, 557, 548], ['dfALU', 571, 560, 549], ['dfALU', 571, 562, 550], ['dfALU', 571, 564, 551], ['L1_ACCESS'], ['dfALU', 576, 557, 553], ['dfALU', 576, 560, 554], ['dfALU', 576, 562, 555], ['dfALU', 576, 564, 556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 582, 581, 559], ['L1_ACCESS'], ['dfALU', 582, 584, 561], ['L1_ACCESS'], ['dfALU', 582, 586, 563], ['L1_ACCESS'], ['dfALU', 582, 588, 565], ['L1_ACCESS'], ['dfALU', 590, 581, 567], ['dfALU', 590, 584, 568], ['dfALU', 590, 586, 569], ['dfALU', 590, 588, 570], ['L1_ACCESS'], ['dfALU', 595, 581, 572], ['dfALU', 595, 584, 573], ['dfALU', 595, 586, 574], ['dfALU', 595, 588, 575], ['L1_ACCESS'], ['dfALU', 600, 581, 577], ['dfALU', 600, 584, 578], ['dfALU', 600, 586, 579], ['dfALU', 600, 588, 580], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 606, 605, 583], ['L1_ACCESS'], ['dfALU', 606, 608, 585], ['L1_ACCESS'], ['dfALU', 606, 610, 587], ['L1_ACCESS'], ['dfALU', 606, 612, 589], ['L1_ACCESS'], ['dfALU', 614, 605, 591], ['dfALU', 614, 608, 592], ['dfALU', 614, 610, 593], ['dfALU', 614, 612, 594], ['L1_ACCESS'], ['dfALU', 619, 605, 596], ['dfALU', 619, 608, 597], ['dfALU', 619, 610, 598], ['dfALU', 619, 612, 599], ['L1_ACCESS'], ['dfALU', 624, 605, 601], ['dfALU', 624, 608, 602], ['dfALU', 624, 610, 603], ['dfALU', 624, 612, 604], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 630, 629, 607], ['L1_ACCESS'], ['dfALU', 630, 632, 609], ['L1_ACCESS'], ['dfALU', 630, 634, 611], ['L1_ACCESS'], ['dfALU', 630, 636, 613], ['L1_ACCESS'], ['dfALU', 638, 629, 615], ['dfALU', 638, 632, 616], ['dfALU', 638, 634, 617], ['dfALU', 638, 636, 618], ['L1_ACCESS'], ['dfALU', 643, 629, 620], ['dfALU', 643, 632, 621], ['dfALU', 643, 634, 622], ['dfALU', 643, 636, 623], ['L1_ACCESS'], ['dfALU', 648, 629, 625], ['dfALU', 648, 632, 626], ['dfALU', 648, 634, 627], ['dfALU', 648, 636, 628], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 654, 653, 631], ['L1_ACCESS'], ['dfALU', 654, 656, 633], ['L1_ACCESS'], ['dfALU', 654, 658, 635], ['L1_ACCESS'], ['dfALU', 654, 660, 637], ['L1_ACCESS'], ['dfALU', 662, 653, 639], ['dfALU', 662, 656, 640], ['dfALU', 662, 658, 641], ['dfALU', 662, 660, 642], ['L1_ACCESS'], ['dfALU', 667, 653, 644], ['dfALU', 667, 656, 645], ['dfALU', 667, 658, 646], ['dfALU', 667, 660, 647], ['L1_ACCESS'], ['dfALU', 672, 653, 649], ['dfALU', 672, 656, 650], ['dfALU', 672, 658, 651], ['dfALU', 672, 660, 652], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 678, 677, 655], ['L1_ACCESS'], ['dfALU', 678, 680, 657], ['L1_ACCESS'], ['dfALU', 678, 682, 659], ['L1_ACCESS'], ['dfALU', 678, 684, 661], ['L1_ACCESS'], ['dfALU', 686, 677, 663], ['dfALU', 686, 680, 664], ['dfALU', 686, 682, 665], ['dfALU', 686, 684, 666], ['L1_ACCESS'], ['dfALU', 691, 677, 668], ['dfALU', 691, 680, 669], ['dfALU', 691, 682, 670], ['dfALU', 691, 684, 671], ['L1_ACCESS'], ['dfALU', 696, 677, 673], ['dfALU', 696, 680, 674], ['dfALU', 696, 682, 675], ['dfALU', 696, 684, 676], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 702, 701, 679], ['L1_ACCESS'], ['dfALU', 702, 704, 681], ['L1_ACCESS'], ['dfALU', 702, 706, 683], ['L1_ACCESS'], ['dfALU', 702, 708, 685], ['L1_ACCESS'], ['dfALU', 710, 701, 687], ['dfALU', 710, 704, 688], ['dfALU', 710, 706, 689], ['dfALU', 710, 708, 690], ['L1_ACCESS'], ['dfALU', 715, 701, 692], ['dfALU', 715, 704, 693], ['dfALU', 715, 706, 694], ['dfALU', 715, 708, 695], ['L1_ACCESS'], ['dfALU', 720, 701, 697], ['dfALU', 720, 704, 698], ['dfALU', 720, 706, 699], ['dfALU', 720, 708, 700], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 726, 725, 703], ['L1_ACCESS'], ['dfALU', 726, 728, 705], ['L1_ACCESS'], ['dfALU', 726, 730, 707], ['L1_ACCESS'], ['dfALU', 726, 732, 709], ['L1_ACCESS'], ['dfALU', 734, 725, 711], ['dfALU', 734, 728, 712], ['dfALU', 734, 730, 713], ['dfALU', 734, 732, 714], ['L1_ACCESS'], ['dfALU', 739, 725, 716], ['dfALU', 739, 728, 717], ['dfALU', 739, 730, 718], ['dfALU', 739, 732, 719], ['L1_ACCESS'], ['dfALU', 744, 725, 721], ['dfALU', 744, 728, 722], ['dfALU', 744, 730, 723], ['dfALU', 744, 732, 724], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 750, 749, 727], ['L1_ACCESS'], ['dfALU', 750, 752, 729], ['L1_ACCESS'], ['dfALU', 750, 754, 731], ['L1_ACCESS'], ['dfALU', 750, 756, 733], ['L1_ACCESS'], ['dfALU', 758, 749, 735], ['dfALU', 758, 752, 736], ['dfALU', 758, 754, 737], ['dfALU', 758, 756, 738], ['L1_ACCESS'], ['dfALU', 763, 749, 740], ['dfALU', 763, 752, 741], ['dfALU', 763, 754, 742], ['dfALU', 763, 756, 743], ['L1_ACCESS'], ['dfALU', 768, 749, 745], ['dfALU', 768, 752, 746], ['dfALU', 768, 754, 747], ['dfALU', 768, 756, 748], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 774, 773, 751], ['L1_ACCESS'], ['dfALU', 774, 776, 753], ['L1_ACCESS'], ['dfALU', 774, 778, 755], ['L1_ACCESS'], ['dfALU', 774, 780, 757], ['L1_ACCESS'], ['dfALU', 782, 773, 759], ['dfALU', 782, 776, 760], ['dfALU', 782, 778, 761], ['dfALU', 782, 780, 762], ['L1_ACCESS'], ['dfALU', 787, 773, 764], ['dfALU', 787, 776, 765], ['dfALU', 787, 778, 766], ['dfALU', 787, 780, 767], ['L1_ACCESS'], ['dfALU', 792, 773, 769], ['dfALU', 792, 776, 770], ['dfALU', 792, 778, 771], ['dfALU', 792, 780, 772], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 798, 797, 775], ['L1_ACCESS'], ['dfALU', 798, 800, 777], ['L1_ACCESS'], ['dfALU', 798, 802, 779], ['L1_ACCESS'], ['dfALU', 798, 804, 781], ['L1_ACCESS'], ['dfALU', 806, 797, 783], ['dfALU', 806, 800, 784], ['dfALU', 806, 802, 785], ['dfALU', 806, 804, 786], ['L1_ACCESS'], ['dfALU', 811, 797, 788], ['dfALU', 811, 800, 789], ['dfALU', 811, 802, 790], ['dfALU', 811, 804, 791], ['L1_ACCESS'], ['dfALU', 816, 797, 793], ['dfALU', 816, 800, 794], ['dfALU', 816, 802, 795], ['dfALU', 816, 804, 796], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 822, 821, 799], ['L1_ACCESS'], ['dfALU', 822, 824, 801], ['L1_ACCESS'], ['dfALU', 822, 826, 803], ['L1_ACCESS'], ['dfALU', 822, 828, 805], ['L1_ACCESS'], ['dfALU', 830, 821, 807], ['dfALU', 830, 824, 808], ['dfALU', 830, 826, 809], ['dfALU', 830, 828, 810], ['L1_ACCESS'], ['dfALU', 835, 821, 812], ['dfALU', 835, 824, 813], ['dfALU', 835, 826, 814], ['dfALU', 835, 828, 815], ['L1_ACCESS'], ['dfALU', 840, 821, 817], ['dfALU', 840, 824, 818], ['dfALU', 840, 826, 819], ['dfALU', 840, 828, 820], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 846, 845, 823], ['L1_ACCESS'], ['dfALU', 846, 848, 825], ['L1_ACCESS'], ['dfALU', 846, 850, 827], ['L1_ACCESS'], ['dfALU', 846, 852, 829], ['L1_ACCESS'], ['dfALU', 854, 845, 831], ['dfALU', 854, 848, 832], ['dfALU', 854, 850, 833], ['dfALU', 854, 852, 834], ['L1_ACCESS'], ['dfALU', 859, 845, 836], ['dfALU', 859, 848, 837], ['dfALU', 859, 850, 838], ['dfALU', 859, 852, 839], ['L1_ACCESS'], ['dfALU', 864, 845, 841], ['dfALU', 864, 848, 842], ['dfALU', 864, 850, 843], ['dfALU', 864, 852, 844], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 870, 869, 847], ['L1_ACCESS'], ['dfALU', 870, 872, 849], ['L1_ACCESS'], ['dfALU', 870, 874, 851], ['L1_ACCESS'], ['dfALU', 870, 876, 853], ['L1_ACCESS'], ['dfALU', 878, 869, 855], ['dfALU', 878, 872, 856], ['dfALU', 878, 874, 857], ['dfALU', 878, 876, 858], ['L1_ACCESS'], ['dfALU', 883, 869, 860], ['dfALU', 883, 872, 861], ['dfALU', 883, 874, 862], ['dfALU', 883, 876, 863], ['L1_ACCESS'], ['dfALU', 888, 869, 865], ['dfALU', 888, 872, 866], ['dfALU', 888, 874, 867], ['dfALU', 888, 876, 868], ['THREAD_SYNC'], ['L1_ACCESS', 493], ['L1_ACCESS', 495], ['L1_ACCESS', 497], ['L1_ACCESS', 499], ['L1_ACCESS', 500], ['L1_ACCESS', 503], ['L1_ACCESS', 506], ['L1_ACCESS', 508], ['THREAD_SYNC'], ['iALU', 487], ['iALU', 903, 65], ['iALU', 489], ['iALU', 905], ['iALU', 71, 906], ['iALU', 69, 905, 70], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['iALU', 908], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 907], ['iALU', 5, 907], ['L2_ACCESS'], ['iALU', 5, 917], ['iALU', 73, 917], ['L2_ACCESS'], ['iALU', 73, 920], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 926, 925, 871], ['L1_ACCESS'], ['dfALU', 926, 928, 873], ['L1_ACCESS'], ['dfALU', 926, 930, 875], ['L1_ACCESS'], ['dfALU', 926, 932, 877], ['L1_ACCESS'], ['dfALU', 934, 925, 879], ['dfALU', 934, 928, 880], ['dfALU', 934, 930, 881], ['dfALU', 934, 932, 882], ['L1_ACCESS'], ['dfALU', 939, 925, 884], ['dfALU', 939, 928, 885], ['dfALU', 939, 930, 886], ['dfALU', 939, 932, 887], ['L1_ACCESS'], ['dfALU', 944, 925, 889], ['dfALU', 944, 928, 890], ['dfALU', 944, 930, 891], ['dfALU', 944, 932, 892], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 950, 949, 927], ['L1_ACCESS'], ['dfALU', 950, 952, 929], ['L1_ACCESS'], ['dfALU', 950, 954, 931], ['L1_ACCESS'], ['dfALU', 950, 956, 933], ['L1_ACCESS'], ['dfALU', 958, 949, 935], ['dfALU', 958, 952, 936], ['dfALU', 958, 954, 937], ['dfALU', 958, 956, 938], ['L1_ACCESS'], ['dfALU', 963, 949, 940], ['dfALU', 963, 952, 941], ['dfALU', 963, 954, 942], ['dfALU', 963, 956, 943], ['L1_ACCESS'], ['dfALU', 968, 949, 945], ['dfALU', 968, 952, 946], ['dfALU', 968, 954, 947], ['dfALU', 968, 956, 948], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 974, 973, 951], ['L1_ACCESS'], ['dfALU', 974, 976, 953], ['L1_ACCESS'], ['dfALU', 974, 978, 955], ['L1_ACCESS'], ['dfALU', 974, 980, 957], ['L1_ACCESS'], ['dfALU', 982, 973, 959], ['dfALU', 982, 976, 960], ['dfALU', 982, 978, 961], ['dfALU', 982, 980, 962], ['L1_ACCESS'], ['dfALU', 987, 973, 964], ['dfALU', 987, 976, 965], ['dfALU', 987, 978, 966], ['dfALU', 987, 980, 967], ['L1_ACCESS'], ['dfALU', 992, 973, 969], ['dfALU', 992, 976, 970], ['dfALU', 992, 978, 971], ['dfALU', 992, 980, 972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 998, 997, 975], ['L1_ACCESS'], ['dfALU', 998, 1000, 977], ['L1_ACCESS'], ['dfALU', 998, 1002, 979], ['L1_ACCESS'], ['dfALU', 998, 1004, 981], ['L1_ACCESS'], ['dfALU', 1006, 997, 983], ['dfALU', 1006, 1000, 984], ['dfALU', 1006, 1002, 985], ['dfALU', 1006, 1004, 986], ['L1_ACCESS'], ['dfALU', 1011, 997, 988], ['dfALU', 1011, 1000, 989], ['dfALU', 1011, 1002, 990], ['dfALU', 1011, 1004, 991], ['L1_ACCESS'], ['dfALU', 1016, 997, 993], ['dfALU', 1016, 1000, 994], ['dfALU', 1016, 1002, 995], ['dfALU', 1016, 1004, 996], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1022, 1021, 999], ['L1_ACCESS'], ['dfALU', 1022, 1024, 1001], ['L1_ACCESS'], ['dfALU', 1022, 1026, 1003], ['L1_ACCESS'], ['dfALU', 1022, 1028, 1005], ['L1_ACCESS'], ['dfALU', 1030, 1021, 1007], ['dfALU', 1030, 1024, 1008], ['dfALU', 1030, 1026, 1009], ['dfALU', 1030, 1028, 1010], ['L1_ACCESS'], ['dfALU', 1035, 1021, 1012], ['dfALU', 1035, 1024, 1013], ['dfALU', 1035, 1026, 1014], ['dfALU', 1035, 1028, 1015], ['L1_ACCESS'], ['dfALU', 1040, 1021, 1017], ['dfALU', 1040, 1024, 1018], ['dfALU', 1040, 1026, 1019], ['dfALU', 1040, 1028, 1020], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1046, 1045, 1023], ['L1_ACCESS'], ['dfALU', 1046, 1048, 1025], ['L1_ACCESS'], ['dfALU', 1046, 1050, 1027], ['L1_ACCESS'], ['dfALU', 1046, 1052, 1029], ['L1_ACCESS'], ['dfALU', 1054, 1045, 1031], ['dfALU', 1054, 1048, 1032], ['dfALU', 1054, 1050, 1033], ['dfALU', 1054, 1052, 1034], ['L1_ACCESS'], ['dfALU', 1059, 1045, 1036], ['dfALU', 1059, 1048, 1037], ['dfALU', 1059, 1050, 1038], ['dfALU', 1059, 1052, 1039], ['L1_ACCESS'], ['dfALU', 1064, 1045, 1041], ['dfALU', 1064, 1048, 1042], ['dfALU', 1064, 1050, 1043], ['dfALU', 1064, 1052, 1044], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1070, 1069, 1047], ['L1_ACCESS'], ['dfALU', 1070, 1072, 1049], ['L1_ACCESS'], ['dfALU', 1070, 1074, 1051], ['L1_ACCESS'], ['dfALU', 1070, 1076, 1053], ['L1_ACCESS'], ['dfALU', 1078, 1069, 1055], ['dfALU', 1078, 1072, 1056], ['dfALU', 1078, 1074, 1057], ['dfALU', 1078, 1076, 1058], ['L1_ACCESS'], ['dfALU', 1083, 1069, 1060], ['dfALU', 1083, 1072, 1061], ['dfALU', 1083, 1074, 1062], ['dfALU', 1083, 1076, 1063], ['L1_ACCESS'], ['dfALU', 1088, 1069, 1065], ['dfALU', 1088, 1072, 1066], ['dfALU', 1088, 1074, 1067], ['dfALU', 1088, 1076, 1068], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1094, 1093, 1071], ['L1_ACCESS'], ['dfALU', 1094, 1096, 1073], ['L1_ACCESS'], ['dfALU', 1094, 1098, 1075], ['L1_ACCESS'], ['dfALU', 1094, 1100, 1077], ['L1_ACCESS'], ['dfALU', 1102, 1093, 1079], ['dfALU', 1102, 1096, 1080], ['dfALU', 1102, 1098, 1081], ['dfALU', 1102, 1100, 1082], ['L1_ACCESS'], ['dfALU', 1107, 1093, 1084], ['dfALU', 1107, 1096, 1085], ['dfALU', 1107, 1098, 1086], ['dfALU', 1107, 1100, 1087], ['L1_ACCESS'], ['dfALU', 1112, 1093, 1089], ['dfALU', 1112, 1096, 1090], ['dfALU', 1112, 1098, 1091], ['dfALU', 1112, 1100, 1092], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1118, 1117, 1095], ['L1_ACCESS'], ['dfALU', 1118, 1120, 1097], ['L1_ACCESS'], ['dfALU', 1118, 1122, 1099], ['L1_ACCESS'], ['dfALU', 1118, 1124, 1101], ['L1_ACCESS'], ['dfALU', 1126, 1117, 1103], ['dfALU', 1126, 1120, 1104], ['dfALU', 1126, 1122, 1105], ['dfALU', 1126, 1124, 1106], ['L1_ACCESS'], ['dfALU', 1131, 1117, 1108], ['dfALU', 1131, 1120, 1109], ['dfALU', 1131, 1122, 1110], ['dfALU', 1131, 1124, 1111], ['L1_ACCESS'], ['dfALU', 1136, 1117, 1113], ['dfALU', 1136, 1120, 1114], ['dfALU', 1136, 1122, 1115], ['dfALU', 1136, 1124, 1116], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1142, 1141, 1119], ['L1_ACCESS'], ['dfALU', 1142, 1144, 1121], ['L1_ACCESS'], ['dfALU', 1142, 1146, 1123], ['L1_ACCESS'], ['dfALU', 1142, 1148, 1125], ['L1_ACCESS'], ['dfALU', 1150, 1141, 1127], ['dfALU', 1150, 1144, 1128], ['dfALU', 1150, 1146, 1129], ['dfALU', 1150, 1148, 1130], ['L1_ACCESS'], ['dfALU', 1155, 1141, 1132], ['dfALU', 1155, 1144, 1133], ['dfALU', 1155, 1146, 1134], ['dfALU', 1155, 1148, 1135], ['L1_ACCESS'], ['dfALU', 1160, 1141, 1137], ['dfALU', 1160, 1144, 1138], ['dfALU', 1160, 1146, 1139], ['dfALU', 1160, 1148, 1140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1166, 1165, 1143], ['L1_ACCESS'], ['dfALU', 1166, 1168, 1145], ['L1_ACCESS'], ['dfALU', 1166, 1170, 1147], ['L1_ACCESS'], ['dfALU', 1166, 1172, 1149], ['L1_ACCESS'], ['dfALU', 1174, 1165, 1151], ['dfALU', 1174, 1168, 1152], ['dfALU', 1174, 1170, 1153], ['dfALU', 1174, 1172, 1154], ['L1_ACCESS'], ['dfALU', 1179, 1165, 1156], ['dfALU', 1179, 1168, 1157], ['dfALU', 1179, 1170, 1158], ['dfALU', 1179, 1172, 1159], ['L1_ACCESS'], ['dfALU', 1184, 1165, 1161], ['dfALU', 1184, 1168, 1162], ['dfALU', 1184, 1170, 1163], ['dfALU', 1184, 1172, 1164], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1190, 1189, 1167], ['L1_ACCESS'], ['dfALU', 1190, 1192, 1169], ['L1_ACCESS'], ['dfALU', 1190, 1194, 1171], ['L1_ACCESS'], ['dfALU', 1190, 1196, 1173], ['L1_ACCESS'], ['dfALU', 1198, 1189, 1175], ['dfALU', 1198, 1192, 1176], ['dfALU', 1198, 1194, 1177], ['dfALU', 1198, 1196, 1178], ['L1_ACCESS'], ['dfALU', 1203, 1189, 1180], ['dfALU', 1203, 1192, 1181], ['dfALU', 1203, 1194, 1182], ['dfALU', 1203, 1196, 1183], ['L1_ACCESS'], ['dfALU', 1208, 1189, 1185], ['dfALU', 1208, 1192, 1186], ['dfALU', 1208, 1194, 1187], ['dfALU', 1208, 1196, 1188], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1214, 1213, 1191], ['L1_ACCESS'], ['dfALU', 1214, 1216, 1193], ['L1_ACCESS'], ['dfALU', 1214, 1218, 1195], ['L1_ACCESS'], ['dfALU', 1214, 1220, 1197], ['L1_ACCESS'], ['dfALU', 1222, 1213, 1199], ['dfALU', 1222, 1216, 1200], ['dfALU', 1222, 1218, 1201], ['dfALU', 1222, 1220, 1202], ['L1_ACCESS'], ['dfALU', 1227, 1213, 1204], ['dfALU', 1227, 1216, 1205], ['dfALU', 1227, 1218, 1206], ['dfALU', 1227, 1220, 1207], ['L1_ACCESS'], ['dfALU', 1232, 1213, 1209], ['dfALU', 1232, 1216, 1210], ['dfALU', 1232, 1218, 1211], ['dfALU', 1232, 1220, 1212], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1238, 1237, 1215], ['L1_ACCESS'], ['dfALU', 1238, 1240, 1217], ['L1_ACCESS'], ['dfALU', 1238, 1242, 1219], ['L1_ACCESS'], ['dfALU', 1238, 1244, 1221], ['L1_ACCESS'], ['dfALU', 1246, 1237, 1223], ['dfALU', 1246, 1240, 1224], ['dfALU', 1246, 1242, 1225], ['dfALU', 1246, 1244, 1226], ['L1_ACCESS'], ['dfALU', 1251, 1237, 1228], ['dfALU', 1251, 1240, 1229], ['dfALU', 1251, 1242, 1230], ['dfALU', 1251, 1244, 1231], ['L1_ACCESS'], ['dfALU', 1256, 1237, 1233], ['dfALU', 1256, 1240, 1234], ['dfALU', 1256, 1242, 1235], ['dfALU', 1256, 1244, 1236], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1262, 1261, 1239], ['L1_ACCESS'], ['dfALU', 1262, 1264, 1241], ['L1_ACCESS'], ['dfALU', 1262, 1266, 1243], ['L1_ACCESS'], ['dfALU', 1262, 1268, 1245], ['L1_ACCESS'], ['dfALU', 1270, 1261, 1247], ['dfALU', 1270, 1264, 1248], ['dfALU', 1270, 1266, 1249], ['dfALU', 1270, 1268, 1250], ['L1_ACCESS'], ['dfALU', 1275, 1261, 1252], ['dfALU', 1275, 1264, 1253], ['dfALU', 1275, 1266, 1254], ['dfALU', 1275, 1268, 1255], ['L1_ACCESS'], ['dfALU', 1280, 1261, 1257], ['dfALU', 1280, 1264, 1258], ['dfALU', 1280, 1266, 1259], ['dfALU', 1280, 1268, 1260], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1286, 1285, 1263], ['L1_ACCESS'], ['dfALU', 1286, 1288, 1265], ['L1_ACCESS'], ['dfALU', 1286, 1290, 1267], ['L1_ACCESS'], ['dfALU', 1286, 1292, 1269], ['L1_ACCESS'], ['dfALU', 1294, 1285, 1271], ['dfALU', 1294, 1288, 1272], ['dfALU', 1294, 1290, 1273], ['dfALU', 1294, 1292, 1274], ['L1_ACCESS'], ['dfALU', 1299, 1285, 1276], ['dfALU', 1299, 1288, 1277], ['dfALU', 1299, 1290, 1278], ['dfALU', 1299, 1292, 1279], ['L1_ACCESS'], ['dfALU', 1304, 1285, 1281], ['dfALU', 1304, 1288, 1282], ['dfALU', 1304, 1290, 1283], ['dfALU', 1304, 1292, 1284], ['THREAD_SYNC'], ['L1_ACCESS', 909], ['L1_ACCESS', 911], ['L1_ACCESS', 913], ['L1_ACCESS', 915], ['L1_ACCESS', 916], ['L1_ACCESS', 919], ['L1_ACCESS', 922], ['L1_ACCESS', 924], ['THREAD_SYNC'], ['iALU', 903], ['iALU', 1319, 65], ['iALU', 905], ['iALU', 1321], ['iALU', 71, 1322], ['iALU', 69, 1321, 70], ['L2_ACCESS'], ['iALU', 1324], ['L2_ACCESS'], ['iALU', 1324], ['L2_ACCESS'], ['iALU', 1324], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 1323], ['iALU', 5, 1323], ['L2_ACCESS'], ['iALU', 5, 1333], ['iALU', 73, 1333], ['L2_ACCESS'], ['iALU', 73, 1336], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 1342, 1341, 1287], ['L1_ACCESS'], ['dfALU', 1342, 1344, 1289], ['L1_ACCESS'], ['dfALU', 1342, 1346, 1291], ['L1_ACCESS'], ['dfALU', 1342, 1348, 1293], ['L1_ACCESS'], ['dfALU', 1350, 1341, 1295], ['dfALU', 1350, 1344, 1296], ['dfALU', 1350, 1346, 1297], ['dfALU', 1350, 1348, 1298], ['L1_ACCESS'], ['dfALU', 1355, 1341, 1300], ['dfALU', 1355, 1344, 1301], ['dfALU', 1355, 1346, 1302], ['dfALU', 1355, 1348, 1303], ['L1_ACCESS'], ['dfALU', 1360, 1341, 1305], ['dfALU', 1360, 1344, 1306], ['dfALU', 1360, 1346, 1307], ['dfALU', 1360, 1348, 1308], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1366, 1365, 1343], ['L1_ACCESS'], ['dfALU', 1366, 1368, 1345], ['L1_ACCESS'], ['dfALU', 1366, 1370, 1347], ['L1_ACCESS'], ['dfALU', 1366, 1372, 1349], ['L1_ACCESS'], ['dfALU', 1374, 1365, 1351], ['dfALU', 1374, 1368, 1352], ['dfALU', 1374, 1370, 1353], ['dfALU', 1374, 1372, 1354], ['L1_ACCESS'], ['dfALU', 1379, 1365, 1356], ['dfALU', 1379, 1368, 1357], ['dfALU', 1379, 1370, 1358], ['dfALU', 1379, 1372, 1359], ['L1_ACCESS'], ['dfALU', 1384, 1365, 1361], ['dfALU', 1384, 1368, 1362], ['dfALU', 1384, 1370, 1363], ['dfALU', 1384, 1372, 1364], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1390, 1389, 1367], ['L1_ACCESS'], ['dfALU', 1390, 1392, 1369], ['L1_ACCESS'], ['dfALU', 1390, 1394, 1371], ['L1_ACCESS'], ['dfALU', 1390, 1396, 1373], ['L1_ACCESS'], ['dfALU', 1398, 1389, 1375], ['dfALU', 1398, 1392, 1376], ['dfALU', 1398, 1394, 1377], ['dfALU', 1398, 1396, 1378], ['L1_ACCESS'], ['dfALU', 1403, 1389, 1380], ['dfALU', 1403, 1392, 1381], ['dfALU', 1403, 1394, 1382], ['dfALU', 1403, 1396, 1383], ['L1_ACCESS'], ['dfALU', 1408, 1389, 1385], ['dfALU', 1408, 1392, 1386], ['dfALU', 1408, 1394, 1387], ['dfALU', 1408, 1396, 1388], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1414, 1413, 1391], ['L1_ACCESS'], ['dfALU', 1414, 1416, 1393], ['L1_ACCESS'], ['dfALU', 1414, 1418, 1395], ['L1_ACCESS'], ['dfALU', 1414, 1420, 1397], ['L1_ACCESS'], ['dfALU', 1422, 1413, 1399], ['dfALU', 1422, 1416, 1400], ['dfALU', 1422, 1418, 1401], ['dfALU', 1422, 1420, 1402], ['L1_ACCESS'], ['dfALU', 1427, 1413, 1404], ['dfALU', 1427, 1416, 1405], ['dfALU', 1427, 1418, 1406], ['dfALU', 1427, 1420, 1407], ['L1_ACCESS'], ['dfALU', 1432, 1413, 1409], ['dfALU', 1432, 1416, 1410], ['dfALU', 1432, 1418, 1411], ['dfALU', 1432, 1420, 1412], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1438, 1437, 1415], ['L1_ACCESS'], ['dfALU', 1438, 1440, 1417], ['L1_ACCESS'], ['dfALU', 1438, 1442, 1419], ['L1_ACCESS'], ['dfALU', 1438, 1444, 1421], ['L1_ACCESS'], ['dfALU', 1446, 1437, 1423], ['dfALU', 1446, 1440, 1424], ['dfALU', 1446, 1442, 1425], ['dfALU', 1446, 1444, 1426], ['L1_ACCESS'], ['dfALU', 1451, 1437, 1428], ['dfALU', 1451, 1440, 1429], ['dfALU', 1451, 1442, 1430], ['dfALU', 1451, 1444, 1431], ['L1_ACCESS'], ['dfALU', 1456, 1437, 1433], ['dfALU', 1456, 1440, 1434], ['dfALU', 1456, 1442, 1435], ['dfALU', 1456, 1444, 1436], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1462, 1461, 1439], ['L1_ACCESS'], ['dfALU', 1462, 1464, 1441], ['L1_ACCESS'], ['dfALU', 1462, 1466, 1443], ['L1_ACCESS'], ['dfALU', 1462, 1468, 1445], ['L1_ACCESS'], ['dfALU', 1470, 1461, 1447], ['dfALU', 1470, 1464, 1448], ['dfALU', 1470, 1466, 1449], ['dfALU', 1470, 1468, 1450], ['L1_ACCESS'], ['dfALU', 1475, 1461, 1452], ['dfALU', 1475, 1464, 1453], ['dfALU', 1475, 1466, 1454], ['dfALU', 1475, 1468, 1455], ['L1_ACCESS'], ['dfALU', 1480, 1461, 1457], ['dfALU', 1480, 1464, 1458], ['dfALU', 1480, 1466, 1459], ['dfALU', 1480, 1468, 1460], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1486, 1485, 1463], ['L1_ACCESS'], ['dfALU', 1486, 1488, 1465], ['L1_ACCESS'], ['dfALU', 1486, 1490, 1467], ['L1_ACCESS'], ['dfALU', 1486, 1492, 1469], ['L1_ACCESS'], ['dfALU', 1494, 1485, 1471], ['dfALU', 1494, 1488, 1472], ['dfALU', 1494, 1490, 1473], ['dfALU', 1494, 1492, 1474], ['L1_ACCESS'], ['dfALU', 1499, 1485, 1476], ['dfALU', 1499, 1488, 1477], ['dfALU', 1499, 1490, 1478], ['dfALU', 1499, 1492, 1479], ['L1_ACCESS'], ['dfALU', 1504, 1485, 1481], ['dfALU', 1504, 1488, 1482], ['dfALU', 1504, 1490, 1483], ['dfALU', 1504, 1492, 1484], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1510, 1509, 1487], ['L1_ACCESS'], ['dfALU', 1510, 1512, 1489], ['L1_ACCESS'], ['dfALU', 1510, 1514, 1491], ['L1_ACCESS'], ['dfALU', 1510, 1516, 1493], ['L1_ACCESS'], ['dfALU', 1518, 1509, 1495], ['dfALU', 1518, 1512, 1496], ['dfALU', 1518, 1514, 1497], ['dfALU', 1518, 1516, 1498], ['L1_ACCESS'], ['dfALU', 1523, 1509, 1500], ['dfALU', 1523, 1512, 1501], ['dfALU', 1523, 1514, 1502], ['dfALU', 1523, 1516, 1503], ['L1_ACCESS'], ['dfALU', 1528, 1509, 1505], ['dfALU', 1528, 1512, 1506], ['dfALU', 1528, 1514, 1507], ['dfALU', 1528, 1516, 1508], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1534, 1533, 1511], ['L1_ACCESS'], ['dfALU', 1534, 1536, 1513], ['L1_ACCESS'], ['dfALU', 1534, 1538, 1515], ['L1_ACCESS'], ['dfALU', 1534, 1540, 1517], ['L1_ACCESS'], ['dfALU', 1542, 1533, 1519], ['dfALU', 1542, 1536, 1520], ['dfALU', 1542, 1538, 1521], ['dfALU', 1542, 1540, 1522], ['L1_ACCESS'], ['dfALU', 1547, 1533, 1524], ['dfALU', 1547, 1536, 1525], ['dfALU', 1547, 1538, 1526], ['dfALU', 1547, 1540, 1527], ['L1_ACCESS'], ['dfALU', 1552, 1533, 1529], ['dfALU', 1552, 1536, 1530], ['dfALU', 1552, 1538, 1531], ['dfALU', 1552, 1540, 1532], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1558, 1557, 1535], ['L1_ACCESS'], ['dfALU', 1558, 1560, 1537], ['L1_ACCESS'], ['dfALU', 1558, 1562, 1539], ['L1_ACCESS'], ['dfALU', 1558, 1564, 1541], ['L1_ACCESS'], ['dfALU', 1566, 1557, 1543], ['dfALU', 1566, 1560, 1544], ['dfALU', 1566, 1562, 1545], ['dfALU', 1566, 1564, 1546], ['L1_ACCESS'], ['dfALU', 1571, 1557, 1548], ['dfALU', 1571, 1560, 1549], ['dfALU', 1571, 1562, 1550], ['dfALU', 1571, 1564, 1551], ['L1_ACCESS'], ['dfALU', 1576, 1557, 1553], ['dfALU', 1576, 1560, 1554], ['dfALU', 1576, 1562, 1555], ['dfALU', 1576, 1564, 1556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1582, 1581, 1559], ['L1_ACCESS'], ['dfALU', 1582, 1584, 1561], ['L1_ACCESS'], ['dfALU', 1582, 1586, 1563], ['L1_ACCESS'], ['dfALU', 1582, 1588, 1565], ['L1_ACCESS'], ['dfALU', 1590, 1581, 1567], ['dfALU', 1590, 1584, 1568], ['dfALU', 1590, 1586, 1569], ['dfALU', 1590, 1588, 1570], ['L1_ACCESS'], ['dfALU', 1595, 1581, 1572], ['dfALU', 1595, 1584, 1573], ['dfALU', 1595, 1586, 1574], ['dfALU', 1595, 1588, 1575], ['L1_ACCESS'], ['dfALU', 1600, 1581, 1577], ['dfALU', 1600, 1584, 1578], ['dfALU', 1600, 1586, 1579], ['dfALU', 1600, 1588, 1580], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1606, 1605, 1583], ['L1_ACCESS'], ['dfALU', 1606, 1608, 1585], ['L1_ACCESS'], ['dfALU', 1606, 1610, 1587], ['L1_ACCESS'], ['dfALU', 1606, 1612, 1589], ['L1_ACCESS'], ['dfALU', 1614, 1605, 1591], ['dfALU', 1614, 1608, 1592], ['dfALU', 1614, 1610, 1593], ['dfALU', 1614, 1612, 1594], ['L1_ACCESS'], ['dfALU', 1619, 1605, 1596], ['dfALU', 1619, 1608, 1597], ['dfALU', 1619, 1610, 1598], ['dfALU', 1619, 1612, 1599], ['L1_ACCESS'], ['dfALU', 1624, 1605, 1601], ['dfALU', 1624, 1608, 1602], ['dfALU', 1624, 1610, 1603], ['dfALU', 1624, 1612, 1604], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1630, 1629, 1607], ['L1_ACCESS'], ['dfALU', 1630, 1632, 1609], ['L1_ACCESS'], ['dfALU', 1630, 1634, 1611], ['L1_ACCESS'], ['dfALU', 1630, 1636, 1613], ['L1_ACCESS'], ['dfALU', 1638, 1629, 1615], ['dfALU', 1638, 1632, 1616], ['dfALU', 1638, 1634, 1617], ['dfALU', 1638, 1636, 1618], ['L1_ACCESS'], ['dfALU', 1643, 1629, 1620], ['dfALU', 1643, 1632, 1621], ['dfALU', 1643, 1634, 1622], ['dfALU', 1643, 1636, 1623], ['L1_ACCESS'], ['dfALU', 1648, 1629, 1625], ['dfALU', 1648, 1632, 1626], ['dfALU', 1648, 1634, 1627], ['dfALU', 1648, 1636, 1628], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1654, 1653, 1631], ['L1_ACCESS'], ['dfALU', 1654, 1656, 1633], ['L1_ACCESS'], ['dfALU', 1654, 1658, 1635], ['L1_ACCESS'], ['dfALU', 1654, 1660, 1637], ['L1_ACCESS'], ['dfALU', 1662, 1653, 1639], ['dfALU', 1662, 1656, 1640], ['dfALU', 1662, 1658, 1641], ['dfALU', 1662, 1660, 1642], ['L1_ACCESS'], ['dfALU', 1667, 1653, 1644], ['dfALU', 1667, 1656, 1645], ['dfALU', 1667, 1658, 1646], ['dfALU', 1667, 1660, 1647], ['L1_ACCESS'], ['dfALU', 1672, 1653, 1649], ['dfALU', 1672, 1656, 1650], ['dfALU', 1672, 1658, 1651], ['dfALU', 1672, 1660, 1652], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1678, 1677, 1655], ['L1_ACCESS'], ['dfALU', 1678, 1680, 1657], ['L1_ACCESS'], ['dfALU', 1678, 1682, 1659], ['L1_ACCESS'], ['dfALU', 1678, 1684, 1661], ['L1_ACCESS'], ['dfALU', 1686, 1677, 1663], ['dfALU', 1686, 1680, 1664], ['dfALU', 1686, 1682, 1665], ['dfALU', 1686, 1684, 1666], ['L1_ACCESS'], ['dfALU', 1691, 1677, 1668], ['dfALU', 1691, 1680, 1669], ['dfALU', 1691, 1682, 1670], ['dfALU', 1691, 1684, 1671], ['L1_ACCESS'], ['dfALU', 1696, 1677, 1673], ['dfALU', 1696, 1680, 1674], ['dfALU', 1696, 1682, 1675], ['dfALU', 1696, 1684, 1676], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1702, 1701, 1679], ['L1_ACCESS'], ['dfALU', 1702, 1704, 1681], ['L1_ACCESS'], ['dfALU', 1702, 1706, 1683], ['L1_ACCESS'], ['dfALU', 1702, 1708, 1685], ['L1_ACCESS'], ['dfALU', 1710, 1701, 1687], ['dfALU', 1710, 1704, 1688], ['dfALU', 1710, 1706, 1689], ['dfALU', 1710, 1708, 1690], ['L1_ACCESS'], ['dfALU', 1715, 1701, 1692], ['dfALU', 1715, 1704, 1693], ['dfALU', 1715, 1706, 1694], ['dfALU', 1715, 1708, 1695], ['L1_ACCESS'], ['dfALU', 1720, 1701, 1697], ['dfALU', 1720, 1704, 1698], ['dfALU', 1720, 1706, 1699], ['dfALU', 1720, 1708, 1700], ['THREAD_SYNC'], ['L1_ACCESS', 1325], ['L1_ACCESS', 1327], ['L1_ACCESS', 1329], ['L1_ACCESS', 1331], ['L1_ACCESS', 1332], ['L1_ACCESS', 1335], ['L1_ACCESS', 1338], ['L1_ACCESS', 1340], ['THREAD_SYNC'], ['iALU', 1319], ['iALU', 1735, 65], ['iALU', 1321], ['iALU', 1737], ['iALU', 71, 1738], ['iALU', 69, 1737, 70], ['L2_ACCESS'], ['iALU', 1740], ['L2_ACCESS'], ['iALU', 1740], ['L2_ACCESS'], ['iALU', 1740], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 1739], ['iALU', 5, 1739], ['L2_ACCESS'], ['iALU', 5, 1749], ['iALU', 73, 1749], ['L2_ACCESS'], ['iALU', 73, 1752], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 1758, 1757, 1703], ['L1_ACCESS'], ['dfALU', 1758, 1760, 1705], ['L1_ACCESS'], ['dfALU', 1758, 1762, 1707], ['L1_ACCESS'], ['dfALU', 1758, 1764, 1709], ['L1_ACCESS'], ['dfALU', 1766, 1757, 1711], ['dfALU', 1766, 1760, 1712], ['dfALU', 1766, 1762, 1713], ['dfALU', 1766, 1764, 1714], ['L1_ACCESS'], ['dfALU', 1771, 1757, 1716], ['dfALU', 1771, 1760, 1717], ['dfALU', 1771, 1762, 1718], ['dfALU', 1771, 1764, 1719], ['L1_ACCESS'], ['dfALU', 1776, 1757, 1721], ['dfALU', 1776, 1760, 1722], ['dfALU', 1776, 1762, 1723], ['dfALU', 1776, 1764, 1724], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1782, 1781, 1759], ['L1_ACCESS'], ['dfALU', 1782, 1784, 1761], ['L1_ACCESS'], ['dfALU', 1782, 1786, 1763], ['L1_ACCESS'], ['dfALU', 1782, 1788, 1765], ['L1_ACCESS'], ['dfALU', 1790, 1781, 1767], ['dfALU', 1790, 1784, 1768], ['dfALU', 1790, 1786, 1769], ['dfALU', 1790, 1788, 1770], ['L1_ACCESS'], ['dfALU', 1795, 1781, 1772], ['dfALU', 1795, 1784, 1773], ['dfALU', 1795, 1786, 1774], ['dfALU', 1795, 1788, 1775], ['L1_ACCESS'], ['dfALU', 1800, 1781, 1777], ['dfALU', 1800, 1784, 1778], ['dfALU', 1800, 1786, 1779], ['dfALU', 1800, 1788, 1780], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1806, 1805, 1783], ['L1_ACCESS'], ['dfALU', 1806, 1808, 1785], ['L1_ACCESS'], ['dfALU', 1806, 1810, 1787], ['L1_ACCESS'], ['dfALU', 1806, 1812, 1789], ['L1_ACCESS'], ['dfALU', 1814, 1805, 1791], ['dfALU', 1814, 1808, 1792], ['dfALU', 1814, 1810, 1793], ['dfALU', 1814, 1812, 1794], ['L1_ACCESS'], ['dfALU', 1819, 1805, 1796], ['dfALU', 1819, 1808, 1797], ['dfALU', 1819, 1810, 1798], ['dfALU', 1819, 1812, 1799], ['L1_ACCESS'], ['dfALU', 1824, 1805, 1801], ['dfALU', 1824, 1808, 1802], ['dfALU', 1824, 1810, 1803], ['dfALU', 1824, 1812, 1804], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1830, 1829, 1807], ['L1_ACCESS'], ['dfALU', 1830, 1832, 1809], ['L1_ACCESS'], ['dfALU', 1830, 1834, 1811], ['L1_ACCESS'], ['dfALU', 1830, 1836, 1813], ['L1_ACCESS'], ['dfALU', 1838, 1829, 1815], ['dfALU', 1838, 1832, 1816], ['dfALU', 1838, 1834, 1817], ['dfALU', 1838, 1836, 1818], ['L1_ACCESS'], ['dfALU', 1843, 1829, 1820], ['dfALU', 1843, 1832, 1821], ['dfALU', 1843, 1834, 1822], ['dfALU', 1843, 1836, 1823], ['L1_ACCESS'], ['dfALU', 1848, 1829, 1825], ['dfALU', 1848, 1832, 1826], ['dfALU', 1848, 1834, 1827], ['dfALU', 1848, 1836, 1828], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1854, 1853, 1831], ['L1_ACCESS'], ['dfALU', 1854, 1856, 1833], ['L1_ACCESS'], ['dfALU', 1854, 1858, 1835], ['L1_ACCESS'], ['dfALU', 1854, 1860, 1837], ['L1_ACCESS'], ['dfALU', 1862, 1853, 1839], ['dfALU', 1862, 1856, 1840], ['dfALU', 1862, 1858, 1841], ['dfALU', 1862, 1860, 1842], ['L1_ACCESS'], ['dfALU', 1867, 1853, 1844], ['dfALU', 1867, 1856, 1845], ['dfALU', 1867, 1858, 1846], ['dfALU', 1867, 1860, 1847], ['L1_ACCESS'], ['dfALU', 1872, 1853, 1849], ['dfALU', 1872, 1856, 1850], ['dfALU', 1872, 1858, 1851], ['dfALU', 1872, 1860, 1852], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1878, 1877, 1855], ['L1_ACCESS'], ['dfALU', 1878, 1880, 1857], ['L1_ACCESS'], ['dfALU', 1878, 1882, 1859], ['L1_ACCESS'], ['dfALU', 1878, 1884, 1861], ['L1_ACCESS'], ['dfALU', 1886, 1877, 1863], ['dfALU', 1886, 1880, 1864], ['dfALU', 1886, 1882, 1865], ['dfALU', 1886, 1884, 1866], ['L1_ACCESS'], ['dfALU', 1891, 1877, 1868], ['dfALU', 1891, 1880, 1869], ['dfALU', 1891, 1882, 1870], ['dfALU', 1891, 1884, 1871], ['L1_ACCESS'], ['dfALU', 1896, 1877, 1873], ['dfALU', 1896, 1880, 1874], ['dfALU', 1896, 1882, 1875], ['dfALU', 1896, 1884, 1876], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1902, 1901, 1879], ['L1_ACCESS'], ['dfALU', 1902, 1904, 1881], ['L1_ACCESS'], ['dfALU', 1902, 1906, 1883], ['L1_ACCESS'], ['dfALU', 1902, 1908, 1885], ['L1_ACCESS'], ['dfALU', 1910, 1901, 1887], ['dfALU', 1910, 1904, 1888], ['dfALU', 1910, 1906, 1889], ['dfALU', 1910, 1908, 1890], ['L1_ACCESS'], ['dfALU', 1915, 1901, 1892], ['dfALU', 1915, 1904, 1893], ['dfALU', 1915, 1906, 1894], ['dfALU', 1915, 1908, 1895], ['L1_ACCESS'], ['dfALU', 1920, 1901, 1897], ['dfALU', 1920, 1904, 1898], ['dfALU', 1920, 1906, 1899], ['dfALU', 1920, 1908, 1900], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1926, 1925, 1903], ['L1_ACCESS'], ['dfALU', 1926, 1928, 1905], ['L1_ACCESS'], ['dfALU', 1926, 1930, 1907], ['L1_ACCESS'], ['dfALU', 1926, 1932, 1909], ['L1_ACCESS'], ['dfALU', 1934, 1925, 1911], ['dfALU', 1934, 1928, 1912], ['dfALU', 1934, 1930, 1913], ['dfALU', 1934, 1932, 1914], ['L1_ACCESS'], ['dfALU', 1939, 1925, 1916], ['dfALU', 1939, 1928, 1917], ['dfALU', 1939, 1930, 1918], ['dfALU', 1939, 1932, 1919], ['L1_ACCESS'], ['dfALU', 1944, 1925, 1921], ['dfALU', 1944, 1928, 1922], ['dfALU', 1944, 1930, 1923], ['dfALU', 1944, 1932, 1924], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1950, 1949, 1927], ['L1_ACCESS'], ['dfALU', 1950, 1952, 1929], ['L1_ACCESS'], ['dfALU', 1950, 1954, 1931], ['L1_ACCESS'], ['dfALU', 1950, 1956, 1933], ['L1_ACCESS'], ['dfALU', 1958, 1949, 1935], ['dfALU', 1958, 1952, 1936], ['dfALU', 1958, 1954, 1937], ['dfALU', 1958, 1956, 1938], ['L1_ACCESS'], ['dfALU', 1963, 1949, 1940], ['dfALU', 1963, 1952, 1941], ['dfALU', 1963, 1954, 1942], ['dfALU', 1963, 1956, 1943], ['L1_ACCESS'], ['dfALU', 1968, 1949, 1945], ['dfALU', 1968, 1952, 1946], ['dfALU', 1968, 1954, 1947], ['dfALU', 1968, 1956, 1948], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1974, 1973, 1951], ['L1_ACCESS'], ['dfALU', 1974, 1976, 1953], ['L1_ACCESS'], ['dfALU', 1974, 1978, 1955], ['L1_ACCESS'], ['dfALU', 1974, 1980, 1957], ['L1_ACCESS'], ['dfALU', 1982, 1973, 1959], ['dfALU', 1982, 1976, 1960], ['dfALU', 1982, 1978, 1961], ['dfALU', 1982, 1980, 1962], ['L1_ACCESS'], ['dfALU', 1987, 1973, 1964], ['dfALU', 1987, 1976, 1965], ['dfALU', 1987, 1978, 1966], ['dfALU', 1987, 1980, 1967], ['L1_ACCESS'], ['dfALU', 1992, 1973, 1969], ['dfALU', 1992, 1976, 1970], ['dfALU', 1992, 1978, 1971], ['dfALU', 1992, 1980, 1972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 1998, 1997, 1975], ['L1_ACCESS'], ['dfALU', 1998, 2000, 1977], ['L1_ACCESS'], ['dfALU', 1998, 2002, 1979], ['L1_ACCESS'], ['dfALU', 1998, 2004, 1981], ['L1_ACCESS'], ['dfALU', 2006, 1997, 1983], ['dfALU', 2006, 2000, 1984], ['dfALU', 2006, 2002, 1985], ['dfALU', 2006, 2004, 1986], ['L1_ACCESS'], ['dfALU', 2011, 1997, 1988], ['dfALU', 2011, 2000, 1989], ['dfALU', 2011, 2002, 1990], ['dfALU', 2011, 2004, 1991], ['L1_ACCESS'], ['dfALU', 2016, 1997, 1993], ['dfALU', 2016, 2000, 1994], ['dfALU', 2016, 2002, 1995], ['dfALU', 2016, 2004, 1996], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2022, 2021, 1999], ['L1_ACCESS'], ['dfALU', 2022, 2024, 2001], ['L1_ACCESS'], ['dfALU', 2022, 2026, 2003], ['L1_ACCESS'], ['dfALU', 2022, 2028, 2005], ['L1_ACCESS'], ['dfALU', 2030, 2021, 2007], ['dfALU', 2030, 2024, 2008], ['dfALU', 2030, 2026, 2009], ['dfALU', 2030, 2028, 2010], ['L1_ACCESS'], ['dfALU', 2035, 2021, 2012], ['dfALU', 2035, 2024, 2013], ['dfALU', 2035, 2026, 2014], ['dfALU', 2035, 2028, 2015], ['L1_ACCESS'], ['dfALU', 2040, 2021, 2017], ['dfALU', 2040, 2024, 2018], ['dfALU', 2040, 2026, 2019], ['dfALU', 2040, 2028, 2020], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2046, 2045, 2023], ['L1_ACCESS'], ['dfALU', 2046, 2048, 2025], ['L1_ACCESS'], ['dfALU', 2046, 2050, 2027], ['L1_ACCESS'], ['dfALU', 2046, 2052, 2029], ['L1_ACCESS'], ['dfALU', 2054, 2045, 2031], ['dfALU', 2054, 2048, 2032], ['dfALU', 2054, 2050, 2033], ['dfALU', 2054, 2052, 2034], ['L1_ACCESS'], ['dfALU', 2059, 2045, 2036], ['dfALU', 2059, 2048, 2037], ['dfALU', 2059, 2050, 2038], ['dfALU', 2059, 2052, 2039], ['L1_ACCESS'], ['dfALU', 2064, 2045, 2041], ['dfALU', 2064, 2048, 2042], ['dfALU', 2064, 2050, 2043], ['dfALU', 2064, 2052, 2044], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2070, 2069, 2047], ['L1_ACCESS'], ['dfALU', 2070, 2072, 2049], ['L1_ACCESS'], ['dfALU', 2070, 2074, 2051], ['L1_ACCESS'], ['dfALU', 2070, 2076, 2053], ['L1_ACCESS'], ['dfALU', 2078, 2069, 2055], ['dfALU', 2078, 2072, 2056], ['dfALU', 2078, 2074, 2057], ['dfALU', 2078, 2076, 2058], ['L1_ACCESS'], ['dfALU', 2083, 2069, 2060], ['dfALU', 2083, 2072, 2061], ['dfALU', 2083, 2074, 2062], ['dfALU', 2083, 2076, 2063], ['L1_ACCESS'], ['dfALU', 2088, 2069, 2065], ['dfALU', 2088, 2072, 2066], ['dfALU', 2088, 2074, 2067], ['dfALU', 2088, 2076, 2068], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2094, 2093, 2071], ['L1_ACCESS'], ['dfALU', 2094, 2096, 2073], ['L1_ACCESS'], ['dfALU', 2094, 2098, 2075], ['L1_ACCESS'], ['dfALU', 2094, 2100, 2077], ['L1_ACCESS'], ['dfALU', 2102, 2093, 2079], ['dfALU', 2102, 2096, 2080], ['dfALU', 2102, 2098, 2081], ['dfALU', 2102, 2100, 2082], ['L1_ACCESS'], ['dfALU', 2107, 2093, 2084], ['dfALU', 2107, 2096, 2085], ['dfALU', 2107, 2098, 2086], ['dfALU', 2107, 2100, 2087], ['L1_ACCESS'], ['dfALU', 2112, 2093, 2089], ['dfALU', 2112, 2096, 2090], ['dfALU', 2112, 2098, 2091], ['dfALU', 2112, 2100, 2092], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2118, 2117, 2095], ['L1_ACCESS'], ['dfALU', 2118, 2120, 2097], ['L1_ACCESS'], ['dfALU', 2118, 2122, 2099], ['L1_ACCESS'], ['dfALU', 2118, 2124, 2101], ['L1_ACCESS'], ['dfALU', 2126, 2117, 2103], ['dfALU', 2126, 2120, 2104], ['dfALU', 2126, 2122, 2105], ['dfALU', 2126, 2124, 2106], ['L1_ACCESS'], ['dfALU', 2131, 2117, 2108], ['dfALU', 2131, 2120, 2109], ['dfALU', 2131, 2122, 2110], ['dfALU', 2131, 2124, 2111], ['L1_ACCESS'], ['dfALU', 2136, 2117, 2113], ['dfALU', 2136, 2120, 2114], ['dfALU', 2136, 2122, 2115], ['dfALU', 2136, 2124, 2116], ['THREAD_SYNC'], ['L1_ACCESS', 1741], ['L1_ACCESS', 1743], ['L1_ACCESS', 1745], ['L1_ACCESS', 1747], ['L1_ACCESS', 1748], ['L1_ACCESS', 1751], ['L1_ACCESS', 1754], ['L1_ACCESS', 1756], ['THREAD_SYNC'], ['iALU', 1735], ['iALU', 2151, 65], ['iALU', 1737], ['iALU', 2153], ['iALU', 71, 2154], ['iALU', 69, 2153, 70], ['L2_ACCESS'], ['iALU', 2156], ['L2_ACCESS'], ['iALU', 2156], ['L2_ACCESS'], ['iALU', 2156], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 2155], ['iALU', 5, 2155], ['L2_ACCESS'], ['iALU', 5, 2165], ['iALU', 73, 2165], ['L2_ACCESS'], ['iALU', 73, 2168], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 2174, 2173, 2119], ['L1_ACCESS'], ['dfALU', 2174, 2176, 2121], ['L1_ACCESS'], ['dfALU', 2174, 2178, 2123], ['L1_ACCESS'], ['dfALU', 2174, 2180, 2125], ['L1_ACCESS'], ['dfALU', 2182, 2173, 2127], ['dfALU', 2182, 2176, 2128], ['dfALU', 2182, 2178, 2129], ['dfALU', 2182, 2180, 2130], ['L1_ACCESS'], ['dfALU', 2187, 2173, 2132], ['dfALU', 2187, 2176, 2133], ['dfALU', 2187, 2178, 2134], ['dfALU', 2187, 2180, 2135], ['L1_ACCESS'], ['dfALU', 2192, 2173, 2137], ['dfALU', 2192, 2176, 2138], ['dfALU', 2192, 2178, 2139], ['dfALU', 2192, 2180, 2140], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2198, 2197, 2175], ['L1_ACCESS'], ['dfALU', 2198, 2200, 2177], ['L1_ACCESS'], ['dfALU', 2198, 2202, 2179], ['L1_ACCESS'], ['dfALU', 2198, 2204, 2181], ['L1_ACCESS'], ['dfALU', 2206, 2197, 2183], ['dfALU', 2206, 2200, 2184], ['dfALU', 2206, 2202, 2185], ['dfALU', 2206, 2204, 2186], ['L1_ACCESS'], ['dfALU', 2211, 2197, 2188], ['dfALU', 2211, 2200, 2189], ['dfALU', 2211, 2202, 2190], ['dfALU', 2211, 2204, 2191], ['L1_ACCESS'], ['dfALU', 2216, 2197, 2193], ['dfALU', 2216, 2200, 2194], ['dfALU', 2216, 2202, 2195], ['dfALU', 2216, 2204, 2196], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2222, 2221, 2199], ['L1_ACCESS'], ['dfALU', 2222, 2224, 2201], ['L1_ACCESS'], ['dfALU', 2222, 2226, 2203], ['L1_ACCESS'], ['dfALU', 2222, 2228, 2205], ['L1_ACCESS'], ['dfALU', 2230, 2221, 2207], ['dfALU', 2230, 2224, 2208], ['dfALU', 2230, 2226, 2209], ['dfALU', 2230, 2228, 2210], ['L1_ACCESS'], ['dfALU', 2235, 2221, 2212], ['dfALU', 2235, 2224, 2213], ['dfALU', 2235, 2226, 2214], ['dfALU', 2235, 2228, 2215], ['L1_ACCESS'], ['dfALU', 2240, 2221, 2217], ['dfALU', 2240, 2224, 2218], ['dfALU', 2240, 2226, 2219], ['dfALU', 2240, 2228, 2220], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2246, 2245, 2223], ['L1_ACCESS'], ['dfALU', 2246, 2248, 2225], ['L1_ACCESS'], ['dfALU', 2246, 2250, 2227], ['L1_ACCESS'], ['dfALU', 2246, 2252, 2229], ['L1_ACCESS'], ['dfALU', 2254, 2245, 2231], ['dfALU', 2254, 2248, 2232], ['dfALU', 2254, 2250, 2233], ['dfALU', 2254, 2252, 2234], ['L1_ACCESS'], ['dfALU', 2259, 2245, 2236], ['dfALU', 2259, 2248, 2237], ['dfALU', 2259, 2250, 2238], ['dfALU', 2259, 2252, 2239], ['L1_ACCESS'], ['dfALU', 2264, 2245, 2241], ['dfALU', 2264, 2248, 2242], ['dfALU', 2264, 2250, 2243], ['dfALU', 2264, 2252, 2244], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2270, 2269, 2247], ['L1_ACCESS'], ['dfALU', 2270, 2272, 2249], ['L1_ACCESS'], ['dfALU', 2270, 2274, 2251], ['L1_ACCESS'], ['dfALU', 2270, 2276, 2253], ['L1_ACCESS'], ['dfALU', 2278, 2269, 2255], ['dfALU', 2278, 2272, 2256], ['dfALU', 2278, 2274, 2257], ['dfALU', 2278, 2276, 2258], ['L1_ACCESS'], ['dfALU', 2283, 2269, 2260], ['dfALU', 2283, 2272, 2261], ['dfALU', 2283, 2274, 2262], ['dfALU', 2283, 2276, 2263], ['L1_ACCESS'], ['dfALU', 2288, 2269, 2265], ['dfALU', 2288, 2272, 2266], ['dfALU', 2288, 2274, 2267], ['dfALU', 2288, 2276, 2268], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2294, 2293, 2271], ['L1_ACCESS'], ['dfALU', 2294, 2296, 2273], ['L1_ACCESS'], ['dfALU', 2294, 2298, 2275], ['L1_ACCESS'], ['dfALU', 2294, 2300, 2277], ['L1_ACCESS'], ['dfALU', 2302, 2293, 2279], ['dfALU', 2302, 2296, 2280], ['dfALU', 2302, 2298, 2281], ['dfALU', 2302, 2300, 2282], ['L1_ACCESS'], ['dfALU', 2307, 2293, 2284], ['dfALU', 2307, 2296, 2285], ['dfALU', 2307, 2298, 2286], ['dfALU', 2307, 2300, 2287], ['L1_ACCESS'], ['dfALU', 2312, 2293, 2289], ['dfALU', 2312, 2296, 2290], ['dfALU', 2312, 2298, 2291], ['dfALU', 2312, 2300, 2292], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2318, 2317, 2295], ['L1_ACCESS'], ['dfALU', 2318, 2320, 2297], ['L1_ACCESS'], ['dfALU', 2318, 2322, 2299], ['L1_ACCESS'], ['dfALU', 2318, 2324, 2301], ['L1_ACCESS'], ['dfALU', 2326, 2317, 2303], ['dfALU', 2326, 2320, 2304], ['dfALU', 2326, 2322, 2305], ['dfALU', 2326, 2324, 2306], ['L1_ACCESS'], ['dfALU', 2331, 2317, 2308], ['dfALU', 2331, 2320, 2309], ['dfALU', 2331, 2322, 2310], ['dfALU', 2331, 2324, 2311], ['L1_ACCESS'], ['dfALU', 2336, 2317, 2313], ['dfALU', 2336, 2320, 2314], ['dfALU', 2336, 2322, 2315], ['dfALU', 2336, 2324, 2316], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2342, 2341, 2319], ['L1_ACCESS'], ['dfALU', 2342, 2344, 2321], ['L1_ACCESS'], ['dfALU', 2342, 2346, 2323], ['L1_ACCESS'], ['dfALU', 2342, 2348, 2325], ['L1_ACCESS'], ['dfALU', 2350, 2341, 2327], ['dfALU', 2350, 2344, 2328], ['dfALU', 2350, 2346, 2329], ['dfALU', 2350, 2348, 2330], ['L1_ACCESS'], ['dfALU', 2355, 2341, 2332], ['dfALU', 2355, 2344, 2333], ['dfALU', 2355, 2346, 2334], ['dfALU', 2355, 2348, 2335], ['L1_ACCESS'], ['dfALU', 2360, 2341, 2337], ['dfALU', 2360, 2344, 2338], ['dfALU', 2360, 2346, 2339], ['dfALU', 2360, 2348, 2340], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2366, 2365, 2343], ['L1_ACCESS'], ['dfALU', 2366, 2368, 2345], ['L1_ACCESS'], ['dfALU', 2366, 2370, 2347], ['L1_ACCESS'], ['dfALU', 2366, 2372, 2349], ['L1_ACCESS'], ['dfALU', 2374, 2365, 2351], ['dfALU', 2374, 2368, 2352], ['dfALU', 2374, 2370, 2353], ['dfALU', 2374, 2372, 2354], ['L1_ACCESS'], ['dfALU', 2379, 2365, 2356], ['dfALU', 2379, 2368, 2357], ['dfALU', 2379, 2370, 2358], ['dfALU', 2379, 2372, 2359], ['L1_ACCESS'], ['dfALU', 2384, 2365, 2361], ['dfALU', 2384, 2368, 2362], ['dfALU', 2384, 2370, 2363], ['dfALU', 2384, 2372, 2364], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2390, 2389, 2367], ['L1_ACCESS'], ['dfALU', 2390, 2392, 2369], ['L1_ACCESS'], ['dfALU', 2390, 2394, 2371], ['L1_ACCESS'], ['dfALU', 2390, 2396, 2373], ['L1_ACCESS'], ['dfALU', 2398, 2389, 2375], ['dfALU', 2398, 2392, 2376], ['dfALU', 2398, 2394, 2377], ['dfALU', 2398, 2396, 2378], ['L1_ACCESS'], ['dfALU', 2403, 2389, 2380], ['dfALU', 2403, 2392, 2381], ['dfALU', 2403, 2394, 2382], ['dfALU', 2403, 2396, 2383], ['L1_ACCESS'], ['dfALU', 2408, 2389, 2385], ['dfALU', 2408, 2392, 2386], ['dfALU', 2408, 2394, 2387], ['dfALU', 2408, 2396, 2388], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2414, 2413, 2391], ['L1_ACCESS'], ['dfALU', 2414, 2416, 2393], ['L1_ACCESS'], ['dfALU', 2414, 2418, 2395], ['L1_ACCESS'], ['dfALU', 2414, 2420, 2397], ['L1_ACCESS'], ['dfALU', 2422, 2413, 2399], ['dfALU', 2422, 2416, 2400], ['dfALU', 2422, 2418, 2401], ['dfALU', 2422, 2420, 2402], ['L1_ACCESS'], ['dfALU', 2427, 2413, 2404], ['dfALU', 2427, 2416, 2405], ['dfALU', 2427, 2418, 2406], ['dfALU', 2427, 2420, 2407], ['L1_ACCESS'], ['dfALU', 2432, 2413, 2409], ['dfALU', 2432, 2416, 2410], ['dfALU', 2432, 2418, 2411], ['dfALU', 2432, 2420, 2412], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2438, 2437, 2415], ['L1_ACCESS'], ['dfALU', 2438, 2440, 2417], ['L1_ACCESS'], ['dfALU', 2438, 2442, 2419], ['L1_ACCESS'], ['dfALU', 2438, 2444, 2421], ['L1_ACCESS'], ['dfALU', 2446, 2437, 2423], ['dfALU', 2446, 2440, 2424], ['dfALU', 2446, 2442, 2425], ['dfALU', 2446, 2444, 2426], ['L1_ACCESS'], ['dfALU', 2451, 2437, 2428], ['dfALU', 2451, 2440, 2429], ['dfALU', 2451, 2442, 2430], ['dfALU', 2451, 2444, 2431], ['L1_ACCESS'], ['dfALU', 2456, 2437, 2433], ['dfALU', 2456, 2440, 2434], ['dfALU', 2456, 2442, 2435], ['dfALU', 2456, 2444, 2436], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2462, 2461, 2439], ['L1_ACCESS'], ['dfALU', 2462, 2464, 2441], ['L1_ACCESS'], ['dfALU', 2462, 2466, 2443], ['L1_ACCESS'], ['dfALU', 2462, 2468, 2445], ['L1_ACCESS'], ['dfALU', 2470, 2461, 2447], ['dfALU', 2470, 2464, 2448], ['dfALU', 2470, 2466, 2449], ['dfALU', 2470, 2468, 2450], ['L1_ACCESS'], ['dfALU', 2475, 2461, 2452], ['dfALU', 2475, 2464, 2453], ['dfALU', 2475, 2466, 2454], ['dfALU', 2475, 2468, 2455], ['L1_ACCESS'], ['dfALU', 2480, 2461, 2457], ['dfALU', 2480, 2464, 2458], ['dfALU', 2480, 2466, 2459], ['dfALU', 2480, 2468, 2460], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2486, 2485, 2463], ['L1_ACCESS'], ['dfALU', 2486, 2488, 2465], ['L1_ACCESS'], ['dfALU', 2486, 2490, 2467], ['L1_ACCESS'], ['dfALU', 2486, 2492, 2469], ['L1_ACCESS'], ['dfALU', 2494, 2485, 2471], ['dfALU', 2494, 2488, 2472], ['dfALU', 2494, 2490, 2473], ['dfALU', 2494, 2492, 2474], ['L1_ACCESS'], ['dfALU', 2499, 2485, 2476], ['dfALU', 2499, 2488, 2477], ['dfALU', 2499, 2490, 2478], ['dfALU', 2499, 2492, 2479], ['L1_ACCESS'], ['dfALU', 2504, 2485, 2481], ['dfALU', 2504, 2488, 2482], ['dfALU', 2504, 2490, 2483], ['dfALU', 2504, 2492, 2484], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2510, 2509, 2487], ['L1_ACCESS'], ['dfALU', 2510, 2512, 2489], ['L1_ACCESS'], ['dfALU', 2510, 2514, 2491], ['L1_ACCESS'], ['dfALU', 2510, 2516, 2493], ['L1_ACCESS'], ['dfALU', 2518, 2509, 2495], ['dfALU', 2518, 2512, 2496], ['dfALU', 2518, 2514, 2497], ['dfALU', 2518, 2516, 2498], ['L1_ACCESS'], ['dfALU', 2523, 2509, 2500], ['dfALU', 2523, 2512, 2501], ['dfALU', 2523, 2514, 2502], ['dfALU', 2523, 2516, 2503], ['L1_ACCESS'], ['dfALU', 2528, 2509, 2505], ['dfALU', 2528, 2512, 2506], ['dfALU', 2528, 2514, 2507], ['dfALU', 2528, 2516, 2508], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2534, 2533, 2511], ['L1_ACCESS'], ['dfALU', 2534, 2536, 2513], ['L1_ACCESS'], ['dfALU', 2534, 2538, 2515], ['L1_ACCESS'], ['dfALU', 2534, 2540, 2517], ['L1_ACCESS'], ['dfALU', 2542, 2533, 2519], ['dfALU', 2542, 2536, 2520], ['dfALU', 2542, 2538, 2521], ['dfALU', 2542, 2540, 2522], ['L1_ACCESS'], ['dfALU', 2547, 2533, 2524], ['dfALU', 2547, 2536, 2525], ['dfALU', 2547, 2538, 2526], ['dfALU', 2547, 2540, 2527], ['L1_ACCESS'], ['dfALU', 2552, 2533, 2529], ['dfALU', 2552, 2536, 2530], ['dfALU', 2552, 2538, 2531], ['dfALU', 2552, 2540, 2532], ['THREAD_SYNC'], ['L1_ACCESS', 2157], ['L1_ACCESS', 2159], ['L1_ACCESS', 2161], ['L1_ACCESS', 2163], ['L1_ACCESS', 2164], ['L1_ACCESS', 2167], ['L1_ACCESS', 2170], ['L1_ACCESS', 2172], ['THREAD_SYNC'], ['iALU', 2151], ['iALU', 2567, 65], ['iALU', 2153], ['iALU', 2569], ['iALU', 71, 2570], ['iALU', 69, 2569, 70], ['L2_ACCESS'], ['iALU', 2572], ['L2_ACCESS'], ['iALU', 2572], ['L2_ACCESS'], ['iALU', 2572], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 2571], ['iALU', 5, 2571], ['L2_ACCESS'], ['iALU', 5, 2581], ['iALU', 73, 2581], ['L2_ACCESS'], ['iALU', 73, 2584], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 2590, 2589, 2535], ['L1_ACCESS'], ['dfALU', 2590, 2592, 2537], ['L1_ACCESS'], ['dfALU', 2590, 2594, 2539], ['L1_ACCESS'], ['dfALU', 2590, 2596, 2541], ['L1_ACCESS'], ['dfALU', 2598, 2589, 2543], ['dfALU', 2598, 2592, 2544], ['dfALU', 2598, 2594, 2545], ['dfALU', 2598, 2596, 2546], ['L1_ACCESS'], ['dfALU', 2603, 2589, 2548], ['dfALU', 2603, 2592, 2549], ['dfALU', 2603, 2594, 2550], ['dfALU', 2603, 2596, 2551], ['L1_ACCESS'], ['dfALU', 2608, 2589, 2553], ['dfALU', 2608, 2592, 2554], ['dfALU', 2608, 2594, 2555], ['dfALU', 2608, 2596, 2556], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2614, 2613, 2591], ['L1_ACCESS'], ['dfALU', 2614, 2616, 2593], ['L1_ACCESS'], ['dfALU', 2614, 2618, 2595], ['L1_ACCESS'], ['dfALU', 2614, 2620, 2597], ['L1_ACCESS'], ['dfALU', 2622, 2613, 2599], ['dfALU', 2622, 2616, 2600], ['dfALU', 2622, 2618, 2601], ['dfALU', 2622, 2620, 2602], ['L1_ACCESS'], ['dfALU', 2627, 2613, 2604], ['dfALU', 2627, 2616, 2605], ['dfALU', 2627, 2618, 2606], ['dfALU', 2627, 2620, 2607], ['L1_ACCESS'], ['dfALU', 2632, 2613, 2609], ['dfALU', 2632, 2616, 2610], ['dfALU', 2632, 2618, 2611], ['dfALU', 2632, 2620, 2612], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2638, 2637, 2615], ['L1_ACCESS'], ['dfALU', 2638, 2640, 2617], ['L1_ACCESS'], ['dfALU', 2638, 2642, 2619], ['L1_ACCESS'], ['dfALU', 2638, 2644, 2621], ['L1_ACCESS'], ['dfALU', 2646, 2637, 2623], ['dfALU', 2646, 2640, 2624], ['dfALU', 2646, 2642, 2625], ['dfALU', 2646, 2644, 2626], ['L1_ACCESS'], ['dfALU', 2651, 2637, 2628], ['dfALU', 2651, 2640, 2629], ['dfALU', 2651, 2642, 2630], ['dfALU', 2651, 2644, 2631], ['L1_ACCESS'], ['dfALU', 2656, 2637, 2633], ['dfALU', 2656, 2640, 2634], ['dfALU', 2656, 2642, 2635], ['dfALU', 2656, 2644, 2636], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2662, 2661, 2639], ['L1_ACCESS'], ['dfALU', 2662, 2664, 2641], ['L1_ACCESS'], ['dfALU', 2662, 2666, 2643], ['L1_ACCESS'], ['dfALU', 2662, 2668, 2645], ['L1_ACCESS'], ['dfALU', 2670, 2661, 2647], ['dfALU', 2670, 2664, 2648], ['dfALU', 2670, 2666, 2649], ['dfALU', 2670, 2668, 2650], ['L1_ACCESS'], ['dfALU', 2675, 2661, 2652], ['dfALU', 2675, 2664, 2653], ['dfALU', 2675, 2666, 2654], ['dfALU', 2675, 2668, 2655], ['L1_ACCESS'], ['dfALU', 2680, 2661, 2657], ['dfALU', 2680, 2664, 2658], ['dfALU', 2680, 2666, 2659], ['dfALU', 2680, 2668, 2660], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2686, 2685, 2663], ['L1_ACCESS'], ['dfALU', 2686, 2688, 2665], ['L1_ACCESS'], ['dfALU', 2686, 2690, 2667], ['L1_ACCESS'], ['dfALU', 2686, 2692, 2669], ['L1_ACCESS'], ['dfALU', 2694, 2685, 2671], ['dfALU', 2694, 2688, 2672], ['dfALU', 2694, 2690, 2673], ['dfALU', 2694, 2692, 2674], ['L1_ACCESS'], ['dfALU', 2699, 2685, 2676], ['dfALU', 2699, 2688, 2677], ['dfALU', 2699, 2690, 2678], ['dfALU', 2699, 2692, 2679], ['L1_ACCESS'], ['dfALU', 2704, 2685, 2681], ['dfALU', 2704, 2688, 2682], ['dfALU', 2704, 2690, 2683], ['dfALU', 2704, 2692, 2684], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2710, 2709, 2687], ['L1_ACCESS'], ['dfALU', 2710, 2712, 2689], ['L1_ACCESS'], ['dfALU', 2710, 2714, 2691], ['L1_ACCESS'], ['dfALU', 2710, 2716, 2693], ['L1_ACCESS'], ['dfALU', 2718, 2709, 2695], ['dfALU', 2718, 2712, 2696], ['dfALU', 2718, 2714, 2697], ['dfALU', 2718, 2716, 2698], ['L1_ACCESS'], ['dfALU', 2723, 2709, 2700], ['dfALU', 2723, 2712, 2701], ['dfALU', 2723, 2714, 2702], ['dfALU', 2723, 2716, 2703], ['L1_ACCESS'], ['dfALU', 2728, 2709, 2705], ['dfALU', 2728, 2712, 2706], ['dfALU', 2728, 2714, 2707], ['dfALU', 2728, 2716, 2708], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2734, 2733, 2711], ['L1_ACCESS'], ['dfALU', 2734, 2736, 2713], ['L1_ACCESS'], ['dfALU', 2734, 2738, 2715], ['L1_ACCESS'], ['dfALU', 2734, 2740, 2717], ['L1_ACCESS'], ['dfALU', 2742, 2733, 2719], ['dfALU', 2742, 2736, 2720], ['dfALU', 2742, 2738, 2721], ['dfALU', 2742, 2740, 2722], ['L1_ACCESS'], ['dfALU', 2747, 2733, 2724], ['dfALU', 2747, 2736, 2725], ['dfALU', 2747, 2738, 2726], ['dfALU', 2747, 2740, 2727], ['L1_ACCESS'], ['dfALU', 2752, 2733, 2729], ['dfALU', 2752, 2736, 2730], ['dfALU', 2752, 2738, 2731], ['dfALU', 2752, 2740, 2732], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2758, 2757, 2735], ['L1_ACCESS'], ['dfALU', 2758, 2760, 2737], ['L1_ACCESS'], ['dfALU', 2758, 2762, 2739], ['L1_ACCESS'], ['dfALU', 2758, 2764, 2741], ['L1_ACCESS'], ['dfALU', 2766, 2757, 2743], ['dfALU', 2766, 2760, 2744], ['dfALU', 2766, 2762, 2745], ['dfALU', 2766, 2764, 2746], ['L1_ACCESS'], ['dfALU', 2771, 2757, 2748], ['dfALU', 2771, 2760, 2749], ['dfALU', 2771, 2762, 2750], ['dfALU', 2771, 2764, 2751], ['L1_ACCESS'], ['dfALU', 2776, 2757, 2753], ['dfALU', 2776, 2760, 2754], ['dfALU', 2776, 2762, 2755], ['dfALU', 2776, 2764, 2756], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2782, 2781, 2759], ['L1_ACCESS'], ['dfALU', 2782, 2784, 2761], ['L1_ACCESS'], ['dfALU', 2782, 2786, 2763], ['L1_ACCESS'], ['dfALU', 2782, 2788, 2765], ['L1_ACCESS'], ['dfALU', 2790, 2781, 2767], ['dfALU', 2790, 2784, 2768], ['dfALU', 2790, 2786, 2769], ['dfALU', 2790, 2788, 2770], ['L1_ACCESS'], ['dfALU', 2795, 2781, 2772], ['dfALU', 2795, 2784, 2773], ['dfALU', 2795, 2786, 2774], ['dfALU', 2795, 2788, 2775], ['L1_ACCESS'], ['dfALU', 2800, 2781, 2777], ['dfALU', 2800, 2784, 2778], ['dfALU', 2800, 2786, 2779], ['dfALU', 2800, 2788, 2780], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2806, 2805, 2783], ['L1_ACCESS'], ['dfALU', 2806, 2808, 2785], ['L1_ACCESS'], ['dfALU', 2806, 2810, 2787], ['L1_ACCESS'], ['dfALU', 2806, 2812, 2789], ['L1_ACCESS'], ['dfALU', 2814, 2805, 2791], ['dfALU', 2814, 2808, 2792], ['dfALU', 2814, 2810, 2793], ['dfALU', 2814, 2812, 2794], ['L1_ACCESS'], ['dfALU', 2819, 2805, 2796], ['dfALU', 2819, 2808, 2797], ['dfALU', 2819, 2810, 2798], ['dfALU', 2819, 2812, 2799], ['L1_ACCESS'], ['dfALU', 2824, 2805, 2801], ['dfALU', 2824, 2808, 2802], ['dfALU', 2824, 2810, 2803], ['dfALU', 2824, 2812, 2804], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2830, 2829, 2807], ['L1_ACCESS'], ['dfALU', 2830, 2832, 2809], ['L1_ACCESS'], ['dfALU', 2830, 2834, 2811], ['L1_ACCESS'], ['dfALU', 2830, 2836, 2813], ['L1_ACCESS'], ['dfALU', 2838, 2829, 2815], ['dfALU', 2838, 2832, 2816], ['dfALU', 2838, 2834, 2817], ['dfALU', 2838, 2836, 2818], ['L1_ACCESS'], ['dfALU', 2843, 2829, 2820], ['dfALU', 2843, 2832, 2821], ['dfALU', 2843, 2834, 2822], ['dfALU', 2843, 2836, 2823], ['L1_ACCESS'], ['dfALU', 2848, 2829, 2825], ['dfALU', 2848, 2832, 2826], ['dfALU', 2848, 2834, 2827], ['dfALU', 2848, 2836, 2828], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2854, 2853, 2831], ['L1_ACCESS'], ['dfALU', 2854, 2856, 2833], ['L1_ACCESS'], ['dfALU', 2854, 2858, 2835], ['L1_ACCESS'], ['dfALU', 2854, 2860, 2837], ['L1_ACCESS'], ['dfALU', 2862, 2853, 2839], ['dfALU', 2862, 2856, 2840], ['dfALU', 2862, 2858, 2841], ['dfALU', 2862, 2860, 2842], ['L1_ACCESS'], ['dfALU', 2867, 2853, 2844], ['dfALU', 2867, 2856, 2845], ['dfALU', 2867, 2858, 2846], ['dfALU', 2867, 2860, 2847], ['L1_ACCESS'], ['dfALU', 2872, 2853, 2849], ['dfALU', 2872, 2856, 2850], ['dfALU', 2872, 2858, 2851], ['dfALU', 2872, 2860, 2852], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2878, 2877, 2855], ['L1_ACCESS'], ['dfALU', 2878, 2880, 2857], ['L1_ACCESS'], ['dfALU', 2878, 2882, 2859], ['L1_ACCESS'], ['dfALU', 2878, 2884, 2861], ['L1_ACCESS'], ['dfALU', 2886, 2877, 2863], ['dfALU', 2886, 2880, 2864], ['dfALU', 2886, 2882, 2865], ['dfALU', 2886, 2884, 2866], ['L1_ACCESS'], ['dfALU', 2891, 2877, 2868], ['dfALU', 2891, 2880, 2869], ['dfALU', 2891, 2882, 2870], ['dfALU', 2891, 2884, 2871], ['L1_ACCESS'], ['dfALU', 2896, 2877, 2873], ['dfALU', 2896, 2880, 2874], ['dfALU', 2896, 2882, 2875], ['dfALU', 2896, 2884, 2876], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2902, 2901, 2879], ['L1_ACCESS'], ['dfALU', 2902, 2904, 2881], ['L1_ACCESS'], ['dfALU', 2902, 2906, 2883], ['L1_ACCESS'], ['dfALU', 2902, 2908, 2885], ['L1_ACCESS'], ['dfALU', 2910, 2901, 2887], ['dfALU', 2910, 2904, 2888], ['dfALU', 2910, 2906, 2889], ['dfALU', 2910, 2908, 2890], ['L1_ACCESS'], ['dfALU', 2915, 2901, 2892], ['dfALU', 2915, 2904, 2893], ['dfALU', 2915, 2906, 2894], ['dfALU', 2915, 2908, 2895], ['L1_ACCESS'], ['dfALU', 2920, 2901, 2897], ['dfALU', 2920, 2904, 2898], ['dfALU', 2920, 2906, 2899], ['dfALU', 2920, 2908, 2900], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2926, 2925, 2903], ['L1_ACCESS'], ['dfALU', 2926, 2928, 2905], ['L1_ACCESS'], ['dfALU', 2926, 2930, 2907], ['L1_ACCESS'], ['dfALU', 2926, 2932, 2909], ['L1_ACCESS'], ['dfALU', 2934, 2925, 2911], ['dfALU', 2934, 2928, 2912], ['dfALU', 2934, 2930, 2913], ['dfALU', 2934, 2932, 2914], ['L1_ACCESS'], ['dfALU', 2939, 2925, 2916], ['dfALU', 2939, 2928, 2917], ['dfALU', 2939, 2930, 2918], ['dfALU', 2939, 2932, 2919], ['L1_ACCESS'], ['dfALU', 2944, 2925, 2921], ['dfALU', 2944, 2928, 2922], ['dfALU', 2944, 2930, 2923], ['dfALU', 2944, 2932, 2924], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 2950, 2949, 2927], ['L1_ACCESS'], ['dfALU', 2950, 2952, 2929], ['L1_ACCESS'], ['dfALU', 2950, 2954, 2931], ['L1_ACCESS'], ['dfALU', 2950, 2956, 2933], ['L1_ACCESS'], ['dfALU', 2958, 2949, 2935], ['dfALU', 2958, 2952, 2936], ['dfALU', 2958, 2954, 2937], ['dfALU', 2958, 2956, 2938], ['L1_ACCESS'], ['dfALU', 2963, 2949, 2940], ['dfALU', 2963, 2952, 2941], ['dfALU', 2963, 2954, 2942], ['dfALU', 2963, 2956, 2943], ['L1_ACCESS'], ['dfALU', 2968, 2949, 2945], ['dfALU', 2968, 2952, 2946], ['dfALU', 2968, 2954, 2947], ['dfALU', 2968, 2956, 2948], ['THREAD_SYNC'], ['L1_ACCESS', 2573], ['L1_ACCESS', 2575], ['L1_ACCESS', 2577], ['L1_ACCESS', 2579], ['L1_ACCESS', 2580], ['L1_ACCESS', 2583], ['L1_ACCESS', 2586], ['L1_ACCESS', 2588], ['THREAD_SYNC'], ['iALU', 2567], ['iALU', 2983, 65], ['iALU', 2569], ['iALU', 2985], ['iALU', 71, 2986], ['iALU', 69, 2985, 70], ['L2_ACCESS'], ['iALU', 2988], ['L2_ACCESS'], ['iALU', 2988], ['L2_ACCESS'], ['iALU', 2988], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 2987], ['iALU', 5, 2987], ['L2_ACCESS'], ['iALU', 5, 2997], ['iALU', 73, 2997], ['L2_ACCESS'], ['iALU', 73, 3000], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 3006, 3005, 2951], ['L1_ACCESS'], ['dfALU', 3006, 3008, 2953], ['L1_ACCESS'], ['dfALU', 3006, 3010, 2955], ['L1_ACCESS'], ['dfALU', 3006, 3012, 2957], ['L1_ACCESS'], ['dfALU', 3014, 3005, 2959], ['dfALU', 3014, 3008, 2960], ['dfALU', 3014, 3010, 2961], ['dfALU', 3014, 3012, 2962], ['L1_ACCESS'], ['dfALU', 3019, 3005, 2964], ['dfALU', 3019, 3008, 2965], ['dfALU', 3019, 3010, 2966], ['dfALU', 3019, 3012, 2967], ['L1_ACCESS'], ['dfALU', 3024, 3005, 2969], ['dfALU', 3024, 3008, 2970], ['dfALU', 3024, 3010, 2971], ['dfALU', 3024, 3012, 2972], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3030, 3029, 3007], ['L1_ACCESS'], ['dfALU', 3030, 3032, 3009], ['L1_ACCESS'], ['dfALU', 3030, 3034, 3011], ['L1_ACCESS'], ['dfALU', 3030, 3036, 3013], ['L1_ACCESS'], ['dfALU', 3038, 3029, 3015], ['dfALU', 3038, 3032, 3016], ['dfALU', 3038, 3034, 3017], ['dfALU', 3038, 3036, 3018], ['L1_ACCESS'], ['dfALU', 3043, 3029, 3020], ['dfALU', 3043, 3032, 3021], ['dfALU', 3043, 3034, 3022], ['dfALU', 3043, 3036, 3023], ['L1_ACCESS'], ['dfALU', 3048, 3029, 3025], ['dfALU', 3048, 3032, 3026], ['dfALU', 3048, 3034, 3027], ['dfALU', 3048, 3036, 3028], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3054, 3053, 3031], ['L1_ACCESS'], ['dfALU', 3054, 3056, 3033], ['L1_ACCESS'], ['dfALU', 3054, 3058, 3035], ['L1_ACCESS'], ['dfALU', 3054, 3060, 3037], ['L1_ACCESS'], ['dfALU', 3062, 3053, 3039], ['dfALU', 3062, 3056, 3040], ['dfALU', 3062, 3058, 3041], ['dfALU', 3062, 3060, 3042], ['L1_ACCESS'], ['dfALU', 3067, 3053, 3044], ['dfALU', 3067, 3056, 3045], ['dfALU', 3067, 3058, 3046], ['dfALU', 3067, 3060, 3047], ['L1_ACCESS'], ['dfALU', 3072, 3053, 3049], ['dfALU', 3072, 3056, 3050], ['dfALU', 3072, 3058, 3051], ['dfALU', 3072, 3060, 3052], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3078, 3077, 3055], ['L1_ACCESS'], ['dfALU', 3078, 3080, 3057], ['L1_ACCESS'], ['dfALU', 3078, 3082, 3059], ['L1_ACCESS'], ['dfALU', 3078, 3084, 3061], ['L1_ACCESS'], ['dfALU', 3086, 3077, 3063], ['dfALU', 3086, 3080, 3064], ['dfALU', 3086, 3082, 3065], ['dfALU', 3086, 3084, 3066], ['L1_ACCESS'], ['dfALU', 3091, 3077, 3068], ['dfALU', 3091, 3080, 3069], ['dfALU', 3091, 3082, 3070], ['dfALU', 3091, 3084, 3071], ['L1_ACCESS'], ['dfALU', 3096, 3077, 3073], ['dfALU', 3096, 3080, 3074], ['dfALU', 3096, 3082, 3075], ['dfALU', 3096, 3084, 3076], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3102, 3101, 3079], ['L1_ACCESS'], ['dfALU', 3102, 3104, 3081], ['L1_ACCESS'], ['dfALU', 3102, 3106, 3083], ['L1_ACCESS'], ['dfALU', 3102, 3108, 3085], ['L1_ACCESS'], ['dfALU', 3110, 3101, 3087], ['dfALU', 3110, 3104, 3088], ['dfALU', 3110, 3106, 3089], ['dfALU', 3110, 3108, 3090], ['L1_ACCESS'], ['dfALU', 3115, 3101, 3092], ['dfALU', 3115, 3104, 3093], ['dfALU', 3115, 3106, 3094], ['dfALU', 3115, 3108, 3095], ['L1_ACCESS'], ['dfALU', 3120, 3101, 3097], ['dfALU', 3120, 3104, 3098], ['dfALU', 3120, 3106, 3099], ['dfALU', 3120, 3108, 3100], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3126, 3125, 3103], ['L1_ACCESS'], ['dfALU', 3126, 3128, 3105], ['L1_ACCESS'], ['dfALU', 3126, 3130, 3107], ['L1_ACCESS'], ['dfALU', 3126, 3132, 3109], ['L1_ACCESS'], ['dfALU', 3134, 3125, 3111], ['dfALU', 3134, 3128, 3112], ['dfALU', 3134, 3130, 3113], ['dfALU', 3134, 3132, 3114], ['L1_ACCESS'], ['dfALU', 3139, 3125, 3116], ['dfALU', 3139, 3128, 3117], ['dfALU', 3139, 3130, 3118], ['dfALU', 3139, 3132, 3119], ['L1_ACCESS'], ['dfALU', 3144, 3125, 3121], ['dfALU', 3144, 3128, 3122], ['dfALU', 3144, 3130, 3123], ['dfALU', 3144, 3132, 3124], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3150, 3149, 3127], ['L1_ACCESS'], ['dfALU', 3150, 3152, 3129], ['L1_ACCESS'], ['dfALU', 3150, 3154, 3131], ['L1_ACCESS'], ['dfALU', 3150, 3156, 3133], ['L1_ACCESS'], ['dfALU', 3158, 3149, 3135], ['dfALU', 3158, 3152, 3136], ['dfALU', 3158, 3154, 3137], ['dfALU', 3158, 3156, 3138], ['L1_ACCESS'], ['dfALU', 3163, 3149, 3140], ['dfALU', 3163, 3152, 3141], ['dfALU', 3163, 3154, 3142], ['dfALU', 3163, 3156, 3143], ['L1_ACCESS'], ['dfALU', 3168, 3149, 3145], ['dfALU', 3168, 3152, 3146], ['dfALU', 3168, 3154, 3147], ['dfALU', 3168, 3156, 3148], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3174, 3173, 3151], ['L1_ACCESS'], ['dfALU', 3174, 3176, 3153], ['L1_ACCESS'], ['dfALU', 3174, 3178, 3155], ['L1_ACCESS'], ['dfALU', 3174, 3180, 3157], ['L1_ACCESS'], ['dfALU', 3182, 3173, 3159], ['dfALU', 3182, 3176, 3160], ['dfALU', 3182, 3178, 3161], ['dfALU', 3182, 3180, 3162], ['L1_ACCESS'], ['dfALU', 3187, 3173, 3164], ['dfALU', 3187, 3176, 3165], ['dfALU', 3187, 3178, 3166], ['dfALU', 3187, 3180, 3167], ['L1_ACCESS'], ['dfALU', 3192, 3173, 3169], ['dfALU', 3192, 3176, 3170], ['dfALU', 3192, 3178, 3171], ['dfALU', 3192, 3180, 3172], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3198, 3197, 3175], ['L1_ACCESS'], ['dfALU', 3198, 3200, 3177], ['L1_ACCESS'], ['dfALU', 3198, 3202, 3179], ['L1_ACCESS'], ['dfALU', 3198, 3204, 3181], ['L1_ACCESS'], ['dfALU', 3206, 3197, 3183], ['dfALU', 3206, 3200, 3184], ['dfALU', 3206, 3202, 3185], ['dfALU', 3206, 3204, 3186], ['L1_ACCESS'], ['dfALU', 3211, 3197, 3188], ['dfALU', 3211, 3200, 3189], ['dfALU', 3211, 3202, 3190], ['dfALU', 3211, 3204, 3191], ['L1_ACCESS'], ['dfALU', 3216, 3197, 3193], ['dfALU', 3216, 3200, 3194], ['dfALU', 3216, 3202, 3195], ['dfALU', 3216, 3204, 3196], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3222, 3221, 3199], ['L1_ACCESS'], ['dfALU', 3222, 3224, 3201], ['L1_ACCESS'], ['dfALU', 3222, 3226, 3203], ['L1_ACCESS'], ['dfALU', 3222, 3228, 3205], ['L1_ACCESS'], ['dfALU', 3230, 3221, 3207], ['dfALU', 3230, 3224, 3208], ['dfALU', 3230, 3226, 3209], ['dfALU', 3230, 3228, 3210], ['L1_ACCESS'], ['dfALU', 3235, 3221, 3212], ['dfALU', 3235, 3224, 3213], ['dfALU', 3235, 3226, 3214], ['dfALU', 3235, 3228, 3215], ['L1_ACCESS'], ['dfALU', 3240, 3221, 3217], ['dfALU', 3240, 3224, 3218], ['dfALU', 3240, 3226, 3219], ['dfALU', 3240, 3228, 3220], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3246, 3245, 3223], ['L1_ACCESS'], ['dfALU', 3246, 3248, 3225], ['L1_ACCESS'], ['dfALU', 3246, 3250, 3227], ['L1_ACCESS'], ['dfALU', 3246, 3252, 3229], ['L1_ACCESS'], ['dfALU', 3254, 3245, 3231], ['dfALU', 3254, 3248, 3232], ['dfALU', 3254, 3250, 3233], ['dfALU', 3254, 3252, 3234], ['L1_ACCESS'], ['dfALU', 3259, 3245, 3236], ['dfALU', 3259, 3248, 3237], ['dfALU', 3259, 3250, 3238], ['dfALU', 3259, 3252, 3239], ['L1_ACCESS'], ['dfALU', 3264, 3245, 3241], ['dfALU', 3264, 3248, 3242], ['dfALU', 3264, 3250, 3243], ['dfALU', 3264, 3252, 3244], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3270, 3269, 3247], ['L1_ACCESS'], ['dfALU', 3270, 3272, 3249], ['L1_ACCESS'], ['dfALU', 3270, 3274, 3251], ['L1_ACCESS'], ['dfALU', 3270, 3276, 3253], ['L1_ACCESS'], ['dfALU', 3278, 3269, 3255], ['dfALU', 3278, 3272, 3256], ['dfALU', 3278, 3274, 3257], ['dfALU', 3278, 3276, 3258], ['L1_ACCESS'], ['dfALU', 3283, 3269, 3260], ['dfALU', 3283, 3272, 3261], ['dfALU', 3283, 3274, 3262], ['dfALU', 3283, 3276, 3263], ['L1_ACCESS'], ['dfALU', 3288, 3269, 3265], ['dfALU', 3288, 3272, 3266], ['dfALU', 3288, 3274, 3267], ['dfALU', 3288, 3276, 3268], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3294, 3293, 3271], ['L1_ACCESS'], ['dfALU', 3294, 3296, 3273], ['L1_ACCESS'], ['dfALU', 3294, 3298, 3275], ['L1_ACCESS'], ['dfALU', 3294, 3300, 3277], ['L1_ACCESS'], ['dfALU', 3302, 3293, 3279], ['dfALU', 3302, 3296, 3280], ['dfALU', 3302, 3298, 3281], ['dfALU', 3302, 3300, 3282], ['L1_ACCESS'], ['dfALU', 3307, 3293, 3284], ['dfALU', 3307, 3296, 3285], ['dfALU', 3307, 3298, 3286], ['dfALU', 3307, 3300, 3287], ['L1_ACCESS'], ['dfALU', 3312, 3293, 3289], ['dfALU', 3312, 3296, 3290], ['dfALU', 3312, 3298, 3291], ['dfALU', 3312, 3300, 3292], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3318, 3317, 3295], ['L1_ACCESS'], ['dfALU', 3318, 3320, 3297], ['L1_ACCESS'], ['dfALU', 3318, 3322, 3299], ['L1_ACCESS'], ['dfALU', 3318, 3324, 3301], ['L1_ACCESS'], ['dfALU', 3326, 3317, 3303], ['dfALU', 3326, 3320, 3304], ['dfALU', 3326, 3322, 3305], ['dfALU', 3326, 3324, 3306], ['L1_ACCESS'], ['dfALU', 3331, 3317, 3308], ['dfALU', 3331, 3320, 3309], ['dfALU', 3331, 3322, 3310], ['dfALU', 3331, 3324, 3311], ['L1_ACCESS'], ['dfALU', 3336, 3317, 3313], ['dfALU', 3336, 3320, 3314], ['dfALU', 3336, 3322, 3315], ['dfALU', 3336, 3324, 3316], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3342, 3341, 3319], ['L1_ACCESS'], ['dfALU', 3342, 3344, 3321], ['L1_ACCESS'], ['dfALU', 3342, 3346, 3323], ['L1_ACCESS'], ['dfALU', 3342, 3348, 3325], ['L1_ACCESS'], ['dfALU', 3350, 3341, 3327], ['dfALU', 3350, 3344, 3328], ['dfALU', 3350, 3346, 3329], ['dfALU', 3350, 3348, 3330], ['L1_ACCESS'], ['dfALU', 3355, 3341, 3332], ['dfALU', 3355, 3344, 3333], ['dfALU', 3355, 3346, 3334], ['dfALU', 3355, 3348, 3335], ['L1_ACCESS'], ['dfALU', 3360, 3341, 3337], ['dfALU', 3360, 3344, 3338], ['dfALU', 3360, 3346, 3339], ['dfALU', 3360, 3348, 3340], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3366, 3365, 3343], ['L1_ACCESS'], ['dfALU', 3366, 3368, 3345], ['L1_ACCESS'], ['dfALU', 3366, 3370, 3347], ['L1_ACCESS'], ['dfALU', 3366, 3372, 3349], ['L1_ACCESS'], ['dfALU', 3374, 3365, 3351], ['dfALU', 3374, 3368, 3352], ['dfALU', 3374, 3370, 3353], ['dfALU', 3374, 3372, 3354], ['L1_ACCESS'], ['dfALU', 3379, 3365, 3356], ['dfALU', 3379, 3368, 3357], ['dfALU', 3379, 3370, 3358], ['dfALU', 3379, 3372, 3359], ['L1_ACCESS'], ['dfALU', 3384, 3365, 3361], ['dfALU', 3384, 3368, 3362], ['dfALU', 3384, 3370, 3363], ['dfALU', 3384, 3372, 3364], ['THREAD_SYNC'], ['L1_ACCESS', 2989], ['L1_ACCESS', 2991], ['L1_ACCESS', 2993], ['L1_ACCESS', 2995], ['L1_ACCESS', 2996], ['L1_ACCESS', 2999], ['L1_ACCESS', 3002], ['L1_ACCESS', 3004], ['THREAD_SYNC'], ['iALU', 2983], ['iALU', 3399, 65], ['iALU', 2985], ['iALU', 3401], ['iALU', 71, 3402], ['iALU', 69, 3401, 70], ['L2_ACCESS'], ['iALU', 3404], ['L2_ACCESS'], ['iALU', 3404], ['L2_ACCESS'], ['iALU', 3404], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 3403], ['iALU', 5, 3403], ['L2_ACCESS'], ['iALU', 5, 3413], ['iALU', 73, 3413], ['L2_ACCESS'], ['iALU', 73, 3416], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 3422, 3421, 3367], ['L1_ACCESS'], ['dfALU', 3422, 3424, 3369], ['L1_ACCESS'], ['dfALU', 3422, 3426, 3371], ['L1_ACCESS'], ['dfALU', 3422, 3428, 3373], ['L1_ACCESS'], ['dfALU', 3430, 3421, 3375], ['dfALU', 3430, 3424, 3376], ['dfALU', 3430, 3426, 3377], ['dfALU', 3430, 3428, 3378], ['L1_ACCESS'], ['dfALU', 3435, 3421, 3380], ['dfALU', 3435, 3424, 3381], ['dfALU', 3435, 3426, 3382], ['dfALU', 3435, 3428, 3383], ['L1_ACCESS'], ['dfALU', 3440, 3421, 3385], ['dfALU', 3440, 3424, 3386], ['dfALU', 3440, 3426, 3387], ['dfALU', 3440, 3428, 3388], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3446, 3445, 3423], ['L1_ACCESS'], ['dfALU', 3446, 3448, 3425], ['L1_ACCESS'], ['dfALU', 3446, 3450, 3427], ['L1_ACCESS'], ['dfALU', 3446, 3452, 3429], ['L1_ACCESS'], ['dfALU', 3454, 3445, 3431], ['dfALU', 3454, 3448, 3432], ['dfALU', 3454, 3450, 3433], ['dfALU', 3454, 3452, 3434], ['L1_ACCESS'], ['dfALU', 3459, 3445, 3436], ['dfALU', 3459, 3448, 3437], ['dfALU', 3459, 3450, 3438], ['dfALU', 3459, 3452, 3439], ['L1_ACCESS'], ['dfALU', 3464, 3445, 3441], ['dfALU', 3464, 3448, 3442], ['dfALU', 3464, 3450, 3443], ['dfALU', 3464, 3452, 3444], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3470, 3469, 3447], ['L1_ACCESS'], ['dfALU', 3470, 3472, 3449], ['L1_ACCESS'], ['dfALU', 3470, 3474, 3451], ['L1_ACCESS'], ['dfALU', 3470, 3476, 3453], ['L1_ACCESS'], ['dfALU', 3478, 3469, 3455], ['dfALU', 3478, 3472, 3456], ['dfALU', 3478, 3474, 3457], ['dfALU', 3478, 3476, 3458], ['L1_ACCESS'], ['dfALU', 3483, 3469, 3460], ['dfALU', 3483, 3472, 3461], ['dfALU', 3483, 3474, 3462], ['dfALU', 3483, 3476, 3463], ['L1_ACCESS'], ['dfALU', 3488, 3469, 3465], ['dfALU', 3488, 3472, 3466], ['dfALU', 3488, 3474, 3467], ['dfALU', 3488, 3476, 3468], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3494, 3493, 3471], ['L1_ACCESS'], ['dfALU', 3494, 3496, 3473], ['L1_ACCESS'], ['dfALU', 3494, 3498, 3475], ['L1_ACCESS'], ['dfALU', 3494, 3500, 3477], ['L1_ACCESS'], ['dfALU', 3502, 3493, 3479], ['dfALU', 3502, 3496, 3480], ['dfALU', 3502, 3498, 3481], ['dfALU', 3502, 3500, 3482], ['L1_ACCESS'], ['dfALU', 3507, 3493, 3484], ['dfALU', 3507, 3496, 3485], ['dfALU', 3507, 3498, 3486], ['dfALU', 3507, 3500, 3487], ['L1_ACCESS'], ['dfALU', 3512, 3493, 3489], ['dfALU', 3512, 3496, 3490], ['dfALU', 3512, 3498, 3491], ['dfALU', 3512, 3500, 3492], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3518, 3517, 3495], ['L1_ACCESS'], ['dfALU', 3518, 3520, 3497], ['L1_ACCESS'], ['dfALU', 3518, 3522, 3499], ['L1_ACCESS'], ['dfALU', 3518, 3524, 3501], ['L1_ACCESS'], ['dfALU', 3526, 3517, 3503], ['dfALU', 3526, 3520, 3504], ['dfALU', 3526, 3522, 3505], ['dfALU', 3526, 3524, 3506], ['L1_ACCESS'], ['dfALU', 3531, 3517, 3508], ['dfALU', 3531, 3520, 3509], ['dfALU', 3531, 3522, 3510], ['dfALU', 3531, 3524, 3511], ['L1_ACCESS'], ['dfALU', 3536, 3517, 3513], ['dfALU', 3536, 3520, 3514], ['dfALU', 3536, 3522, 3515], ['dfALU', 3536, 3524, 3516], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3542, 3541, 3519], ['L1_ACCESS'], ['dfALU', 3542, 3544, 3521], ['L1_ACCESS'], ['dfALU', 3542, 3546, 3523], ['L1_ACCESS'], ['dfALU', 3542, 3548, 3525], ['L1_ACCESS'], ['dfALU', 3550, 3541, 3527], ['dfALU', 3550, 3544, 3528], ['dfALU', 3550, 3546, 3529], ['dfALU', 3550, 3548, 3530], ['L1_ACCESS'], ['dfALU', 3555, 3541, 3532], ['dfALU', 3555, 3544, 3533], ['dfALU', 3555, 3546, 3534], ['dfALU', 3555, 3548, 3535], ['L1_ACCESS'], ['dfALU', 3560, 3541, 3537], ['dfALU', 3560, 3544, 3538], ['dfALU', 3560, 3546, 3539], ['dfALU', 3560, 3548, 3540], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3566, 3565, 3543], ['L1_ACCESS'], ['dfALU', 3566, 3568, 3545], ['L1_ACCESS'], ['dfALU', 3566, 3570, 3547], ['L1_ACCESS'], ['dfALU', 3566, 3572, 3549], ['L1_ACCESS'], ['dfALU', 3574, 3565, 3551], ['dfALU', 3574, 3568, 3552], ['dfALU', 3574, 3570, 3553], ['dfALU', 3574, 3572, 3554], ['L1_ACCESS'], ['dfALU', 3579, 3565, 3556], ['dfALU', 3579, 3568, 3557], ['dfALU', 3579, 3570, 3558], ['dfALU', 3579, 3572, 3559], ['L1_ACCESS'], ['dfALU', 3584, 3565, 3561], ['dfALU', 3584, 3568, 3562], ['dfALU', 3584, 3570, 3563], ['dfALU', 3584, 3572, 3564], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3590, 3589, 3567], ['L1_ACCESS'], ['dfALU', 3590, 3592, 3569], ['L1_ACCESS'], ['dfALU', 3590, 3594, 3571], ['L1_ACCESS'], ['dfALU', 3590, 3596, 3573], ['L1_ACCESS'], ['dfALU', 3598, 3589, 3575], ['dfALU', 3598, 3592, 3576], ['dfALU', 3598, 3594, 3577], ['dfALU', 3598, 3596, 3578], ['L1_ACCESS'], ['dfALU', 3603, 3589, 3580], ['dfALU', 3603, 3592, 3581], ['dfALU', 3603, 3594, 3582], ['dfALU', 3603, 3596, 3583], ['L1_ACCESS'], ['dfALU', 3608, 3589, 3585], ['dfALU', 3608, 3592, 3586], ['dfALU', 3608, 3594, 3587], ['dfALU', 3608, 3596, 3588], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3614, 3613, 3591], ['L1_ACCESS'], ['dfALU', 3614, 3616, 3593], ['L1_ACCESS'], ['dfALU', 3614, 3618, 3595], ['L1_ACCESS'], ['dfALU', 3614, 3620, 3597], ['L1_ACCESS'], ['dfALU', 3622, 3613, 3599], ['dfALU', 3622, 3616, 3600], ['dfALU', 3622, 3618, 3601], ['dfALU', 3622, 3620, 3602], ['L1_ACCESS'], ['dfALU', 3627, 3613, 3604], ['dfALU', 3627, 3616, 3605], ['dfALU', 3627, 3618, 3606], ['dfALU', 3627, 3620, 3607], ['L1_ACCESS'], ['dfALU', 3632, 3613, 3609], ['dfALU', 3632, 3616, 3610], ['dfALU', 3632, 3618, 3611], ['dfALU', 3632, 3620, 3612], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3638, 3637, 3615], ['L1_ACCESS'], ['dfALU', 3638, 3640, 3617], ['L1_ACCESS'], ['dfALU', 3638, 3642, 3619], ['L1_ACCESS'], ['dfALU', 3638, 3644, 3621], ['L1_ACCESS'], ['dfALU', 3646, 3637, 3623], ['dfALU', 3646, 3640, 3624], ['dfALU', 3646, 3642, 3625], ['dfALU', 3646, 3644, 3626], ['L1_ACCESS'], ['dfALU', 3651, 3637, 3628], ['dfALU', 3651, 3640, 3629], ['dfALU', 3651, 3642, 3630], ['dfALU', 3651, 3644, 3631], ['L1_ACCESS'], ['dfALU', 3656, 3637, 3633], ['dfALU', 3656, 3640, 3634], ['dfALU', 3656, 3642, 3635], ['dfALU', 3656, 3644, 3636], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3662, 3661, 3639], ['L1_ACCESS'], ['dfALU', 3662, 3664, 3641], ['L1_ACCESS'], ['dfALU', 3662, 3666, 3643], ['L1_ACCESS'], ['dfALU', 3662, 3668, 3645], ['L1_ACCESS'], ['dfALU', 3670, 3661, 3647], ['dfALU', 3670, 3664, 3648], ['dfALU', 3670, 3666, 3649], ['dfALU', 3670, 3668, 3650], ['L1_ACCESS'], ['dfALU', 3675, 3661, 3652], ['dfALU', 3675, 3664, 3653], ['dfALU', 3675, 3666, 3654], ['dfALU', 3675, 3668, 3655], ['L1_ACCESS'], ['dfALU', 3680, 3661, 3657], ['dfALU', 3680, 3664, 3658], ['dfALU', 3680, 3666, 3659], ['dfALU', 3680, 3668, 3660], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3686, 3685, 3663], ['L1_ACCESS'], ['dfALU', 3686, 3688, 3665], ['L1_ACCESS'], ['dfALU', 3686, 3690, 3667], ['L1_ACCESS'], ['dfALU', 3686, 3692, 3669], ['L1_ACCESS'], ['dfALU', 3694, 3685, 3671], ['dfALU', 3694, 3688, 3672], ['dfALU', 3694, 3690, 3673], ['dfALU', 3694, 3692, 3674], ['L1_ACCESS'], ['dfALU', 3699, 3685, 3676], ['dfALU', 3699, 3688, 3677], ['dfALU', 3699, 3690, 3678], ['dfALU', 3699, 3692, 3679], ['L1_ACCESS'], ['dfALU', 3704, 3685, 3681], ['dfALU', 3704, 3688, 3682], ['dfALU', 3704, 3690, 3683], ['dfALU', 3704, 3692, 3684], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3710, 3709, 3687], ['L1_ACCESS'], ['dfALU', 3710, 3712, 3689], ['L1_ACCESS'], ['dfALU', 3710, 3714, 3691], ['L1_ACCESS'], ['dfALU', 3710, 3716, 3693], ['L1_ACCESS'], ['dfALU', 3718, 3709, 3695], ['dfALU', 3718, 3712, 3696], ['dfALU', 3718, 3714, 3697], ['dfALU', 3718, 3716, 3698], ['L1_ACCESS'], ['dfALU', 3723, 3709, 3700], ['dfALU', 3723, 3712, 3701], ['dfALU', 3723, 3714, 3702], ['dfALU', 3723, 3716, 3703], ['L1_ACCESS'], ['dfALU', 3728, 3709, 3705], ['dfALU', 3728, 3712, 3706], ['dfALU', 3728, 3714, 3707], ['dfALU', 3728, 3716, 3708], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3734, 3733, 3711], ['L1_ACCESS'], ['dfALU', 3734, 3736, 3713], ['L1_ACCESS'], ['dfALU', 3734, 3738, 3715], ['L1_ACCESS'], ['dfALU', 3734, 3740, 3717], ['L1_ACCESS'], ['dfALU', 3742, 3733, 3719], ['dfALU', 3742, 3736, 3720], ['dfALU', 3742, 3738, 3721], ['dfALU', 3742, 3740, 3722], ['L1_ACCESS'], ['dfALU', 3747, 3733, 3724], ['dfALU', 3747, 3736, 3725], ['dfALU', 3747, 3738, 3726], ['dfALU', 3747, 3740, 3727], ['L1_ACCESS'], ['dfALU', 3752, 3733, 3729], ['dfALU', 3752, 3736, 3730], ['dfALU', 3752, 3738, 3731], ['dfALU', 3752, 3740, 3732], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3758, 3757, 3735], ['L1_ACCESS'], ['dfALU', 3758, 3760, 3737], ['L1_ACCESS'], ['dfALU', 3758, 3762, 3739], ['L1_ACCESS'], ['dfALU', 3758, 3764, 3741], ['L1_ACCESS'], ['dfALU', 3766, 3757, 3743], ['dfALU', 3766, 3760, 3744], ['dfALU', 3766, 3762, 3745], ['dfALU', 3766, 3764, 3746], ['L1_ACCESS'], ['dfALU', 3771, 3757, 3748], ['dfALU', 3771, 3760, 3749], ['dfALU', 3771, 3762, 3750], ['dfALU', 3771, 3764, 3751], ['L1_ACCESS'], ['dfALU', 3776, 3757, 3753], ['dfALU', 3776, 3760, 3754], ['dfALU', 3776, 3762, 3755], ['dfALU', 3776, 3764, 3756], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3782, 3781, 3759], ['L1_ACCESS'], ['dfALU', 3782, 3784, 3761], ['L1_ACCESS'], ['dfALU', 3782, 3786, 3763], ['L1_ACCESS'], ['dfALU', 3782, 3788, 3765], ['L1_ACCESS'], ['dfALU', 3790, 3781, 3767], ['dfALU', 3790, 3784, 3768], ['dfALU', 3790, 3786, 3769], ['dfALU', 3790, 3788, 3770], ['L1_ACCESS'], ['dfALU', 3795, 3781, 3772], ['dfALU', 3795, 3784, 3773], ['dfALU', 3795, 3786, 3774], ['dfALU', 3795, 3788, 3775], ['L1_ACCESS'], ['dfALU', 3800, 3781, 3777], ['dfALU', 3800, 3784, 3778], ['dfALU', 3800, 3786, 3779], ['dfALU', 3800, 3788, 3780], ['THREAD_SYNC'], ['L1_ACCESS', 3405], ['L1_ACCESS', 3407], ['L1_ACCESS', 3409], ['L1_ACCESS', 3411], ['L1_ACCESS', 3412], ['L1_ACCESS', 3415], ['L1_ACCESS', 3418], ['L1_ACCESS', 3420], ['THREAD_SYNC'], ['iALU', 3399], ['iALU', 3815, 65], ['iALU', 3401], ['iALU', 3817], ['iALU', 71, 3818], ['iALU', 69, 3817, 70], ['L2_ACCESS'], ['iALU', 3820], ['L2_ACCESS'], ['iALU', 3820], ['L2_ACCESS'], ['iALU', 3820], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 3819], ['iALU', 5, 3819], ['L2_ACCESS'], ['iALU', 5, 3829], ['iALU', 73, 3829], ['L2_ACCESS'], ['iALU', 73, 3832], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 3838, 3837, 3783], ['L1_ACCESS'], ['dfALU', 3838, 3840, 3785], ['L1_ACCESS'], ['dfALU', 3838, 3842, 3787], ['L1_ACCESS'], ['dfALU', 3838, 3844, 3789], ['L1_ACCESS'], ['dfALU', 3846, 3837, 3791], ['dfALU', 3846, 3840, 3792], ['dfALU', 3846, 3842, 3793], ['dfALU', 3846, 3844, 3794], ['L1_ACCESS'], ['dfALU', 3851, 3837, 3796], ['dfALU', 3851, 3840, 3797], ['dfALU', 3851, 3842, 3798], ['dfALU', 3851, 3844, 3799], ['L1_ACCESS'], ['dfALU', 3856, 3837, 3801], ['dfALU', 3856, 3840, 3802], ['dfALU', 3856, 3842, 3803], ['dfALU', 3856, 3844, 3804], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3862, 3861, 3839], ['L1_ACCESS'], ['dfALU', 3862, 3864, 3841], ['L1_ACCESS'], ['dfALU', 3862, 3866, 3843], ['L1_ACCESS'], ['dfALU', 3862, 3868, 3845], ['L1_ACCESS'], ['dfALU', 3870, 3861, 3847], ['dfALU', 3870, 3864, 3848], ['dfALU', 3870, 3866, 3849], ['dfALU', 3870, 3868, 3850], ['L1_ACCESS'], ['dfALU', 3875, 3861, 3852], ['dfALU', 3875, 3864, 3853], ['dfALU', 3875, 3866, 3854], ['dfALU', 3875, 3868, 3855], ['L1_ACCESS'], ['dfALU', 3880, 3861, 3857], ['dfALU', 3880, 3864, 3858], ['dfALU', 3880, 3866, 3859], ['dfALU', 3880, 3868, 3860], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3886, 3885, 3863], ['L1_ACCESS'], ['dfALU', 3886, 3888, 3865], ['L1_ACCESS'], ['dfALU', 3886, 3890, 3867], ['L1_ACCESS'], ['dfALU', 3886, 3892, 3869], ['L1_ACCESS'], ['dfALU', 3894, 3885, 3871], ['dfALU', 3894, 3888, 3872], ['dfALU', 3894, 3890, 3873], ['dfALU', 3894, 3892, 3874], ['L1_ACCESS'], ['dfALU', 3899, 3885, 3876], ['dfALU', 3899, 3888, 3877], ['dfALU', 3899, 3890, 3878], ['dfALU', 3899, 3892, 3879], ['L1_ACCESS'], ['dfALU', 3904, 3885, 3881], ['dfALU', 3904, 3888, 3882], ['dfALU', 3904, 3890, 3883], ['dfALU', 3904, 3892, 3884], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3910, 3909, 3887], ['L1_ACCESS'], ['dfALU', 3910, 3912, 3889], ['L1_ACCESS'], ['dfALU', 3910, 3914, 3891], ['L1_ACCESS'], ['dfALU', 3910, 3916, 3893], ['L1_ACCESS'], ['dfALU', 3918, 3909, 3895], ['dfALU', 3918, 3912, 3896], ['dfALU', 3918, 3914, 3897], ['dfALU', 3918, 3916, 3898], ['L1_ACCESS'], ['dfALU', 3923, 3909, 3900], ['dfALU', 3923, 3912, 3901], ['dfALU', 3923, 3914, 3902], ['dfALU', 3923, 3916, 3903], ['L1_ACCESS'], ['dfALU', 3928, 3909, 3905], ['dfALU', 3928, 3912, 3906], ['dfALU', 3928, 3914, 3907], ['dfALU', 3928, 3916, 3908], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3934, 3933, 3911], ['L1_ACCESS'], ['dfALU', 3934, 3936, 3913], ['L1_ACCESS'], ['dfALU', 3934, 3938, 3915], ['L1_ACCESS'], ['dfALU', 3934, 3940, 3917], ['L1_ACCESS'], ['dfALU', 3942, 3933, 3919], ['dfALU', 3942, 3936, 3920], ['dfALU', 3942, 3938, 3921], ['dfALU', 3942, 3940, 3922], ['L1_ACCESS'], ['dfALU', 3947, 3933, 3924], ['dfALU', 3947, 3936, 3925], ['dfALU', 3947, 3938, 3926], ['dfALU', 3947, 3940, 3927], ['L1_ACCESS'], ['dfALU', 3952, 3933, 3929], ['dfALU', 3952, 3936, 3930], ['dfALU', 3952, 3938, 3931], ['dfALU', 3952, 3940, 3932], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3958, 3957, 3935], ['L1_ACCESS'], ['dfALU', 3958, 3960, 3937], ['L1_ACCESS'], ['dfALU', 3958, 3962, 3939], ['L1_ACCESS'], ['dfALU', 3958, 3964, 3941], ['L1_ACCESS'], ['dfALU', 3966, 3957, 3943], ['dfALU', 3966, 3960, 3944], ['dfALU', 3966, 3962, 3945], ['dfALU', 3966, 3964, 3946], ['L1_ACCESS'], ['dfALU', 3971, 3957, 3948], ['dfALU', 3971, 3960, 3949], ['dfALU', 3971, 3962, 3950], ['dfALU', 3971, 3964, 3951], ['L1_ACCESS'], ['dfALU', 3976, 3957, 3953], ['dfALU', 3976, 3960, 3954], ['dfALU', 3976, 3962, 3955], ['dfALU', 3976, 3964, 3956], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 3982, 3981, 3959], ['L1_ACCESS'], ['dfALU', 3982, 3984, 3961], ['L1_ACCESS'], ['dfALU', 3982, 3986, 3963], ['L1_ACCESS'], ['dfALU', 3982, 3988, 3965], ['L1_ACCESS'], ['dfALU', 3990, 3981, 3967], ['dfALU', 3990, 3984, 3968], ['dfALU', 3990, 3986, 3969], ['dfALU', 3990, 3988, 3970], ['L1_ACCESS'], ['dfALU', 3995, 3981, 3972], ['dfALU', 3995, 3984, 3973], ['dfALU', 3995, 3986, 3974], ['dfALU', 3995, 3988, 3975], ['L1_ACCESS'], ['dfALU', 4000, 3981, 3977], ['dfALU', 4000, 3984, 3978], ['dfALU', 4000, 3986, 3979], ['dfALU', 4000, 3988, 3980], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4006, 4005, 3983], ['L1_ACCESS'], ['dfALU', 4006, 4008, 3985], ['L1_ACCESS'], ['dfALU', 4006, 4010, 3987], ['L1_ACCESS'], ['dfALU', 4006, 4012, 3989], ['L1_ACCESS'], ['dfALU', 4014, 4005, 3991], ['dfALU', 4014, 4008, 3992], ['dfALU', 4014, 4010, 3993], ['dfALU', 4014, 4012, 3994], ['L1_ACCESS'], ['dfALU', 4019, 4005, 3996], ['dfALU', 4019, 4008, 3997], ['dfALU', 4019, 4010, 3998], ['dfALU', 4019, 4012, 3999], ['L1_ACCESS'], ['dfALU', 4024, 4005, 4001], ['dfALU', 4024, 4008, 4002], ['dfALU', 4024, 4010, 4003], ['dfALU', 4024, 4012, 4004], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4030, 4029, 4007], ['L1_ACCESS'], ['dfALU', 4030, 4032, 4009], ['L1_ACCESS'], ['dfALU', 4030, 4034, 4011], ['L1_ACCESS'], ['dfALU', 4030, 4036, 4013], ['L1_ACCESS'], ['dfALU', 4038, 4029, 4015], ['dfALU', 4038, 4032, 4016], ['dfALU', 4038, 4034, 4017], ['dfALU', 4038, 4036, 4018], ['L1_ACCESS'], ['dfALU', 4043, 4029, 4020], ['dfALU', 4043, 4032, 4021], ['dfALU', 4043, 4034, 4022], ['dfALU', 4043, 4036, 4023], ['L1_ACCESS'], ['dfALU', 4048, 4029, 4025], ['dfALU', 4048, 4032, 4026], ['dfALU', 4048, 4034, 4027], ['dfALU', 4048, 4036, 4028], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4054, 4053, 4031], ['L1_ACCESS'], ['dfALU', 4054, 4056, 4033], ['L1_ACCESS'], ['dfALU', 4054, 4058, 4035], ['L1_ACCESS'], ['dfALU', 4054, 4060, 4037], ['L1_ACCESS'], ['dfALU', 4062, 4053, 4039], ['dfALU', 4062, 4056, 4040], ['dfALU', 4062, 4058, 4041], ['dfALU', 4062, 4060, 4042], ['L1_ACCESS'], ['dfALU', 4067, 4053, 4044], ['dfALU', 4067, 4056, 4045], ['dfALU', 4067, 4058, 4046], ['dfALU', 4067, 4060, 4047], ['L1_ACCESS'], ['dfALU', 4072, 4053, 4049], ['dfALU', 4072, 4056, 4050], ['dfALU', 4072, 4058, 4051], ['dfALU', 4072, 4060, 4052], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4078, 4077, 4055], ['L1_ACCESS'], ['dfALU', 4078, 4080, 4057], ['L1_ACCESS'], ['dfALU', 4078, 4082, 4059], ['L1_ACCESS'], ['dfALU', 4078, 4084, 4061], ['L1_ACCESS'], ['dfALU', 4086, 4077, 4063], ['dfALU', 4086, 4080, 4064], ['dfALU', 4086, 4082, 4065], ['dfALU', 4086, 4084, 4066], ['L1_ACCESS'], ['dfALU', 4091, 4077, 4068], ['dfALU', 4091, 4080, 4069], ['dfALU', 4091, 4082, 4070], ['dfALU', 4091, 4084, 4071], ['L1_ACCESS'], ['dfALU', 4096, 4077, 4073], ['dfALU', 4096, 4080, 4074], ['dfALU', 4096, 4082, 4075], ['dfALU', 4096, 4084, 4076], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4102, 4101, 4079], ['L1_ACCESS'], ['dfALU', 4102, 4104, 4081], ['L1_ACCESS'], ['dfALU', 4102, 4106, 4083], ['L1_ACCESS'], ['dfALU', 4102, 4108, 4085], ['L1_ACCESS'], ['dfALU', 4110, 4101, 4087], ['dfALU', 4110, 4104, 4088], ['dfALU', 4110, 4106, 4089], ['dfALU', 4110, 4108, 4090], ['L1_ACCESS'], ['dfALU', 4115, 4101, 4092], ['dfALU', 4115, 4104, 4093], ['dfALU', 4115, 4106, 4094], ['dfALU', 4115, 4108, 4095], ['L1_ACCESS'], ['dfALU', 4120, 4101, 4097], ['dfALU', 4120, 4104, 4098], ['dfALU', 4120, 4106, 4099], ['dfALU', 4120, 4108, 4100], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4126, 4125, 4103], ['L1_ACCESS'], ['dfALU', 4126, 4128, 4105], ['L1_ACCESS'], ['dfALU', 4126, 4130, 4107], ['L1_ACCESS'], ['dfALU', 4126, 4132, 4109], ['L1_ACCESS'], ['dfALU', 4134, 4125, 4111], ['dfALU', 4134, 4128, 4112], ['dfALU', 4134, 4130, 4113], ['dfALU', 4134, 4132, 4114], ['L1_ACCESS'], ['dfALU', 4139, 4125, 4116], ['dfALU', 4139, 4128, 4117], ['dfALU', 4139, 4130, 4118], ['dfALU', 4139, 4132, 4119], ['L1_ACCESS'], ['dfALU', 4144, 4125, 4121], ['dfALU', 4144, 4128, 4122], ['dfALU', 4144, 4130, 4123], ['dfALU', 4144, 4132, 4124], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4150, 4149, 4127], ['L1_ACCESS'], ['dfALU', 4150, 4152, 4129], ['L1_ACCESS'], ['dfALU', 4150, 4154, 4131], ['L1_ACCESS'], ['dfALU', 4150, 4156, 4133], ['L1_ACCESS'], ['dfALU', 4158, 4149, 4135], ['dfALU', 4158, 4152, 4136], ['dfALU', 4158, 4154, 4137], ['dfALU', 4158, 4156, 4138], ['L1_ACCESS'], ['dfALU', 4163, 4149, 4140], ['dfALU', 4163, 4152, 4141], ['dfALU', 4163, 4154, 4142], ['dfALU', 4163, 4156, 4143], ['L1_ACCESS'], ['dfALU', 4168, 4149, 4145], ['dfALU', 4168, 4152, 4146], ['dfALU', 4168, 4154, 4147], ['dfALU', 4168, 4156, 4148], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4174, 4173, 4151], ['L1_ACCESS'], ['dfALU', 4174, 4176, 4153], ['L1_ACCESS'], ['dfALU', 4174, 4178, 4155], ['L1_ACCESS'], ['dfALU', 4174, 4180, 4157], ['L1_ACCESS'], ['dfALU', 4182, 4173, 4159], ['dfALU', 4182, 4176, 4160], ['dfALU', 4182, 4178, 4161], ['dfALU', 4182, 4180, 4162], ['L1_ACCESS'], ['dfALU', 4187, 4173, 4164], ['dfALU', 4187, 4176, 4165], ['dfALU', 4187, 4178, 4166], ['dfALU', 4187, 4180, 4167], ['L1_ACCESS'], ['dfALU', 4192, 4173, 4169], ['dfALU', 4192, 4176, 4170], ['dfALU', 4192, 4178, 4171], ['dfALU', 4192, 4180, 4172], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4198, 4197, 4175], ['L1_ACCESS'], ['dfALU', 4198, 4200, 4177], ['L1_ACCESS'], ['dfALU', 4198, 4202, 4179], ['L1_ACCESS'], ['dfALU', 4198, 4204, 4181], ['L1_ACCESS'], ['dfALU', 4206, 4197, 4183], ['dfALU', 4206, 4200, 4184], ['dfALU', 4206, 4202, 4185], ['dfALU', 4206, 4204, 4186], ['L1_ACCESS'], ['dfALU', 4211, 4197, 4188], ['dfALU', 4211, 4200, 4189], ['dfALU', 4211, 4202, 4190], ['dfALU', 4211, 4204, 4191], ['L1_ACCESS'], ['dfALU', 4216, 4197, 4193], ['dfALU', 4216, 4200, 4194], ['dfALU', 4216, 4202, 4195], ['dfALU', 4216, 4204, 4196], ['THREAD_SYNC'], ['L1_ACCESS', 3821], ['L1_ACCESS', 3823], ['L1_ACCESS', 3825], ['L1_ACCESS', 3827], ['L1_ACCESS', 3828], ['L1_ACCESS', 3831], ['L1_ACCESS', 3834], ['L1_ACCESS', 3836], ['THREAD_SYNC'], ['iALU', 3815], ['iALU', 4231, 65], ['iALU', 3817], ['iALU', 4233], ['iALU', 71, 4234], ['iALU', 69, 4233, 70], ['L2_ACCESS'], ['iALU', 4236], ['L2_ACCESS'], ['iALU', 4236], ['L2_ACCESS'], ['iALU', 4236], ['L2_ACCESS'], ['L2_ACCESS'], ['iALU', 72, 4235], ['iALU', 5, 4235], ['L2_ACCESS'], ['iALU', 5, 4245], ['iALU', 73, 4245], ['L2_ACCESS'], ['iALU', 73, 4248], ['L2_ACCESS'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 4254, 4253, 4199], ['L1_ACCESS'], ['dfALU', 4254, 4256, 4201], ['L1_ACCESS'], ['dfALU', 4254, 4258, 4203], ['L1_ACCESS'], ['dfALU', 4254, 4260, 4205], ['L1_ACCESS'], ['dfALU', 4262, 4253, 4207], ['dfALU', 4262, 4256, 4208], ['dfALU', 4262, 4258, 4209], ['dfALU', 4262, 4260, 4210], ['L1_ACCESS'], ['dfALU', 4267, 4253, 4212], ['dfALU', 4267, 4256, 4213], ['dfALU', 4267, 4258, 4214], ['dfALU', 4267, 4260, 4215], ['L1_ACCESS'], ['dfALU', 4272, 4253, 4217], ['dfALU', 4272, 4256, 4218], ['dfALU', 4272, 4258, 4219], ['dfALU', 4272, 4260, 4220], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4278, 4277, 4255], ['L1_ACCESS'], ['dfALU', 4278, 4280, 4257], ['L1_ACCESS'], ['dfALU', 4278, 4282, 4259], ['L1_ACCESS'], ['dfALU', 4278, 4284, 4261], ['L1_ACCESS'], ['dfALU', 4286, 4277, 4263], ['dfALU', 4286, 4280, 4264], ['dfALU', 4286, 4282, 4265], ['dfALU', 4286, 4284, 4266], ['L1_ACCESS'], ['dfALU', 4291, 4277, 4268], ['dfALU', 4291, 4280, 4269], ['dfALU', 4291, 4282, 4270], ['dfALU', 4291, 4284, 4271], ['L1_ACCESS'], ['dfALU', 4296, 4277, 4273], ['dfALU', 4296, 4280, 4274], ['dfALU', 4296, 4282, 4275], ['dfALU', 4296, 4284, 4276], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4302, 4301, 4279], ['L1_ACCESS'], ['dfALU', 4302, 4304, 4281], ['L1_ACCESS'], ['dfALU', 4302, 4306, 4283], ['L1_ACCESS'], ['dfALU', 4302, 4308, 4285], ['L1_ACCESS'], ['dfALU', 4310, 4301, 4287], ['dfALU', 4310, 4304, 4288], ['dfALU', 4310, 4306, 4289], ['dfALU', 4310, 4308, 4290], ['L1_ACCESS'], ['dfALU', 4315, 4301, 4292], ['dfALU', 4315, 4304, 4293], ['dfALU', 4315, 4306, 4294], ['dfALU', 4315, 4308, 4295], ['L1_ACCESS'], ['dfALU', 4320, 4301, 4297], ['dfALU', 4320, 4304, 4298], ['dfALU', 4320, 4306, 4299], ['dfALU', 4320, 4308, 4300], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4326, 4325, 4303], ['L1_ACCESS'], ['dfALU', 4326, 4328, 4305], ['L1_ACCESS'], ['dfALU', 4326, 4330, 4307], ['L1_ACCESS'], ['dfALU', 4326, 4332, 4309], ['L1_ACCESS'], ['dfALU', 4334, 4325, 4311], ['dfALU', 4334, 4328, 4312], ['dfALU', 4334, 4330, 4313], ['dfALU', 4334, 4332, 4314], ['L1_ACCESS'], ['dfALU', 4339, 4325, 4316], ['dfALU', 4339, 4328, 4317], ['dfALU', 4339, 4330, 4318], ['dfALU', 4339, 4332, 4319], ['L1_ACCESS'], ['dfALU', 4344, 4325, 4321], ['dfALU', 4344, 4328, 4322], ['dfALU', 4344, 4330, 4323], ['dfALU', 4344, 4332, 4324], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4350, 4349, 4327], ['L1_ACCESS'], ['dfALU', 4350, 4352, 4329], ['L1_ACCESS'], ['dfALU', 4350, 4354, 4331], ['L1_ACCESS'], ['dfALU', 4350, 4356, 4333], ['L1_ACCESS'], ['dfALU', 4358, 4349, 4335], ['dfALU', 4358, 4352, 4336], ['dfALU', 4358, 4354, 4337], ['dfALU', 4358, 4356, 4338], ['L1_ACCESS'], ['dfALU', 4363, 4349, 4340], ['dfALU', 4363, 4352, 4341], ['dfALU', 4363, 4354, 4342], ['dfALU', 4363, 4356, 4343], ['L1_ACCESS'], ['dfALU', 4368, 4349, 4345], ['dfALU', 4368, 4352, 4346], ['dfALU', 4368, 4354, 4347], ['dfALU', 4368, 4356, 4348], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4374, 4373, 4351], ['L1_ACCESS'], ['dfALU', 4374, 4376, 4353], ['L1_ACCESS'], ['dfALU', 4374, 4378, 4355], ['L1_ACCESS'], ['dfALU', 4374, 4380, 4357], ['L1_ACCESS'], ['dfALU', 4382, 4373, 4359], ['dfALU', 4382, 4376, 4360], ['dfALU', 4382, 4378, 4361], ['dfALU', 4382, 4380, 4362], ['L1_ACCESS'], ['dfALU', 4387, 4373, 4364], ['dfALU', 4387, 4376, 4365], ['dfALU', 4387, 4378, 4366], ['dfALU', 4387, 4380, 4367], ['L1_ACCESS'], ['dfALU', 4392, 4373, 4369], ['dfALU', 4392, 4376, 4370], ['dfALU', 4392, 4378, 4371], ['dfALU', 4392, 4380, 4372], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4398, 4397, 4375], ['L1_ACCESS'], ['dfALU', 4398, 4400, 4377], ['L1_ACCESS'], ['dfALU', 4398, 4402, 4379], ['L1_ACCESS'], ['dfALU', 4398, 4404, 4381], ['L1_ACCESS'], ['dfALU', 4406, 4397, 4383], ['dfALU', 4406, 4400, 4384], ['dfALU', 4406, 4402, 4385], ['dfALU', 4406, 4404, 4386], ['L1_ACCESS'], ['dfALU', 4411, 4397, 4388], ['dfALU', 4411, 4400, 4389], ['dfALU', 4411, 4402, 4390], ['dfALU', 4411, 4404, 4391], ['L1_ACCESS'], ['dfALU', 4416, 4397, 4393], ['dfALU', 4416, 4400, 4394], ['dfALU', 4416, 4402, 4395], ['dfALU', 4416, 4404, 4396], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4422, 4421, 4399], ['L1_ACCESS'], ['dfALU', 4422, 4424, 4401], ['L1_ACCESS'], ['dfALU', 4422, 4426, 4403], ['L1_ACCESS'], ['dfALU', 4422, 4428, 4405], ['L1_ACCESS'], ['dfALU', 4430, 4421, 4407], ['dfALU', 4430, 4424, 4408], ['dfALU', 4430, 4426, 4409], ['dfALU', 4430, 4428, 4410], ['L1_ACCESS'], ['dfALU', 4435, 4421, 4412], ['dfALU', 4435, 4424, 4413], ['dfALU', 4435, 4426, 4414], ['dfALU', 4435, 4428, 4415], ['L1_ACCESS'], ['dfALU', 4440, 4421, 4417], ['dfALU', 4440, 4424, 4418], ['dfALU', 4440, 4426, 4419], ['dfALU', 4440, 4428, 4420], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4446, 4445, 4423], ['L1_ACCESS'], ['dfALU', 4446, 4448, 4425], ['L1_ACCESS'], ['dfALU', 4446, 4450, 4427], ['L1_ACCESS'], ['dfALU', 4446, 4452, 4429], ['L1_ACCESS'], ['dfALU', 4454, 4445, 4431], ['dfALU', 4454, 4448, 4432], ['dfALU', 4454, 4450, 4433], ['dfALU', 4454, 4452, 4434], ['L1_ACCESS'], ['dfALU', 4459, 4445, 4436], ['dfALU', 4459, 4448, 4437], ['dfALU', 4459, 4450, 4438], ['dfALU', 4459, 4452, 4439], ['L1_ACCESS'], ['dfALU', 4464, 4445, 4441], ['dfALU', 4464, 4448, 4442], ['dfALU', 4464, 4450, 4443], ['dfALU', 4464, 4452, 4444], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4470, 4469, 4447], ['L1_ACCESS'], ['dfALU', 4470, 4472, 4449], ['L1_ACCESS'], ['dfALU', 4470, 4474, 4451], ['L1_ACCESS'], ['dfALU', 4470, 4476, 4453], ['L1_ACCESS'], ['dfALU', 4478, 4469, 4455], ['dfALU', 4478, 4472, 4456], ['dfALU', 4478, 4474, 4457], ['dfALU', 4478, 4476, 4458], ['L1_ACCESS'], ['dfALU', 4483, 4469, 4460], ['dfALU', 4483, 4472, 4461], ['dfALU', 4483, 4474, 4462], ['dfALU', 4483, 4476, 4463], ['L1_ACCESS'], ['dfALU', 4488, 4469, 4465], ['dfALU', 4488, 4472, 4466], ['dfALU', 4488, 4474, 4467], ['dfALU', 4488, 4476, 4468], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4494, 4493, 4471], ['L1_ACCESS'], ['dfALU', 4494, 4496, 4473], ['L1_ACCESS'], ['dfALU', 4494, 4498, 4475], ['L1_ACCESS'], ['dfALU', 4494, 4500, 4477], ['L1_ACCESS'], ['dfALU', 4502, 4493, 4479], ['dfALU', 4502, 4496, 4480], ['dfALU', 4502, 4498, 4481], ['dfALU', 4502, 4500, 4482], ['L1_ACCESS'], ['dfALU', 4507, 4493, 4484], ['dfALU', 4507, 4496, 4485], ['dfALU', 4507, 4498, 4486], ['dfALU', 4507, 4500, 4487], ['L1_ACCESS'], ['dfALU', 4512, 4493, 4489], ['dfALU', 4512, 4496, 4490], ['dfALU', 4512, 4498, 4491], ['dfALU', 4512, 4500, 4492], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4518, 4517, 4495], ['L1_ACCESS'], ['dfALU', 4518, 4520, 4497], ['L1_ACCESS'], ['dfALU', 4518, 4522, 4499], ['L1_ACCESS'], ['dfALU', 4518, 4524, 4501], ['L1_ACCESS'], ['dfALU', 4526, 4517, 4503], ['dfALU', 4526, 4520, 4504], ['dfALU', 4526, 4522, 4505], ['dfALU', 4526, 4524, 4506], ['L1_ACCESS'], ['dfALU', 4531, 4517, 4508], ['dfALU', 4531, 4520, 4509], ['dfALU', 4531, 4522, 4510], ['dfALU', 4531, 4524, 4511], ['L1_ACCESS'], ['dfALU', 4536, 4517, 4513], ['dfALU', 4536, 4520, 4514], ['dfALU', 4536, 4522, 4515], ['dfALU', 4536, 4524, 4516], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4542, 4541, 4519], ['L1_ACCESS'], ['dfALU', 4542, 4544, 4521], ['L1_ACCESS'], ['dfALU', 4542, 4546, 4523], ['L1_ACCESS'], ['dfALU', 4542, 4548, 4525], ['L1_ACCESS'], ['dfALU', 4550, 4541, 4527], ['dfALU', 4550, 4544, 4528], ['dfALU', 4550, 4546, 4529], ['dfALU', 4550, 4548, 4530], ['L1_ACCESS'], ['dfALU', 4555, 4541, 4532], ['dfALU', 4555, 4544, 4533], ['dfALU', 4555, 4546, 4534], ['dfALU', 4555, 4548, 4535], ['L1_ACCESS'], ['dfALU', 4560, 4541, 4537], ['dfALU', 4560, 4544, 4538], ['dfALU', 4560, 4546, 4539], ['dfALU', 4560, 4548, 4540], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4566, 4565, 4543], ['L1_ACCESS'], ['dfALU', 4566, 4568, 4545], ['L1_ACCESS'], ['dfALU', 4566, 4570, 4547], ['L1_ACCESS'], ['dfALU', 4566, 4572, 4549], ['L1_ACCESS'], ['dfALU', 4574, 4565, 4551], ['dfALU', 4574, 4568, 4552], ['dfALU', 4574, 4570, 4553], ['dfALU', 4574, 4572, 4554], ['L1_ACCESS'], ['dfALU', 4579, 4565, 4556], ['dfALU', 4579, 4568, 4557], ['dfALU', 4579, 4570, 4558], ['dfALU', 4579, 4572, 4559], ['L1_ACCESS'], ['dfALU', 4584, 4565, 4561], ['dfALU', 4584, 4568, 4562], ['dfALU', 4584, 4570, 4563], ['dfALU', 4584, 4572, 4564], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4590, 4589, 4567], ['L1_ACCESS'], ['dfALU', 4590, 4592, 4569], ['L1_ACCESS'], ['dfALU', 4590, 4594, 4571], ['L1_ACCESS'], ['dfALU', 4590, 4596, 4573], ['L1_ACCESS'], ['dfALU', 4598, 4589, 4575], ['dfALU', 4598, 4592, 4576], ['dfALU', 4598, 4594, 4577], ['dfALU', 4598, 4596, 4578], ['L1_ACCESS'], ['dfALU', 4603, 4589, 4580], ['dfALU', 4603, 4592, 4581], ['dfALU', 4603, 4594, 4582], ['dfALU', 4603, 4596, 4583], ['L1_ACCESS'], ['dfALU', 4608, 4589, 4585], ['dfALU', 4608, 4592, 4586], ['dfALU', 4608, 4594, 4587], ['dfALU', 4608, 4596, 4588], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4614, 4613, 4591], ['L1_ACCESS'], ['dfALU', 4614, 4616, 4593], ['L1_ACCESS'], ['dfALU', 4614, 4618, 4595], ['L1_ACCESS'], ['dfALU', 4614, 4620, 4597], ['L1_ACCESS'], ['dfALU', 4622, 4613, 4599], ['dfALU', 4622, 4616, 4600], ['dfALU', 4622, 4618, 4601], ['dfALU', 4622, 4620, 4602], ['L1_ACCESS'], ['dfALU', 4627, 4613, 4604], ['dfALU', 4627, 4616, 4605], ['dfALU', 4627, 4618, 4606], ['dfALU', 4627, 4620, 4607], ['L1_ACCESS'], ['dfALU', 4632, 4613, 4609], ['dfALU', 4632, 4616, 4610], ['dfALU', 4632, 4618, 4611], ['dfALU', 4632, 4620, 4612], ['THREAD_SYNC'], ['L1_ACCESS', 4237], ['L1_ACCESS', 4239], ['L1_ACCESS', 4241], ['L1_ACCESS', 4243], ['L1_ACCESS', 4244], ['L1_ACCESS', 4247], ['L1_ACCESS', 4250], ['L1_ACCESS', 4252], ['THREAD_SYNC'], ['iALU', 4231], ['iALU', 4647, 65], ['iALU', 4233], ['iALU'], ['L1_ACCESS'], ['iALU'], ['L1_ACCESS', 67], ['L1_ACCESS', 29], ['dfALU', 4654, 4653, 4615], ['L1_ACCESS'], ['dfALU', 4654, 4656, 4617], ['L1_ACCESS'], ['dfALU', 4654, 4658, 4619], ['L1_ACCESS'], ['dfALU', 4654, 4660, 4621], ['L1_ACCESS'], ['dfALU', 4662, 4653, 4623], ['dfALU', 4662, 4656, 4624], ['dfALU', 4662, 4658, 4625], ['dfALU', 4662, 4660, 4626], ['L1_ACCESS'], ['dfALU', 4667, 4653, 4628], ['dfALU', 4667, 4656, 4629], ['dfALU', 4667, 4658, 4630], ['dfALU', 4667, 4660, 4631], ['L1_ACCESS'], ['dfALU', 4672, 4653, 4633], ['dfALU', 4672, 4656, 4634], ['dfALU', 4672, 4658, 4635], ['dfALU', 4672, 4660, 4636], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4678, 4677, 4655], ['L1_ACCESS'], ['dfALU', 4678, 4680, 4657], ['L1_ACCESS'], ['dfALU', 4678, 4682, 4659], ['L1_ACCESS'], ['dfALU', 4678, 4684, 4661], ['L1_ACCESS'], ['dfALU', 4686, 4677, 4663], ['dfALU', 4686, 4680, 4664], ['dfALU', 4686, 4682, 4665], ['dfALU', 4686, 4684, 4666], ['L1_ACCESS'], ['dfALU', 4691, 4677, 4668], ['dfALU', 4691, 4680, 4669], ['dfALU', 4691, 4682, 4670], ['dfALU', 4691, 4684, 4671], ['L1_ACCESS'], ['dfALU', 4696, 4677, 4673], ['dfALU', 4696, 4680, 4674], ['dfALU', 4696, 4682, 4675], ['dfALU', 4696, 4684, 4676], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4702, 4701, 4679], ['L1_ACCESS'], ['dfALU', 4702, 4704, 4681], ['L1_ACCESS'], ['dfALU', 4702, 4706, 4683], ['L1_ACCESS'], ['dfALU', 4702, 4708, 4685], ['L1_ACCESS'], ['dfALU', 4710, 4701, 4687], ['dfALU', 4710, 4704, 4688], ['dfALU', 4710, 4706, 4689], ['dfALU', 4710, 4708, 4690], ['L1_ACCESS'], ['dfALU', 4715, 4701, 4692], ['dfALU', 4715, 4704, 4693], ['dfALU', 4715, 4706, 4694], ['dfALU', 4715, 4708, 4695], ['L1_ACCESS'], ['dfALU', 4720, 4701, 4697], ['dfALU', 4720, 4704, 4698], ['dfALU', 4720, 4706, 4699], ['dfALU', 4720, 4708, 4700], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4726, 4725, 4703], ['L1_ACCESS'], ['dfALU', 4726, 4728, 4705], ['L1_ACCESS'], ['dfALU', 4726, 4730, 4707], ['L1_ACCESS'], ['dfALU', 4726, 4732, 4709], ['L1_ACCESS'], ['dfALU', 4734, 4725, 4711], ['dfALU', 4734, 4728, 4712], ['dfALU', 4734, 4730, 4713], ['dfALU', 4734, 4732, 4714], ['L1_ACCESS'], ['dfALU', 4739, 4725, 4716], ['dfALU', 4739, 4728, 4717], ['dfALU', 4739, 4730, 4718], ['dfALU', 4739, 4732, 4719], ['L1_ACCESS'], ['dfALU', 4744, 4725, 4721], ['dfALU', 4744, 4728, 4722], ['dfALU', 4744, 4730, 4723], ['dfALU', 4744, 4732, 4724], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4750, 4749, 4727], ['L1_ACCESS'], ['dfALU', 4750, 4752, 4729], ['L1_ACCESS'], ['dfALU', 4750, 4754, 4731], ['L1_ACCESS'], ['dfALU', 4750, 4756, 4733], ['L1_ACCESS'], ['dfALU', 4758, 4749, 4735], ['dfALU', 4758, 4752, 4736], ['dfALU', 4758, 4754, 4737], ['dfALU', 4758, 4756, 4738], ['L1_ACCESS'], ['dfALU', 4763, 4749, 4740], ['dfALU', 4763, 4752, 4741], ['dfALU', 4763, 4754, 4742], ['dfALU', 4763, 4756, 4743], ['L1_ACCESS'], ['dfALU', 4768, 4749, 4745], ['dfALU', 4768, 4752, 4746], ['dfALU', 4768, 4754, 4747], ['dfALU', 4768, 4756, 4748], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4774, 4773, 4751], ['L1_ACCESS'], ['dfALU', 4774, 4776, 4753], ['L1_ACCESS'], ['dfALU', 4774, 4778, 4755], ['L1_ACCESS'], ['dfALU', 4774, 4780, 4757], ['L1_ACCESS'], ['dfALU', 4782, 4773, 4759], ['dfALU', 4782, 4776, 4760], ['dfALU', 4782, 4778, 4761], ['dfALU', 4782, 4780, 4762], ['L1_ACCESS'], ['dfALU', 4787, 4773, 4764], ['dfALU', 4787, 4776, 4765], ['dfALU', 4787, 4778, 4766], ['dfALU', 4787, 4780, 4767], ['L1_ACCESS'], ['dfALU', 4792, 4773, 4769], ['dfALU', 4792, 4776, 4770], ['dfALU', 4792, 4778, 4771], ['dfALU', 4792, 4780, 4772], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4798, 4797, 4775], ['L1_ACCESS'], ['dfALU', 4798, 4800, 4777], ['L1_ACCESS'], ['dfALU', 4798, 4802, 4779], ['L1_ACCESS'], ['dfALU', 4798, 4804, 4781], ['L1_ACCESS'], ['dfALU', 4806, 4797, 4783], ['dfALU', 4806, 4800, 4784], ['dfALU', 4806, 4802, 4785], ['dfALU', 4806, 4804, 4786], ['L1_ACCESS'], ['dfALU', 4811, 4797, 4788], ['dfALU', 4811, 4800, 4789], ['dfALU', 4811, 4802, 4790], ['dfALU', 4811, 4804, 4791], ['L1_ACCESS'], ['dfALU', 4816, 4797, 4793], ['dfALU', 4816, 4800, 4794], ['dfALU', 4816, 4802, 4795], ['dfALU', 4816, 4804, 4796], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4822, 4821, 4799], ['L1_ACCESS'], ['dfALU', 4822, 4824, 4801], ['L1_ACCESS'], ['dfALU', 4822, 4826, 4803], ['L1_ACCESS'], ['dfALU', 4822, 4828, 4805], ['L1_ACCESS'], ['dfALU', 4830, 4821, 4807], ['dfALU', 4830, 4824, 4808], ['dfALU', 4830, 4826, 4809], ['dfALU', 4830, 4828, 4810], ['L1_ACCESS'], ['dfALU', 4835, 4821, 4812], ['dfALU', 4835, 4824, 4813], ['dfALU', 4835, 4826, 4814], ['dfALU', 4835, 4828, 4815], ['L1_ACCESS'], ['dfALU', 4840, 4821, 4817], ['dfALU', 4840, 4824, 4818], ['dfALU', 4840, 4826, 4819], ['dfALU', 4840, 4828, 4820], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4846, 4845, 4823], ['L1_ACCESS'], ['dfALU', 4846, 4848, 4825], ['L1_ACCESS'], ['dfALU', 4846, 4850, 4827], ['L1_ACCESS'], ['dfALU', 4846, 4852, 4829], ['L1_ACCESS'], ['dfALU', 4854, 4845, 4831], ['dfALU', 4854, 4848, 4832], ['dfALU', 4854, 4850, 4833], ['dfALU', 4854, 4852, 4834], ['L1_ACCESS'], ['dfALU', 4859, 4845, 4836], ['dfALU', 4859, 4848, 4837], ['dfALU', 4859, 4850, 4838], ['dfALU', 4859, 4852, 4839], ['L1_ACCESS'], ['dfALU', 4864, 4845, 4841], ['dfALU', 4864, 4848, 4842], ['dfALU', 4864, 4850, 4843], ['dfALU', 4864, 4852, 4844], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4870, 4869, 4847], ['L1_ACCESS'], ['dfALU', 4870, 4872, 4849], ['L1_ACCESS'], ['dfALU', 4870, 4874, 4851], ['L1_ACCESS'], ['dfALU', 4870, 4876, 4853], ['L1_ACCESS'], ['dfALU', 4878, 4869, 4855], ['dfALU', 4878, 4872, 4856], ['dfALU', 4878, 4874, 4857], ['dfALU', 4878, 4876, 4858], ['L1_ACCESS'], ['dfALU', 4883, 4869, 4860], ['dfALU', 4883, 4872, 4861], ['dfALU', 4883, 4874, 4862], ['dfALU', 4883, 4876, 4863], ['L1_ACCESS'], ['dfALU', 4888, 4869, 4865], ['dfALU', 4888, 4872, 4866], ['dfALU', 4888, 4874, 4867], ['dfALU', 4888, 4876, 4868], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4894, 4893, 4871], ['L1_ACCESS'], ['dfALU', 4894, 4896, 4873], ['L1_ACCESS'], ['dfALU', 4894, 4898, 4875], ['L1_ACCESS'], ['dfALU', 4894, 4900, 4877], ['L1_ACCESS'], ['dfALU', 4902, 4893, 4879], ['dfALU', 4902, 4896, 4880], ['dfALU', 4902, 4898, 4881], ['dfALU', 4902, 4900, 4882], ['L1_ACCESS'], ['dfALU', 4907, 4893, 4884], ['dfALU', 4907, 4896, 4885], ['dfALU', 4907, 4898, 4886], ['dfALU', 4907, 4900, 4887], ['L1_ACCESS'], ['dfALU', 4912, 4893, 4889], ['dfALU', 4912, 4896, 4890], ['dfALU', 4912, 4898, 4891], ['dfALU', 4912, 4900, 4892], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4918, 4917, 4895], ['L1_ACCESS'], ['dfALU', 4918, 4920, 4897], ['L1_ACCESS'], ['dfALU', 4918, 4922, 4899], ['L1_ACCESS'], ['dfALU', 4918, 4924, 4901], ['L1_ACCESS'], ['dfALU', 4926, 4917, 4903], ['dfALU', 4926, 4920, 4904], ['dfALU', 4926, 4922, 4905], ['dfALU', 4926, 4924, 4906], ['L1_ACCESS'], ['dfALU', 4931, 4917, 4908], ['dfALU', 4931, 4920, 4909], ['dfALU', 4931, 4922, 4910], ['dfALU', 4931, 4924, 4911], ['L1_ACCESS'], ['dfALU', 4936, 4917, 4913], ['dfALU', 4936, 4920, 4914], ['dfALU', 4936, 4922, 4915], ['dfALU', 4936, 4924, 4916], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4942, 4941, 4919], ['L1_ACCESS'], ['dfALU', 4942, 4944, 4921], ['L1_ACCESS'], ['dfALU', 4942, 4946, 4923], ['L1_ACCESS'], ['dfALU', 4942, 4948, 4925], ['L1_ACCESS'], ['dfALU', 4950, 4941, 4927], ['dfALU', 4950, 4944, 4928], ['dfALU', 4950, 4946, 4929], ['dfALU', 4950, 4948, 4930], ['L1_ACCESS'], ['dfALU', 4955, 4941, 4932], ['dfALU', 4955, 4944, 4933], ['dfALU', 4955, 4946, 4934], ['dfALU', 4955, 4948, 4935], ['L1_ACCESS'], ['dfALU', 4960, 4941, 4937], ['dfALU', 4960, 4944, 4938], ['dfALU', 4960, 4946, 4939], ['dfALU', 4960, 4948, 4940], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4966, 4965, 4943], ['L1_ACCESS'], ['dfALU', 4966, 4968, 4945], ['L1_ACCESS'], ['dfALU', 4966, 4970, 4947], ['L1_ACCESS'], ['dfALU', 4966, 4972, 4949], ['L1_ACCESS'], ['dfALU', 4974, 4965, 4951], ['dfALU', 4974, 4968, 4952], ['dfALU', 4974, 4970, 4953], ['dfALU', 4974, 4972, 4954], ['L1_ACCESS'], ['dfALU', 4979, 4965, 4956], ['dfALU', 4979, 4968, 4957], ['dfALU', 4979, 4970, 4958], ['dfALU', 4979, 4972, 4959], ['L1_ACCESS'], ['dfALU', 4984, 4965, 4961], ['dfALU', 4984, 4968, 4962], ['dfALU', 4984, 4970, 4963], ['dfALU', 4984, 4972, 4964], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 4990, 4989, 4967], ['L1_ACCESS'], ['dfALU', 4990, 4992, 4969], ['L1_ACCESS'], ['dfALU', 4990, 4994, 4971], ['L1_ACCESS'], ['dfALU', 4990, 4996, 4973], ['L1_ACCESS'], ['dfALU', 4998, 4989, 4975], ['dfALU', 4998, 4992, 4976], ['dfALU', 4998, 4994, 4977], ['dfALU', 4998, 4996, 4978], ['L1_ACCESS'], ['dfALU', 5003, 4989, 4980], ['dfALU', 5003, 4992, 4981], ['dfALU', 5003, 4994, 4982], ['dfALU', 5003, 4996, 4983], ['L1_ACCESS'], ['dfALU', 5008, 4989, 4985], ['dfALU', 5008, 4992, 4986], ['dfALU', 5008, 4994, 4987], ['dfALU', 5008, 4996, 4988], ['L1_ACCESS'], ['L1_ACCESS'], ['dfALU', 5014, 5013, 4991], ['L1_ACCESS'], ['dfALU', 5014, 5016, 4993], ['L1_ACCESS'], ['dfALU', 5014, 5018, 4995], ['L1_ACCESS'], ['dfALU', 5014, 5020, 4997], ['L1_ACCESS'], ['dfALU', 5022, 5013, 4999], ['dfALU', 5022, 5016, 5000], ['dfALU', 5022, 5018, 5001], ['dfALU', 5022, 5020, 5002], ['L1_ACCESS'], ['dfALU', 5027, 5013, 5004], ['dfALU', 5027, 5016, 5005], ['dfALU', 5027, 5018, 5006], ['dfALU', 5027, 5020, 5007], ['L1_ACCESS'], ['dfALU', 5032, 5013, 5009], ['dfALU', 5032, 5016, 5010], ['dfALU', 5032, 5018, 5011], ['dfALU', 5032, 5020, 5012], ['iALU', 18, 4652], ['iALU', 5037, 4651], ['iALU', 5038, 17], ['iALU', 5039, 4650], ['iALU', 5037, 2], ['iALU', 17, 4650], ['iALU', 5042, 1], ['iALU', 5043, 5041], ['diALU', 0], ['iALU', 5040], ['diALU', 5045, 5046], ['GLOB_MEM_ACCESS', 5047], ['dfALU', 5048, 7], ['dfALU', 5015, 6, 5049], ['GLOB_MEM_ACCESS', 5050], ['iALU', 5042], ['iALU', 5052, 1], ['iALU', 5053, 5041], ['diALU', 0], ['iALU', 5040], ['diALU', 5055, 5056], ['GLOB_MEM_ACCESS'], ['dfALU', 5058, 7], ['dfALU', 5023, 6, 5059], ['GLOB_MEM_ACCESS', 5060], ['iALU', 5042], ['iALU', 5062, 1], ['iALU', 5063, 5041], ['diALU', 0], ['iALU', 5040], ['diALU', 5065, 5066], ['GLOB_MEM_ACCESS'], ['dfALU', 5068, 7], ['dfALU', 5028, 6, 5069], ['GLOB_MEM_ACCESS', 5070], ['iALU', 5042], ['iALU', 5072, 1], ['iALU', 5073, 5041], ['diALU', 0], ['iALU', 5040], ['diALU', 5075, 5076], ['GLOB_MEM_ACCESS'], ['dfALU', 5078, 7], ['dfALU', 5033, 6, 5079], ['GLOB_MEM_ACCESS', 5080], ['L1_ACCESS'], ['iALU', 5082], ['iALU', 5083], ['iALU', 5040], ['diALU', 5085, 5084], ['diALU', 0], ['diALU', 5086], ['diALU', 5087, 5088], ['iALU', 5037], ['iALU', 5090, 2], ['iALU', 5043, 5091], ['GLOB_MEM_ACCESS', 5089], ['dfALU', 5093, 7], ['dfALU', 5017, 6, 5094], ['GLOB_MEM_ACCESS', 5095], ['iALU', 5053, 5091], ['GLOB_MEM_ACCESS'], ['dfALU', 5098, 7], ['dfALU', 5024, 6, 5099], ['GLOB_MEM_ACCESS', 5100], ['iALU', 5063, 5091], ['GLOB_MEM_ACCESS'], ['dfALU', 5103, 7], ['dfALU', 5029, 6, 5104], ['GLOB_MEM_ACCESS', 5105], ['iALU', 5073, 5091], ['GLOB_MEM_ACCESS'], ['dfALU', 5108, 7], ['dfALU', 5034, 6, 5109], ['GLOB_MEM_ACCESS', 5110], ['iALU', 5083], ['iALU', 5040], ['diALU', 5113, 5112], ['diALU', 5114], ['diALU', 5087, 5115], ['iALU', 5037], ['iALU', 5117, 2], ['iALU', 5043, 5118], ['GLOB_MEM_ACCESS', 5116], ['dfALU', 5120, 7], ['dfALU', 5019, 6, 5121], ['GLOB_MEM_ACCESS', 5122], ['iALU', 5053, 5118], ['GLOB_MEM_ACCESS'], ['dfALU', 5125, 7], ['dfALU', 5025, 6, 5126], ['GLOB_MEM_ACCESS', 5127], ['iALU', 5063, 5118], ['GLOB_MEM_ACCESS'], ['dfALU', 5130, 7], ['dfALU', 5030, 6, 5131], ['GLOB_MEM_ACCESS', 5132], ['iALU', 5073, 5118], ['GLOB_MEM_ACCESS'], ['dfALU', 5135, 7], ['dfALU', 5035, 6, 5136], ['GLOB_MEM_ACCESS', 5137], ['iALU', 5083], ['iALU', 5040], ['diALU', 5139, 5140], ['diALU', 5141], ['diALU', 5087, 5142], ['iALU', 5037], ['iALU', 5144, 2], ['iALU', 5043, 5145], ['GLOB_MEM_ACCESS', 5143], ['dfALU', 5147, 7], ['dfALU', 5021, 6, 5148], ['GLOB_MEM_ACCESS', 5149], ['iALU', 5053, 5145], ['GLOB_MEM_ACCESS'], ['dfALU', 5152, 7], ['dfALU', 5026, 6, 5153], ['GLOB_MEM_ACCESS', 5154], ['iALU', 5063, 5145], ['GLOB_MEM_ACCESS'], ['dfALU', 5157, 7], ['dfALU', 5031, 6, 5158], ['GLOB_MEM_ACCESS', 5159], ['iALU', 5073, 5145], ['GLOB_MEM_ACCESS'], ['dfALU', 5162, 7], ['dfALU', 5036, 6, 5163], ['GLOB_MEM_ACCESS', 5164]] 

    if ns == 6:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter, stats) = core.time_compute(CPU_tasklist2, simianEngine.now, True)
      print "Time for a single kernel call = ", time_iter
      time = time_iter*nb_iter
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 8:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      time = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")   
    elif ns == 10:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6 
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 12:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 14:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 16:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 18:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += time_iter_21*nb_iter_21/gridsize_21 + time_iter_22*nb_iter_22/gridsize_22
      time += time_iter_23*nb_iter_23/gridsize_23 + time_iter_24*nb_iter_24/gridsize_24
      time += time_iter_25*nb_iter_25/gridsize_25 + time_iter_26*nb_iter_26/gridsize_26
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 20:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += time_iter_21*nb_iter_21/gridsize_21 + time_iter_22*nb_iter_22/gridsize_22
      time += time_iter_23*nb_iter_23/gridsize_23 + time_iter_24*nb_iter_24/gridsize_24
      time += time_iter_25*nb_iter_25/gridsize_25 + time_iter_26*nb_iter_26/gridsize_26
      time += time_iter_27*nb_iter_27/gridsize_27 + time_iter_28*nb_iter_28/gridsize_28
      time += time_iter_29*nb_iter_29/gridsize_29 + time_iter_30*nb_iter_30/gridsize_30
      time += time_iter_31*nb_iter_31/gridsize_31 + time_iter_32*nb_iter_32/gridsize_32
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 22:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += time_iter_21*nb_iter_21/gridsize_21 + time_iter_22*nb_iter_22/gridsize_22
      time += time_iter_23*nb_iter_23/gridsize_23 + time_iter_24*nb_iter_24/gridsize_24
      time += time_iter_25*nb_iter_25/gridsize_25 + time_iter_26*nb_iter_26/gridsize_26
      time += time_iter_27*nb_iter_27/gridsize_27 + time_iter_28*nb_iter_28/gridsize_28
      time += time_iter_29*nb_iter_29/gridsize_29 + time_iter_30*nb_iter_30/gridsize_30
      time += time_iter_31*nb_iter_31/gridsize_31 + time_iter_32*nb_iter_32/gridsize_32
      time += time_iter_33*nb_iter_33/gridsize_33 + time_iter_34*nb_iter_34/gridsize_34
      time += time_iter_35*nb_iter_35/gridsize_35 + time_iter_36*nb_iter_36/gridsize_36
      time += time_iter_37*nb_iter_37/gridsize_37 + time_iter_38*nb_iter_38/gridsize_38
      time += time_iter_39*nb_iter_39/gridsize_39
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 24:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47   
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += time_iter_21*nb_iter_21/gridsize_21 + time_iter_22*nb_iter_22/gridsize_22
      time += time_iter_23*nb_iter_23/gridsize_23 + time_iter_24*nb_iter_24/gridsize_24
      time += time_iter_25*nb_iter_25/gridsize_25 + time_iter_26*nb_iter_26/gridsize_26
      time += time_iter_27*nb_iter_27/gridsize_27 + time_iter_28*nb_iter_28/gridsize_28
      time += time_iter_29*nb_iter_29/gridsize_29 + time_iter_30*nb_iter_30/gridsize_30
      time += time_iter_31*nb_iter_31/gridsize_31 + time_iter_32*nb_iter_32/gridsize_32
      time += time_iter_33*nb_iter_33/gridsize_33 + time_iter_34*nb_iter_34/gridsize_34
      time += time_iter_35*nb_iter_35/gridsize_35 + time_iter_36*nb_iter_36/gridsize_36
      time += time_iter_37*nb_iter_37/gridsize_37 + time_iter_38*nb_iter_38/gridsize_38
      time += time_iter_39*nb_iter_39/gridsize_39 + time_iter_40*nb_iter_40/gridsize_40
      time += time_iter_41*nb_iter_41/gridsize_41 + time_iter_42*nb_iter_42/gridsize_42
      time += time_iter_43*nb_iter_43/gridsize_43 + time_iter_44*nb_iter_44/gridsize_44
      time += time_iter_45*nb_iter_45/gridsize_45 + time_iter_46*nb_iter_46/gridsize_46
      time += time_iter_47*nb_iter_47/gridsize_47
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 26:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += time_iter_21*nb_iter_21/gridsize_21 + time_iter_22*nb_iter_22/gridsize_22
      time += time_iter_23*nb_iter_23/gridsize_23 + time_iter_24*nb_iter_24/gridsize_24
      time += time_iter_25*nb_iter_25/gridsize_25 + time_iter_26*nb_iter_26/gridsize_26
      time += time_iter_27*nb_iter_27/gridsize_27 + time_iter_28*nb_iter_28/gridsize_28
      time += time_iter_29*nb_iter_29/gridsize_29 + time_iter_30*nb_iter_30/gridsize_30
      time += time_iter_31*nb_iter_31/gridsize_31 + time_iter_32*nb_iter_32/gridsize_32
      time += time_iter_33*nb_iter_33/gridsize_33 + time_iter_34*nb_iter_34/gridsize_34
      time += time_iter_35*nb_iter_35/gridsize_35 + time_iter_36*nb_iter_36/gridsize_36
      time += time_iter_37*nb_iter_37/gridsize_37 + time_iter_38*nb_iter_38/gridsize_38
      time += time_iter_39*nb_iter_39/gridsize_39 + time_iter_40*nb_iter_40/gridsize_40
      time += time_iter_41*nb_iter_41/gridsize_41 + time_iter_42*nb_iter_42/gridsize_42
      time += time_iter_43*nb_iter_43/gridsize_43 + time_iter_44*nb_iter_44/gridsize_44
      time += time_iter_45*nb_iter_45/gridsize_45 + time_iter_46*nb_iter_46/gridsize_46
      time += time_iter_47*nb_iter_47/gridsize_47 + time_iter_48*nb_iter_48/gridsize_48
      time += time_iter_49*nb_iter_49/gridsize_49 + time_iter_50*nb_iter_50/gridsize_50
      time += time_iter_51*nb_iter_51/gridsize_51 + time_iter_52*nb_iter_52/gridsize_52
      time += time_iter_53*nb_iter_53/gridsize_53 + time_iter_54*nb_iter_54/gridsize_54
      time += time_iter_55*nb_iter_55/gridsize_55 + time_iter_56*nb_iter_56/gridsize_56
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 28:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_57 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_57,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_58 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_58,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_59 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_59,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      (time_iter_57, stats) = core.time_compute(CPU_tasklist2_57, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_57 = ", time_iter_57
      (time_iter_58, stats) = core.time_compute(CPU_tasklist2_58, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_58 = ", time_iter_58
      (time_iter_59, stats) = core.time_compute(CPU_tasklist2_59, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_59 = ", time_iter_59
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += (3.0/7.0)*time_iter_13*nb_iter_13/gridsize_13 + (3.0/7.0)*time_iter_14*nb_iter_14/gridsize_14
      time += (3.0/7.0)*time_iter_15*nb_iter_15/gridsize_15 + (3.0/7.0)*time_iter_16*nb_iter_16/gridsize_16
      time += (3.0/7.0)*time_iter_17*nb_iter_17/gridsize_17 + (3.0/7.0)*time_iter_18*nb_iter_18/gridsize_18
      time += (3.0/7.0)*time_iter_19*nb_iter_19/gridsize_19 + (3.0/7.0)*time_iter_20*nb_iter_20/gridsize_20
      time += (3.0/7.0)*time_iter_21*nb_iter_21/gridsize_21 + (3.0/7.0)*time_iter_22*nb_iter_22/gridsize_22
      time += (3.0/7.0)*time_iter_23*nb_iter_23/gridsize_23 + (3.0/7.0)*time_iter_24*nb_iter_24/gridsize_24
      time += (3.0/7.0)*time_iter_25*nb_iter_25/gridsize_25 + (3.0/7.0)*time_iter_26*nb_iter_26/gridsize_26
      time += (3.0/7.0)*time_iter_27*nb_iter_27/gridsize_27 + (3.0/7.0)*time_iter_28*nb_iter_28/gridsize_28
      time += (3.0/7.0)*time_iter_29*nb_iter_29/gridsize_29 + (3.0/7.0)*time_iter_30*nb_iter_30/gridsize_30
      time += (3.0/7.0)*time_iter_31*nb_iter_31/gridsize_31 + (3.0/7.0)*time_iter_32*nb_iter_32/gridsize_32
      time += (3.0/7.0)*time_iter_33*nb_iter_33/gridsize_33 + (3.0/7.0)*time_iter_34*nb_iter_34/gridsize_34
      time += (3.0/7.0)*time_iter_35*nb_iter_35/gridsize_35 + (3.0/7.0)*time_iter_36*nb_iter_36/gridsize_36
      time += (3.0/11.0)*time_iter_37*nb_iter_37/gridsize_37 + (3.0/11.0)*time_iter_38*nb_iter_38/gridsize_38
      time += (3.0/11.0)*time_iter_39*nb_iter_39/gridsize_39 + (3.0/11.0)*time_iter_40*nb_iter_40/gridsize_40
      time += (3.0/11.0)*time_iter_41*nb_iter_41/gridsize_41 + (3.0/11.0)*time_iter_42*nb_iter_42/gridsize_42
      time += (3.0/11.0)*time_iter_43*nb_iter_43/gridsize_43 + (3.0/11.0)*time_iter_44*nb_iter_44/gridsize_44
      time += (3.0/11.0)*time_iter_45*nb_iter_45/gridsize_45 + (3.0/11.0)*time_iter_46*nb_iter_46/gridsize_46
      time += (3.0/11.0)*time_iter_47*nb_iter_47/gridsize_47 + (3.0/11.0)*time_iter_48*nb_iter_48/gridsize_48
      time += (3.0/11.0)*time_iter_49*nb_iter_49/gridsize_49 + (3.0/11.0)*time_iter_50*nb_iter_50/gridsize_50
      time += (3.0/11.0)*time_iter_51*nb_iter_51/gridsize_51 + (3.0/11.0)*time_iter_52*nb_iter_52/gridsize_52
      time += (3.0/11.0)*time_iter_53*nb_iter_53/gridsize_53 + (3.0/11.0)*time_iter_54*nb_iter_54/gridsize_54
      time += (3.0/11.0)*time_iter_55*nb_iter_55/gridsize_55 + (3.0/11.0)*time_iter_56*nb_iter_56/gridsize_56
      time += (3.0/11.0)*time_iter_57*nb_iter_57/gridsize_57 + (3.0/11.0)*time_iter_58*nb_iter_58/gridsize_58
      time += (3.0/11.0)*time_iter_59*nb_iter_59/gridsize_59
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 30:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_57 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_57,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_58 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_58,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_59 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_59,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_60 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_60,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_61 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_61,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_62 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_62,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_63 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_63,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_64 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_64,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_65 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_65,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_66 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_66,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_67 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_67,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_68 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_68,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_69 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_69,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      (time_iter_57, stats) = core.time_compute(CPU_tasklist2_57, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_57 = ", time_iter_57
      (time_iter_58, stats) = core.time_compute(CPU_tasklist2_58, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_58 = ", time_iter_58
      (time_iter_59, stats) = core.time_compute(CPU_tasklist2_59, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_59 = ", time_iter_59
      (time_iter_60, stats) = core.time_compute(CPU_tasklist2_60, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_60 = ", time_iter_60
      (time_iter_61, stats) = core.time_compute(CPU_tasklist2_61, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_61 = ", time_iter_61
      (time_iter_62, stats) = core.time_compute(CPU_tasklist2_62, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_62 = ", time_iter_62
      (time_iter_63, stats) = core.time_compute(CPU_tasklist2_63, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_63 = ", time_iter_63
      (time_iter_64, stats) = core.time_compute(CPU_tasklist2_64, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_64 = ", time_iter_64
      (time_iter_65, stats) = core.time_compute(CPU_tasklist2_65, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_65 = ", time_iter_65
      (time_iter_66, stats) = core.time_compute(CPU_tasklist2_66, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_66 = ", time_iter_66
      (time_iter_67, stats) = core.time_compute(CPU_tasklist2_67, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_67 = ", time_iter_67
      (time_iter_68, stats) = core.time_compute(CPU_tasklist2_68, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_68 = ", time_iter_68
      (time_iter_69, stats) = core.time_compute(CPU_tasklist2_69, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_69 = ", time_iter_69
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += (3.0/7.0)*time_iter_15*nb_iter_15/gridsize_15 + (3.0/7.0)*time_iter_16*nb_iter_16/gridsize_16
      time += (3.0/7.0)*time_iter_17*nb_iter_17/gridsize_17 + (3.0/7.0)*time_iter_18*nb_iter_18/gridsize_18
      time += (3.0/7.0)*time_iter_19*nb_iter_19/gridsize_19 + (3.0/7.0)*time_iter_20*nb_iter_20/gridsize_20
      time += (3.0/7.0)*time_iter_21*nb_iter_21/gridsize_21 + (3.0/7.0)*time_iter_22*nb_iter_22/gridsize_22
      time += (3.0/7.0)*time_iter_23*nb_iter_23/gridsize_23 + (3.0/7.0)*time_iter_24*nb_iter_24/gridsize_24
      time += (3.0/7.0)*time_iter_25*nb_iter_25/gridsize_25 + (3.0/7.0)*time_iter_26*nb_iter_26/gridsize_26
      time += (3.0/7.0)*time_iter_27*nb_iter_27/gridsize_27 + (3.0/7.0)*time_iter_28*nb_iter_28/gridsize_28
      time += (3.0/7.0)*time_iter_29*nb_iter_29/gridsize_29 + (3.0/7.0)*time_iter_30*nb_iter_30/gridsize_30
      time += (3.0/7.0)*time_iter_31*nb_iter_31/gridsize_31 + (3.0/7.0)*time_iter_32*nb_iter_32/gridsize_32
      time += (3.0/7.0)*time_iter_33*nb_iter_33/gridsize_33 + (3.0/7.0)*time_iter_34*nb_iter_34/gridsize_34
      time += (3.0/7.0)*time_iter_35*nb_iter_35/gridsize_35 + (3.0/7.0)*time_iter_36*nb_iter_36/gridsize_36
      time += (3.0/7.0)*time_iter_37*nb_iter_37/gridsize_37 + (3.0/7.0)*time_iter_38*nb_iter_38/gridsize_38
      time += (3.0/7.0)*time_iter_39*nb_iter_39/gridsize_39 + (3.0/7.0)*time_iter_40*nb_iter_40/gridsize_40
      time += (3.0/7.0)*time_iter_41*nb_iter_41/gridsize_41 + (3.0/7.0)*time_iter_42*nb_iter_42/gridsize_42
      time += (3.0/11.0)*time_iter_43*nb_iter_43/gridsize_43 + (3.0/11.0)*time_iter_44*nb_iter_44/gridsize_44
      time += (3.0/11.0)*time_iter_45*nb_iter_45/gridsize_45 + (3.0/11.0)*time_iter_46*nb_iter_46/gridsize_46
      time += (3.0/11.0)*time_iter_47*nb_iter_47/gridsize_47 + (3.0/11.0)*time_iter_48*nb_iter_48/gridsize_48
      time += (3.0/11.0)*time_iter_49*nb_iter_49/gridsize_49 + (3.0/11.0)*time_iter_50*nb_iter_50/gridsize_50
      time += (3.0/11.0)*time_iter_51*nb_iter_51/gridsize_51 + (3.0/11.0)*time_iter_52*nb_iter_52/gridsize_52
      time += (3.0/11.0)*time_iter_53*nb_iter_53/gridsize_53 + (3.0/11.0)*time_iter_54*nb_iter_54/gridsize_54
      time += (3.0/11.0)*time_iter_55*nb_iter_55/gridsize_55 + (3.0/11.0)*time_iter_56*nb_iter_56/gridsize_56
      time += (3.0/11.0)*time_iter_57*nb_iter_57/gridsize_57 + (3.0/11.0)*time_iter_58*nb_iter_58/gridsize_58
      time += (3.0/11.0)*time_iter_59*nb_iter_59/gridsize_59 + (3.0/11.0)*time_iter_60*nb_iter_60/gridsize_60
      time += (3.0/11.0)*time_iter_61*nb_iter_61/gridsize_61 + (3.0/11.0)*time_iter_62*nb_iter_62/gridsize_62
      time += (3.0/11.0)*time_iter_63*nb_iter_63/gridsize_63 + (3.0/11.0)*time_iter_64*nb_iter_64/gridsize_64
      time += (3.0/11.0)*time_iter_65*nb_iter_65/gridsize_65 + (3.0/11.0)*time_iter_66*nb_iter_66/gridsize_66
      time += (3.0/11.0)*time_iter_67*nb_iter_67/gridsize_67 + (3.0/11.0)*time_iter_68*nb_iter_68/gridsize_68
      time += (3.0/11.0)*time_iter_69*nb_iter_69/gridsize_69
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 32:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_57 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_57,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_58 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_58,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_59 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_59,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_60 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_60,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_61 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_61,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      (time_iter_57, stats) = core.time_compute(CPU_tasklist2_57, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_57 = ", time_iter_57
      (time_iter_58, stats) = core.time_compute(CPU_tasklist2_58, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_58 = ", time_iter_58
      (time_iter_59, stats) = core.time_compute(CPU_tasklist2_59, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_59 = ", time_iter_59
      (time_iter_60, stats) = core.time_compute(CPU_tasklist2_60, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_60 = ", time_iter_60
      (time_iter_61, stats) = core.time_compute(CPU_tasklist2_61, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_61 = ", time_iter_61
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += (7.0/7.0)*time_iter_15*nb_iter_15/gridsize_15 + (7.0/7.0)*time_iter_16*nb_iter_16/gridsize_16
      time += (7.0/7.0)*time_iter_17*nb_iter_17/gridsize_17 + (7.0/7.0)*time_iter_18*nb_iter_18/gridsize_18
      time += (7.0/7.0)*time_iter_19*nb_iter_19/gridsize_19 + (7.0/7.0)*time_iter_20*nb_iter_20/gridsize_20
      time += (7.0/7.0)*time_iter_21*nb_iter_21/gridsize_21 + (7.0/7.0)*time_iter_22*nb_iter_22/gridsize_22
      time += (7.0/7.0)*time_iter_23*nb_iter_23/gridsize_23 + (7.0/7.0)*time_iter_24*nb_iter_24/gridsize_24
      time += (7.0/7.0)*time_iter_25*nb_iter_25/gridsize_25 + (7.0/7.0)*time_iter_26*nb_iter_26/gridsize_26
      time += (7.0/7.0)*time_iter_27*nb_iter_27/gridsize_27 + (7.0/7.0)*time_iter_28*nb_iter_28/gridsize_28
      time += (7.0/7.0)*time_iter_29*nb_iter_29/gridsize_29 + (7.0/7.0)*time_iter_30*nb_iter_30/gridsize_30
      time += (7.0/7.0)*time_iter_31*nb_iter_31/gridsize_31 + (7.0/7.0)*time_iter_32*nb_iter_32/gridsize_32
      time += (7.0/7.0)*time_iter_33*nb_iter_33/gridsize_33 + (7.0/7.0)*time_iter_34*nb_iter_34/gridsize_34
      time += (7.0/7.0)*time_iter_35*nb_iter_35/gridsize_35 + (7.0/11.0)*time_iter_36*nb_iter_36/gridsize_36
      time += (7.0/11.0)*time_iter_37*nb_iter_37/gridsize_37 + (7.0/11.0)*time_iter_38*nb_iter_38/gridsize_38
      time += (7.0/11.0)*time_iter_39*nb_iter_39/gridsize_39 + (7.0/11.0)*time_iter_40*nb_iter_40/gridsize_40
      time += (7.0/11.0)*time_iter_41*nb_iter_41/gridsize_41 + (7.0/11.0)*time_iter_42*nb_iter_42/gridsize_42
      time += (7.0/11.0)*time_iter_43*nb_iter_43/gridsize_43 + (7.0/11.0)*time_iter_44*nb_iter_44/gridsize_44
      time += (7.0/11.0)*time_iter_45*nb_iter_45/gridsize_45 + (7.0/11.0)*time_iter_46*nb_iter_46/gridsize_46
      time += (7.0/11.0)*time_iter_47*nb_iter_47/gridsize_47 + (7.0/11.0)*time_iter_48*nb_iter_48/gridsize_48
      time += (7.0/11.0)*time_iter_49*nb_iter_49/gridsize_49 + (7.0/11.0)*time_iter_50*nb_iter_50/gridsize_50
      time += (7.0/11.0)*time_iter_51*nb_iter_51/gridsize_51 + (7.0/11.0)*time_iter_52*nb_iter_52/gridsize_52
      time += (7.0/11.0)*time_iter_53*nb_iter_53/gridsize_53 + (7.0/11.0)*time_iter_54*nb_iter_54/gridsize_54
      time += (7.0/11.0)*time_iter_55*nb_iter_55/gridsize_55 + (7.0/11.0)*time_iter_56*nb_iter_56/gridsize_56
      time += (7.0/11.0)*time_iter_57*nb_iter_57/gridsize_57 + (7.0/11.0)*time_iter_58*nb_iter_58/gridsize_58
      time += (7.0/11.0)*time_iter_59*nb_iter_59/gridsize_59 + (7.0/11.0)*time_iter_60*nb_iter_60/gridsize_60
      time += (7.0/11.0)*time_iter_61*nb_iter_61/gridsize_61
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 34:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_57 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_57,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_58 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_58,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_59 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_59,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_60 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_60,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_61 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_61,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_62 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_62,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_63 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_63,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_64 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_64,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_65 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_65,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_66 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_66,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_67 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_67,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_68 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_68,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_69 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_69,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_70 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_70,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_71 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_71,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_72 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_72,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_73 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_73,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_74 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_74,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_75 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_75,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_76 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_76,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_77 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_77,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_78 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_78,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_79 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_79,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_80 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_80,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_81 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_81,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_82 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_82,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_83 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_83,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_84 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_84,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_85 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_85,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_86 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_86,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_87 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_87,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_88 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_88,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_89 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_89,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      # Compute time for a single iteration
      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      (time_iter_57, stats) = core.time_compute(CPU_tasklist2_57, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_57 = ", time_iter_57
      (time_iter_58, stats) = core.time_compute(CPU_tasklist2_58, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_58 = ", time_iter_58
      (time_iter_59, stats) = core.time_compute(CPU_tasklist2_59, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_59 = ", time_iter_59
      (time_iter_60, stats) = core.time_compute(CPU_tasklist2_60, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_60 = ", time_iter_60
      (time_iter_61, stats) = core.time_compute(CPU_tasklist2_61, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_61 = ", time_iter_61
      (time_iter_62, stats) = core.time_compute(CPU_tasklist2_62, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_62 = ", time_iter_62
      (time_iter_63, stats) = core.time_compute(CPU_tasklist2_63, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_63 = ", time_iter_63
      (time_iter_64, stats) = core.time_compute(CPU_tasklist2_64, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_64 = ", time_iter_64
      (time_iter_65, stats) = core.time_compute(CPU_tasklist2_65, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_65 = ", time_iter_65
      (time_iter_66, stats) = core.time_compute(CPU_tasklist2_66, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_66 = ", time_iter_66
      (time_iter_67, stats) = core.time_compute(CPU_tasklist2_67, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_67 = ", time_iter_67
      (time_iter_68, stats) = core.time_compute(CPU_tasklist2_68, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_68 = ", time_iter_68
      (time_iter_69, stats) = core.time_compute(CPU_tasklist2_69, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_69 = ", time_iter_69
      (time_iter_70, stats) = core.time_compute(CPU_tasklist2_70, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_70 = ", time_iter_70
      (time_iter_71, stats) = core.time_compute(CPU_tasklist2_71, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_71 = ", time_iter_71
      (time_iter_72, stats) = core.time_compute(CPU_tasklist2_72, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_72 = ", time_iter_72
      (time_iter_73, stats) = core.time_compute(CPU_tasklist2_73, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_73 = ", time_iter_73
      (time_iter_74, stats) = core.time_compute(CPU_tasklist2_74, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_74 = ", time_iter_74
      (time_iter_75, stats) = core.time_compute(CPU_tasklist2_75, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_75 = ", time_iter_75
      (time_iter_76, stats) = core.time_compute(CPU_tasklist2_76, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_76 = ", time_iter_76
      (time_iter_77, stats) = core.time_compute(CPU_tasklist2_77, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_77 = ", time_iter_77
      (time_iter_78, stats) = core.time_compute(CPU_tasklist2_78, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_78 = ", time_iter_78
      (time_iter_79, stats) = core.time_compute(CPU_tasklist2_79, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_79 = ", time_iter_79
      (time_iter_80, stats) = core.time_compute(CPU_tasklist2_80, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_80 = ", time_iter_80
      (time_iter_81, stats) = core.time_compute(CPU_tasklist2_81, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_81 = ", time_iter_81
      (time_iter_82, stats) = core.time_compute(CPU_tasklist2_82, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_82 = ", time_iter_82
      (time_iter_83, stats) = core.time_compute(CPU_tasklist2_83, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_83 = ", time_iter_83
      (time_iter_84, stats) = core.time_compute(CPU_tasklist2_84, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_84 = ", time_iter_84
      (time_iter_85, stats) = core.time_compute(CPU_tasklist2_85, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_85 = ", time_iter_85
      (time_iter_86, stats) = core.time_compute(CPU_tasklist2_86, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_86 = ", time_iter_86
      (time_iter_87, stats) = core.time_compute(CPU_tasklist2_87, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_87 = ", time_iter_87
      (time_iter_88, stats) = core.time_compute(CPU_tasklist2_88, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_88 = ", time_iter_88
      (time_iter_89, stats) = core.time_compute(CPU_tasklist2_89, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_89 = ", time_iter_89
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += (3.0/7.0)*time_iter_19*nb_iter_19/gridsize_19 + (3.0/7.0)*time_iter_20*nb_iter_20/gridsize_20
      time += (3.0/7.0)*time_iter_21*nb_iter_21/gridsize_21 + (3.0/7.0)*time_iter_22*nb_iter_22/gridsize_22
      time += (3.0/7.0)*time_iter_23*nb_iter_23/gridsize_23 + (3.0/7.0)*time_iter_24*nb_iter_24/gridsize_24
      time += (3.0/7.0)*time_iter_25*nb_iter_25/gridsize_25 + (3.0/7.0)*time_iter_26*nb_iter_26/gridsize_26
      time += (3.0/7.0)*time_iter_27*nb_iter_27/gridsize_27 + (3.0/7.0)*time_iter_28*nb_iter_28/gridsize_28
      time += (3.0/7.0)*time_iter_29*nb_iter_29/gridsize_29 + (3.0/7.0)*time_iter_30*nb_iter_30/gridsize_30
      time += (3.0/7.0)*time_iter_31*nb_iter_31/gridsize_31 + (3.0/7.0)*time_iter_32*nb_iter_32/gridsize_32
      time += (3.0/7.0)*time_iter_33*nb_iter_33/gridsize_33 + (3.0/7.0)*time_iter_34*nb_iter_34/gridsize_34
      time += (3.0/7.0)*time_iter_35*nb_iter_35/gridsize_35 + (3.0/7.0)*time_iter_36*nb_iter_36/gridsize_36
      time += (3.0/7.0)*time_iter_37*nb_iter_37/gridsize_37 + (3.0/7.0)*time_iter_38*nb_iter_38/gridsize_38
      time += (3.0/7.0)*time_iter_39*nb_iter_39/gridsize_39 + (3.0/7.0)*time_iter_40*nb_iter_40/gridsize_40
      time += (3.0/7.0)*time_iter_41*nb_iter_41/gridsize_41 + (3.0/7.0)*time_iter_42*nb_iter_42/gridsize_42
      time += (3.0/7.0)*time_iter_43*nb_iter_43/gridsize_43 + (3.0/7.0)*time_iter_44*nb_iter_44/gridsize_44
      time += (3.0/7.0)*time_iter_45*nb_iter_45/gridsize_45 + (3.0/7.0)*time_iter_46*nb_iter_46/gridsize_46
      time += (3.0/7.0)*time_iter_47*nb_iter_47/gridsize_47 + (3.0/7.0)*time_iter_48*nb_iter_48/gridsize_48
      time += (3.0/7.0)*time_iter_49*nb_iter_49/gridsize_49 + (3.0/7.0)*time_iter_50*nb_iter_50/gridsize_50
      time += (3.0/7.0)*time_iter_51*nb_iter_51/gridsize_51 + (3.0/7.0)*time_iter_52*nb_iter_52/gridsize_52
      time += (3.0/7.0)*time_iter_53*nb_iter_53/gridsize_53 + (3.0/7.0)*time_iter_54*nb_iter_54/gridsize_54
      time += (3.0/11.0)*time_iter_55*nb_iter_55/gridsize_55 + (3.0/11.0)*time_iter_56*nb_iter_56/gridsize_56
      time += (3.0/11.0)*time_iter_57*nb_iter_57/gridsize_57 + (3.0/11.0)*time_iter_58*nb_iter_58/gridsize_58
      time += (3.0/11.0)*time_iter_59*nb_iter_59/gridsize_59 + (3.0/11.0)*time_iter_60*nb_iter_60/gridsize_60
      time += (3.0/11.0)*time_iter_61*nb_iter_61/gridsize_61 + (3.0/11.0)*time_iter_62*nb_iter_62/gridsize_62
      time += (3.0/11.0)*time_iter_63*nb_iter_63/gridsize_63 + (3.0/11.0)*time_iter_64*nb_iter_64/gridsize_64
      time += (3.0/11.0)*time_iter_65*nb_iter_65/gridsize_65 + (3.0/11.0)*time_iter_66*nb_iter_66/gridsize_66
      time += (3.0/11.0)*time_iter_67*nb_iter_67/gridsize_67 + (3.0/11.0)*time_iter_68*nb_iter_68/gridsize_68
      time += (3.0/11.0)*time_iter_69*nb_iter_69/gridsize_69 + (3.0/11.0)*time_iter_70*nb_iter_70/gridsize_70
      time += (3.0/11.0)*time_iter_71*nb_iter_71/gridsize_71 + (3.0/11.0)*time_iter_72*nb_iter_72/gridsize_72
      time += (3.0/11.0)*time_iter_73*nb_iter_73/gridsize_73 + (3.0/11.0)*time_iter_74*nb_iter_74/gridsize_74
      time += (3.0/11.0)*time_iter_75*nb_iter_75/gridsize_75 + (3.0/11.0)*time_iter_76*nb_iter_76/gridsize_76
      time += (3.0/11.0)*time_iter_77*nb_iter_77/gridsize_77 + (3.0/11.0)*time_iter_78*nb_iter_78/gridsize_78
      time += (3.0/11.0)*time_iter_79*nb_iter_79/gridsize_79 + (3.0/11.0)*time_iter_80*nb_iter_80/gridsize_80
      time += (3.0/11.0)*time_iter_81*nb_iter_81/gridsize_81 + (3.0/11.0)*time_iter_82*nb_iter_82/gridsize_82
      time += (3.0/11.0)*time_iter_83*nb_iter_83/gridsize_83 + (3.0/11.0)*time_iter_84*nb_iter_84/gridsize_84
      time += (3.0/11.0)*time_iter_85*nb_iter_85/gridsize_85 + (3.0/11.0)*time_iter_86*nb_iter_86/gridsize_86
      time += (3.0/11.0)*time_iter_87*nb_iter_87/gridsize_87 + (3.0/11.0)*time_iter_88*nb_iter_88/gridsize_88
      time += (3.0/11.0)*time_iter_89*nb_iter_89/gridsize_89
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 36:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_57 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_57,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_58 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_58,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_59 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_59,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_60 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_60,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_61 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_61,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_62 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_62,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_63 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_63,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_64 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_64,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_65 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_65,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_66 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_66,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_67 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_67,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_68 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_68,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_69 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_69,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_70 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_70,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_71 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_71,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_72 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_72,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_73 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_73,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_74 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_74,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_75 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_75,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_76 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_76,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_77 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_77,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_78 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_78,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_79 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_79,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_80 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_80,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_81 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_81,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_82 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_82,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_83 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_83,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_84 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_84,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_85 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_85,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_86 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_86,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_87 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_87,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_88 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_88,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_89 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_89,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_90 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_90,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_91 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_91,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_92 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_92,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_93 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_93,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_94 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_94,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_95 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_95,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_96 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_96,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_97 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_97,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_98 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_98,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_99 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_99,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]  

      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      (time_iter_57, stats) = core.time_compute(CPU_tasklist2_57, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_57 = ", time_iter_57
      (time_iter_58, stats) = core.time_compute(CPU_tasklist2_58, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_58 = ", time_iter_58
      (time_iter_59, stats) = core.time_compute(CPU_tasklist2_59, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_59 = ", time_iter_59
      (time_iter_60, stats) = core.time_compute(CPU_tasklist2_60, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_60 = ", time_iter_60
      (time_iter_61, stats) = core.time_compute(CPU_tasklist2_61, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_61 = ", time_iter_61
      (time_iter_62, stats) = core.time_compute(CPU_tasklist2_62, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_62 = ", time_iter_62
      (time_iter_63, stats) = core.time_compute(CPU_tasklist2_63, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_63 = ", time_iter_63
      (time_iter_64, stats) = core.time_compute(CPU_tasklist2_64, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_64 = ", time_iter_64
      (time_iter_65, stats) = core.time_compute(CPU_tasklist2_65, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_65 = ", time_iter_65
      (time_iter_66, stats) = core.time_compute(CPU_tasklist2_66, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_66 = ", time_iter_66
      (time_iter_67, stats) = core.time_compute(CPU_tasklist2_67, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_67 = ", time_iter_67
      (time_iter_68, stats) = core.time_compute(CPU_tasklist2_68, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_68 = ", time_iter_68
      (time_iter_69, stats) = core.time_compute(CPU_tasklist2_69, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_69 = ", time_iter_69
      (time_iter_70, stats) = core.time_compute(CPU_tasklist2_70, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_70 = ", time_iter_70
      (time_iter_71, stats) = core.time_compute(CPU_tasklist2_71, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_71 = ", time_iter_71
      (time_iter_72, stats) = core.time_compute(CPU_tasklist2_72, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_72 = ", time_iter_72
      (time_iter_73, stats) = core.time_compute(CPU_tasklist2_73, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_73 = ", time_iter_73
      (time_iter_74, stats) = core.time_compute(CPU_tasklist2_74, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_74 = ", time_iter_74
      (time_iter_75, stats) = core.time_compute(CPU_tasklist2_75, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_75 = ", time_iter_75
      (time_iter_76, stats) = core.time_compute(CPU_tasklist2_76, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_76 = ", time_iter_76
      (time_iter_77, stats) = core.time_compute(CPU_tasklist2_77, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_77 = ", time_iter_77
      (time_iter_78, stats) = core.time_compute(CPU_tasklist2_78, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_78 = ", time_iter_78
      (time_iter_79, stats) = core.time_compute(CPU_tasklist2_79, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_79 = ", time_iter_79
      (time_iter_80, stats) = core.time_compute(CPU_tasklist2_80, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_80 = ", time_iter_80
      (time_iter_81, stats) = core.time_compute(CPU_tasklist2_81, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_81 = ", time_iter_81
      (time_iter_82, stats) = core.time_compute(CPU_tasklist2_82, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_82 = ", time_iter_82
      (time_iter_83, stats) = core.time_compute(CPU_tasklist2_83, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_83 = ", time_iter_83
      (time_iter_84, stats) = core.time_compute(CPU_tasklist2_84, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_84 = ", time_iter_84
      (time_iter_85, stats) = core.time_compute(CPU_tasklist2_85, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_85 = ", time_iter_85
      (time_iter_86, stats) = core.time_compute(CPU_tasklist2_86, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_86 = ", time_iter_86
      (time_iter_87, stats) = core.time_compute(CPU_tasklist2_87, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_87 = ", time_iter_87
      (time_iter_88, stats) = core.time_compute(CPU_tasklist2_88, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_88 = ", time_iter_88
      (time_iter_89, stats) = core.time_compute(CPU_tasklist2_89, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_89 = ", time_iter_89
      (time_iter_90, stats) = core.time_compute(CPU_tasklist2_90, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_90 = ", time_iter_90
      (time_iter_91, stats) = core.time_compute(CPU_tasklist2_91, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_91 = ", time_iter_91
      (time_iter_92, stats) = core.time_compute(CPU_tasklist2_92, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_92 = ", time_iter_92
      (time_iter_93, stats) = core.time_compute(CPU_tasklist2_93, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_93 = ", time_iter_93
      (time_iter_94, stats) = core.time_compute(CPU_tasklist2_94, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_94 = ", time_iter_94
      (time_iter_95, stats) = core.time_compute(CPU_tasklist2_95, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_95 = ", time_iter_95
      (time_iter_96, stats) = core.time_compute(CPU_tasklist2_96, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_96 = ", time_iter_96
      (time_iter_97, stats) = core.time_compute(CPU_tasklist2_97, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_97 = ", time_iter_97
      (time_iter_98, stats) = core.time_compute(CPU_tasklist2_98, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_98 = ", time_iter_98
      (time_iter_99, stats) = core.time_compute(CPU_tasklist2_99, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_99 = ", time_iter_99
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += (3.0/7.0)*time_iter_21*nb_iter_21/gridsize_21 + (3.0/7.0)*time_iter_22*nb_iter_22/gridsize_22
      time += (3.0/7.0)*time_iter_23*nb_iter_23/gridsize_23 + (3.0/7.0)*time_iter_24*nb_iter_24/gridsize_24
      time += (3.0/7.0)*time_iter_25*nb_iter_25/gridsize_25 + (3.0/7.0)*time_iter_26*nb_iter_26/gridsize_26
      time += (3.0/7.0)*time_iter_27*nb_iter_27/gridsize_27 + (3.0/7.0)*time_iter_28*nb_iter_28/gridsize_28
      time += (3.0/7.0)*time_iter_29*nb_iter_29/gridsize_29 + (3.0/7.0)*time_iter_30*nb_iter_30/gridsize_30
      time += (3.0/7.0)*time_iter_31*nb_iter_31/gridsize_31 + (3.0/7.0)*time_iter_32*nb_iter_32/gridsize_32
      time += (3.0/7.0)*time_iter_33*nb_iter_33/gridsize_33 + (3.0/7.0)*time_iter_34*nb_iter_34/gridsize_34
      time += (3.0/7.0)*time_iter_35*nb_iter_35/gridsize_35 + (3.0/7.0)*time_iter_36*nb_iter_36/gridsize_36
      time += (3.0/7.0)*time_iter_37*nb_iter_37/gridsize_37 + (3.0/7.0)*time_iter_38*nb_iter_38/gridsize_38
      time += (3.0/7.0)*time_iter_39*nb_iter_39/gridsize_39 + (3.0/7.0)*time_iter_40*nb_iter_40/gridsize_40
      time += (3.0/7.0)*time_iter_41*nb_iter_41/gridsize_41 + (3.0/7.0)*time_iter_42*nb_iter_42/gridsize_42
      time += (3.0/7.0)*time_iter_43*nb_iter_43/gridsize_43 + (3.0/7.0)*time_iter_44*nb_iter_44/gridsize_44
      time += (3.0/7.0)*time_iter_45*nb_iter_45/gridsize_45 + (3.0/7.0)*time_iter_46*nb_iter_46/gridsize_46
      time += (3.0/7.0)*time_iter_47*nb_iter_47/gridsize_47 + (3.0/7.0)*time_iter_48*nb_iter_48/gridsize_48
      time += (3.0/7.0)*time_iter_49*nb_iter_49/gridsize_49 + (3.0/7.0)*time_iter_50*nb_iter_50/gridsize_50
      time += (3.0/7.0)*time_iter_51*nb_iter_51/gridsize_51 + (3.0/7.0)*time_iter_52*nb_iter_52/gridsize_52
      time += (3.0/7.0)*time_iter_53*nb_iter_53/gridsize_53 + (3.0/7.0)*time_iter_54*nb_iter_54/gridsize_54
      time += (3.0/7.0)*time_iter_55*nb_iter_55/gridsize_55 + (3.0/7.0)*time_iter_56*nb_iter_56/gridsize_56
      time += (3.0/7.0)*time_iter_57*nb_iter_57/gridsize_57 + (3.0/7.0)*time_iter_58*nb_iter_58/gridsize_58
      time += (3.0/7.0)*time_iter_59*nb_iter_59/gridsize_59 + (3.0/7.0)*time_iter_60*nb_iter_60/gridsize_60
      time += (3.0/11.0)*time_iter_61*nb_iter_61/gridsize_61 + (3.0/11.0)*time_iter_62*nb_iter_62/gridsize_62
      time += (3.0/11.0)*time_iter_63*nb_iter_63/gridsize_63 + (3.0/11.0)*time_iter_64*nb_iter_64/gridsize_64
      time += (3.0/11.0)*time_iter_65*nb_iter_65/gridsize_65 + (3.0/11.0)*time_iter_66*nb_iter_66/gridsize_66
      time += (3.0/11.0)*time_iter_67*nb_iter_67/gridsize_67 + (3.0/11.0)*time_iter_68*nb_iter_68/gridsize_68
      time += (3.0/11.0)*time_iter_69*nb_iter_69/gridsize_69 + (3.0/11.0)*time_iter_70*nb_iter_70/gridsize_70
      time += (3.0/11.0)*time_iter_71*nb_iter_71/gridsize_71 + (3.0/11.0)*time_iter_72*nb_iter_72/gridsize_72
      time += (3.0/11.0)*time_iter_73*nb_iter_73/gridsize_73 + (3.0/11.0)*time_iter_74*nb_iter_74/gridsize_74
      time += (3.0/11.0)*time_iter_75*nb_iter_75/gridsize_75 + (3.0/11.0)*time_iter_76*nb_iter_76/gridsize_76
      time += (3.0/11.0)*time_iter_77*nb_iter_77/gridsize_77 + (3.0/11.0)*time_iter_78*nb_iter_78/gridsize_78
      time += (3.0/11.0)*time_iter_79*nb_iter_79/gridsize_79 + (3.0/11.0)*time_iter_80*nb_iter_80/gridsize_80
      time += (3.0/11.0)*time_iter_81*nb_iter_81/gridsize_81 + (3.0/11.0)*time_iter_82*nb_iter_82/gridsize_82
      time += (3.0/11.0)*time_iter_83*nb_iter_83/gridsize_83 + (3.0/11.0)*time_iter_84*nb_iter_84/gridsize_84
      time += (3.0/11.0)*time_iter_85*nb_iter_85/gridsize_85 + (3.0/11.0)*time_iter_86*nb_iter_86/gridsize_86
      time += (3.0/11.0)*time_iter_87*nb_iter_87/gridsize_87 + (3.0/11.0)*time_iter_88*nb_iter_88/gridsize_88
      time += (3.0/11.0)*time_iter_89*nb_iter_89/gridsize_89 + (3.0/11.0)*time_iter_90*nb_iter_90/gridsize_90
      time += (3.0/11.0)*time_iter_91*nb_iter_91/gridsize_91 + (3.0/11.0)*time_iter_92*nb_iter_92/gridsize_92
      time += (3.0/11.0)*time_iter_93*nb_iter_93/gridsize_93 + (3.0/11.0)*time_iter_94*nb_iter_94/gridsize_94
      time += (3.0/11.0)*time_iter_95*nb_iter_95/gridsize_95 + (3.0/11.0)*time_iter_96*nb_iter_96/gridsize_96
      time += (3.0/11.0)*time_iter_97*nb_iter_97/gridsize_97 + (3.0/11.0)*time_iter_98*nb_iter_98/gridsize_98
      time += (3.0/11.0)*time_iter_99*nb_iter_99/gridsize_99
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    elif ns == 38:
      # For now consider only work done on the GPU
      CPU_tasklist1 = [['DEVICE_ALLOC', 0, A_dev_size*8],
                       ['DEVICE_ALLOC', 0, b_dev_size*8],
                       ['DEVICE_TRANSFER', 0, A_dev_size*8],
                       ['DEVICE_TRANSFER', 0, b_dev_size*8]]
      CPU_tasklist2_1 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_1,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_2 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_2,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_3 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_3,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_4 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_4,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_5 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_5,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_6 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_6,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_7 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_7,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_8 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_8,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_9 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_9,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_10 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_10,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_11 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_11,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_12 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_12,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_13 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_13,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_14 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_14,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_15 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_15,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_16 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_16,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_17 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_17,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_18 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_18,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_19 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_19,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_20 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_20,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_21 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_21,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_22 = [['KERNEL_CALL', 0, GPU_tasklist, blocksize, gridsize_22,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_23 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_23,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_24 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_24,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_25 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_25,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_26 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_26,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_27 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_27,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_28 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_28,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_29 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_29,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_30 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_30,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_31 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_31,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_32 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_32,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_33 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_33,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_34 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_34,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_35 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_35,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_36 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_36,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_37 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_37,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_38 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_38,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_39 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_39,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_40 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_40,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_41 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_41,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_42 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_42,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_43 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_43,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_44 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_44,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_45 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_45,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_46 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_46,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_47 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_47,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_48 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_48,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_49 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_49,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_50 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_50,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_51 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_51,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_52 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_52,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_53 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_53,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_54 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_54,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_55 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_55,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_56 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_56,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_57 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_57,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_58 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_58,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_59 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_59,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_60 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_60,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_61 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_61,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_62 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_62,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_63 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_63,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_64 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_64,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_65 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_65,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_66 = [['KERNEL_CALL', 0, GPU_tasklist2, blocksize, gridsize_66,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_67 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_67,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_68 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_68,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_69 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_69,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_70 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_70,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_71 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_71,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_72 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_72,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_73 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_73,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_74 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_74,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_75 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_75,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_76 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_76,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_77 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_77,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_78 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_78,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_79 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_79,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_80 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_80,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_81 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_81,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_82 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_82,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_83 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_83,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_84 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_84,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_85 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_85,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_86 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_86,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_87 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_87,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_88 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_88,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_89 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_89,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_90 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_90,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_91 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_91,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_92 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_92,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_93 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_93,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_94 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_94,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_95 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_95,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_96 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_96,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_97 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_97,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_98 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_98,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_99 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_99,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_100 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_100,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_101 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_101,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_102 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_102,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_103 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_103,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_104 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_104,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_105 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_105,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_106 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_106,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_107 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_107,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_108 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_108,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist2_109 = [['KERNEL_CALL', 0, GPU_tasklist3, blocksize, gridsize_109,regcount],['DEVICE_SYNC', 0]]
      CPU_tasklist3 = [['DEVICE_TRANSFER', 0, b_dev_size*8],
                      ['DEVICE_ALLOC', 0, -A_dev_size*8],
                      ['DEVICE_ALLOC', 0, -b_dev_size*8]]

      (time_init, stats) = core.time_compute(CPU_tasklist1, simianEngine.now, True)
      this.sleep(time_init)
      print "Time for initialization = ", time_init
      (time_iter_1, stats) = core.time_compute(CPU_tasklist2_1, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_1 = ", time_iter_1
      (time_iter_2, stats) = core.time_compute(CPU_tasklist2_2, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_2 = ", time_iter_2
      (time_iter_3, stats) = core.time_compute(CPU_tasklist2_3, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_3 = ", time_iter_3
      (time_iter_4, stats) = core.time_compute(CPU_tasklist2_4, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_4 = ", time_iter_4
      (time_iter_5, stats) = core.time_compute(CPU_tasklist2_5, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_5 = ", time_iter_5
      (time_iter_6, stats) = core.time_compute(CPU_tasklist2_6, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_6 = ", time_iter_6
      (time_iter_7, stats) = core.time_compute(CPU_tasklist2_7, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_7 = ", time_iter_7
      (time_iter_8, stats) = core.time_compute(CPU_tasklist2_8, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_8 = ", time_iter_8
      (time_iter_9, stats) = core.time_compute(CPU_tasklist2_9, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_9 = ", time_iter_9
      (time_iter_10, stats) = core.time_compute(CPU_tasklist2_10, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_10 = ", time_iter_10
      (time_iter_11, stats) = core.time_compute(CPU_tasklist2_11, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_11 = ", time_iter_11
      (time_iter_12, stats) = core.time_compute(CPU_tasklist2_12, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_12 = ", time_iter_12
      (time_iter_13, stats) = core.time_compute(CPU_tasklist2_13, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_13 = ", time_iter_13
      (time_iter_14, stats) = core.time_compute(CPU_tasklist2_14, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_14 = ", time_iter_14
      (time_iter_15, stats) = core.time_compute(CPU_tasklist2_15, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_15 = ", time_iter_15
      (time_iter_16, stats) = core.time_compute(CPU_tasklist2_16, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_16 = ", time_iter_16
      (time_iter_17, stats) = core.time_compute(CPU_tasklist2_17, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_17 = ", time_iter_17
      (time_iter_18, stats) = core.time_compute(CPU_tasklist2_18, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_18 = ", time_iter_18
      (time_iter_19, stats) = core.time_compute(CPU_tasklist2_19, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_19 = ", time_iter_19
      (time_iter_20, stats) = core.time_compute(CPU_tasklist2_20, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_20 = ", time_iter_20
      (time_iter_21, stats) = core.time_compute(CPU_tasklist2_21, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_21 = ", time_iter_21
      (time_iter_22, stats) = core.time_compute(CPU_tasklist2_22, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_22 = ", time_iter_22
      (time_iter_23, stats) = core.time_compute(CPU_tasklist2_23, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_23 = ", time_iter_23
      (time_iter_24, stats) = core.time_compute(CPU_tasklist2_24, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_24 = ", time_iter_24
      (time_iter_25, stats) = core.time_compute(CPU_tasklist2_25, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_25 = ", time_iter_25
      (time_iter_26, stats) = core.time_compute(CPU_tasklist2_26, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_26 = ", time_iter_26
      (time_iter_27, stats) = core.time_compute(CPU_tasklist2_27, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_27 = ", time_iter_27
      (time_iter_28, stats) = core.time_compute(CPU_tasklist2_28, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_28 = ", time_iter_28
      (time_iter_29, stats) = core.time_compute(CPU_tasklist2_29, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_29 = ", time_iter_29
      (time_iter_30, stats) = core.time_compute(CPU_tasklist2_30, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_30 = ", time_iter_30
      (time_iter_31, stats) = core.time_compute(CPU_tasklist2_31, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_31 = ", time_iter_31
      (time_iter_32, stats) = core.time_compute(CPU_tasklist2_32, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_32 = ", time_iter_32
      (time_iter_33, stats) = core.time_compute(CPU_tasklist2_33, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_33 = ", time_iter_33
      (time_iter_34, stats) = core.time_compute(CPU_tasklist2_34, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_34 = ", time_iter_34
      (time_iter_35, stats) = core.time_compute(CPU_tasklist2_35, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_35 = ", time_iter_35
      (time_iter_36, stats) = core.time_compute(CPU_tasklist2_36, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_36 = ", time_iter_36
      (time_iter_37, stats) = core.time_compute(CPU_tasklist2_37, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_37 = ", time_iter_37
      (time_iter_38, stats) = core.time_compute(CPU_tasklist2_38, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_38 = ", time_iter_38
      (time_iter_39, stats) = core.time_compute(CPU_tasklist2_39, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_39 = ", time_iter_39
      (time_iter_40, stats) = core.time_compute(CPU_tasklist2_40, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_40 = ", time_iter_40
      (time_iter_41, stats) = core.time_compute(CPU_tasklist2_41, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_41 = ", time_iter_41
      (time_iter_42, stats) = core.time_compute(CPU_tasklist2_42, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_42 = ", time_iter_42
      (time_iter_43, stats) = core.time_compute(CPU_tasklist2_43, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_43 = ", time_iter_43
      (time_iter_44, stats) = core.time_compute(CPU_tasklist2_44, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_44 = ", time_iter_44
      (time_iter_45, stats) = core.time_compute(CPU_tasklist2_45, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_45 = ", time_iter_45
      (time_iter_46, stats) = core.time_compute(CPU_tasklist2_46, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_46 = ", time_iter_46
      (time_iter_47, stats) = core.time_compute(CPU_tasklist2_47, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_47 = ", time_iter_47
      (time_iter_48, stats) = core.time_compute(CPU_tasklist2_48, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_48 = ", time_iter_48
      (time_iter_49, stats) = core.time_compute(CPU_tasklist2_49, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_49 = ", time_iter_49
      (time_iter_50, stats) = core.time_compute(CPU_tasklist2_50, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_50 = ", time_iter_50
      (time_iter_51, stats) = core.time_compute(CPU_tasklist2_51, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_51 = ", time_iter_51
      (time_iter_52, stats) = core.time_compute(CPU_tasklist2_52, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_52 = ", time_iter_52
      (time_iter_53, stats) = core.time_compute(CPU_tasklist2_53, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_53 = ", time_iter_53
      (time_iter_54, stats) = core.time_compute(CPU_tasklist2_54, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_54 = ", time_iter_54
      (time_iter_55, stats) = core.time_compute(CPU_tasklist2_55, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_55 = ", time_iter_55
      (time_iter_56, stats) = core.time_compute(CPU_tasklist2_56, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_56 = ", time_iter_56
      (time_iter_57, stats) = core.time_compute(CPU_tasklist2_57, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_57 = ", time_iter_57
      (time_iter_58, stats) = core.time_compute(CPU_tasklist2_58, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_58 = ", time_iter_58
      (time_iter_59, stats) = core.time_compute(CPU_tasklist2_59, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_59 = ", time_iter_59
      (time_iter_60, stats) = core.time_compute(CPU_tasklist2_60, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_60 = ", time_iter_60
      (time_iter_61, stats) = core.time_compute(CPU_tasklist2_61, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_61 = ", time_iter_61
      (time_iter_62, stats) = core.time_compute(CPU_tasklist2_62, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_62 = ", time_iter_62
      (time_iter_63, stats) = core.time_compute(CPU_tasklist2_63, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_63 = ", time_iter_63
      (time_iter_64, stats) = core.time_compute(CPU_tasklist2_64, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_64 = ", time_iter_64
      (time_iter_65, stats) = core.time_compute(CPU_tasklist2_65, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_65 = ", time_iter_65
      (time_iter_66, stats) = core.time_compute(CPU_tasklist2_66, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_66 = ", time_iter_66
      (time_iter_67, stats) = core.time_compute(CPU_tasklist2_67, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_67 = ", time_iter_67
      (time_iter_68, stats) = core.time_compute(CPU_tasklist2_68, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_68 = ", time_iter_68
      (time_iter_69, stats) = core.time_compute(CPU_tasklist2_69, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_69 = ", time_iter_69
      (time_iter_70, stats) = core.time_compute(CPU_tasklist2_70, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_70 = ", time_iter_70
      (time_iter_71, stats) = core.time_compute(CPU_tasklist2_71, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_71 = ", time_iter_71
      (time_iter_72, stats) = core.time_compute(CPU_tasklist2_72, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_72 = ", time_iter_72
      (time_iter_73, stats) = core.time_compute(CPU_tasklist2_73, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_73 = ", time_iter_73
      (time_iter_74, stats) = core.time_compute(CPU_tasklist2_74, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_74 = ", time_iter_74
      (time_iter_75, stats) = core.time_compute(CPU_tasklist2_75, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_75 = ", time_iter_75
      (time_iter_76, stats) = core.time_compute(CPU_tasklist2_76, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_76 = ", time_iter_76
      (time_iter_77, stats) = core.time_compute(CPU_tasklist2_77, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_77 = ", time_iter_77
      (time_iter_78, stats) = core.time_compute(CPU_tasklist2_78, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_78 = ", time_iter_78
      (time_iter_79, stats) = core.time_compute(CPU_tasklist2_79, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_79 = ", time_iter_79
      (time_iter_80, stats) = core.time_compute(CPU_tasklist2_80, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_80 = ", time_iter_80
      (time_iter_81, stats) = core.time_compute(CPU_tasklist2_81, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_81 = ", time_iter_81
      (time_iter_82, stats) = core.time_compute(CPU_tasklist2_82, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_82 = ", time_iter_82
      (time_iter_83, stats) = core.time_compute(CPU_tasklist2_83, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_83 = ", time_iter_83
      (time_iter_84, stats) = core.time_compute(CPU_tasklist2_84, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_84 = ", time_iter_84
      (time_iter_85, stats) = core.time_compute(CPU_tasklist2_85, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_85 = ", time_iter_85
      (time_iter_86, stats) = core.time_compute(CPU_tasklist2_86, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_86 = ", time_iter_86
      (time_iter_87, stats) = core.time_compute(CPU_tasklist2_87, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_87 = ", time_iter_87
      (time_iter_88, stats) = core.time_compute(CPU_tasklist2_88, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_88 = ", time_iter_88
      (time_iter_89, stats) = core.time_compute(CPU_tasklist2_89, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_89 = ", time_iter_89
      (time_iter_90, stats) = core.time_compute(CPU_tasklist2_90, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_90 = ", time_iter_90
      (time_iter_91, stats) = core.time_compute(CPU_tasklist2_91, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_91 = ", time_iter_91
      (time_iter_92, stats) = core.time_compute(CPU_tasklist2_92, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_92 = ", time_iter_92
      (time_iter_93, stats) = core.time_compute(CPU_tasklist2_93, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_93 = ", time_iter_93
      (time_iter_94, stats) = core.time_compute(CPU_tasklist2_94, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_94 = ", time_iter_94
      (time_iter_95, stats) = core.time_compute(CPU_tasklist2_95, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_95 = ", time_iter_95
      (time_iter_96, stats) = core.time_compute(CPU_tasklist2_96, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_96 = ", time_iter_96
      (time_iter_97, stats) = core.time_compute(CPU_tasklist2_97, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_97 = ", time_iter_97
      (time_iter_98, stats) = core.time_compute(CPU_tasklist2_98, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_98 = ", time_iter_98
      (time_iter_99, stats) = core.time_compute(CPU_tasklist2_99, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_99 = ", time_iter_99
      (time_iter_100, stats) = core.time_compute(CPU_tasklist2_100, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_100 = ", time_iter_100
      (time_iter_101, stats) = core.time_compute(CPU_tasklist2_101, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_101 = ", time_iter_101
      (time_iter_102, stats) = core.time_compute(CPU_tasklist2_102, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_102 = ", time_iter_102
      (time_iter_103, stats) = core.time_compute(CPU_tasklist2_103, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_103 = ", time_iter_103
      (time_iter_104, stats) = core.time_compute(CPU_tasklist2_104, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_104 = ", time_iter_104
      (time_iter_105, stats) = core.time_compute(CPU_tasklist2_105, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_105 = ", time_iter_105
      (time_iter_106, stats) = core.time_compute(CPU_tasklist2_106, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_106 = ", time_iter_106
      (time_iter_107, stats) = core.time_compute(CPU_tasklist2_107, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_107 = ", time_iter_107
      (time_iter_108, stats) = core.time_compute(CPU_tasklist2_108, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_108 = ", time_iter_108
      (time_iter_109, stats) = core.time_compute(CPU_tasklist2_109, simianEngine.now, True)
      print "Time for a single kernel call with gridsize_109 = ", time_iter_109
      time  = time_iter_1*nb_iter_1/gridsize_1 + time_iter_2*nb_iter_2/gridsize_2
      time += time_iter_3*nb_iter_3/gridsize_3 + time_iter_4*nb_iter_4/gridsize_4
      time += time_iter_5*nb_iter_5/gridsize_5 + time_iter_6*nb_iter_6/gridsize_6
      time += time_iter_7*nb_iter_7/gridsize_7 + time_iter_8*nb_iter_8/gridsize_8
      time += time_iter_9*nb_iter_9/gridsize_9 + time_iter_10*nb_iter_10/gridsize_10
      time += time_iter_11*nb_iter_11/gridsize_11 + time_iter_12*nb_iter_12/gridsize_12
      time += time_iter_13*nb_iter_13/gridsize_13 + time_iter_14*nb_iter_14/gridsize_14
      time += time_iter_15*nb_iter_15/gridsize_15 + time_iter_16*nb_iter_16/gridsize_16
      time += time_iter_17*nb_iter_17/gridsize_17 + time_iter_18*nb_iter_18/gridsize_18
      time += time_iter_19*nb_iter_19/gridsize_19 + time_iter_20*nb_iter_20/gridsize_20
      time += time_iter_21*nb_iter_21/gridsize_21 + time_iter_22*nb_iter_22/gridsize_22
      time += (3.0/7.0)*time_iter_23*nb_iter_23/gridsize_23 + (3.0/7.0)*time_iter_24*nb_iter_24/gridsize_24
      time += (3.0/7.0)*time_iter_25*nb_iter_25/gridsize_25 + (3.0/7.0)*time_iter_26*nb_iter_26/gridsize_26
      time += (3.0/7.0)*time_iter_27*nb_iter_27/gridsize_27 + (3.0/7.0)*time_iter_28*nb_iter_28/gridsize_28
      time += (3.0/7.0)*time_iter_29*nb_iter_29/gridsize_29 + (3.0/7.0)*time_iter_30*nb_iter_30/gridsize_30
      time += (3.0/7.0)*time_iter_31*nb_iter_31/gridsize_31 + (3.0/7.0)*time_iter_32*nb_iter_32/gridsize_32
      time += (3.0/7.0)*time_iter_33*nb_iter_33/gridsize_33 + (3.0/7.0)*time_iter_34*nb_iter_34/gridsize_34
      time += (3.0/7.0)*time_iter_35*nb_iter_35/gridsize_35 + (3.0/7.0)*time_iter_36*nb_iter_36/gridsize_36
      time += (3.0/7.0)*time_iter_37*nb_iter_37/gridsize_37 + (3.0/7.0)*time_iter_38*nb_iter_38/gridsize_38
      time += (3.0/7.0)*time_iter_39*nb_iter_39/gridsize_39 + (3.0/7.0)*time_iter_40*nb_iter_40/gridsize_40
      time += (3.0/7.0)*time_iter_41*nb_iter_41/gridsize_41 + (3.0/7.0)*time_iter_42*nb_iter_42/gridsize_42
      time += (3.0/7.0)*time_iter_43*nb_iter_43/gridsize_43 + (3.0/7.0)*time_iter_44*nb_iter_44/gridsize_44
      time += (3.0/7.0)*time_iter_45*nb_iter_45/gridsize_45 + (3.0/7.0)*time_iter_46*nb_iter_46/gridsize_46
      time += (3.0/7.0)*time_iter_47*nb_iter_47/gridsize_47 + (3.0/7.0)*time_iter_48*nb_iter_48/gridsize_48
      time += (3.0/7.0)*time_iter_49*nb_iter_49/gridsize_49 + (3.0/7.0)*time_iter_50*nb_iter_50/gridsize_50
      time += (3.0/7.0)*time_iter_51*nb_iter_51/gridsize_51 + (3.0/7.0)*time_iter_52*nb_iter_52/gridsize_52
      time += (3.0/7.0)*time_iter_53*nb_iter_53/gridsize_53 + (3.0/7.0)*time_iter_54*nb_iter_54/gridsize_54
      time += (3.0/7.0)*time_iter_55*nb_iter_55/gridsize_55 + (3.0/7.0)*time_iter_56*nb_iter_56/gridsize_56
      time += (3.0/7.0)*time_iter_57*nb_iter_57/gridsize_57 + (3.0/7.0)*time_iter_58*nb_iter_58/gridsize_58
      time += (3.0/7.0)*time_iter_59*nb_iter_59/gridsize_59 + (3.0/7.0)*time_iter_60*nb_iter_60/gridsize_60
      time += (3.0/7.0)*time_iter_61*nb_iter_61/gridsize_61 + (3.0/7.0)*time_iter_62*nb_iter_62/gridsize_62
      time += (3.0/7.0)*time_iter_63*nb_iter_63/gridsize_63 + (3.0/7.0)*time_iter_64*nb_iter_64/gridsize_64
      time += (3.0/7.0)*time_iter_65*nb_iter_65/gridsize_65 + (3.0/7.0)*time_iter_66*nb_iter_66/gridsize_66
      time += (7.0/11.0)*time_iter_67*nb_iter_67/gridsize_67 + (7.0/11.0)*time_iter_68*nb_iter_68/gridsize_68
      time += (7.0/11.0)*time_iter_69*nb_iter_69/gridsize_69 + (7.0/11.0)*time_iter_70*nb_iter_70/gridsize_70
      time += (7.0/11.0)*time_iter_71*nb_iter_71/gridsize_71 + (7.0/11.0)*time_iter_72*nb_iter_72/gridsize_72
      time += (7.0/11.0)*time_iter_73*nb_iter_73/gridsize_73 + (7.0/11.0)*time_iter_74*nb_iter_74/gridsize_74
      time += (7.0/11.0)*time_iter_75*nb_iter_75/gridsize_75 + (7.0/11.0)*time_iter_76*nb_iter_76/gridsize_76
      time += (7.0/11.0)*time_iter_77*nb_iter_77/gridsize_77 + (7.0/11.0)*time_iter_78*nb_iter_78/gridsize_78
      time += (7.0/11.0)*time_iter_79*nb_iter_79/gridsize_79 + (7.0/11.0)*time_iter_80*nb_iter_80/gridsize_80
      time += (7.0/11.0)*time_iter_81*nb_iter_81/gridsize_81 + (7.0/11.0)*time_iter_82*nb_iter_82/gridsize_82
      time += (7.0/11.0)*time_iter_83*nb_iter_83/gridsize_83 + (7.0/11.0)*time_iter_84*nb_iter_84/gridsize_84
      time += (7.0/11.0)*time_iter_85*nb_iter_85/gridsize_85 + (7.0/11.0)*time_iter_86*nb_iter_86/gridsize_86
      time += (7.0/11.0)*time_iter_87*nb_iter_87/gridsize_87 + (7.0/11.0)*time_iter_88*nb_iter_88/gridsize_88
      time += (7.0/11.0)*time_iter_89*nb_iter_89/gridsize_89 + (7.0/11.0)*time_iter_90*nb_iter_90/gridsize_90
      time += (3.0/11.0)*time_iter_91*nb_iter_91/gridsize_91 + (3.0/11.0)*time_iter_92*nb_iter_92/gridsize_92
      time += (3.0/11.0)*time_iter_93*nb_iter_93/gridsize_93 + (3.0/11.0)*time_iter_94*nb_iter_94/gridsize_94
      time += (3.0/11.0)*time_iter_95*nb_iter_95/gridsize_95 + (3.0/11.0)*time_iter_96*nb_iter_96/gridsize_96
      time += (3.0/11.0)*time_iter_97*nb_iter_97/gridsize_97 + (3.0/11.0)*time_iter_98*nb_iter_98/gridsize_98
      time += (3.0/11.0)*time_iter_99*nb_iter_99/gridsize_99 + (3.0/11.0)*time_iter_100*nb_iter_100/gridsize_100
      time += (3.0/11.0)*time_iter_101*nb_iter_101/gridsize_101 + (3.0/11.0)*time_iter_102*nb_iter_102/gridsize_102
      time += (3.0/11.0)*time_iter_103*nb_iter_103/gridsize_103 + (3.0/11.0)*time_iter_104*nb_iter_104/gridsize_104
      time += (3.0/11.0)*time_iter_105*nb_iter_105/gridsize_105 + (3.0/11.0)*time_iter_106*nb_iter_106/gridsize_106
      time += (3.0/11.0)*time_iter_107*nb_iter_107/gridsize_107 + (3.0/11.0)*time_iter_108*nb_iter_108/gridsize_108
      time += (3.0/11.0)*time_iter_109*nb_iter_109/gridsize_109
      this.sleep(time)
      (time_finalize, stats) = core.time_compute(CPU_tasklist3, simianEngine.now, True)
      print "Time for finalization = ", time_finalize
      this.sleep(time_finalize)
      this.entity.out.write("Time: "+str(simianEngine.now)+ ":\t "+this.entity.name+" "+str(this.entity.num)+\
                       " computations completed on core id "+str(0)+"; execution time: "+\
                       str(time)+"; Thread Efficiency: "+str(stats['Thread Efficiency'])+"\n")
    else:
      print "Warning: unsupported ns value!"
      sys.exit(1)
 
def GPUTest_Handler(self, msg, *args):
#    self.createProcess("STENCIL", STENCIL) #MR: comment out2 
#    self.startProcess("STENCIL", self) #MR: comment out   
    self.createProcess("LU_APP", LU_APP)
    self.startProcess("LU_APP", self)

################################
# "MAIN"
################################


modeldict = { "model_name"    : "n01",
              "sim_time"      : 1000000,
              "use_mpi"       : False,
              "intercon_type" : "Bypass",
              "torus"         : configs.cielo_intercon,
              "host_type"     : "CieloNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
	      "debug_options" : []
            }

# 1. Add a compute node to the engine
#simianEngine.addEntity("Node", nodes.TTNNode, 0, modeldict, 1,1,1,1,1,1,1,1,1,1) #MR: comment out
simianEngine.addEntity("Node", nodes.MLIntelPlusGPUNode, 0, modeldict, 1,1,1,1,1,1,1,1,1,1)


# 2. Create a GPUtest Service on the node
simianEngine.attachService(nodes.Node, "GPUTest_Handler" , GPUTest_Handler)

simianEngine.schedService(0, "GPUTest_Handler", None, "Node", 0)
    
# 3. Run simx
simianEngine.run()
simianEngine.exit()

