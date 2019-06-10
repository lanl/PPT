"""
*********** Performance Prediction Toolkit PPT *********

File: gpu_app.py
Description: The application which is to run on the accelerator (GPU)
Author: Yehia Arafa 
"""

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *
path.append('configs/')
import string
from GPU_config import *

import nodes


############################
# Simian Engine parameters #
############################

simName, startTime, endTime, minDelay = "perf", 0.0, 1000000.0, 0.000001
simianEngine = Simian(simName, startTime, endTime, minDelay)


#############################################
# The application to run on the modeled GPU #
#############################################


def app(this, arg, *args): #Rodinia->Gaussain 
	'''
	We are simulating only the kernel call and not taking into account the CPU to GPU memory allocation and byte transfered transfer.
	This will added for the CPU+GPU version. 
	'''
	num_register = 14		 # Number of registers available for the applcation  		
	static_shared_mem = 2048 # Static shared memory available for the application 
	phit_l2 = 0.803          #--> Estimation from real device <loads & stores>
	#phit_l2 = 0
	node = this.entity
	core = node.cores[0]
	vector_size = 1024
	block_size = 16
	grid_size = 65536
	
	tasklist = [['PARAM_MEM_ACCESS', 'LOAD'], ['PARAM_MEM_ACCESS', 'LOAD'], ['PARAM_MEM_ACCESS', 'LOAD'], ['PARAM_MEM_ACCESS', 'LOAD'], ['PARAM_MEM_ACCESS', 'LOAD'], ['iALU', 9], ['iALU', 9], ['iALU', 9], ['iALU', 9], ['iALU', 9], ['iALU', 9, 3], ['iALU', 9, 10, 4], ['iALU', 9], ['iALU', 9], ['iALU', 9], ['iALU', 9], ['iALU', 9], ['iALU', 9, 3, 4], ['iALU', 9], ['iALU', 9], ['iALU', 9, 4, 9], ['iALU', 9, 20], ['iALU', 9, 21, 3], ['iALU', 9, 22, 4], ['iALU', 9, 23], ['dALU', 10, 24], ['iALU', 9, 16, 4], ['iALU', 9, 4, 3, 26], ['iALU', 9, 27], ['dALU', 10, 28], ['GLOB_MEM_ACCESS', 'LOAD', 29], ['GLOB_MEM_ACCESS', 'LOAD', 25], ['fALU', 9, 31, 30], ['iALU', 9, 26, 22], ['iALU', 9, 33], ['dALU', 10, 34], ['GLOB_MEM_ACCESS', 'LOAD', 35], ['fALU', 9, 36, 32], ['GLOB_MEM_ACCESS', 'STORE', 37], ['iALU', 9], ['iALU', 9], ['dALU', 10, 34], ['iALU', 9, 4], ['dALU', 10, 42], ['GLOB_MEM_ACCESS', 'LOAD', 43], ['GLOB_MEM_ACCESS', 'LOAD', 41], ['fALU', 9, 45, 44], ['iALU', 9, 21], ['dALU', 10, 47], ['GLOB_MEM_ACCESS', 'LOAD', 48], ['fALU', 9, 49, 46], ['GLOB_MEM_ACCESS', 'STORE', 50]]

	GPU_tasklist = [['KERNEL_CALL', 0, tasklist, block_size, grid_size, num_register, static_shared_mem, phit_l2]]
	core.time_compute(GPU_tasklist, simianEngine.now, True)


######################################################################################
# The Handler which have the application's tasklist and the target config GPU inside #
######################################################################################

def GPU_APP_Handler(self, msg, *args):
	self.createProcess("app", app)
	gpu_config = get_gpu_config(getattr(sys.modules[__name__], "K40m")) #K40m is one of the classes in [configs/GPU_config]
	self.generate_target_accelerator(gpu_config)
	self.startProcess("app", self) 


if __name__ == "__main__":
	'''
	For now, we choose to have a node (GPUNode) that have only one accelerator with a dummy host (GPUCore) 
	and an interconnect (cielo_intercon)
	''' 

	modeldict = { 
            "model_name"    : "n01",
            "sim_time"      : 1000000,
            "use_mpi"       : False,
            "intercon_type" : "Bypass",
            "torus"         : configs.cielo_intercon,
            "host_type"     : "CieloNode",
            "load_libraries": set(["mpi"]),
            "mpiopt"        : configs.gemini_mpiopt,
            "debug_options" : []
            } 
    
    #########
    # STEPS #
    #########

    # 1. Add a compute node to the engine
	simianEngine.addEntity("Node", nodes.GPUNode, 0, modeldict, 1,1,1,1,1,1,1,1,1,1)

    # 2. Create a new GPU_APP Service on the node
	simianEngine.attachService(nodes.Node, "GPU_APP_Handler" , GPU_APP_Handler)
	
    # 3. Scheduling the GPU_APP service to run at time 0 
	simianEngine.schedService(0, "GPU_APP_Handler", None, "Node", 0)
 
    # 4. Run simian
	simianEngine.run()

    # 5. Exit simian
	simianEngine.exit()
