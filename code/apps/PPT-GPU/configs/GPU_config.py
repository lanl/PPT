"""
*********** Performance Prediction Toolkit PPT *********

File: GPU_config.py
Description: Target GPU configurations
Author: Yehia Arafa 
"""

import sys 
from arch_latencies_config import *


def get_gpu_config(gpu):
    new_gpu = gpu()
    config = new_gpu.populate_config()
    return config


class K40m():

    def populate_config(self):
        config = {}
        config["gpu_name"] = "K40m"
        config["gpu_arch"] = "Kepler" #This name must be one of the classes defined in 'arch_latencies_config'

        mem_latencies = get_mem_latencies(getattr(sys.modules[__name__], config["gpu_arch"]))
       
        config["num_SM"]                     = 15           # Number of Streaming Multiprocessors 
        config["num_SP_per_SM"]              = 192          # Number of Single Precision cores per multiprocessor  
        config["num_SF_per_SM"]              = 32           # Number of Special Function units per multiprocessor 
        config["num_DP_per_SM"]              = 64           # Number of Double Precision cores per multiprocessor 
        config["num_load_store_units"]       = 32           # Number of Load & Store units per multiprocessor
        config["num_warp_schedulers"]        = 4            # Number of warp schedulers available (Max number of warps that can be executed concurrently)
        config["num_inst_per_warp"]          = 2            # Number of instructions that can be issued simultaneously to a given warp 
        config["clockspeed"]                 = 667*10**6    # GPU clock speed in Hertz

        config["num_registers"]              = 65536        # Number of registers available 
        
        config["l1_cache_size"]              = 24000        # L1 cache size in Bytes  
        config["l2_cache_size"]              = 1.5*10**6    # L2 cache size in Bytes
        config["global_mem_size"]            = 12288*10**6	# Global memory size in Byte
        config["shared_mem_size"]            = 49152		# Shared memory size in Bytes per multiprocessor 

        config["l1_mem_latency"]             = mem_latencies["l1_cache_access"]
        config["l2_mem_latency"]             = mem_latencies["l2_cache_access"]
        config["l2_to_global_mem_latency"]   = mem_latencies["global_mem_latency"] - mem_latencies["l2_cache_access"]
        config["local_mem_latency"]          = mem_latencies["local_mem_latency"]
        config["const_mem_latency"]          = mem_latencies["constant_mem_latency"]
        config["tex_mem_latency"]            = mem_latencies["texture_mem_latency"]
        config["tex_cache_latency"]          = mem_latencies["texture_cache_latency"]
        config["shared_mem_latency"]         = mem_latencies["shared_mem_latency"]
        
        config["warp_size"]                  = 32		    # Number of threads in a warp
        config["max_num_warps_per_SM"]       = 64		    # Max number of warps resident on a single SM
        config["max_num_block_per_SM"]       = 32		    # Max number of blocks queued on a single SM 
        config["max_num_threads_per_block"]  = 1024		    # Max number of (software) threads in a block 
        config["max_num_threads_per_SM"]     = 2048		    # Max number of threads queued or active on a single SM
        
        config["global_mem_return_queue"]    = 128		    # Number of memory concurrent transfer from the memory queue
        config["num_memory_ports"]           = 1            # Number of memory ports

        return config


class Titanx():

    def populate_config(self):
        config = {}
        config["gpu_name"] = "TitianX"
        config["gpu_arch"] = "Maxwell" #This name must be one of the classes defined in 'arch_latencies_config'

        mem_latencies = get_mem_latencies(getattr(sys.modules[__name__], config["gpu_arch"]))
       
        config["num_SM"]                     = 24           # Number of Streaming Multiprocessors 
        config["num_SP_per_SM"]              = 128          # Number of Single Precision cores per multiprocessor  
        config["num_SF_per_SM"]              = 32           # Number of Special Function usints per multiprocessor 
        config["num_DP_per_SM"]              = 32           # Number of Double Precision cores per multiprocessor 
        config["num_load_store_units"]       = 32           # Number of Load & Store units per multiprocessor
        config["num_warp_schedulers"]        = 4            # Number of warp schedulers available (Max number of warps that can be executed concurrently)
        config["num_inst_per_warp"]          = 2            # Number of instructions that can be issued simultaneously to a given warp 
        config["clockspeed"]                 = 1000*10**6   # GPU clock speed in Hertz

        config["num_registers"]              = 65536        # Number of registers available 
        
        config["l1_cache_size"]              = 24000        # L1 cache size in Bytes  
        config["l2_cache_size"]              = 2*10**6	    # L2 cache size in Bytes
        config["global_mem_size"]            = 12288*10**6	# Global memory size in Byte
        config["shared_mem_size"]            = 98304		# Shared memory size in Bytes per multiprocessor 

        config["l1_mem_latency"]             = mem_latencies["l1_cache_access"]
        config["l2_mem_latency"]             = mem_latencies["l2_cache_access"]
        config["l2_to_global_mem_latency"]   = mem_latencies["global_mem_latency"] - mem_latencies["l2_cache_access"]
        config["local_mem_latency"]          = mem_latencies["local_mem_latency"]
        config["const_mem_latency"]          = mem_latencies["constant_mem_latency"]
        config["tex_mem_latency"]            = mem_latencies["texture_mem_latency"]
        config["tex_cache_latency"]          = mem_latencies["texture_cache_latency"]
        config["shared_mem_latency"]         = mem_latencies["shared_mem_latency"]
        
        config["warp_size"]                  = 32		    # Number of threads in a warp
        config["max_num_warps_per_SM "]      = 64		    # Max number of warps resident on a single SM
        config["max_num_block_per_SM"]       = 32		    # Max number of blocks queued on a single SM 
        config["max_num_threads_per_block"]  = 1024		    # Max number of (software) threads in a block 
        config["max_num_threads_per_SM"]     = 2048		    # Max number of threads queued or active on a single SM
        
        config["global_mem_return_queue"]    = 128		    # Number of memory concurrent transfer from the memory queue
        config["num_memory_ports "]          = 1

        return config


