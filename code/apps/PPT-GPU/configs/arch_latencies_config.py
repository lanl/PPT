"""
*********** Performance Prediction Toolkit PPT *********

File: arch_latencies_config.py
Description: Has the latencies for various ALU and memory operations for the target architecture/generation
Author: Yehia Arafa 
comments: 
  please cite this paper if you are going to use these reported latencies: 

  Y. Arafa, A.A. badawy, G. Chennupati, N. Santhi, and S. Eidenbenz, 
  "Instructions' latencies characterization for nvidia gpgpus," 2019, [online]. Available: https://arxiv.org/abs/1905.08778
"""

def get_alu_latencies(generation):
    new_gpu = generation()
    alu_latencies = new_gpu.populate_alu_latencies()
    return alu_latencies
    


def get_mem_latencies(generation):
    new_gpu = generation()
    mem_latencies = new_gpu.populate_mem_latencies()
    return mem_latencies


class Kepler():
    
    def populate_alu_latencies(self):
        latencies = {}
        latencies["add"]      = 9
        latencies["sub"]      = 9
        latencies["min"]      = 9
        latencies["max"]      = 9
        latencies["mul"]      = 9
        latencies["mad"]      = 9
        latencies["sdiv"]     = 149
        latencies["srem"]     = 132
        latencies["abs"]      = 16
        latencies["udiv"]     = 131
        latencies["urem"]     = 116
        latencies["and"]      = 9 
        latencies["or"]       = 9 
        latencies["not"]      = 9 
        latencies["xor"]      = 9
        latencies["cnot"]     = 18 
        latencies["shl"]      = 9
        latencies["shr"]      = 9
        latencies["fadd"]     = 9 
        latencies["fsub"]     = 9
        latencies["fmul"]     = 9
        latencies["fmad"]     = 9
        latencies["fma"]      = 9
        latencies["fdiv"]     = 418 
        latencies["dadd"]     = 10 
        latencies["dsub"]     = 10
        latencies["dmin"]     = 10
        latencies["dmax"]     = 10
        latencies["dmul"]     = 10
        latencies["dmad"]     = 10
        latencies["dfma"]     = 10
        latencies["ddiv"]     = 445 
        latencies["add.cc"]   = 9
        latencies["addc"]     = 9  
        latencies["sub.cc"]   = 9
        latencies["subc"]     = 18  
        latencies["mad.cc"]   = 9
        latencies["rcp"]      = 377   
        latencies["sqrt"]     = 432
        latencies["fasqrt"]   = 49
        latencies["rsqrt"]    = 40
        latencies["sin"]      = 18
        latencies["cos"]      = 18
        latencies["lg2"]      = 40
        latencies["ex2"]      = 49
        latencies["copysign"] = 21
        latencies["mul24"]    = 22
        latencies["mad24"]    = 22
        latencies["mulhi"]    = 9
        latencies["mul64hi"]  = 226
        latencies["sad"]      = 9
        latencies["popc"]     = 9
        latencies["clz"]      = 20
        latencies["bfe"]      = 9
        latencies["bfi"]      = 9
        latencies["bfind"]    = 9
        latencies["bbrev"]    = 9
        latencies["mov"]      = 9
        latencies["shfl"]     = 9
        latencies["cvta"]     = 9
        latencies["cvt"]      = 9
        latencies["setp"]     = 9
        latencies["selp"]     = 9
        return latencies

    def populate_mem_latencies(self):
        latencies = {}
        latencies["l1_cache_access"]        = 32
        latencies["l2_cache_access"]        = 188
        latencies["global_mem_latency"]     = 331
        latencies["local_mem_latency"]      = 331
        latencies["constant_mem_latency"]   = 16
        latencies["texture_mem_latency"]    = 317
        latencies["texture_cache_latency"]  = 112
        latencies["shared_mem_latency"]     = 26
        return latencies 

class Maxwell():
    
    def populate_alu_latencies(self):
        latencies = {}
        latencies["add"]      = 6
        latencies["sub"]      = 6
        latencies["min"]      = 6
        latencies["max"]      = 6
        latencies["mul"]      = 13
        latencies["mad"]      = 13
        latencies["sdiv"]     = 150
        latencies["srem"]     = 141
        latencies["abs"]      = 13
        latencies["udiv"]     = 136
        latencies["urem"]     = 127
        latencies["and"]      = 6 
        latencies["or"]       = 6 
        latencies["not"]      = 6 
        latencies["xor"]      = 6
        latencies["cnot"]     = 6 
        latencies["shl"]      = 6
        latencies["shr"]      = 6
        latencies["fadd"]     = 6 
        latencies["fsub"]     = 6
        latencies["fmul"]     = 6
        latencies["fmad"]     = 6
        latencies["fma"]      = 6
        latencies["fdiv"]     = 450 
        latencies["dadd"]     = 48 
        latencies["dsub"]     = 48
        latencies["dmin"]     = 48
        latencies["dmax"]     = 48
        latencies["dmul"]     = 48
        latencies["dmad"]     = 48
        latencies["dfma"]     = 48
        latencies["ddiv"]     = 709 
        latencies["add.cc"]   = 6
        latencies["addc"]     = 6  
        latencies["sub.cc"]   = 6
        latencies["subc"]     = 12  
        latencies["mad.cc"]   = 13
        latencies["rcp"]      = 347   
        latencies["sqrt"]     = 360
        latencies["fasqrt"]   = 47
        latencies["rsqrt"]    = 34
        latencies["sin"]      = 15
        latencies["cos"]      = 15
        latencies["lg2"]      = 34
        latencies["ex2"]      = 40
        latencies["copysign"] = 20
        latencies["mul24"]    = 21
        latencies["mad24"]    = 18
        latencies["mulhi"]    = 18
        latencies["mul64hi"]  = 106
        latencies["sad"]      = 6
        latencies["popc"]     = 13
        latencies["clz"]      = 19
        latencies["bfe"]      = 6
        latencies["bfi"]      = 6
        latencies["bfind"]    = 6
        latencies["bbrev"]    = 6
        latencies["mov"]      = 6
        latencies["shfl"]     = 6
        latencies["cvta"]     = 6
        latencies["setp"]     = 6
        latencies["selp"]     = 6
        return latencies

    def populate_mem_latencies(self):
        latencies = {}
        latencies["l1_cache_access"]        = 82
        latencies["l2_cache_access"]        = 225
        latencies["global_mem_latency"]     = 375
        latencies["local_mem_latency"]      = 375
        latencies["constant_mem_latency"]   = 20
        latencies["texture_mem_latency"]    = 357
        latencies["texture_cache_latency"]  = 95
        latencies["shared_mem_latency"]     = 24
        return latencies 

class Pascal():
    
    def populate_alu_latencies(self):
        latencies = {}
        latencies["add"]      = 6
        latencies["sub"]      = 6
        latencies["min"]      = 6
        latencies["max"]      = 6
        latencies["mul"]      = 13
        latencies["mad"]      = 13
        latencies["sdiv"]     = 153
        latencies["srem"]     = 144
        latencies["abs"]      = 13
        latencies["udiv"]     = 139
        latencies["urem"]     = 130
        latencies["and"]      = 6 
        latencies["or"]       = 6 
        latencies["not"]      = 6 
        latencies["xor"]      = 6
        latencies["cnot"]     = 6 
        latencies["shl"]      = 6
        latencies["shr"]      = 6
        latencies["fadd"]     = 6 
        latencies["fsub"]     = 6
        latencies["fmul"]     = 6
        latencies["ffmad"]    = 6
        latencies["fma"]      = 6
        latencies["fdiv"]     = 408 
        latencies["dadd"]     = 8 
        latencies["dsub"]     = 8
        latencies["dmin"]     = 8
        latencies["dmax"]     = 8
        latencies["dmul"]     = 8
        latencies["dmad"]     = 8
        latencies["dfma"]     = 8
        latencies["ddiv"]     = 545
        latencies["hfadd"]    = 6
        latencies["hfsub"]    = 6  
        latencies["hfmul"]    = 6
        latencies["hffma"]    = 6  
        latencies["add.cc"]   = 6
        latencies["addc"]     = 6  
        latencies["sub.cc"]   = 6
        latencies["subc"]     = 12  
        latencies["mad.cc"]   = 13
        latencies["rcp"]      = 266   
        latencies["sqrt"]     = 282
        latencies["fasqrt"]   = 35
        latencies["rsqrt"]    = 35
        latencies["sin"]      = 15
        latencies["cos"]      = 15
        latencies["lg2"]      = 35
        latencies["ex2"]      = 41
        latencies["copysign"] = 20
        latencies["mul24"]    = 21
        latencies["mad24"]    = 18
        latencies["mulhi"]    = 18
        latencies["mul64hi"]  = 118
        latencies["sad"]      = 6
        latencies["popc"]     = 13
        latencies["clz"]      = 18
        latencies["bfe"]      = 6
        latencies["bfi"]      = 6
        latencies["bfind"]    = 6
        latencies["bbrev"]    = 6
        latencies["mov"]      = 6
        latencies["shfl"]     = 6
        latencies["cvta"]     = 6
        latencies["setp"]     = 6
        return latencies

    def populate_mem_latencies(self):
        latencies = {}
        latencies["l1_cache_access"]        = 81
        latencies["l2_cache_access"]        = 252
        latencies["global_mem_latency"]     = 486
        latencies["local_mem_latency"]      = 486
        latencies["constant_mem_latency"]   = 12
        latencies["texture_mem_latency"]    = 470
        latencies["texture_cache_latency"]  = 87
        latencies["shared_mem_latency"]     = 25
        return latencies 
        

