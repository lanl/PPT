##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##############################################################################


import os, sys, math, time
from scipy import special as sp


def dump_output(pred_out):

    kernel_prefix = str(pred_out["kernel_id"])+"_"+pred_out["ISA"] +"_g"+pred_out["granularity"]
    outF = open(os.path.join(pred_out["app_path"], "kernel_"+kernel_prefix+".out"), "w+")

    print("kernel name:", pred_out["kernel_name"], file=outF)
    
    print("\n- Total GPU computations is divided into " + str(pred_out["total_num_workloads"])+\
                " thread block(s) running on " + str(pred_out["active_SMs"]) + " SM(s)", file=outF)
    
    print("\n- Modeled SM-0 running", pred_out["num_workloads_per_SM_new"], "thread block(s):", file=outF)
    print("\t* allocated max active thread block(s):", pred_out["allocated_active_blocks_per_SM"], file=outF)
    print("\t* allocated max active warps per thread block:", pred_out["allocated_active_warps_per_block"], file=outF)

    print("\n- Occupancy of SM-0:", file=outF)
    print("\t* Thread block Limit SM:", pred_out["max_active_blocks_per_SM"], file=outF)
    print("\t* Thread block limit registers:", pred_out["blocks_per_SM_limit_regs"], file=outF)
    print("\t* Thread block limit shared memory:", pred_out["blocks_per_SM_limit_smem"], file=outF)
    print("\t* Thread block limit warps:", pred_out["blocks_per_SM_limit_warps"], file=outF)
    print("\t* theoretical max active thread block(s):", pred_out["th_active_blocks"], file=outF)
    print("\t* theoretical max active warps per SM:", pred_out["th_active_warps"], file=outF)
    print("\t* theoretical occupancy:", pred_out["th_occupancy"],"%", file=outF)
    print("\t* achieved active warps per SM:", round(pred_out["achieved_active_warps"], 2), file=outF)
    print("\t* achieved occupancy:", round(pred_out["achieved_occupancy"], 2),"%", file=outF)

    print("\n- Memory Performance:", file=outF)
    print("\t* unified L1 cache hit rate:", round((pred_out["memory_stats"]["umem_hit_rate"]*100),2),"%", file=outF)
    print("\t* unified L1 cache hit rate for read transactions (global memory accesses):", round((pred_out["memory_stats"]["gmem_hit_rate_lds"]*100),2),"%", file=outF)
    if pred_out["memory_stats"]["lmem_used"]:
        print("\t* unified L1 cache hit rate (global memory accesses):", round((pred_out["memory_stats"]["gmem_hit_rate"]*100),2),"%", file=outF)
        
    print("\t* L2 cache hit rate:", round((pred_out["memory_stats"]["hit_rate_l2"]*100),2),"%", file=outF)
    
    print("\n\t* Global Memory Requests:", file=outF)
    print("\t\t** GMEM read requests:", pred_out["memory_stats"]["gmem_ld_reqs"], file=outF)
    print("\t\t** GMEM write requests:", pred_out["memory_stats"]["gmem_st_reqs"], file=outF)
    print("\t\t** GMEM total requests:", pred_out["memory_stats"]["gmem_tot_reqs"], file=outF)

    print("\n\t* Global Memory Transactions:", file=outF)
    print("\t\t** GMEM read transactions:", pred_out["memory_stats"]["gmem_ld_trans"], file=outF)
    print("\t\t** GMEM write transactions:", pred_out["memory_stats"]["gmem_st_trans"], file=outF)
    print("\t\t** GMEM total transactions:", pred_out["memory_stats"]["gmem_tot_trans"], file=outF)

    print("\n\t* Global Memory Divergence:", file=outF)
    print("\t\t** number of read transactions per read requests: "+ str(pred_out["memory_stats"]["gmem_ld_diverg"])+\
        " ("+str(round(((pred_out["memory_stats"]["gmem_ld_diverg"]/32)*100),2))+"%)", file=outF)
    print("\t\t** number of write transactions per write requests: "+ str(pred_out["memory_stats"]["gmem_st_diverg"])+\
        " ("+str(round(((pred_out["memory_stats"]["gmem_st_diverg"]/32)*100),2))+"%)", file=outF)

    print("\n\t* L2 Cache Transactions (for global memory accesses):", file=outF)
    print("\t\t** L2 read transactions:", pred_out["memory_stats"]["l2_ld_trans_gmem"], file=outF)
    print("\t\t** L2 write transactions:", pred_out["memory_stats"]["l2_st_trans_gmem"], file=outF)
    print("\t\t** L2 total transactions:", pred_out["memory_stats"]["l2_tot_trans_gmem"], file=outF)

    print("\n\t* DRAM Transactions (for global memory accesses):", file=outF)
    print("\t\t** DRAM total transactions:", pred_out["memory_stats"]["dram_tot_trans_gmem"], file=outF)

    print("\n\t* Total number of global atomic requests:", pred_out["memory_stats"]["atom_tot_reqs"], file=outF)
    print("\t* Total number of global reduction requests:", pred_out["memory_stats"]["red_tot_reqs"], file=outF)
    print("\t* Global memory atomic and reduction transactions:", pred_out["memory_stats"]["atom_red_tot_trans"], file=outF)

    print("\n- Kernel cycles:", file=outF)
    print("\t* GPU active cycles (min):", place_value(int(pred_out["gpu_act_cycles_min"])), file=outF)
    print("\t* GPU active cycles (max):", place_value(int(pred_out["gpu_act_cycles_max"])), file=outF)
    print("\t* SM active cycles (sum):", place_value(int(pred_out["sm_act_cycles.sum"])), file=outF)
    print("\t* SM elapsed cycles (sum):", place_value(int(pred_out["sm_elp_cycles.sum"])), file=outF)
    
    print("\n- Warp instructions executed:", place_value(int(pred_out["tot_warps_instructions_executed"])), file=outF)
    print("- Thread instructions executed:", place_value(int(pred_out["tot_threads_instructions_executed"])), file=outF)
    print("- Instructions executed per clock cycle (IPC):", round(pred_out["tot_ipc"], 3), file=outF)
    print("- Clock cycles per instruction (CPI): ", round(pred_out["tot_cpi"], 3), file=outF)
    print("- Total instructions executed per seconds (MIPS):", int(round((pred_out["tot_throughput_ips"]/1000000), 3)), file=outF)
    print("- Kernel execution time:", round((pred_out["execution_time_sec"]*1000000),4), "us", file=outF)

    print("\n- Simulation Time:", file=outF)
    print("\t* Memory model:", round(pred_out["simulation_time"]["memory"], 3), "sec,", convert_sec(pred_out["simulation_time"]["memory"]), file=outF)
    print("\t* Compute model:", round(pred_out["simulation_time"]["compute"], 3), "sec,", convert_sec(pred_out["simulation_time"]["compute"]), file=outF)


def place_value(number): 
    return ("{:,}".format(number))


def convert_sec(seconds): 
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def print_config_error(config_name, flag=0):
	if flag == 1:
		print("\n[Error]\nGPU Compute Capabilty \"" +config_name+"\" is not supported")
		sys.exit(1)
	elif flag == 2:
		print("\n[Error]\n\""+config_name+"\" is not defined in the hardware compute capability file")
		sys.exit(1)
	else:
		print("\n[Error]\n\""+config_name+"\" config is not defined in the hardware configuration file")
		sys.exit(1)


def print_warning(arg1, arg2, flag=False):
	if flag:
		print("\n[Warning]\n\"" + arg1 + "\" is not defined in the config file "+\
		"assuming L1 cache is "+ arg2 + "\n")
	else:
		print("\n[Warning]\n\"" + arg1 + "\" can't be more than " + arg2\
		 	+" registers\n assuming \"" + arg1 + "\" = " + arg2 + "\n")


def ceil(x, s):
	return s * math.ceil(float(x)/s)


def floor(x, s):
    return s * math.floor(float(x)/s)


def qfunc(arg):
    return 0.5-0.5*sp.erf(arg/1.41421)


def ncr(n, m):
    '''
    n choose m
    '''
    if(m>n): return 0
    r = 1
    for j in range(1,m+1):
        try:
            r *= (n-m+j)/float(j)
        except FloatingPointError:
            continue
    return r


class stack_el(object):
    def __init__ (self,data):
        self.address = data[0]
        self.access_time = data[1]
        self.next_el = None
        self.prev_el = None
    def __str__(self):
        return "(%s %d)" %(self.address,self.access_time)


class Stack(object):
    def __init__ (self):
        self.elements = []
        self.sp = {} #dictionary of stack pointers

    def push(self,address,t,el=None):
        if not el is None:
            se = el 
        else:
            se = stack_el((address,t))

        if not self.sp == {}:  #stack is not empty
            se.next_el = self.sp["top"]
            se.prev_el = None
            self.sp["top"].prev_el = se

        else: #stack empty
            se.next_el = None
            se.prev_el = None

        self.sp["top"] = se
        self.sp[se.access_time] = se

    def update(self,last_access,now,address):
        try:
            se = self.sp.pop(last_access) #pop deletes key, and returns value
            assert(se.address == address)
            assert(se.access_time == last_access)
            d = 0 #calculate distance from se to top of stack
            tmp = se
            while (not tmp.prev_el is None):
                d += 1
                tmp = tmp.prev_el
            if not se.prev_el is None: #remove from linked list
                se.prev_el.next_el = se.next_el
            else: 
                #the element is already at the top of the stack. 
                #just update access time and return depth
                se.access_time = now
                self.sp[se.access_time] = se
                return d
            if not se.next_el is None:
                se.next_el.prev_el = se.prev_el
            #update access time
            se.access_time = now
            #create an entry for popped element at the top of the stack
            self.push(None,None,el=se)
            #return the depth  of this element before it was brought to the top
            return d

        except KeyError:
            print ("internal error. (%s %d) not found in dictionary" %(address,last_access))
            quit()