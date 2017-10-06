# Copyright (c) 2014. Los Alamos National Security, LLC. 

# This material was produced under U.S. Government contract DE-AC52-06NA25396
# for Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
# National Security, LLC for the U.S. Department of Energy. The U.S. Government 
# has rights to use, reproduce, and distribute this software.  

# NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, 
# EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
# If software is modified to produce derivative works, such modified software should
# be clearly marked, so as not to confuse it with the version available from LANL.

# Additionally, this library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License v 2.1 as published by the 
# Free Software Foundation. Accordingly, this library is distributed in the hope that 
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE.txt for more details.

"""
*********** Performance Prediction Toolkit PPT *********

File: nodes.py
Description: main library of compute node definitions.

Comments:
 2015-11-06: included into repository, contains 
 	0. 		AMD Opteron (for Mustang)
	i. 		KNLNode
	ii.		CieloNode
	iii.	MacProNode
	v. 		TTNNode (Titan)
"""




from entity import Entity
#import Jason's node
#from node import Node
from interconnect import *
from mpi import *

import processors
import threadsim
import math


startup_overhead    = 270*10**-6
scheduling_overhead = 10**-6

          
class AMDOpteron(Host):
    """
    Class that represents a KnightsLanding compute node
    """
    def __init__(self, baseInfo, hpcsim_dict, *arg):
        super(AMDOpteron, self).__init__(baseInfo, hpcsim_dict, *arg)
        self.memory_footprint =  0       # Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        
        self.num_cores 				=  	24		# number of cores on the node
        self.memorysize 			= 	2600 * self.num_cores  # total memory on node  in MB
        self.filesystem_access_time =  	1.0      # TBD; filesystem access time in seconds
        self.interconnect_bandwidth = 	40/8 * 10**8 # Specs say 40 Gb/s; speed of interconnect in bits/sec
        self.interconnect_latency 	= 	10**-9      # TBD; Delay in getting packet ready in sec
        
        self.out.write("AMDOpteron node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.AMDOpteronCore(self))
        self.out.write(str(self.num_cores)+" Opteron cores generated at time "+str(self.engine.now)+"\n\n")    	
            

class KNLNode(Host):
    """
    Class that represents a KnightsLanding compute node
    """
    def __init__(self, baseInfo, hpcsim_dict, *arg):
        super(KNLNode, self).__init__(baseInfo, hpcsim_dict, *arg)
        self.memory_footprint =  0       # Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        
        self.num_cores =  64	# number of cores on the node
        self.memorysize = 512 * self.num_cores  # total memory on node  in MB
        self.filesystem_access_time =  1.0      # filesystem access time in seconds
        self.interconnect_bandwidth = 1.0 * 10**8      # speed of interconnect in bits/sec
        self.interconnect_latency = 10**-9      # Delay in getting packet ready in sec
        
        self.out.write("KNL node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.KNLCore(self))
        self.out.write(str(self.num_cores)+" KNL cores generated at time "+str(self.engine.now)+"\n\n")
        
    def mem_alloc(self, size):
        """
        Allocates or unallocates memory of size 
        """
        self.memory_footprint += size
        if self.memory_footprint > self.memorysize:
            if size < 0: # still too high, but at least in unalloc
                print "Warning: KNLNode", self, " still out of memory at time ", simx.get_now()
                return True
            else:
                #print "Warning: KNLNode", self, " ran out of memory at time ", simx.get_now()
                return False
        else:
            if self.memory_footprint - size > self.memorysize:
                # We are back to normal memory use
                print "Warning: KNLNode", node, " has recovered from swap mode ", simx.get_now()
            return True


class CieloNode(Host):
    """
    Class that represents a Cielo compute node
    """
    def __init__(self, baseInfo, hpcsim_dict, *arg):
        super(CieloNode, self).__init__(baseInfo, hpcsim_dict, *arg)
#    def __init__(self, baseInfo, *args):
#        super(CieloNode, self).__init__(baseInfo)
        self.memory_footprint =  0       # Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        self.refcount = 0
        self.num_cores = 16                     # number of cores on the node
        self.memorysize = 32000  		# total memory on node  in MB

        self.interconnect_bandwidth = 6.8 * 10**10      # speed of interconnect in bits/sec
        self.interconnect_latency = 10**-6              # Delay in getting packet ready in sec
        self.interconnect_latency_mpi = 1.5 * 10**-6    # Delay in getting MPI packet ready in sec

	# This number - needs to look more
        self.filesystem_access_time =  1.0      # filesystem access time in seconds
        self.activethreads = 0
        self.hwthreads = 0
        self.thread_pool = 0
        
        #print "Created CieloNode ", self.num
        #self.out.write("Cielo node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.CieloCore(self))
             self.hwthreads += self.cores[i].hwthreads
        #self.out.write(str(self.num_cores)+" Cielo cores generated at time "+str(self.engine.now)+"\n")
        
    def mem_alloc(self, size):
        """
        Allocates or unallocates memory of size 
        """
        self.memory_footprint += size
        if self.memory_footprint > self.memorysize:
            if size < 0: # still too high, but at least in unalloc
                print "Warning: CieloNode", self, " still out of memory at time ", simx.get_now()
                return True
            else:
                #print "Warning: CieloNode", self, " ran out of memory at time ", simx.get_now()
                return False
        else:
            if self.memory_footprint - size > self.memorysize:
                # We are back to normal memory use
                print "Warning: CieloNode", node, " has recovered from swap mode ", simx.get_now()
            return True
    
    # TODO move to abstract class for node    
    def spawn_threads(self, num_threads):
    	time = 0.0
    	if self.thread_pool < num_threads:
    		self.thread_pool = num_threads
    		time = startup_overhead
    	for i in range(num_threads):
    		self.cores[i%self.num_cores].activethreads += 1
    	self.activethreads += num_threads
    	return time
    
    # TODO move to abstract class for node  
    def unspawn_threads(self, num_threads):
    	for i in range(num_threads):
    		self.cores[i%self.num_cores].activethreads -= 1
    	self.activethreads -= num_threads
    
    # TODO move to abstract class for node  
    def thread_efficiency(self):
	    """
	    Gives the efficiency back as a function of the number of active threads.
	    Function chosen as inverse of active threads. This is a cheap way of
	    mimicking time slicing.
	    """
	
	    efficiency = 0.0
	    if self.activethreads <=self.hwthreads:
	      efficiency = 1.0
	    else:
	    	# TODO: include penalty to running more software threads than available
	    	# 			hardware threads
	    	efficiency = math.pow(.9,float(self.activethreads)/float(self.hwthreads))*float(self.hwthreads)/float(self.activethreads)
	
	    return efficiency
	   
    # TODO move to abstract class for node       
    def time_compute(self, construct):
        time = 0.0
        for item in construct:
            if item[0] == "parallel_for":
                num_threads = item[1]
                num_iter                = item[2]
                scheduling      = item[3]
                use_proc                = item[4]
                tasklist                = item[5]
                if use_proc:
                    time += threadsim.exec_loop(self, num_threads, num_iter, scheduling, tasklist,item[6],item[7])
                else:
                    time += threadsim.exec_loop_simple(self, num_threads, num_iter, scheduling, tasklist)
            elif item[0] == "parallel_tasks":
                num_threads = item[1]
                tasklist                = item[2]
                time += threadsim.exec_tasks(self, num_threads, tasklist,item[3],item[4])
            else:
                print('Warning: construct item', item, 'cannot be parsed, ignoring it')
        return time

 
class MLIntelNode(Host):
    """
    Class that represents a Moonlight compute node
    """
    def __init__(self, baseInfo, hpcsim_dict, *arg):
        super(MLIntelNode, self).__init__(baseInfo, hpcsim_dict, *arg)
#    def __init__(self, baseInfo, *args):
#        super(CieloNode, self).__init__(baseInfo)
        self.memory_footprint =  0       # Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        
        self.num_cores = 16                     # number of cores on the node
        self.memorysize = 32768  		# total memory on node  in MB

        self.interconnect_bandwidth = 1.0 * 10**10      # speed of interconnect in bits/sec
        self.interconnect_latency = 10**-6              # Delay in getting packet ready in sec
        self.interconnect_latency_mpi = 1.5 * 10**-6    # Delay in getting MPI packet ready in sec

	# This number - needs to look more
        self.filesystem_access_time =  1.0      # filesystem access time in seconds
        
        #print "Created MLIntelNode ", self.num
        #self.out.write("Moonlight Intel node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.MLIntelCore(self))
        #self.out.write(str(self.num_cores)+" Moonlight Intel cores generated at time "+str(self.engine.now)+"\n")
        
    def mem_alloc(self, size):
        """
        Allocates or unallocates memory of size 
        """
        self.memory_footprint += size
        if self.memory_footprint > self.memorysize:
            if size < 0: # still too high, but at least in unalloc
                print "Warning: MLIntelNode", self, " still out of memory at time ", simx.get_now()
                return True
            else:
                #print "Warning: MLIntelNode", self, " ran out of memory at time ", simx.get_now()
                return False
        else:
            if self.memory_footprint - size > self.memorysize:
                # We are back to normal memory use
                print "Warning: MLIntelNode", node, " has recovered from swap mode ", simx.get_now()
            return True


class MBPNode(Host):
    """
    Class that represents a MacPro compute node
    """
    def __init__(self, baseInfo, hpcsim_dict, *arg):
        super(MBPNode, self).__init__(baseInfo, hpcsim_dict, *arg)
        self.memory_footprint =  0       # Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        
        self.num_cores = 12                     # number of cores on the node
        self.memorysize = 64000  		# total memory on node  in MB

        self.interconnect_bandwidth = 6.8 * 10**10      # speed of interconnect in bits/sec
        self.interconnect_latency = 10**-6              # Delay in getting packet ready in sec
        self.interconnect_latency_mpi = 1.5 * 10**-6    # Delay in getting MPI packet ready in sec

	# This number - needs to look more
        self.filesystem_access_time =  1.0      # filesystem access time in seconds
        
        self.out.write("MacPro node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.MacProCore(self))
        self.out.write(str(self.num_cores)+" MacPro cores generated at time "+str(self.engine.now)+"\n")
        
    def mem_alloc(self, size):
        """
        Allocates or unallocates memory of size 
        """
        self.memory_footprint += size
        if self.memory_footprint > self.memorysize:
            if size < 0: # still too high, but at least in unalloc
                print "Warning: MacProNode", self, " still out of memory at time ", simx.get_now()
                return True
            else:
                #print "Warning: MacProNode", self, " ran out of memory at time ", simx.get_now()
                return False
        else:
            if self.memory_footprint - size > self.memorysize:
                # We are back to normal memory use
                print "Warning: MacProNode", node, " has recovered from swap mode ", simx.get_now()
            return True
        
 
class TTNNode(Host):
    """
    Class that represents a Titan compute node
    """
    def __init__(self, baseInfo, hpcsim_dict, *arg):
        super(TTNNode, self).__init__(baseInfo, hpcsim_dict, *arg)
        import accelerators
        self.memory_footprint =  0       		# Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        
        self.num_cores  = 16 #63                    	 # number of cores on the node
        self.memorysize = 32*1024  			             # total memory on node  in MB

        self.filesystem_access_time =  1.0      	     # filesystem access time in seconds
        self.interconnect_bandwidth = 4.5 * 10**9     	 # 2.9 to 5.8 GB/sec per direction
        self.interconnect_latency   = 2.6 * 10**-6     	 # about 1.27*10**-6 s (nearest nodes pair) 
							                             # and 3.88*10**-6 s (farthest nodes pair) on a quiet network
        self.PCIe_bandwidth = 100*10**9#6.23*10**9                # 8 GB/s at maximum speed (if using x16 PCI 2.0???)
        self.PCIe_latency = 10*10**-6
        self.num_accelerators = 1                        # currently 1 Tesla K20X attached to each node
        
        self.out.write("TTN node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.TTNCore(self))
        self.out.write(str(self.num_cores)+" TTN cores generated at time "+str(self.engine.now)+"\n")
        
        self.accelerators = []
    	for i in range(self.num_accelerators):
    		self.accelerators.append(accelerators.Pascal(self, i)) # Change back to k20x
    	self.out.write(str(self.num_accelerators)+" TTN Pascal     accelerator(s) generated at time "+str(self.engine.now)+"\n")
        
    def mem_alloc(self, size):
        """
        Allocates or unallocates memory of size 
        """
        self.memory_footprint += size
        if self.memory_footprint > self.memorysize:
            if size < 0: # still too high, but at least in unalloc
                print "Warning: TTNNode", self, " still out of memory at time ", simx.get_now()
                return True
            else:
                #print "Warning: TTNNode", self, " ran out of memory at time ", simx.get_now()
                return False
        else:
            if self.memory_footprint - size > self.memorysize:
                # We are back to normal memory use
                print "Warning: TTNNode", node, " has recovered from swap mode ", simx.get_now()
            return True


