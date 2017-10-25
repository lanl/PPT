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
	i. 		KNLNode
	ii.		CieloNode
	iii.	MacProNode
	v. 		TTNNode (Titan)
"""




from entity import Entity
import processors

class Node(Entity):
    """
    Base class for Node. Actual nodes should derive from this. The application can add handlers to this base class.
    """
    def __init__(self, baseInfo, *args):
        super(Node, self).__init__(baseInfo)

            




class KNLNode(Node):
    """
    Class that represents a KnightsLanding compute node
    """
    def __init__(self, baseInfo, *args):
        super(KNLNode, self).__init__(baseInfo)
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
        

class CieloNode(Node):
    """
    Class that represents a Cielo compute node
    """
    def __init__(self, baseInfo, *args):
        super(CieloNode, self).__init__(baseInfo)
        self.memory_footprint =  0       # Number of bytes allocated in memory
        
        #
        #  PARAMETERS
        #
        
        self.num_cores = 16                     # number of cores on the node
        self.memorysize = 32000  		# total memory on node  in MB

        self.interconnect_bandwidth = 6.8 * 10**10      # speed of interconnect in bits/sec
        self.interconnect_latency = 10**-6              # Delay in getting packet ready in sec
        self.interconnect_latency_mpi = 1.5 * 10**-6    # Delay in getting MPI packet ready in sec

	# This number - needs to look more
        self.filesystem_access_time =  1.0      # filesystem access time in seconds
        
        self.out.write("Cielo node generated at time "+str(self.engine.now)+"\n")
        # So let's generate the cores
        self.cores = []
        for i in range(self.num_cores):
             self.cores.append(processors.CieloCore(self))
        self.out.write(str(self.num_cores)+" Cielo cores generated at time "+str(self.engine.now)+"\n")
        
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
        
 
class MBPNode(Node):
    """
    Class that represents a MacPro compute node
    """
    def __init__(self, baseInfo, *args):
        super(MBPNode, self).__init__(baseInfo)
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
        
 
class TTNNode(Node):
    """
    Class that represents a Titan compute node
    """
    def __init__(self, baseInfo, *args):
        super(TTNNode, self).__init__(baseInfo)
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


