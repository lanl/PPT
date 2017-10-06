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

File: accelerators.py
Description: main library of GPU accelerator definitions.

Comments:
 2015-11-06: included into repository
"""


import math
from json.encoder import INFINITY
	
class Accelerator(object): 
	"""
	A Simian resource that represents a hardware accelerator
	"""
	def __init__(self, node, id):
		super(Accelerator,self).__init__()
		self.node =	 node 		# needed for Node/Accelerator communication
		self.waiting_processes = [] 	# list of processes waiting to be executed
		self.id = id
		self.num_SM = 1
		self.mem_op = []
		self.mem_capacity = [1,1]
		self.ram_cycles = [0,1]
		self.max_transfers=0
		

	class Warp(object):
	 	"""
	 	Class that represents a Warp being executed on an SM
	 	"""
	 	def __init__(self, gpu, block, task_list, id):
	 		self.gpu = gpu
	 		# True if warp still has computations to run
	 		self.active = True
	 		self.id = id
	 		# Number of cycles before the warp can be issued a new instruction
	 		self.stalled = 0
	 		self.block = block
	 		# Current instruction in the task_list
	 		self.current_inst = 0
	 		self.task_list = task_list
	 		self.completions = []
	 		self.syncing = False
			self.max_dep = 0
	 		
 		# Return whether the current warp is still active
 		def is_active(self):
 			return self.active
 		# Advance computations on the current warp by one clock cycle
 		def step(self, cycles):
 			res = False
 			issued = 0
 			while issued < self.gpu.nb_ins_per_warp:
 				if not self.process_inst(cycles):
 					break
 				res = True
 				issued += 1
 			return res
 		
 		def process_inst(self, cycles):
 			res = False
 			#print self.current_inst,"/",len(self.task_list)
 			if self.stalled < cycles:
 			 	if self.syncing:
 			 		pass
	 			elif self.current_inst == len(self.task_list):
	 				self.active = False
	 			else:
	 				inst = self.task_list[self.current_inst]
	 				max_dep = 0
	 				for i in inst[1:]:
	 					if i>= len(self.completions):
	 						print("[ERROR] with instruction: ", inst, "at index", i)
	 					if max_dep < self.completions[i]:
	 						max_dep = self.completions[i]
	 				if inst[0] == 'THREAD_SYNC':
	 					max_dep = self.max_dep 
	 				# Current instruction depends on a result not yet computed
	 				if cycles < max_dep:
	 					self.stalled = max_dep
	 				# Current instruction is safe to execute
	 				else:
		 				if inst[0] == 'GLOB_MEM_ACCESS':
		 					if self.block.request_LSU(cycles,1):
		 						ok, mem_time = self.gpu.mem_op_time(cycles)
		 						if ok:
			 						self.gpu.mem_op.append(cycles+mem_time)
				 					self.completions.append(cycles+mem_time)
				 					if self.max_dep < self.completions[-1]:
				 						self.max_dep = self.completions[-1] 
				 					res = True
				 					self.current_inst += 1
						elif inst[0] == 'fALU':
							if self.block.request_SPU(cycles, 1):
								self.completions.append(cycles+self.gpu.cycles_per_fALU)
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					res = True
			 					self.current_inst += 1
						elif inst[0] == 'iALU':
							if self.block.request_SPU(cycles, 1):#self.gpu.cycles_per_iALU):
								self.completions.append(cycles+self.gpu.cycles_per_iALU)
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					res = True 
			 					#print "[",str(cycles),"][",str(self.id),"] iALU"
			 					self.current_inst += 1
						elif inst[0] == 'dfALU':
							if self.block.request_DPU(cycles, 1):
								self.completions.append(cycles+self.gpu.cycles_per_fALU)
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					res = True 
			 					#print "[",str(cycles),"][",str(self.id),"] dfALU"
			 					self.current_inst += 1
						elif inst[0] == 'diALU':
							if self.block.request_DPU(cycles, 1):
								self.completions.append(cycles+self.gpu.cycles_per_iALU)
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					res = True 
			 					#print "[",str(cycles),"][",str(self.id),"] diALU"
			 					self.current_inst += 1
						elif inst[0] == 'SFU':
							if self.block.request_SFU(cycles, 1):
								self.completions.append(cycles+self.gpu.cycles_per_spec_inst)
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					res = True 
			 					self.current_inst += 1
						elif inst[0] == 'L1_ACCESS':
		 					self.completions.append(cycles+self.gpu.cache_cycles[0])
		 					if self.max_dep < self.completions[-1]:
		 						self.max_dep = self.completions[-1]
		 					res = True 
		 					self.current_inst += 1
						elif inst[0] == 'L2_ACCESS':
		 					self.completions.append(cycles+self.gpu.cache_cycles[1])
		 					if self.max_dep < self.completions[-1]:
		 						self.max_dep = self.completions[-1]
		 					res = True 
		 					self.current_inst += 1
						elif inst[0] == 'THREAD_SYNC':
		 					self.completions.append(0)
							self.block.sync_cnter += 1
							self.syncing = True
		 					res = True 
		 					self.current_inst += 1
						################
						else:
							print 'Warning: unknown task list item', inst,' cannot be parsed, ignoring it'
							self.current_inst += 1
			return res
		
	class Block(object):
		def __init__(self, gpu, id, nb_warps, task_list):
			self.gpu = gpu
			self.nb_warps = nb_warps
			self.sync_cnter = 0
			self.id = id
			self.active = True
			self.end_computations = 0.
			self.actual_end = 0
			self.warp_list = []
			for i in range(self.nb_warps):
				self.warp_list.append(K20X.Warp(self.gpu, self, task_list,str(self.id)+"-"+str(i)))
		def is_active(self, cycles):
			if not self.active:
				return False
			actual_end = 0
			for warp in self.warp_list:
				if warp.is_active():
					return True
				actual_end = max(actual_end, warp.completions[-1])
			self.active = False
			self.end_computations = cycles
			self.actual_end = actual_end
			return False
		def step(self, issued, cycles):
			self_issued = 0
			for warp in self.warp_list:
				if warp.is_active():
					if issued+self_issued >= self.gpu.nb_warp_schedulers:
						break
					if warp.step(cycles):
						self_issued += 1
			# Check whether all warps have reached a sync point
			if self.sync_cnter == self.nb_warps:
				# Allow warps to resume computations
				self.sync_cnter = 0
				for warp in self.warp_list:
					warp.syncing = False
			return self_issued
		def request_SPU(self, cycles, delay):
			res = False
			for i in range(len(self.gpu.SP_units)):
				if self.gpu.SP_units[i]<=cycles:
					self.gpu.SP_units[i] = cycles+delay
					res = True
					break
			return res
			
		def request_DPU(self, cycles, delay):
			res = False
			for i in range(len(self.gpu.DP_units)):
				if self.gpu.DP_units[i]<=cycles:
					self.gpu.DP_units[i] = cycles+delay
					res = True
					break
			return res
		def request_SFU(self, cycles, delay):
			res = False
			for i in range(len(self.gpu.SF_units)):
				if self.gpu.SF_units[i]<=cycles:
					self.gpu.SF_units[i] = cycles+delay
					res = True
					break
			return res
		def request_LSU(self, cycles, delay):
			res = False
			for i in range(len(self.gpu.LS_units)):
				if self.gpu.LS_units[i]<=cycles:
					#for j in range(len(self.gpu.mem_ports)):
					#	if self.gpu.mem_ports[j]<=cycles:
					#		self.gpu.mem_ports[j] = cycles+self.gpu.ram_cycles
					self.gpu.LS_units[i] = cycles+delay
					res = True
					break
					#		res = True
					#		break
					#break
			return res
													
	def mem_alloc(self, size):
		"""
		Allocate or deallocate memory on the GPU
		size is given in Bytes
		"""
		if self.memory_use + size > self.memorysize:
			print "Warning: unable to allocate memory on device ", str(self.id), " of node ", str(self.node.num), "."
		else:
			self.memory_use += size	
			
	def transfer(self, size):
		"""
		Transfer size bytes between the CPU and the GPU
		Return the time taken by the transfer
		"""
		if size > self.memory_use:
			print "Warning: attempting to transfer more bytes than allocated on device ", str(self.id), " of node ", str(self.node.num), "."
		return size/ self.node.PCIe_bandwidth + self.node.PCIe_latency
	
	# Create nb_warps Warps for the current kernel
	def spawn_blocks(self, nb_blocks, nb_warps_per_block, tasklist):
	 	block_list = []
	 	for i in range(nb_blocks):
	 		block_list.append(Accelerator.Block(self, i, nb_warps_per_block, tasklist))
	 	return block_list
	 
	# Return True if any warp in the list is still active
	def has_active_blocks(self, block_list, cycles):
	 	for block in block_list:
	 		if block.is_active(cycles):
	 			return True
	 	return False
	 
	# Advance computations on all warps by one cycle
	def step(self, block_list, cycles, stats):
		warps_issued = 0
	 	for block in block_list :
	 		if warps_issued >= self.nb_warp_schedulers:
	 			break
	 		if block.is_active(cycles):
	 			warps_issued+=block.step(warps_issued, cycles)
	 	stats['IPC'] += warps_issued
	
	
	def mem_op_time(self, cycles):
		time = 0
		ok = True
		while len(self.mem_op) >0 and self.mem_op[0]<=cycles:
			self.mem_op.pop()
		if len(self.mem_op)+1>self.max_transfers:
			self.max_transfers = len(self.mem_op)
		if len(self.mem_op)>=self.mem_capacity[1	]:
			ok=False
	 	time = self.ram_cycles[0]+(float(max(len(self.mem_op)-self.mem_capacity[0], 0.))/float(self.mem_capacity[1]))*(self.ram_cycles[1]-self.ram_cycles[0])
	 	#print("Estimated time for mem op is: "+str(time)+" op queue has "+str(len(self.mem_op))+" pending operations, capacity is at: "+str(min(1.,float(len(self.mem_op))/float(self.mem_capacity))*100))
	 	return ok, time
		
	def kernel_call(self, tasklist, blocksize, gridsize, regcount, start):
		"""
		Compute the time spent to complete the given kernel
		"""
		stats = {}
		stats['IPC'] = 0.
		# Add the kernel launch overhead to the time of first computations
		actual_start = start + self.kernel_launch_overhead
		# Update values in SM_availability to avoid preemptive computations
		for i in range(self.num_SM):
			self.SM_availability[i] = max(self.SM_availability[i], actual_start)
		# Number of warps in a block	
		nb_warps_per_block = int((blocksize+self.warp_size-1)/self.warp_size)
		nb_registers_per_block = blocksize*regcount
		# Number of blocks that can run concurrently on a single SM
		nb_blocks_per_sm = min(self.max_num_block_per_SM, int(self.max_num_threads_per_SM/blocksize))
		nb_blocks_per_sm = min(nb_blocks_per_sm, int(self.num_registers/nb_registers_per_block))
		nb_blocks_per_sm = min(nb_blocks_per_sm, (gridsize+self.num_SM-1)/self.num_SM)
		# Total number of warps per SM given the number of blocks
		nb_warps_per_sm = nb_blocks_per_sm*nb_warps_per_block
		self.node.out.write("Allocated "+str(nb_blocks_per_sm)+" simultaneous blocks per SM for a total of "+str(nb_warps_per_sm)+" warps.\n")
		# Total number of workloads for this kernel
		nb_work_loads = int((nb_blocks_per_sm -1 + gridsize)/nb_blocks_per_sm)
			
		block_list = self.spawn_blocks(nb_blocks_per_sm, nb_warps_per_block, tasklist)
		
		cycles = 0.
		# Scheduling a workload requires writing parameters from global mem to special registers
		# Add latency for a memory load to the cycles
		cycles += self.ram_cycles[0]+max(0.,float(nb_blocks_per_sm-self.mem_capacity[0]))/float(self.mem_capacity[1])*(self.ram_cycles[1]-self.ram_cycles[0])
		while self.has_active_blocks(block_list, cycles):
			self.step(block_list, cycles, stats)
			cycles+=1
		
		overlap = 0.
		block_average = 0.
		mem_write_delay = 0
		for block in block_list:
			overlap += cycles - block.end_computations
			block_average += block.end_computations
			mem_write_delay = max(mem_write_delay, block.actual_end-cycles)
		overlap /= len(block_list)
		block_average /= len(block_list)
		overlap /= cycles
			
		workload_time = (cycles)*1/self.clockspeed
		block_average = (block_average)*1/self.clockspeed
		self.node.out.write("BLOCK AVERAGE = "+ str(block_average)+"\n")
		self.node.out.write("GPU computations for a single workload took "+str(cycles)+" cycles.\n")
		self.node.out.write("GPU computations for a single workload took "+str(workload_time)+" s.\n")
		self.node.out.write("Total number of workloads "+str(nb_work_loads)+".\n")
		self.node.out.write("Total number of instructions issued for a single workload: "+str(stats['IPC'])+".\n")
		self.node.out.write("Potential workload overlap: "+str(overlap*100)+"%.\n")
		stats['IPC'] /= cycles
		self.node.out.write("Overall IPC "+str(stats['IPC'])+".\n")
		
		nb_workload_per_SM = int((nb_work_loads+self.num_SM-1)/self.num_SM)
		work_loads_left = nb_work_loads
		actual_workload_time = 0.
		
		for i in range(nb_workload_per_SM):
			if i==0:
				actual_workload_time+=min(self.num_SM, work_loads_left)*workload_time
			else:
				actual_workload_time += min(self.num_SM, work_loads_left)*(workload_time-self.ram_cycles[0]*1/self.clockspeed)*(1-overlap)
			work_loads_left -= self.num_SM
			
		actual_workload_time /= nb_work_loads
			
		self.node.out.write("Actual workload time: "+str(actual_workload_time)+" s.\n")
		
		rounded_nb_workloads = int(nb_work_loads/self.num_SM)*self.num_SM
		remainder = nb_work_loads-rounded_nb_workloads
		
		last_job_end = 0.0
		for i in range(nb_work_loads):
			# Schedule current workload onto the most available SM
			self.SM_availability[0] += actual_workload_time
			# Get the end time of the last job to schedule before sorting
			if i == nb_work_loads-1:
				last_job_end = self.SM_availability[0]
			self.SM_availability.sort()
		
		if remainder>0:
			self.SM_availability[0] += block_average
			last_job_end = self.SM_availability[0]
			self.SM_availability.sort()
		
		time = last_job_end-start+mem_write_delay*1/self.clockspeed
			
		self.node.out.write("Total GPU computations took "+str(time)+" s.\n")
		self.node.out.write("Maximum number of concurrent transfers: "+str(self.max_transfers)+"\n")

		return time


class K20X(Accelerator):
	"""
	A SimX resource	 that represents a K20X accelerator
	"""
	def __init__(self, node, id):
		super(K20X, self).__init__(node, id)
		#
		#  PARAMETERS
		#
		self.num_SM             = 14				# Number of streaming multiprocessors
		self.num_SP_per_SM      = 192				# Number of Single Precision cores per SM
		self.num_SF_per_SM		= 32
		self.num_DP_per_SM      = 64				# Number of Double Precision cores per SM
		self.clockspeed         = 732*10**6			# Hertz 

		self.cycles_per_iALU      = 10				# Integer ALU operation latency (3 for int/long ADD, int MUL and int FMAD, 6 for long MUL, no atomic long FMAD)
		self.cycles_per_fALU      = 10				# Float ALU operation latency
		self.cycles_per_spec_inst = 40				# Average latency for special instructions
		
		self.cache_levels 	= 2						# number of cache levels
		self.cache_sizes 	= [64*10**3, 1.5*10**6] # list of cache sizes
		self.cache_page_sizes	= [64, 1024] 		# list of page sizes for the different cache levels[bytes]
		self.num_registers 	= 65536					# number of registers (single precision, holds an int) [4 bytes]
		
		self.cache_cycles 	= [30, 175]				# list of cache access cycles per level
		self.ram_cycles 	= [230,1000]	#[270,400]			# number of cycles to load from main memory
		self.mem_capacity	= [8,48]	#[6,48]
		self.ram_page_size 	= 1024					# page size for main memory access [bytes]
		self.memorysize		= 6*10**9				# Global memory size in Bytes
		self.memory_use		= 0	

		self.warp_size		 	= 32				# Number of threads in a warp (similar to vector width)
		self.max_num_block_per_SM 	= 16			# Max number of blocks queued on a single SM
		self.max_num_threads_per_block 	= 1024		# Max number of (software) threads in a block
		self.max_num_threads_per_SM	= 2048			# Max number of threads queued or active on a single SM
													# Although 32 blocks can be allocated to a given SM, the 
													# total number of threads cannot exceed 2048
		self.kernel_launch_overhead = 3.5*10**-6	# Overhead for launching a kernel on the GPU
		self.SM_availability = []					# For each SM in the GPU contains the time at which the SM will 
													# be available for computations
		self.num_load_store_units = 32
		for i in range(self.num_SM):
			self.SM_availability.append(0.0)		# Initial value for SM_availability is 0, will be updated before use
		self.num_memory_ports = 24
		self.mem_ports = []
		for i in range(self.num_memory_ports):
			self.mem_ports.append(0.)
		self.nb_warp_schedulers = 4					# Number of warp schedulers available
		self.nb_ins_per_warp	= 2					# Number of instructions that can be issued simultaneously to a given warp			
		self.SP_units = []
		num_warps_SP_capabilities = int(self.num_SP_per_SM/self.warp_size)
		for i in range(num_warps_SP_capabilities):
			self.SP_units.append(0.)
		self.DP_units = []
		num_warps_DP_capabilities = int(self.num_DP_per_SM/self.warp_size)
		for i in range(num_warps_DP_capabilities):
			self.DP_units.append(0.)
		self.SF_units = []
		num_warps_SF_capabilities = int(self.num_SF_per_SM/self.warp_size)
		for i in range(num_warps_SF_capabilities):
			self.SF_units.append(0.)
		num_warps_LS_capabilities = int(self.num_load_store_units)
		self.LS_units = []
		for i in range(num_warps_LS_capabilities):
			self.LS_units.append(0.)


class K40(Accelerator):
	"""
	A SimX resource	 that represents a K20X accelerator
	"""
	def __init__(self, node, id):
		super(K40, self).__init__(node, id)
		#
		#  PARAMETERS
		#
		self.num_SM             = 15				# Number of streaming multiprocessors
		self.num_SP_per_SM      = 192				# Number of Single Precision cores per SM
		self.num_SF_per_SM		= 32
		self.num_DP_per_SM      = 64				# Number of Double Precision cores per SM
		self.clockspeed         = 745*10**6			# Hertz 

		self.cycles_per_iALU      = 9				# Integer ALU operation latency (3 for int/long ADD, int MUL and int FMAD, 6 for long MUL, no atomic long FMAD)
		self.cycles_per_fALU      = 9				# Float ALU operation latency
		self.cycles_per_spec_inst = 18				# Average latency for special instructions
		
		self.cache_levels 	= 2						# number of cache levels
		self.cache_sizes 	= [64*1024, 1.5*2**20]  # list of cache sizes
		self.cache_page_sizes	= [64, 1024] 		# list of page sizes for the different cache levels[bytes]
		self.num_registers 	= 65536					# number of registers (single precision, holds an int) [4 bytes]
		
		self.cache_cycles 	= [30, 175]				# list of cache access cycles per level
		self.ram_cycles 	= [230,1000]	#[270,400]			# number of cycles to load from main memory
		self.mem_capacity	= [10,40]	#[6,48]
		self.ram_page_size 	= 1024					# page size for main memory access [bytes]
		self.memorysize		= 12*2**30				# Global memory size in Bytes
		self.memory_use		= 0	

		self.warp_size		 	= 32				# Number of threads in a warp (similar to vector width)
		self.max_num_block_per_SM 	= 16			# Max number of blocks queued on a single SM
		self.max_num_threads_per_block 	= 1024		# Max number of (software) threads in a block
		self.max_num_threads_per_SM	= 2048			# Max number of threads queued or active on a single SM
													# Although 32 blocks can be allocated to a given SM, the 
													# total number of threads cannot exceed 2048
		self.kernel_launch_overhead = 3.5*10**-6	# Overhead for launching a kernel on the GPU
		self.SM_availability = []					# For each SM in the GPU contains the time at which the SM will 
													# be available for computations
		self.num_load_store_units = 32
		for i in range(self.num_SM):
			self.SM_availability.append(0.0)		# Initial value for SM_availability is 0, will be updated before use
		#self.num_memory_ports = 24
		#self.mem_ports = []
		#for i in range(self.num_memory_ports):
		#	self.mem_ports.append(0.)
		self.nb_warp_schedulers = 4					# Number of warp schedulers available
		self.nb_ins_per_warp	= 2					# Number of instructions that can be issued simultaneously to a given warp			
		self.SP_units = []
		num_warps_SP_capabilities = int(self.num_SP_per_SM/self.warp_size)
		for i in range(num_warps_SP_capabilities):
			self.SP_units.append(0.)
		self.DP_units = []
		num_warps_DP_capabilities = int(self.num_DP_per_SM/self.warp_size)
		for i in range(num_warps_DP_capabilities):
			self.DP_units.append(0.)
		self.SF_units = []
		num_warps_SF_capabilities = int(self.num_SF_per_SM/self.warp_size)
		for i in range(num_warps_SF_capabilities):
			self.SF_units.append(0.)
		num_warps_LS_capabilities = int(self.num_load_store_units)
		self.LS_units = []
		for i in range(num_warps_LS_capabilities):
			self.LS_units.append(0.)
			
class K6000(Accelerator):
	"""
	A SimX resource	 that represents a K20X accelerator
	"""
	def __init__(self, node, id):
		super(K6000, self).__init__(node, id)
		#
		#  PARAMETERS
		#
		self.num_SM             = 15				# Number of streaming multiprocessors
		self.num_SP_per_SM      = 192				# Number of Single Precision cores per SM
		self.num_SF_per_SM		= 32
		self.num_DP_per_SM      = 64				# Number of Double Precision cores per SM
		self.clockspeed         = 901.5*10**6		# Hertz 

		self.cycles_per_iALU      = 9				# Integer ALU operation latency
		self.cycles_per_fALU      = 9				# Float ALU operation latency
		self.cycles_per_spec_inst = 18				# Average latency for special instructions
		
		self.cache_levels 	= 2						# number of cache levels
		self.cache_sizes 	= [64*1024, 1.5*2**20]  # list of cache sizes
		self.cache_page_sizes	= [64, 1024] 		# list of page sizes for the different cache levels[bytes]
		self.num_registers 	= 65536					# number of registers (single precision, holds an int or a single) [4 bytes]
		
		self.cache_cycles 	= [30, 175]				# list of cache access cycles per level
		self.ram_cycles 	= [230,1000]	#[270,400]			# number of cycles to load from main memory
		self.mem_capacity	= [10,44]	#[6,48]
		self.ram_page_size 	= 1024					# page size for main memory access [bytes]
		self.memorysize		= 12*2**30				# Global memory size in Bytes
		self.memory_use		= 0	

		self.warp_size		 	= 32				# Number of threads in a warp (similar to vector width)
		self.max_num_block_per_SM 	= 16			# Max number of blocks queued on a single SM
		self.max_num_threads_per_block 	= 1024		# Max number of (software) threads in a block
		self.max_num_threads_per_SM	= 2048			# Max number of threads queued or active on a single SM
													# Although 32 blocks can be allocated to a given SM, the 
													# total number of threads cannot exceed 2048
		self.kernel_launch_overhead = 3.5*10**-6	# Overhead for launching a kernel on the GPU
		self.SM_availability = []					# For each SM in the GPU contains the time at which the SM will 
													# be available for computations
		self.num_load_store_units = 32
		for i in range(self.num_SM):
			self.SM_availability.append(0.0)		# Initial value for SM_availability is 0, will be updated before use
		self.num_memory_ports = 24
		self.mem_ports = []
		for i in range(self.num_memory_ports):
			self.mem_ports.append(0.)
		self.nb_warp_schedulers = 4					# Number of warp schedulers available
		self.nb_ins_per_warp	= 2					# Number of instructions that can be issued simultaneously to a given warp			
		self.SP_units = []
		num_warps_SP_capabilities = int(self.num_SP_per_SM/self.warp_size)
		for i in range(num_warps_SP_capabilities):
			self.SP_units.append(0.)
		self.DP_units = []
		num_warps_DP_capabilities = int(self.num_DP_per_SM/self.warp_size)
		for i in range(num_warps_DP_capabilities):
			self.DP_units.append(0.)
		self.SF_units = []
		num_warps_SF_capabilities = int(self.num_SF_per_SM/self.warp_size)
		for i in range(num_warps_SF_capabilities):
			self.SF_units.append(0.)
		num_warps_LS_capabilities = int(self.num_load_store_units)
		self.LS_units = []
		for i in range(num_warps_LS_capabilities):
			self.LS_units.append(0.)
			
			
class M2090(Accelerator):
	"""
	A SimX resource	 that represents a K20X accelerator
	"""
	def __init__(self, node, id):
		super(M2090, self).__init__(node, id)
		#
		#  PARAMETERS
		#
		self.num_SM             = 16				# Number of streaming multiprocessors
		self.num_SP_per_SM      = 32				# Number of Single Precision cores per SM
		self.num_SF_per_SM		= 4
		self.num_DP_per_SM      = 32				# Number of Double Precision cores per SM
		self.clockspeed         = 650*10**6		# Hertz 

		self.cycles_per_iALU      = 18				# Integer ALU operation latency
		self.cycles_per_fALU      = 18				# Float ALU operation latency
		self.cycles_per_spec_inst = 36				# Average latency for special instructions
		
		self.cache_levels 	= 2						# number of cache levels
		self.cache_sizes 	= [64*1024, 1.5*2**20]  # list of cache sizes
		self.cache_page_sizes	= [64, 1024] 		# list of page sizes for the different cache levels[bytes]
		self.num_registers 	= 65536					# number of registers (single precision, holds an int or a single) [4 bytes]
		
		self.cache_cycles 	= [25, 175]				# list of cache access cycles per level
		self.ram_cycles 	= [140,800]				# number of cycles to load from main memory
		self.mem_capacity	= [4,12]	
		self.ram_page_size 	= 1024					# page size for main memory access [bytes]
		self.memorysize		= 6*2**30				# Global memory size in Bytes
		self.memory_use		= 0	

		self.warp_size		 	= 32				# Number of threads in a warp (similar to vector width)
		self.max_num_block_per_SM 	= 8				# Max number of blocks queued on a single SM
		self.max_num_threads_per_block 	= 1024		# Max number of (software) threads in a block
		self.max_num_threads_per_SM	= 1536			# Max number of threads queued or active on a single SM
													# Although 32 blocks can be allocated to a given SM, the 
													# total number of threads cannot exceed 2048
		self.kernel_launch_overhead = 3.5*10**-6	# Overhead for launching a kernel on the GPU
		self.SM_availability = []					# For each SM in the GPU contains the time at which the SM will 
													# be available for computations
		self.num_load_store_units = 16
		for i in range(self.num_SM):
			self.SM_availability.append(0.0)		# Initial value for SM_availability is 0, will be updated before use
		self.num_memory_ports = 1
		self.mem_ports = []
		for i in range(self.num_memory_ports):
			self.mem_ports.append(0.)
		self.nb_warp_schedulers = 2					# Number of warp schedulers available
		self.nb_ins_per_warp	= 1					# Number of instructions that can be issued simultaneously to a given warp			
		self.SP_units = []
		num_warps_SP_capabilities = int(self.num_SP_per_SM/self.warp_size)
		for i in range(num_warps_SP_capabilities):
			self.SP_units.append(0.)
		self.DP_units = []
		num_warps_DP_capabilities = int(self.num_DP_per_SM/self.warp_size)
		for i in range(num_warps_DP_capabilities):
			self.DP_units.append(0.)
		self.SF_units = []
		num_warps_SF_capabilities = int(self.num_SF_per_SM/self.warp_size)
		for i in range(num_warps_SF_capabilities):
			self.SF_units.append(0.)
		num_warps_LS_capabilities = int(self.num_load_store_units)
		self.LS_units = []
		for i in range(num_warps_LS_capabilities):
			self.LS_units.append(0.)


class Pascal(Accelerator):
	"""
	A SimX resource	 that represents a K20X accelerator
	"""
	def __init__(self, node, id):
		super(Pascal, self).__init__(node, id)
		#
		#  PARAMETERS
		#
		self.num_SM             = 16				# Number of streaming multiprocessors
		self.num_SP_per_SM      = 256				# Number of Single Precision cores per SM
		self.num_SF_per_SM		= 64
		self.num_DP_per_SM      = 128				# Number of Double Precision cores per SM
		self.clockspeed         = 900*10**6			# Hertz 

		self.cycles_per_iALU      = 9				# Integer ALU operation latency (3 for int/long ADD, int MUL and int FMAD, 6 for long MUL, no atomic long FMAD)
		self.cycles_per_fALU      = 9				# Float ALU operation latency
		self.cycles_per_spec_inst = 18				# Average latency for special instructions
		
		self.cache_levels 	= 2						# number of cache levels
		self.cache_sizes 	= [64*1024, 1.5*2**20]  # list of cache sizes
		self.cache_page_sizes	= [64, 1024] 		# list of page sizes for the different cache levels[bytes]
		self.num_registers 	= 65536					# number of registers (single precision, holds an int) [4 bytes]
		
		self.cache_cycles 	= [30, 175]				# list of cache access cycles per level
		self.ram_cycles 	= [230,1000]	#[270,400]			# number of cycles to load from main memory
		self.mem_capacity	= [30,120]	#[6,48]
		self.ram_page_size 	= 1024					# page size for main memory access [bytes]
		self.memorysize		= 24*2**30				# Global memory size in Bytes
		self.memory_use		= 0	

		self.warp_size		 	= 32				# Number of threads in a warp (similar to vector width)
		self.max_num_block_per_SM 	= 32			# Max number of blocks queued on a single SM
		self.max_num_threads_per_block 	= 1024		# Max number of (software) threads in a block
		self.max_num_threads_per_SM	= 2048			# Max number of threads queued or active on a single SM
													# Although 32 blocks can be allocated to a given SM, the 
													# total number of threads cannot exceed 2048
		self.kernel_launch_overhead = 3.5*10**-6	# Overhead for launching a kernel on the GPU
		self.SM_availability = []					# For each SM in the GPU contains the time at which the SM will 
													# be available for computations
		self.num_load_store_units = 64
		for i in range(self.num_SM):
			self.SM_availability.append(0.0)		# Initial value for SM_availability is 0, will be updated before use
		#self.num_memory_ports = 24
		#self.mem_ports = []
		#for i in range(self.num_memory_ports):
		#	self.mem_ports.append(0.)
		self.nb_warp_schedulers = 4					# Number of warp schedulers available
		self.nb_ins_per_warp	= 2					# Number of instructions that can be issued simultaneously to a given warp			
		self.SP_units = []
		num_warps_SP_capabilities = int(self.num_SP_per_SM/self.warp_size)
		for i in range(num_warps_SP_capabilities):
			self.SP_units.append(0.)
		self.DP_units = []
		num_warps_DP_capabilities = int(self.num_DP_per_SM/self.warp_size)
		for i in range(num_warps_DP_capabilities):
			self.DP_units.append(0.)
		self.SF_units = []
		num_warps_SF_capabilities = int(self.num_SF_per_SM/self.warp_size)
		for i in range(num_warps_SF_capabilities):
			self.SF_units.append(0.)
		num_warps_LS_capabilities = int(self.num_load_store_units)
		self.LS_units = []
		for i in range(num_warps_LS_capabilities):
			self.LS_units.append(0.)			
			