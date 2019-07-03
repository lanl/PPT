# Copyright (c) 2017, Los Alamos National Security, LLC
# All rights reserved.
# Copyright 2017. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
# Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
*********** Performance Prediction Toolkit PPT *********

File: accelerators.py
Description: Predict the performance of the target gpu on a predifned application
"""


import math


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
		self.max_transfers=0


	def allocate_device_mem(self, size):
		"""
		Allocate or deallocate memory on the GPU
		size is given in Bytes
		"""
		if self.memory_use + size > self.global_mem_size:
			print "Warning: unable to allocate memory on device ", str(self.id), " of node ", str(self.node.num), "."
		else:
			self.memory_use += size


	def transfer_to_device(self, size):
		"""
		Transfer size bytes between the CPU and the GPU
		Return the time taken by the transfer
		"""
		if size > self.memory_use:
			print "Warning: attempting to transfer more bytes than allocated on device ", str(self.id), " of node ", str(self.node.num), "."
		return size/ self.node.PCIe_bandwidth + self.node.PCIe_latency


	def kernel_call(self, tasklist, blocksize, gridsize, regcount, staticSMem, phit_l2, start):
		"""
		Compute the time spent and IPC executed to complete the given kernel
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
		
		# Number of registers in a block = number of threads in a block * number registers per thread
		nb_registers_per_block = blocksize*regcount
		
		#Shared memory per block
		shared_memory_per_block = min(staticSMem, self.shared_mem_size) 
		
		# Number of blocks that can run concurrently on a single SM
		nb_blocks_per_sm = min(self.max_num_block_per_SM, int(self.max_num_threads_per_SM/blocksize))
		nb_blocks_per_sm = min(nb_blocks_per_sm, int(self.num_registers/nb_registers_per_block))
		nb_blocks_per_sm = min(nb_blocks_per_sm, int(gridsize+self.num_SM-1)/self.num_SM)
		if shared_memory_per_block!=0:
			nb_blocks_per_sm = min(nb_blocks_per_sm, int(self.shared_mem_size/shared_memory_per_block))
		
		# Total number of warps per SM given the number of blocks
		nb_warps_per_sm = min(nb_blocks_per_sm*nb_warps_per_block, self.max_num_warps_per_SM)

		self.node.out.write("Allocated "+str(nb_blocks_per_sm)+" simultaneous blocks per SM for a total of "+str(nb_warps_per_sm)+" warps.\n")
		
		# Total number of workloads for this kernel
		nb_work_loads = int((gridsize + nb_blocks_per_sm -1)/nb_blocks_per_sm)	
		
		AMT = (phit_l2 * self.l2_mem_latency) + (1-phit_l2)* (self.l2_mem_latency + self.l2_to_global_mem_latency)
		self.average_memory_latency = AMT
	
		# Initilaize the blocks 	
		block_list = self.spawn_blocks(nb_blocks_per_sm, nb_warps_per_block, tasklist)
		cycles = 0.
				
		#process the instruction in the task by the active blocks every cycle
		while self.blockList_has_active_warps(block_list, cycles):	
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
		
		workload_time = (cycles)*1/self.clockspeed
		block_average = (block_average)*1/self.clockspeed
		nb_inst = stats['IPC']

		self.node.out.write("BLOCK AVERAGE = "+ str(block_average)+"\n")
		self.node.out.write("GPU computations for a single workload took "+str(cycles)+" cycles.\n")
		self.node.out.write("GPU computations for a single workload took "+str(workload_time)+" s.\n")
		self.node.out.write("Total number of workloads "+str(nb_work_loads)+".\n")
		self.node.out.write("Total number of instructions issued for a single workload: "+str(stats['IPC'])+".\n")
		self.node.out.write("Potential workload overlap: "+str(overlap*100)+"%.\n")
		stats['IPC'] = nb_inst/cycles
		self.node.out.write("Overall IPC "+str(stats['IPC'])+".\n")
		inst_per_second = self.clockspeed * stats['IPC']
		self.node.out.write("Overall MIPS: "+str(inst_per_second/100000)+".\n")
		
		nb_workload_per_SM = int((nb_work_loads+self.num_SM-1)/self.num_SM)
		work_loads_left = nb_work_loads
		actual_workload_time = 0.

		for i in range(nb_workload_per_SM):
			if i==0:
				actual_workload_time+=min(self.num_SM, work_loads_left)*workload_time
			else:
				actual_workload_time += min(self.num_SM, work_loads_left)*(workload_time*1/self.clockspeed)*(1-overlap)
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


	def spawn_blocks(self, nb_blocks, nb_warps_per_block, tasklist):
	 	"""
		- Create number of Blocks (nb_blocks) to run on 1 SM each has certain number of warps (nb_warps_per_block) 
		- Return a list of these blocks
		"""
		block_list = []
	 	for i in range(nb_blocks):
	 		block_list.append(Accelerator.Block(self, i, nb_warps_per_block, tasklist))
	 	return block_list


	def blockList_has_active_warps(self, block_list, cycles):
		"""
		Return True if any warp in the block list is active
		"""
		for block in block_list:
	 		if block.is_active(cycles): # a block is active if it has any active warp
	 			return True
	 	return False
	

	def step(self, block_list, cycles, stats):
		"""
		Advance computations on all warps by one cycle
		"""
		warps_issued = 0 #number of warps running concurrently
	 	for block in block_list :
	 		if warps_issued >= self.num_warp_schedulers:
	 			break
	 		warps_issued+=block.step(warps_issued, cycles)
	 	stats['IPC'] += warps_issued
	
	
	class Block(object):
		"""
		Class that represents a Block (CTA) being executed on an SM
		"""
		def __init__(self, gpu, id, nb_warps, task_list):
			self.gpu = gpu
			self.nb_warps = nb_warps 
			self.sync_cnter = 0
			self.id = id
			self.active = True
			self.end_computations = 0.
			self.actual_end = 0 # ??? 
			self.warp_list = []
			for i in range(self.nb_warps):
				self.warp_list.append(gpu.Warp(self.gpu, self, task_list,str(self.id)+"-"+str(i)))
		
		# Return whether the block has an active warp or not
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
		
		#Advance computation for the active warps in the block by one cycle
		def step(self, issued, cycles):
			self_issued = 0
			for warp in self.warp_list:
				if warp.is_active():
					if issued+self_issued >= self.gpu.num_warp_schedulers:
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
	
		def request_unit(self, cycles, delay, units):
			result = False
			for i in range(len(units)):
				if units[i]<=cycles:
					units[i] = cycles+delay
					result = True
					break
			return result


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

 		# Return whether the warp is still active
 		def is_active(self):
 			return self.active
 		
		# Advance computations on the current warp by one clock cycle
 		def step(self, cycles):
 			res = False
 			inst_issued = 0 #number of inst that can be issued at the same time  
 			while inst_issued < self.gpu.num_inst_per_warp:
 				if not self.process_inst(cycles):
 					break
 				res = True
 				inst_issued += 1
 			return res

 		def process_inst(self, cycles):
 			result = False
 			if self.stalled < cycles:
 			 	if self.syncing:
 			 		pass
	 			elif self.current_inst == len(self.task_list):
	 				self.active = False
	 			else:
	 				inst = self.task_list[self.current_inst]
	 				max_dep = 0
	 				for i in inst[2:]:
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
							if inst[1] == 'LOAD' or inst[1] == 'STORE':
								latency = self.gpu.average_memory_latency
							else:
								latency = (self.gpu.l2_mem_latency+self.gpu.l2_to_global_mem_latency)
		 					result = self.perform_mem_request(cycles, latency) 
						elif inst[0] == 'PARAM_MEM_ACCESS':
							latency = self.gpu.const_mem_latency
							result = self.perform_mem_request(cycles, latency) 
						elif inst[0] == 'SHARED_MEM_ACCESS':
							latency = self.gpu.shared_mem_latency
							result = self.perform_mem_request(cycles, latency, False)
						elif inst[0] == 'LOCAL_MEM_ACCESS':
		 					latency = self.gpu.local_mem_latency
							result = self.perform_mem_request(cycles, latency)
						elif inst[0] == 'CONST_MEM_ACCESS':
		 					latency = self.gpu.const_mem_latency
							result = self.perform_mem_request(cycles, latency)	
						elif inst[0] == 'TEX_MEM_ACCESS':
		 					latency = self.gpu.tex_mem_latency
							result = self.perform_mem_request(cycles, latency)		 
						elif inst[0] == 'fALU' or inst[0] == 'iALU':
							if self.block.request_unit(cycles, 1, self.gpu.SP_units):
								self.completions.append(cycles+inst[1]) 
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					result = True
			 					self.current_inst += 1
						elif inst[0] == 'dALU':
							if self.block.request_unit(cycles, 1, self.gpu.DP_units):
								self.completions.append(cycles+inst[1])
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					result = True
			 					self.current_inst += 1
						elif inst[0] == 'SFU':
							if self.block.request_unit(cycles, 1, self.gpu.SF_units):
								self.completions.append(cycles+inst[1])
			 					if self.max_dep < self.completions[-1]:
			 						self.max_dep = self.completions[-1]
			 					result = True
			 					self.current_inst += 1
						elif inst[0] == 'THREAD_SYNC':
		 					self.completions.append(0)
							self.block.sync_cnter += 1
							self.syncing = True
		 					result = True
		 					self.current_inst += 1
						else:
							print 'Warning: unknown task list item', inst,' cannot be parsed, ignoring it'
							self.current_inst += 1
			return result

		# return True if the request is fulfilled
		def perform_mem_request(self, cycles, latency, useLDST = True):
			result = False
			if not useLDST:
				self.completions.append(cycles+latency)
		 		if self.max_dep < self.completions[-1]:
		 			self.max_dep = self.completions[-1]
				result = True
		 		self.current_inst += 1
				return result
			else:
				if self.block.request_unit(cycles, 1, self.gpu.LS_units):
		 			ok, mem_time = self.gpu.memory_access_time(cycles,latency)
		 			if ok:
			 			self.gpu.mem_op.append(cycles+mem_time)
				 		self.completions.append(cycles+mem_time)
				 		if self.max_dep < self.completions[-1]:
				 			self.max_dep = self.completions[-1]
				 		result = True
						self.current_inst += 1
				return result

	
	def memory_access_time(self, cycles, latency):
		"""
		Return time needed to access memory
		"""
		time = 0
		ok = True
		while len(self.mem_op) >0 and self.mem_op[0]<=cycles:
			self.mem_op.pop()
		if len(self.mem_op)+1>self.max_transfers:
			self.max_transfers = len(self.mem_op)
		if len(self.mem_op)>=self.global_mem_return_queue:
			ok=False
	 	time = latency 
		return ok, time



class GPU(Accelerator):
	"""
	A Simian resource that represent a GPU 
	"""
	def __init__(self, node, id, gpu_config):
		super(GPU, self).__init__(node, id)
		#
		#  Target GPU Configurations
		#
		self.num_SM = gpu_config["num_SM"] 
		self.num_SP_per_SM = gpu_config["num_SP_per_SM"]  
		self.num_SF_per_SM = gpu_config["num_SF_per_SM"] 
		self.num_DP_per_SM = gpu_config["num_DP_per_SM"]
		self.num_load_store_units = gpu_config["num_load_store_units"]
		self.num_warp_schedulers = gpu_config["num_warp_schedulers"]
		self.num_inst_per_warp = gpu_config["num_inst_per_warp"]
		self.clockspeed = gpu_config["clockspeed"]

		self.num_registers = gpu_config["num_registers"]

		self.l1_cache_size = gpu_config["l1_cache_size"]
		self.l2_cache_size = gpu_config["l2_cache_size"]
		self.global_mem_size = gpu_config["global_mem_size"]
		self.shared_mem_size = gpu_config["shared_mem_size"]
		
		self.l1_mem_latency = gpu_config["l1_mem_latency"]
		self.l2_mem_latency = gpu_config["l2_mem_latency"]
		self.l2_to_global_mem_latency = gpu_config["l2_to_global_mem_latency"]
		self.local_mem_latency = gpu_config["local_mem_latency"]
		self.const_mem_latency = gpu_config["const_mem_latency"]
		self.tex_mem_latency = gpu_config["tex_mem_latency"]
		self.shared_mem_latency = gpu_config["shared_mem_latency"]
		self.average_memory_latency	= 0  # Average Number of cycles after applying AMT, it will be upated later

		self.warp_size = gpu_config["warp_size"]
		self.max_num_warps_per_SM = gpu_config["max_num_warps_per_SM"]
		self.max_num_block_per_SM = gpu_config["max_num_block_per_SM"] 
		self.max_num_threads_per_block = gpu_config["max_num_threads_per_block"]
		self.max_num_threads_per_SM	= gpu_config["max_num_threads_per_SM"] 
													
		self.memory_use = 0  # Allocated memory, it will updated later
		self.global_mem_return_queue = gpu_config["global_mem_return_queue"]
		self.num_memory_ports = gpu_config["num_memory_ports"]
		
		self.kernel_launch_overhead = 3.5*10**-6	# Overhead for launching a kernel on the GPU
		
		self.SM_availability = []  # For each SM in the GPU contains the time at which the SM will be available for computations
		for i in range(self.num_SM):
			self.SM_availability.append(0.0)  # Initial value for SM_availability is 0, will be updated before use

		self.mem_ports = []
		for i in range(self.num_memory_ports):
			self.mem_ports.append(0.)
		
		self.SP_units = []
		num_warps_SP_capabilities = int(self.num_SP_per_SM/self.warp_size)
		num_warps_SP_capabilities = max(num_warps_SP_capabilities, 1)
		for i in range(num_warps_SP_capabilities):
			self.SP_units.append(0.)
		
		self.DP_units = []
		num_warps_DP_capabilities = int(self.num_DP_per_SM/self.warp_size)
		num_warps_DP_capabilities = max(num_warps_DP_capabilities, 1)
		for i in range(num_warps_DP_capabilities):
			self.DP_units.append(0.)
		
		self.SF_units = []
		num_warps_SF_capabilities = int(self.num_SF_per_SM/self.warp_size)
		num_warps_SF_capabilities = max(num_warps_SF_capabilities, 1)
		for i in range(num_warps_SF_capabilities):
			self.SF_units.append(0.)
		num_warps_LS_capabilities = int(self.num_load_store_units)
		
		self.LS_units = []
		for i in range(num_warps_LS_capabilities):
			self.LS_units.append(0.)
