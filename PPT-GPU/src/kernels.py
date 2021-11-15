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


import sys, time, importlib
from simian import Entity, Simian
from .helper_methods import *
from .memory_model import *
from .blocks import Block
from .warp_scheduler import Scheduler


class Kernel(Entity):

	def __init__(self, base_info, gpuNode, kernel_info):
		super(Kernel, self).__init__(base_info)
		# print("kernel %s, %s inits on Entity %d, Rank %d" % (kernel_info['kernel_name'], self.name, self.num, self.engine.rank))
		# sys.stdout.flush()

		## There is 1 Acc in Node replicated for each Kernel
		self.gpuNode = gpuNode
		self.acc = self.gpuNode.accelerators[0] 

		# get kernel_info
		self.kernel_id_real = self.num
		self.kernel_name = kernel_info['kernel_name']
		self.kernel_id = int(kernel_info["kernel_id"])
		self.mem_traces_dir_path = kernel_info['mem_traces_dir_path']
		self.kernel_grid_size = kernel_info['grid_size']
		self.kernel_block_size = kernel_info['block_size']
		self.kernel_num_regs = kernel_info['num_regs']
		self.kernel_smem_size = kernel_info['smem_size']
		self.ISA = "PTX" if kernel_info['ISA'] == 1 else "SASS"
		self.ptx_file_path = kernel_info["ptx_file_path"]
		self.sass_file_path = kernel_info["sass_file_path"]
		if kernel_info["granularity"] == "1":
			self.simulation_granularity = "OTB" #One Thread Block
		elif kernel_info["granularity"] == "2":
			self.simulation_granularity = "AcTB" #Active Thread Block
		elif kernel_info["granularity"] == "3":
			self.simulation_granularity = "AlTB" #All Thread Block

		# self.simulation_granularity = "TBSW" if kernel_info["granularity"] == "1" else "TBS"

		## kernel local predictions outputs
		self.pred_out = {}
		pred_out = self.pred_out

		## init predictions outputs
		pred_out["app_path"] = kernel_info['app_path']
		pred_out["kernel_id"] = self.kernel_id 
		pred_out["kernel_name"] = self.kernel_name
		pred_out["ISA"] = self.ISA
		pred_out["granularity"] = kernel_info["granularity"]
		pred_out["total_num_workloads"] = 0
		pred_out["active_SMs"] = 0
		pred_out["max_active_blocks_per_SM"] = self.acc.max_active_blocks_per_SM
		pred_out["blocks_per_SM_limit_warps"] = 0.0
		pred_out["blocks_per_SM_limit_regs"] = 0.0
		pred_out["blocks_per_SM_limit_smem"] = 0.0
		pred_out["th_active_blocks"] = 0
		pred_out["th_active_warps"] = 0
		pred_out["th_active_threads"] = 0
		pred_out["th_occupancy"] = 0.0
		pred_out["allocated_active_blocks_per_SM"] = 0
		pred_out["allocated_active_warps_per_block"] = 0
		pred_out["achieved_active_warps"] = 0.0
		pred_out["achieved_occupancy"] = 0.0
		pred_out["num_workloads_per_SM_orig"] = 0
		pred_out["num_workloads_per_SM_new"] = 0
		pred_out["active_cycles"] = 0
		pred_out["warps_instructions_executed"] = 0
		pred_out["threads_instructions_executed"] = 0
		pred_out["ipc"] = 0
		pred_out["l1_cache_bypassed"] = self.acc.l1_cache_bypassed
		pred_out["comp_cycles"] = 0
		pred_out["gpu_act_cycles"] = 0
		pred_out["gpu_elp_cycles"] = 0
		pred_out["sm_act_cycles.sum"] = 0
		pred_out["sm_elp_cycles.sum"] = 0
		pred_out["last_inst_delay"] = 0
		pred_out["tot_warps_instructions_executed"] = 0
		pred_out["tot_ipc"] = 0
		pred_out["tot_cpi"] = 0.0
		pred_out["tot_throughput_ips"] = 0.0
		pred_out["execution_time_sec"] = 0.0
		pred_out["AMAT"] = 0
		pred_out["ACPAO"] = 0
		pred_out["memory_stats"] = {}
		pred_out["simulation_time"] = {}
		

		if self.kernel_block_size > self.acc.max_block_size:
			print_warning("block_size",str(self.acc.max_block_size))
			self.kernel_block_size = self.acc.max_block_size

		if self.kernel_num_regs > self.acc.max_registers_per_thread:
			print_warning("num_registers",str(self.acc.max_registers_per_thread))
			self.kernel_num_regs = self.acc.max_registers_per_thread

		if self.kernel_smem_size > self.acc.shared_mem_size:
			print_warning("shared_mem_bytes",str(self.acc.shared_mem_size))
			self.kernel_smem_size = self.acc.shared_mem_size

		pred_out["total_num_workloads"]  = self.kernel_grid_size
		pred_out["active_SMs"] = min(self.acc.num_SMs, pred_out["total_num_workloads"])
		pred_out["allocated_active_warps_per_block"] = int(ceil((float(self.kernel_block_size)/float(self.acc.warp_size)),1))
		pred_out["blocks_per_SM_limit_warps"] = int(min(pred_out["max_active_blocks_per_SM"],\
				int(floor((self.acc.max_active_warps_per_SM/pred_out["allocated_active_warps_per_block"]),1))))

		if self.kernel_num_regs == 0: pred_out["blocks_per_SM_limit_regs"] = pred_out["max_active_blocks_per_SM"]
		else:
			allocated_regs_per_warp = ceil((self.kernel_num_regs*self.acc.warp_size),self.acc.register_allocation_size)
			allocated_regs_per_SM = int(floor((self.acc.max_registers_per_block/allocated_regs_per_warp),\
				self.acc.num_warp_schedulers_per_SM))
			pred_out["blocks_per_SM_limit_regs"] = int(floor((allocated_regs_per_SM/pred_out\
				["allocated_active_warps_per_block"]),1) * floor((self.acc.max_registers_per_SM/\
				self.acc.max_registers_per_block),1))

		if self.kernel_smem_size == 0:
			pred_out["blocks_per_SM_limit_smem"] = pred_out["max_active_blocks_per_SM"]
		else:
			smem_per_block = ceil(self.kernel_smem_size, self.acc.smem_allocation_size)
			pred_out["blocks_per_SM_limit_smem"] = int(floor((self.acc.shared_mem_size/smem_per_block),1))

		pred_out["allocated_active_blocks_per_SM"] = min(pred_out["blocks_per_SM_limit_warps"],\
													pred_out["blocks_per_SM_limit_regs"],\
													pred_out["blocks_per_SM_limit_smem"])

		pred_out["th_active_blocks"] = pred_out["allocated_active_blocks_per_SM"]
		pred_out["th_active_warps"]=pred_out["allocated_active_blocks_per_SM"] * pred_out["allocated_active_warps_per_block"]
		pred_out["th_occupancy"] =int(ceil((float(pred_out["th_active_warps"])/float(\
			self.acc.max_active_warps_per_SM) * 100),1))
		
		pred_out["num_workloads_per_SM_orig"] = float(pred_out["total_num_workloads"])/float(self.acc.num_SMs)
		pred_out["num_workloads_per_SM_orig"] = int(ceil(pred_out["num_workloads_per_SM_orig"],1))
		
		pred_out["allocated_active_blocks_per_SM"] = min(pred_out["allocated_active_blocks_per_SM"], pred_out["num_workloads_per_SM_orig"])
		pred_out["allocated_active_blocks_per_SM"] = int(ceil(pred_out["allocated_active_blocks_per_SM"], 1))
		
		## allocate the num_workloads_per_SM_new according to the simulation granularity
		if self.simulation_granularity == "OTB":
			pred_out["allocated_active_blocks_per_SM"] = 1
			pred_out["num_workloads_per_SM_new"] = 1
		elif self.simulation_granularity == "AcTB":
			pred_out["num_workloads_per_SM_new"] = pred_out["allocated_active_blocks_per_SM"] 
		elif self.simulation_granularity == "AlTB":
			pred_out["num_workloads_per_SM_new"] = pred_out["num_workloads_per_SM_orig"]

		## initilaizing kernel's warp scehdulers 
		self.warp_scheduler = Scheduler(self.acc.num_warp_schedulers_per_SM, self.acc.warp_scheduling_policy)



	def kernel_call(self, data, name, num):
		
		pred_out = self.pred_out

		if self.ISA == "PTX":
			ptx_parser = importlib.import_module("ISA_parser.ptx_parser")
			self.kernel_tasklist, gmem_reqs = ptx_parser.parse(units_latency = self.acc.units_latency, ptx_instructions = self.acc.ptx_isa,\
																ptx_path = self.ptx_file_path, num_warps = pred_out["allocated_active_warps_per_block"])

		elif self.ISA == "SASS":
			sass_parser = importlib.import_module("ISA_parser.sass_parser")
			self.kernel_tasklist, gmem_reqs = sass_parser.parse(units_latency = self.acc.units_latency, sass_instructions = self.acc.sass_isa,\
																sass_path = self.sass_file_path, num_warps = pred_out["allocated_active_warps_per_block"])
													
		###### ---- memory performance predictions ---- ######
		tic = time.time()
		pred_out["memory_stats"] = get_memory_perf(pred_out["kernel_id"], self.mem_traces_dir_path, pred_out["total_num_workloads"], self.acc.num_SMs,\
													self.acc.l1_cache_size, self.acc.l1_cache_line_size, self.acc.l1_cache_associativity,\
													self.acc.l2_cache_size, self.acc.l2_cache_line_size, self.acc.l2_cache_associativity,\
													gmem_reqs, int(pred_out["num_workloads_per_SM_orig"]), int(pred_out["num_workloads_per_SM_new"]))
		toc = time.time()
		pred_out["simulation_time"]["memory"] = (toc - tic)

		# AMAT: Average Memory Access Time (Cycles)
		if pred_out["memory_stats"]["gmem_tot_reqs"] != 0:

			highly_divergent_degree = 17
			l2_parallelism = 1
			dram_parallelism = 1
			if pred_out["memory_stats"]["gmem_ld_diverg"] >= self.acc.num_dram_channels\
			or pred_out["memory_stats"]["gmem_st_diverg"] >= self.acc.num_dram_channels\
			or pred_out["memory_stats"]["gmem_tot_diverg"] >= highly_divergent_degree:
				l2_parallelism = pred_out["memory_stats"]["gmem_tot_diverg"] if pred_out["memory_stats"]["gmem_tot_diverg"] < self.acc.num_dram_channels else self.acc.num_dram_channels
				dram_parallelism = pred_out["memory_stats"]["gmem_tot_diverg"] if pred_out["memory_stats"]["gmem_tot_diverg"] < self.acc.num_dram_channels else self.acc.num_dram_channels
				# l2_parallelism = self.num_dram_channels
				# dram_parallelism = self.num_dram_channels
				# l2_parallelism = self.num_l2_partitions
				# l2_parallelism = pred_out["memory_stats"]["gmem_tot_diverg"]
				# dram_parallelism = pred_out["memory_stats"]["gmem_tot_diverg"]

			l1_cycles_no_contention = (pred_out["memory_stats"]["l1_sm_trans_gmem"]) * self.acc.l1_cache_access_latency
			l2_cycles_no_contention = pred_out["memory_stats"]["l2_tot_trans_gmem"] * self.acc.l2_cache_from_l1_access_latency * (1/l2_parallelism)
			dram_cycles_no_contention = pred_out["memory_stats"]["dram_tot_trans_gmem"] * self.acc.dram_mem_from_l2_access_latency * (1/dram_parallelism)
			
			mem_cycles_no_contention = max(l1_cycles_no_contention, l2_cycles_no_contention) 
			mem_cycles_no_contention = max(mem_cycles_no_contention, dram_cycles_no_contention)
			mem_cycles_no_contention = ceil(mem_cycles_no_contention, 1)
			
			dram_service_latency = self.acc.dram_clockspeed * (self.acc.l2_cache_line_size / self.acc.dram_bandwidth)
			dram_queuing_delay_cycles = pred_out["memory_stats"]["dram_tot_trans_gmem"] * dram_service_latency * (1/dram_parallelism)

			mem_cycles_ovhds = dram_queuing_delay_cycles
		
			noc_service_latency = self.acc.dram_clockspeed * (self.acc.l1_cache_line_size / self.acc.noc_bandwidth)
			noc_queueing_delay_cycles = pred_out["memory_stats"]["l2_tot_trans_gmem"] * noc_service_latency * (1/l2_parallelism)

			if noc_queueing_delay_cycles > (self.acc.l2_cache_from_l1_access_latency + self.acc.dram_mem_from_l2_access_latency):
				mem_cycles_ovhds += noc_queueing_delay_cycles

			mem_cycles_ovhds = ceil(mem_cycles_ovhds, 1)

			tot_mem_cycles = ceil((mem_cycles_no_contention + mem_cycles_ovhds), 1)
			
			pred_out["AMAT"] = tot_mem_cycles/pred_out["memory_stats"]["gmem_tot_reqs"]
			pred_out["AMAT"] = ceil(pred_out["AMAT"], 1)

				
		# ACPAO: Average Cycles Per Atomic Operation
		# ACPAO = atomic operations latency / total atomic requests
		# atomic operations latency= (atomic & redcutions transactions * access latency of atomic & red requests)
		if pred_out["memory_stats"]["atom_red_tot_trans"] != 0:
			pred_out["ACPAO"] = (self.acc.atomic_op_access_latency * pred_out["memory_stats"]["atom_red_tot_trans"])\
							/(pred_out["memory_stats"]["atom_tot_reqs"] + pred_out["memory_stats"]["red_tot_reqs"])



		###### ---- compute performance predictions ---- ######
		tic = time.time()
		block_list = self.spawn_blocks(self.acc, pred_out["num_workloads_per_SM_new"], pred_out["allocated_active_warps_per_block"],\
										self.kernel_tasklist, self.kernel_id_real, self.ISA, pred_out["AMAT"], pred_out["ACPAO"])

		
		## before we do anything we need to activate Blocks up to active blocks
		for i in range(pred_out["allocated_active_blocks_per_SM"]):
			block_list[i].active = True
			block_list[i].waiting_to_execute = False

		pred_out["comp_cycles"] = self.acc.TB_launch_overhead
		## process instructions of the tasklist by the active blocks every cycle
		while self.blockList_has_active_warps(block_list):

			## compute the list warps in active blocks
			current_active_block_list = []
			current_warp_list = []
			current_active_blocks = 0

			for block in block_list:
				if current_active_blocks >= pred_out["allocated_active_blocks_per_SM"]:
					break
				
				## all warps inside this block finished execution
				if not block.is_active() and not block.is_waiting_to_execute():
					continue
				
				## block is ready to execute
				if not block.is_active() and block.is_waiting_to_execute():
					block.active = True
					block.waiting_to_execute = False
					## add latency of scheduling a new TB 
					pred_out["comp_cycles"] += self.acc.TB_launch_overhead / pred_out["allocated_active_blocks_per_SM"]
				
				## this block still has warps executing; add its warps to the warp list
				if block.is_active() and not block.is_waiting_to_execute():
					current_active_block_list.append(block)
					block_active_warp_list = []
					for warp in block.warp_list:
						if warp.is_active():
							block_active_warp_list.append(warp)
					current_warp_list += block_active_warp_list
					current_active_blocks += 1

			## pass warps belonging to the active blocks to the warp scheduler to step the computations
			instructions_executed = self.warp_scheduler.step(current_warp_list, pred_out["active_cycles"]) 
			pred_out["warps_instructions_executed"] += instructions_executed

			for block in current_active_block_list:
				pred_out["achieved_active_warps"] += block.count_active_warps()

			## next cycles
			pred_out["active_cycles"] += 1

		pred_out["achieved_active_warps"] = pred_out["achieved_active_warps"] / pred_out["active_cycles"]
		pred_out["achieved_occupancy"]= (float(pred_out["achieved_active_warps"]) / float(self.acc.max_active_warps_per_SM)) * 100

		#TODO: has to be done in a more logical way per TB
		last_inst_delay = 0
		for block in block_list:
			last_inst_delay_act_min = max(last_inst_delay, block.actual_end - pred_out["active_cycles"])
			last_inst_delay_act_max = max(last_inst_delay, block.actual_end)

		act_cycles_min = pred_out["active_cycles"] + pred_out["comp_cycles"] + last_inst_delay_act_min
		act_cycles_max = pred_out["active_cycles"] + pred_out["comp_cycles"] + last_inst_delay_act_max

		avg_instructions_executed_per_block = pred_out["warps_instructions_executed"] / len(block_list)

		num_workloads_left = pred_out["num_workloads_per_SM_orig"] - pred_out["num_workloads_per_SM_new"]
		
		if num_workloads_left > 0:
			remaining_cycles = ceil((num_workloads_left/pred_out["num_workloads_per_SM_new"]),1)
			pred_out["gpu_act_cycles_min"] = act_cycles_min * remaining_cycles
			pred_out["gpu_act_cycles_max"] = act_cycles_max * remaining_cycles
		else:
			pred_out["gpu_act_cycles_min"] = act_cycles_min
			pred_out["gpu_act_cycles_max"] = act_cycles_max

		pred_out["sm_act_cycles.sum"] = pred_out["gpu_act_cycles_max"] * pred_out["active_SMs"]
		pred_out["sm_elp_cycles.sum"] = pred_out["gpu_act_cycles_max"] * self.acc.num_SMs
		pred_out["tot_warps_instructions_executed"] = avg_instructions_executed_per_block * pred_out["total_num_workloads"]
		pred_out["tot_threads_instructions_executed"] = (pred_out["tot_warps_instructions_executed"] * self.kernel_block_size) / pred_out["allocated_active_warps_per_block"]
		pred_out["tot_ipc"] = pred_out["tot_warps_instructions_executed"] * (1.0/pred_out["sm_act_cycles.sum"])
		pred_out["tot_cpi"] = 1 * (1.0/pred_out["tot_ipc"])
		pred_out["tot_throughput_ips"] = pred_out["tot_ipc"] * self.acc.GPU_clockspeed
		pred_out["execution_time_sec"] = pred_out["sm_elp_cycles.sum"] * (1.0/self.acc.GPU_clockspeed)

		toc = time.time()
		pred_out["simulation_time"]["compute"] = (toc - tic)

		## commit results
		dump_output(pred_out)



	def spawn_blocks(self, gpu, blocks_per_SM, warps_per_block, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat):
		'''
		return a list of Blocks to run on one SM each with allocated number of warps
		'''
		block_list = []
		for i in range(blocks_per_SM):
			block_list.append(Block(gpu, i, warps_per_block, self.acc.num_warp_schedulers_per_SM, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat))
		return block_list


	def blockList_has_active_warps(self, block_list):
		'''
		return true if any block in the block list is active
		'''
		for block in block_list:
			if block.is_active(): ## block is active if it has any active warp
				return True
		return False

