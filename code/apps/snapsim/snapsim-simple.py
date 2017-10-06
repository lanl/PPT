"""
 SNAPSim-simple, an application simulator of the SNAP radiation transport miniapp.
 
 January 20, 2016: start of SNAPSim-simple, which copies the SNAP structure much
 more directly and also uses the MPI module
 
 
 Date: February 26, 2016
"""
# To run: python [SNAPSIM py file name]
# Output: To screen and to snapsim.0.out

version = "2016.01.20.1"

# May need to set environment variable PYTHONPATH
# Add to the path .../[Simian], ..../[Simian]/SimianPie
# Need greenlet and may need to add a path
# setenv PYTHONPATH /Users/rzerr/working/ldrd/Simian-v1.4.6:/Users/rzerr/working/ldrd/Simian-v1.4.6/SimianPie:.

# MPI Example: mpirun -np 3 python2.7 SNAPOSSimV7.py 

# Set up path  variables; PPT applications expect it in this fashion.
import sys
from sys import path
path.append('../../simian/simian-master/SimianPie')
path.append('../../hardware')
path.append('../../hardware/interconnect')
path.append('../../middleware/threading')

sys.dont_write_bytecode = True

#import simian
from simian import Simian 
import clusters
import nodes

# imports for this specific application
import math

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

#######################
# Parameters
#######################

nsteps = 5	 #10   # number of time-steps to be simulated
nmom = 4	 #4	   # anisotropic scattering order
nx = 2	   # discretization units (aka cells) in x-dimension
ny = 2		   # discretization units (aka cells) in y-dimension
nz = 2	   # discretization units (aka cells) in z-dimension
ichunk = 1		   # number of cells per chunk in x direction
jchunk = 1		   # number of cells per chunk in y direction
kchunk = 1		   # number of cells per chunk in z direction
ng = 2			   # number of energy groups
nang = 24		   # number of discrete ordinates per octant
iitm = 2		   # number of inner iterations before convergence
oitm = 2		   # number of outer iterations before convergence

# assume 3D problem always . . . Stephan asks: what exactly do these do?
noct = 8		   # number of octants
ncor = 4		   # number of starting corners
cmom = nmom**2	   # number of flux moments

# Interconnect: very simple model here for testing before connecting to MPI model
interconnect_delay = 3*U_SEC

############################################
# Helper consts
SYNC_INTERVAL = 1		# wait time for thread efficiency
PSI_SIZE_PER_CELL = 1	# Number of data bytes per cell, determines size of MPI msg
STAT_FREQUENCY = 1		# Statistics frequency
# number of spatial chunks in each dimension
# note in SNAP nychunks-->npey and nzchunks--npez
nxchunks = int(math.ceil(float(nx)/ichunk))
nychunks = int(math.ceil(float(ny)/jchunk))
nzchunks = int(math.ceil(float(nz)/kchunk))

#assignment = {}	  # assignment[chunk_id] = (node_id, core_id) # global const
########################
# 0. Simian Initialization

simName, startTime, endTime, minDelay, useMPI = \
  "snapsim", 0.0, 100000000000.0, 0.1, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

###############################################################################
###############################################################################
# SNAPSim master process
###############################################################################

def snapsim_master_process(this):
	"""
	This process runs on the single master node and organizes the SNAP compute node
	processes in two phases: inner iteration and cross-group scattering.
	It computes the complete predicted runtime as a loop of those two functions.
	It is created by the simulation main.
	"""
	entity = this.entity
	#################
	# 1. Inner loop 
	#####################
	time_before_inner = simianEngine.now
	total_cores_required = nychunks * nzchunks # TODO: Check with Joe
	cum_core_count = 0
	for node_id in xrange(cluster.num_nodes):
		cum_core_count += cluster.cores[node_id]
		if total_cores_required <= cum_core_count:
			break
	nodes_required = node_id+1
	print "Master: begin inner with CN required: ",nodes_required
	# compute on how many compute nodes we need to start the SNAP_CN_handler
	
	timestep_i, octant, group = 0, 0, 0 
	msg = ["start inner", timestep_i, octant, group]	
	for node_id in xrange(nodes_required):
		print "create start inner", node_id
		entity.reqService(
				minDelay,			# add min-delay
				"SNAP_CN_Handler",  # Handler/Service name
				msg,				# this is the data we send
				"Node",			   	# Target entity type
				node_id				# Target entity id
				)
	
	response_count = 0
	while response_count < nodes_required:
		print "Master hibernate ", response_count 
		this.hibernate()
		print "Master woke up ", response_count
		while entity.msg_q != []:
		  msg = entity.msg_q.pop()
		  if msg[0] == 'inner_compute_done':
			response_count += 1

	time_after_inner = simianEngine.now
	time_per_inner_loop = time_after_inner - time_before_inner
		
	#################
	# 2. Cross-group Scatter Computation
	#####################
	time_before_scatter = simianEngine.now
	print "Master: begin scatter"
	for node_id in xrange(nodes_required):
		entity.reqService(minDelay, "SNAP_CN_Handler",
						  ['cross_compute'],
						  "Node", node_id)
	
	response_count = 0
	while response_count < nodes_required:
		this.hibernate()
		while entity.msg_q != []:
		  msg = entity.msg_q.pop()
		  if msg[0] == 'cross_compute_done':
			response_count += 1
		
	time_after_scatter = simianEngine.now
	time_per_scatter = time_after_scatter - time_before_scatter
		
	#################
	# 3. Entire SNAP running time
	#####################
	time = 0
	for timestep_i in range(nsteps):
		for outer_i in range(oitm):
			for inner_i in range(iitm):
				time += time_per_inner_loop
			time += time_per_scatter
	# could we have said?: 
	#	time = (time_per_inner_loop * iitm + time_per_scatter) * oitm * nsteps 
	# yes, but it would have been less intuitive. 
	# In any case, we need to deduct what we have already advanced in time
	time -= time_per_inner_loop + time_per_scatter
	this.sleep(time)
	print "Simian_master done at simulated time: ", simianEngine.now


###############################################################################
###############################################################################
# assignment function
###############################################################################

def assignment(chunk_id):
	"""
	Returns (node_id, local_core_id) of where the chunk is supposed to be executed on
	"""
	(group, octant, z, y, x) = chunk_id
	global_core_id = z * nychunks + y	
							# this is limiting to what SNAP always does, should allow
							# for oversubscription of cores 
	lo_core_id = 0
	hi_core_id = 0
	for node_id in xrange(cluster.num_nodes):
		lo_core_id = hi_core_id 
		hi_core_id += cores[node_id]
		if hi_core_id > global_core_id:
			local_core_id = global_core_id - lo_core_id	
			break 
	return (node_id, local_core_id)
		
###############################################################################
###############################################################################
# compute_dependencies
###############################################################################

def compute_dependencies(chunk_id):
	"""
	Returns the chunks required before specified chunk_id can be processed
	and returns the chunks that immediately depend on specified chunk_id.

	Incoming: chunk_id=(group, octant, z, y, x)
			 requiredFlag = true/false
			 nxc, nyc, nzc = number of chunks in x, y, and z, respectively
	Outgoing: dep_list and req_list
	"""
	nzc, nyc, nxc	 = nzchunks, nychunks, nxchunks #legacy names
	(group, octant, z, y, x) = chunk_id
	idir = octant % 2 # based on original definition of octant = 2*cor_i + id_i
	dep_list = []
	req_list = []
	#
	#return (req_list, dep_list)
	#
	# The group stuff isn't set up correctly for threads; needs fixed
	# Set the requirements/dependents at x==0, where sweep returns in
	# positive x-direction for same group
	if x == 0:
		if idir == 1:
			req_list.append((group, octant-1, z, y, x))
		if idir == 0:
			dep_list.append((group, octant+1, z, y, x))
	#
	# Set up the requirements/dependents at x==nxc-1, where sweep starts in
	# negative x-direction for next group
	if x == nxc-1:
		if idir == 0 and group > 0:
			req_list.append((group-1, octant+1, z, y, x))
		if idir == 1 and group < ng-1:
			dep_list.append((group+1, octant-1, z, y, x))
	#
	# Set up the requirements/dependents for switching starting corners, where
	# sweeps start anew from group 0
	if x == nxc-1 and (octant>0 and octant<noct-1):
		if idir == 0 and group == 0:
			req_list.append((ng-1, octant-1, z, y, x))
		if idir == 1 and group == ng-1:
			dep_list.append((0, octant+1, z, y, x))
	#
	# Set spatial chunk requirements/dependents for a given octant and group
	if octant == 0:
		if x < nxc-1:
			req_list.append((group, octant, z, y, x+1))
		if x > 0:
			dep_list.append((group, octant, z, y, x-1))
		if y < nyc-1:
			req_list.append((group, octant, z, y+1, x))
		if y > 0:
			dep_list.append((group, octant, z, y-1, x))
		if z < nzc-1:
			req_list.append((group, octant, z+1, y, x))
		if z > 0:
			dep_list.append((group, octant, z-1, y, x))
	#
	if octant == 1:
		if x > 0:
			req_list.append((group, octant, z, y, x-1))
		if x < nxc-1:
			dep_list.append((group, octant, z, y, x+1))
		if y < nyc-1:
			req_list.append((group, octant, z, y+1, x))
		if y > 0:
			dep_list.append((group, octant, z, y-1, x))
		if z < nzc-1:
			req_list.append((group, octant, z+1, y, x))
		if z > 0:
			dep_list.append((group, octant, z-1, y, x))
	#
	if octant == 2:
		if x < nxc-1:
			req_list.append((group, octant, z, y, x+1))
		if x > 0:
			dep_list.append((group, octant, z, y, x-1))
		if y > 0:
			req_list.append((group, octant, z, y-1, x))
		if y < nyc-1:
			dep_list.append((group, octant, z, y+1, x))
		if z < nzc-1:
			req_list.append((group, octant, z+1, y, x))
		if z > 0:
			dep_list.append((group, octant, z-1, y, x))
	#
	if octant == 3:
		if x > 0:
			req_list.append((group, octant, z, y, x-1))
		if x < nxc-1:
			dep_list.append((group, octant, z, y, x+1))
		if y > 0:
			req_list.append((group, octant, z, y-1, x))
		if y < nyc-1:
			dep_list.append((group, octant, z, y+1, x))
		if z < nzc-1:
			req_list.append((group, octant, z+1, y, x))
		if z > 0:
			dep_list.append((group, octant, z-1, y, x))
	#
	if octant == 4:
		if x < nxc-1:
			req_list.append((group, octant, z, y, x+1))
		if x > 0:
			dep_list.append((group, octant, z, y, x-1))
		if y < nyc-1:
			req_list.append((group, octant, z, y+1, x))
		if y > 0:
			dep_list.append((group, octant, z, y-1, x))
		if z > 0:
			req_list.append((group, octant, z-1, y, x))
		if z < nzc-1:
			dep_list.append((group, octant, z+1, y, x))
	#
	if octant == 5:
		if x > 0:
			req_list.append((group, octant, z, y, x-1))
		if x < nxc-1:
			dep_list.append((group, octant, z, y, x+1))
		if y < nyc-1:
			req_list.append((group, octant, z, y+1, x))
		if y > 0:
			dep_list.append((group, octant, z, y-1, x))
		if z > 0:
			req_list.append((group, octant, z-1, y, x))
		if z < nzc-1:
			dep_list.append((group, octant, z+1, y, x))
	#
	if octant == 6:
		if x < nxc-1:
			req_list.append((group, octant, z, y, x+1))
		if x > 0:
			dep_list.append((group, octant, z, y, x-1))
		if y > 0:
			req_list.append((group, octant, z, y-1, x))
		if y < nyc-1:
			dep_list.append((group, octant, z, y+1, x))
		if z > 0:
			req_list.append((group, octant, z-1, y, x))
		if z < nzc-1:
			dep_list.append((group, octant, z+1, y, x))
	#
	if octant == 7:
		if x > 0:
			req_list.append((group, octant, z, y, x-1))
		if x < nxc-1:
			dep_list.append((group, octant, z, y, x+1))
		if y > 0:
			req_list.append((group, octant, z, y-1, x))
		if y < nyc-1:
			dep_list.append((group, octant, z, y+1, x))
		if z > 0:
			req_list.append((group, octant, z-1, y, x))
		if z < nzc-1:
			dep_list.append((group, octant, z+1, y, x))
	#
	# Return the requested list type
	return (req_list, dep_list)


###############################################################################
###############################################################################
# SNAP Process
###############################################################################

def SNAP_Process(this, proc_info, *args):
	# Again args is not used, but Simian attempts to send it more arguments
	# SNAP_Process receives a list of chunk ids to be executed and dependencies
	# and neighboring rank ids
	"""
	Individual process for an update of a chunks of cells
	"""
	# 0. Initializations
	# 
	#proc_infos =  [timestep_i, octant_i, group_i],proc_id
	timestep_i, octant, group = proc_info[0], proc_info[1], proc_info[2]
	core_id, global_core_id = args[0], args[1]
	node = this.entity
	entity = this.entity
	core = node.cores[core_id]

	# 1. Advance a small waiting time so other synchronized threads get on board
	#	   and don't screw up the thread_efficiency
	#this.sleep(core.time_compute([['CPU', SYNC_INTERVAL]]))

	# 2.Put together task list to send to hardware 
	num_index_vars = 10			   # number of index variables
								   # decent enough guess for now (12/23 rjz)
	# Per cell basis
	num_float_vars = 11 + 2*cmom + (19 + cmom)*nang
	# Per cell basis
	index_loads = 55
	# Per cell basis
	float_loads = 26*nang + 9 + 4*(cmom-1)*nang + 2*(cmom-1)
	avg_dist = 1			# average distance in arrays between accessed elements 
	avg_reuse_dist = 1		# avg number of unique loads between two consecutive
							#	 accesses of the same element (ie use different
							#	 weights for int and float loads)
	stdev_reuse_dist = 1	# stddev number of unique loads between two
							#   consecutive accesses of the same element
	int_alu_ops = 89*nang + 35*(nang%2) + 321 + \
				(15*nang + 4*(nang%2) + 23)*(cmom-1) # Per cell basis
	float_alu_ops = 7*nang + 2*(cmom-1)*nang - (cmom-1) + 2 # Per cell basis
	float_vector_ops = (21 + 2*(cmom-1))*nang + 2*(cmom-1) + 10 # Per cell basis

	tasklist_per_interior_cell = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
								 ['VECTOR', float_vector_ops, nang],
								 ['MEM_ACCESS', num_index_vars, num_float_vars,
								  avg_dist, avg_reuse_dist, stdev_reuse_dist,
								  index_loads, float_loads, False] ]
	# ['alloc', 1024]\ TODO: put memory in at some point. Probably at the very
	#	  very beginning
	tasklist_per_border_cell = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
								['VECTOR', float_vector_ops, nang],
								['MEM_ACCESS', num_index_vars, num_float_vars,
								avg_dist, avg_reuse_dist, stdev_reuse_dist,
								index_loads, float_loads, True] ]
	# Difference in tasklists is the Init_file set to True. In border cells, more
	#	  float elements will need to be loaded from main

	# 3. Loop over cells in chunk to compute the time that is spent for the
	#	   entire chunk, then perform a sleep (premature optimization ...)
	time = 0
	time_per_border_cell, stats_border = \
		core.time_compute(tasklist_per_border_cell, True)
	time_per_interior_cell, stats_interior = \
		core.time_compute(tasklist_per_interior_cell, True)

	num_border = 0
	num_interior = 0
	for z in range(kchunk):
		for y in range(jchunk):
			for x in range(ichunk):
				if (z==0) or (y==0) or (x==0):
					num_border += 1
					time += time_per_border_cell
				else:
					time += time_per_interior_cell
					num_interior += 1
	# TODO: note this is wrong for thread-efficiency, will deal with this later
	chunk_time = time
	print "Chunk time : ", chunk_time

	#############
	# 4. Start main loop: 
	# wait for requireds, compute chunk, send to dependents, determine next chunk, repeat
	
	y = global_core_id % nychunks
	z = int(global_core_id / nychunks)
	x = nxchunks-1 # hard-codes current SNAP method
	num_chunks_in_core = 0
	cur_chunk_id = (group, octant, z, y, x)

	while cur_chunk_id <> None:
		print entity.engine.now, "Core ", global_core_id," with chunk", cur_chunk_id
		num_chunks_in_core += 1
		(reqs, deps) = compute_dependencies(cur_chunk_id)
		#print "reqs:", reqs 
		#print "deps:", deps
		while not (set(reqs) <= set(entity.msg_q[core_id])): 
			# set makes this unordered for comparison purposes
			print entity.engine.now, "Core ", global_core_id," into hibernate", entity.msg_q[core_id]
			this.hibernate() # this is like blocking mpi_recv
			print entity.engine.now, "Core ", global_core_id," woken up ", entity.msg_q[core_id]
		for req_id in reqs:
			#print entity.msg_q[core_id]
			del entity.msg_q[core_id][req_id]
		# We have all requireds, so we can mimic the computation time
		# has to be atomic, ie cant be interrupted by hibernate wake-ups
		target_after_sleep_time = entity.engine.now + chunk_time
		this.sleep(chunk_time)
		while target_after_sleep_time > entity.engine.now:
			this.hibernate() # because we did schedule a wake up at the right time
		print entity.engine.now, "Core ", global_core_id," executed chunk ", cur_chunk_id, " with deps, reqs ", deps, reqs
		# Now we communicate this to dependents: mpi_sends 
		for dep_id in deps:
			if (node.num, core_id) <> assignment(dep_id):
				(dest_node_id, dest_proc_id) = assignment(dep_id)
				node.reqService(
					interconnect_delay, "SNAP_CN_Handler",
					["chunk received", cur_chunk_id, dest_proc_id],
					"Node", dest_node_id)
		# We find the next chunk id for this rank process
		
		entity.msg_q[core_id][cur_chunk_id] = "rec"
		cur_chunk_id = None
		for dep_id in deps:
			if (node.num, core_id)== assignment(dep_id): # this is our next chunk
				cur_chunk_id = dep_id
				break
	# This rank is done, we tell the CN Handler
	print "Core ", global_core_id," finds no more chunks to execute at time ", simianEngine.now
	node.reqService(
				0, "SNAP_CN_Handler",
				["inner rank done", core_id, num_chunks_in_core])

###############################################################################
###############################################################################
# SNAP_CN_Handler
###############################################################################

def SNAP_CN_Handler(self, msg, *args):
	# 
	# Creates SNAP_Rank processes
	#

	# Type 1: Start a process for inner SNAP loop
	if msg[0] == 'start inner':
		timestep_i, octant_i, group_i = msg[1], msg[2], msg[3] 
		proc_infos =  [timestep_i, octant_i, group_i]
		num_cores = len(self.cores)	 
		self.msg_q = {} # create message q
		self.inner_results = {} # keep data per core
		self.total_chunks, self.max_chunks = 0, 0
		tot_cores = 0
		
		for node_id in range(self.num):
			tot_cores += cores[node_id]
		self.req_local_cores = min(nychunks*nzchunks - tot_cores, num_cores)
		# this math is required to get the correct number for the last used CN
		for core_id in xrange(self.req_local_cores):
			global_core_id= tot_cores + core_id
			print "SNAPCNHandler creates SNAPProcess ", self.num, core_id
			self.msg_q[core_id] = {}
			self.createProcess("snap_proc"+str(core_id), SNAP_Process)
			self.startProcess("snap_proc"+str(core_id), proc_infos, \
				core_id, global_core_id)

	# Type 2: a chunk is done process finished, put into q for process
	elif msg[0] == 'chunk received':
		chunk_id , proc_id = msg[1], msg[2]
		print self.engine.now, ":A CN_Handler with chunk_id, proc_id", msg
		core = self.cores[proc_id % self.num_cores]
		self.msg_q[proc_id][chunk_id]= "rec"
		self.wakeProcess("snap_proc"+str(proc_id))
	
	# Type 3: inner loop done by a core
	elif msg[0] == 'inner rank done':
		print "Inner rank done msg:", msg
		proc_id, num_chunks = msg[1], msg[2]
		self.inner_results[proc_id] = num_chunks
		self.total_chunks += num_chunks
		self.max_chunks = max(self.max_chunks, num_chunks)
		if len(self.inner_results) == self.req_local_cores: # all are done
				self.reqService(minDelay, "Master_Handler",
						['inner_compute_done', self.num, self.total_chunks], "Master", 0) 
	
	# Type 4: Cross compute phase
	elif msg[0] == 'cross_compute':
		maxtime = 0.0
		for core_id in self.inner_results:
			num_elements = 5 # this is a magic number
			core = self.cores[core_id]
			cells_on_core = self.inner_results[core_id]*ichunk*jchunk*kchunk
			cross_compute_tasklist = \
				[['fALU', cells_on_core*num_elements*ng**2]]
			(time, stats) = core.time_compute(cross_compute_tasklist, True)
			maxtime = max(maxtime, time)
		time = max(time, minDelay)
		# Thus only slowest core determines the elapse time, we report to master
		#	with delay time
		self.reqService(minDelay, "Master_Handler",
						['cross_compute_done', self.num], "Master", 0)


###############################################################################
###############################################################################
# MasterNode
###############################################################################

class MasterNode(simianEngine.Entity):

	def __init__(self, baseInfo, *args):
		super(MasterNode, self).__init__(baseInfo)
		self.msg_q = []
		self.createProcess("snapsim_master_process", snapsim_master_process)
		self.startProcess("snapsim_master_process")	#no arguments

	def Master_Handler(self, msg, *args):
		# args is artificial
		# Put message into q and wake the master process up again
		self.msg_q.insert(0, msg)
		# print simianEngine.now, ": inserted msg", msg
		self.wakeProcess("snapsim_master_process")


###############################################################################
###############################################################################


# "MAIN"
###############################################################################

# 1. Choose and instantiate the Cluster that we want to simulate 

print "\nSNAPSim run with Simian PDES Engine\nVersion =", version
#cluster = clusters.MiniTrinity(simianEngine)
cluster = clusters.SingleCielo(simianEngine)

# determine number of cores on each node and the total number of cores
cores = cluster.cores # Is a dictionary {node_id: number of cores}
total_cores = 0
for node_id in cores:
	total_cores += cores[node_id]

# print the computational resources used
print "\nNodes =", cluster.num_nodes, "\nCores =", total_cores, "\n" \
	"Cores per node =", cores, "\n\n", \
	"Begin SNAPSim loops\n--------------------------------"

# 2. Create a Snap Handler Service on each node
simianEngine.attachService(nodes.Node, "SNAP_CN_Handler" , SNAP_CN_Handler)

# 3. Create Master Node
# This in turn creates and starts the main process	snapsim_master_process
simianEngine.addEntity("Master", MasterNode, 0)

# 4. Run simx
simianEngine.run()
simianEngine.exit()

