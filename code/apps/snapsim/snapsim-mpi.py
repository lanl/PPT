"""
 SNAPSim, an application simulator of the SNAP radiation transport miniapp.
"""

# To run: python [SNAPSim py file name]
# Output: To screen and to file snapsim-mpi.#.out

version = "mpi-2016.07.21.1"

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

#path.append('../../simian/simian-master/SimianPie')
#path.append('../../hardware')
#path.append('../../hardware/interconnect')
#path.append('../../hardware/interconnect/configs')
#path.append('../../middleware/mpi')
#path.append('../../middleware/threading')

#sys.dont_write_bytecode = True

import math

###############################################################################
###############################################################################


# Problem input
###############################################################################

iparam = {}

f = open(sys.argv[1], 'r')
for line in f:
  line2 = line.split()
  iparam.update({line2[0]:int(line2[2])})
f.close()

nx = iparam['nx']
ny = iparam['ny']
nz = iparam['nz']
ichunk = iparam['ichunk']
jchunk = iparam['jchunk']
kchunk = iparam['kchunk']
nmom = iparam['nmom']
nang = iparam['nang']
ng = iparam['ng']
nsteps = iparam['nsteps']
iitm = iparam['iitm']
oitm = iparam['oitm']
noct = iparam['noct']
ncor = iparam['ncor']

# Setup from problem input
###############################################################################

                       # assume 3D problem always for now
cmom = nmom**2         # number of flux moments

nxchunks = int(math.ceil(float(nx)/ichunk))  # number of spatial chunks in x
nychunks = int(math.ceil(float(ny)/jchunk))  # SNAP's npey
nzchunks = int(math.ceil(float(nz)/kchunk))  # SNAP's npez

total_ranks = nychunks * nzchunks            # number of MPI ranks

total_chunks_per_rank = nxchunks*noct*ng

# Helper constants (none really used right now)
###############################################################################

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

SYNC_INTERVAL = 1      # wait time for thread efficiency
PSI_SIZE_PER_CELL = 1  # data bytes per cell, determines size of MPI msg
STAT_FREQUENCY = 1     # Statistics frequency

###############################################################################


# compute_dependencies
###############################################################################

def compute_dependencies(chunk_id):
  """
  Returns the chunks required (req_list) before specified chunk_id can be
  processed and returns the chunks that immediately depend (dep_list) on
  specified chunk_id.

  Incoming: chunk_id=(group, octant, z, y, x)
  Outgoing: dep_list and req_list
  """

  (group, octant, z, y, x) = chunk_id
  idir = octant % 2

  req_list = []
  dep_list = []

  # The group stuff isn't set up correctly for threads; needs fixed

  # Set up the requirements/dependents at x==nxchunks-1, where sweep starts in
  #   negative x-direction for next group
  if x == nxchunks-1:
    if idir == 0 and group > 0:
      req_list.append((group-1, octant+1, z, y, x))
    if idir == 1 and group < ng-1:
      dep_list.append((group+1, octant-1, z, y, x))

  # Set the requirements/dependents at x==0, where sweep returns in
  #   positive x-direction for same group
  if x == 0:
    if idir == 1:
      req_list.append((group, octant-1, z, y, x))
    if idir == 0:
      dep_list.append((group, octant+1, z, y, x))

  # Set up the requirements/dependents for switching starting corners, where
  #   sweeps start anew from group 0
  if x == nxchunks-1 and (octant>0 and octant<noct-1):
    if idir == 0 and group == 0:
      req_list.append((ng-1, octant-1, z, y, x))
    if idir == 1 and group == ng-1:
      dep_list.append((0, octant+1, z, y, x))

  # Set spatial chunk requirements/dependents for a given octant and group
  # 0: -x, -y, -z
  if octant == 0:
    if x < nxchunks-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y < nychunks-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z < nzchunks-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  # 1: +x, -y, -z
  if octant == 1:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxchunks-1:
      dep_list.append((group, octant, z, y, x+1))
    if y < nychunks-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z < nzchunks-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  # 2: -x, +y, -z
  if octant == 2:
    if x < nxchunks-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nychunks-1:
      dep_list.append((group, octant, z, y+1, x))
    if z < nzchunks-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  # 3: +x, +y, -z
  if octant == 3:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxchunks-1:
      dep_list.append((group, octant, z, y, x+1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nychunks-1:
      dep_list.append((group, octant, z, y+1, x))
    if z < nzchunks-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  # 4: -x, -y, +z
  if octant == 4:
    if x < nxchunks-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y < nychunks-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzchunks-1:
      dep_list.append((group, octant, z+1, y, x))

  # 5: +x, -y, +z
  if octant == 5:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxchunks-1:
      dep_list.append((group, octant, z, y, x+1))
    if y < nychunks-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzchunks-1:
      dep_list.append((group, octant, z+1, y, x))

  # 6: -x, +y, +z
  if octant == 6:
    if x < nxchunks-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nychunks-1:
      dep_list.append((group, octant, z, y+1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzchunks-1:
      dep_list.append((group, octant, z+1, y, x))

  # 7: +x, +y, +z
  if octant == 7:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxchunks-1:
      dep_list.append((group, octant, z, y, x+1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nychunks-1:
      dep_list.append((group, octant, z, y+1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzchunks-1:
      dep_list.append((group, octant, z+1, y, x))

  # Return the requested list type
  return (req_list, dep_list)

###############################################################################


# compute_chunk_time
###############################################################################

def compute_chunk_time(core):
  """
  Put together the task list to send to hardware. Contains all the formulas for
  the flops, iops, and memory ops performed per chunk.

  Incoming: core -- core id, each core will perform this calc for one chunk
  Outgoing: chunk_time -- predicted time to solve one work chunk, which is
                          a set of cells, all angles for one octant, one group 
  """

  nijk = ichunk*jchunk*kchunk
  nij  = ichunk*jchunk
  nik  = ichunk*kchunk
  njk  = jchunk*kchunk

  num_index_vars = 10         # number of index variables
  num_float_vars = (11 + 2*cmom + (19 + cmom)*nang)*nijk

  index_loads = 15 + 7*nij + 6*nik + 1*njk + 11*nijk
  float_loads = (26*nang + 9 + 4*(cmom-1)*nang + 2*(cmom-1))*nijk

  avg_dist = 8.08             # average distance in arrays between accessed
                              #   elements in bytes; assume stride 1 word with
                              #   a little extra cost

  # avg number of unique loads between two consecutive accesses of the same
  #   element (ie use different weights for int and float loads); assume about
  #   10 variables between reuse
  if nmom == 4:
    avg_reuse_dist = nang
  else:
    avg_reuse_dist = nang/(2.0*(4-nmom))

  stdev_reuse_dist = nang     # stddev number of unique loads between two
                              #   consecutive accesses of the same element;
                              #   assume our reuse guess above is pretty good

  L1_hitrate = 1.0 - (1.0/(1.0*avg_reuse_dist))
  L2_hitrate = 1.0 - (1.0/(2.0*avg_reuse_dist))
  L3_hitrate = 1.0 - (1.0/(3.0*avg_reuse_dist))

  #int_alu_ops      = ( 89*nang + 35*(nang%2) + 205 +
  #                     (15*nang + 4*(nang%2) + 23)*(cmom-1) ) * nijk + \
  #                   15*nik + 47*njk + 13*nij + 41
  int_alu_ops      = (205 + 23*(cmom-1))*nijk + 15*nik + 47*njk + 13*nij + 41
  int_vector_ops   = (20*nang + 4*nang*(cmom-1))*nijk
  float_alu_ops    = (7*nang + 2*(cmom-1)*nang - (cmom-1) - 2)*nijk + 4
  float_vector_ops = 2*(((11 + (cmom-1))*nang + (cmom-1) + 5)*nijk)

  tasklist_per_chunk = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                         ['INTVEC', int_vector_ops, 1],
                         ['VECTOR', float_vector_ops, 1],
                         ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False] ]
  # TODO: put memory in at some point. Probably at the beginning.

  # Difference in tasklists is the Init_file set to True. In border cells, more
  #   float elements will need to be loaded from main

  # Loop over the cells, adding time for interior and border cells
  print "\ntime_compute for sweeping chunk"
  chunk_time, stats_chunk = core.time_compute(tasklist_per_chunk, True)

  # Return
  return chunk_time

###############################################################################


# compute_outer_time
###############################################################################

def compute_outer_time(core):
  """
  Put together the task list to send to hardware. Contains all the formulas for
  the flops, iops, and memory ops performed per outer source update.

  Incoming: core -- core id, each core will perform this calc its cells
  Outgoing: outer_time -- predicted time to update outer source
  """

  cells_on_core = nx*jchunk*kchunk

  int_alu_ops   = 8*cells_on_core*cmom*ng**2
  float_alu_ops = 8*cells_on_core*cmom*ng**2

  num_index_vars = 5
  num_float_vars = 2*(cells_on_core*cmom*ng + cells_on_core*cmom*ng**2)
  avg_dist = 8.0
  avg_reuse_dist = 2.0
  stdev_reuse_dist = 8.0

  L1_hitrate = 1.0 - (1.0/(1.0*avg_reuse_dist))
  L2_hitrate = 1.0 - (1.0/(2.0*avg_reuse_dist))
  L3_hitrate = 1.0 - (1.0/(3.0*avg_reuse_dist))

  index_loads = 5.0
  float_loads = 2*(cells_on_core*cmom*ng + cells_on_core*cmom*ng**2)

  cross_compute_tasklist = [ ['iALU', int_alu_ops],
                             ['fALU', float_alu_ops],
                             ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
                              num_index_vars, num_float_vars, avg_dist,
                              index_loads, float_loads, False] ]

  print "\ntime_compute for outer source"
  cross_compute_time = core.time_compute(cross_compute_tasklist)

  # Return
  return cross_compute_time

###############################################################################


# snap_process
###############################################################################

def snap_process(mpi_comm_world):
  """
  Driver for the SNAP nested loop structure. Computes the time for a work
  chunk Each core then starts the inner loop process. The cores know their own
  first work chunk, and uses that to determine what chunks it requires to be
  finished and which chunks depend on it from compute_dependencies. The process
  waits for required chunks, computes (sleeps, really) its current chunk's time,
  then passes a message to the dependent cores/chunks. Add in the time to
  compute the cross-group scatter (outer source). Once the process is laid out
  for a single inner iteration, perform a generic looping over
  the time steps-->outers-->inners, accumulating the appropriate time.

  Input: mpi_comm_world -- communicator from the MPI model
  Output: nothing returned, but total simulation time printed
  """

  n = mpi_comm_size(mpi_comm_world)                     # total # ranks
  p = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

  # Compute time to sweep a single work chunk
  timestep_i, octant, group = 0, 0, 0
  host = mpi_ext_host(mpi_comm_world)
  core_id = p % cores_per_host
  core = host.cores[core_id]

  chunk_time = compute_chunk_time(core)

  # Start main loop: Wait for required chunks to be completed, complete own
  #   chunk, send to dependents, determine next chunk on core, repeat.
  ############################

  # Initialize timing and the received chunk dictionary
  time_before_inner = mpi_wtime(mpi_comm_world)
  recvd_chunks = {}

  # Determine starting chunk for the core
  y = global_core_id % nychunks
  z = int(global_core_id / nychunks)
  x = nxchunks-1 # hard-codes current SNAP method
  num_chunks_in_core = 0
  cur_chunk_id = (group, octant, z, y, x)

  # Start the loop over core's chunks. cur_chunk_id is initialized to value
  #   above. Later reset until the core has no more work, where loop ends.
  while cur_chunk_id != None:

    #print "Core ", global_core_id, " with chunk", cur_chunk_id

    # Determine reqs and dependents
    num_chunks_in_core += 1
    (reqs, deps) = compute_dependencies(cur_chunk_id)

    #print "reqs:", reqs 
    #print "deps:", deps

    # Wait to receive from upstream neighbors. Once all reqs are received,
    #   clear the recvd_chunks dictionary.
    #   ('set' makes this unordered for comparision purposes)
    while not (set(reqs) <= set(recvd_chunks)):
      r = mpi_recv(mpi_comm_world)
      recvd_chunks[r["data"]] = "rec"
    for req_id in reqs:
      del recvd_chunks[req_id]

    # We have all requireds, so we can mimic the computation time--i.e., just
    #   sleep for the time it takes to compute over the work chunk. This model
    #   assumes the cache effects across chunks are minimal, probably a safe
    #   assumption.
    #print p, " now sleeping for chunk ", cur_chunk_id
    mpi_ext_sleep(chunk_time, mpi_comm_world)

    #print "Core ", global_core_id," executed chunk ", cur_chunk_id, \
    #      " at time ", mpi_wtime(mpi_comm_world)

    # Communicate to the dependents
    for dep_id in deps:

      # Set tuple for sending chunk and destination chunk from deps. Use
      #   receiving chunk info to determine destination rank.
      (mygrp, myoct, myz, myy, myx) = cur_chunk_id
      (group, octant, z, y, x) = dep_id

      dest_rank = z * nychunks + y

      # Set data size according to grid direction
      if dest_rank == p:
        data_size = 0.0
      else:
        if myy == y and myz != z:
          data_size = nang*ichunk*jchunk*8.0
        elif myy != y and myz == z:
          data_size = nang*ichunk*kchunk*8.0
        else:
          print "ERROR: snap_process: myy!=y and myz!=z. Should not be here."
          sys.exit(1)

      # Send message
      if dest_rank != p:
        mpi_send(dest_rank, cur_chunk_id, data_size, mpi_comm_world)

    # Add the current chunk to recvd_chunks for the next chunk, because no
    #   message gets sent for chunks on the same process
    recvd_chunks[cur_chunk_id] = "rec"

    # Determine the next cur_chunk_id for this process as the one in the deps
    #   list that is on the same process (can only be one)
    cur_chunk_id = None
    for dep_id in deps:
      (group, octant, z, y, x) = dep_id
      dest_rank = z * nychunks + y
      if dest_rank == p:
        cur_chunk_id = dep_id
        break

  # This rank is done, print the finished status
  print "\nCore ", global_core_id, \
        " done at time ", mpi_wtime(mpi_comm_world), \
        " after executing ", num_chunks_in_core, " chunks"

  if num_chunks_in_core != total_chunks_per_rank:
    print "ERROR: snap_process: incorrect number of chunks swept."
    sys.exit(1)

  # We synchronize to let all ranks finish inner loops
  mpi_barrier(mpi_comm_world)
  time_after_inner = mpi_wtime(mpi_comm_world)
  time_per_inner_loop = time_after_inner - time_before_inner

  # Compute cross-group scattering (outer source)
  cross_compute_time = compute_outer_time(core)
  mpi_ext_sleep(cross_compute_time, mpi_comm_world)
  mpi_barrier(mpi_comm_world)
  time_after_cross = mpi_wtime(mpi_comm_world)
  time_per_scatter= time_after_cross - time_after_inner

  # Compute the entire duration (only on one rank)
  if p == 0:

    time = 0.0
    for timestep_i in range(nsteps):
      for outer_i in range(oitm):
        for inner_i in range(iitm):
          time += time_per_inner_loop
        time += time_per_scatter

    # Process 0 has already advanced time by one inner and one scatter, so
    #   we deduct that from time and put mpi to sleep to simulate that time
    time -= time_per_inner_loop + time_per_scatter
    mpi_ext_sleep(time, mpi_comm_world)

    # Print the results to screen
    print "\nEnd results:"
    print "============"
    print "Total time (sec):", mpi_wtime(mpi_comm_world)
    print "Time per inner loop (sec): ", time_per_inner_loop
    print "Time for crossgroup scatter (sec): ", time_per_scatter

  # Finalize mpi and the simulation
  mpi_finalize(mpi_comm_world)

###############################################################################


# MAIN
###############################################################################

modeldict = { "model_name"    : "n01",
              "sim_time"      : 1000000,
              #"use_mpi"       : False,
              "use_mpi"       : False,
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              #"mpi_path"      : "/usr/lib/libmpich.so.12",
#              "node_type"     : "CieloNode",
#              "host_type"     : "mpihost",
#              "intercon_type" : "crossbar",
#              "crossbar"      : { "nhosts" : 50 },
#              "mpiopt"        : { "min_pktsz"     : 2048,
#                                  "max_pktsz"     : 10000,
#                                  "resend_intv"   : 0.01,
#                                  "resend_trials" : 10,
#                                  "call_time"     : 1e-6,
#                                  "max_injection" : 1e9 } }
              "intercon_type" : "Gemini",
              "torus"         : configs.cielo_intercon,
              "host_type"     : "CieloNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
            }

print "\nSNAPSim run with Simian PDES Engine\nVersion = ", version

# We use __builtin__ to create a truly global variable that the host module
#   can inherit from.
#import __builtin__
#__builtin__.nodeType = "default"
#if "node_type" in modeldict:
#  __builtin__.nodeType = modeldict["node_type"]

cluster = Cluster(modeldict)

total_hosts = cluster.intercon.num_hosts()

cores_per_host = 16
total_cores = total_hosts*cores_per_host

if total_ranks >= total_cores:
  print "ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run this \
         job (p=%d)" % total_hosts, cores_per_host, total_cores, total_ranks
  sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(total_ranks)]

cluster.start_mpi(hostmap, snap_process)
cluster.run()

###############################################################################
###############################################################################
