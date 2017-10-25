#!/usr/bin/python2.7
"""
 CLOVERLEAFSim, an application simulator of the CLOVERLEAF radiation transport miniapp.
"""

# To run: python [CLOVERLEAFSim py file name]
# Output: To screen and to file CloverLeafsim-mpi.#.out

version = "mpi-2016.07.21.1"

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
path.append('../../simian/simian-master/SimianPie')
path.append('../../hardware')
path.append('../../hardware/interconnect')
path.append('../../hardware/interconnect/configs')
path.append('../../middleware/mpi')
path.append('../../middleware/threading')
from ppt import *
import configs

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
  if(len(line2) >= 3):
    iparam.update({line2[0]:int(line2[2])})
f.close()

nx = iparam['nx']
ny = iparam['ny']
nsteps = iparam['nsteps']
nproc = iparam['nproc']

# Setup from problem input
###############################################################################

mesh_ratio = nx/ny
chunk_x = 1
chunk_y = 1
split_found = False
for c in range(1,nproc):
  if(nproc%c == 0):
    factor_x = int(nproc/c)
    factor_y = c
  if(float(factor_x)/float(factor_y) <= mesh_ratio):
    chunk_y = c
    chunk_x = nproc/c
    split_found = True
    break

if(chunk_y == nproc or not split_found):
  if(mesh_ratio >= 1.0):
    chunk_x = nproc
    chunk_y = 1
  else:
    chunk_x = 1
    chunk_y = nproc

nchunk = chunk_x*chunk_y
chunk_size_avg = nx*ny/nchunk

delta_x = int(nx/chunk_x)
delta_y = int(ny/chunk_y)
mod_x = nx%chunk_x
mod_y = ny%chunk_y

chunk_size = [0]*nproc
chunk_lx   = [0]*nproc
chunk_ly   = [0]*nproc
for i in range(chunk_x):
  for j in range(chunk_y):
    idx = j*chunk_x + i
    if (i < mod_x):
      chunk_lx[idx] = delta_x + 1
    else:
      chunk_lx[idx] = delta_x
    if (j < mod_y):
      chunk_ly[idx] = delta_y + 1
    else:
      chunk_ly[idx] = delta_y

    chunk_size[idx] = chunk_lx[idx]*chunk_ly[idx]

north_neighbor_rank = [0]*nproc
south_neighbor_rank = [0]*nproc
east_neighbor_rank  = [0]*nproc
west_neighbor_rank  = [0]*nproc

on_northern_boundary = [False]*nproc
on_southern_boundary = [False]*nproc
on_eastern_boundary  = [False]*nproc
on_western_boundary  = [False]*nproc

for p in range(nproc):
  px = p%chunk_x
  py = (p-px)/chunk_x

  if(px > 0):
    west_neighbor_rank[p] = py*chunk_x + (px-1)
  else:
    west_neighbor_rank[p]  = -1
    on_western_boundary[p] = True
  if(px < chunk_x-1):
    east_neighbor_rank[p]  = py*chunk_x + (px+1)
  else:
    east_neighbor_rank[p]  = -1
    on_eastern_boundary[p] = True
  
  if(py > 0):
    south_neighbor_rank[p]  = (py-1)*chunk_x + px
  else:
    south_neighbor_rank[p]  = -1
    on_southern_boundary[p] = True
  if(py < chunk_y-1):
    north_neighbor_rank[p]  = (py+1)*chunk_x + px
  else:
    north_neighbor_rank[p]  = -1
    on_northern_boundary[p] = True

print("Mesh ratio of", mesh_ratio)
print("Decomposing mesh into", chunk_x, "by", chunk_y, "chunks")

###############################################################################



# compute_chunk_time
###############################################################################

def compute_chunk_time(core,rank):
  """
  Put together the task list to send to hardware. Contains all the formulas for
  the flops, iops, and memory ops performed per chunk.
  """

  maxPtsPerCore = 1024**3
  ptsPerCore = nx*ny
  assert (ptsPerCore < maxPtsPerCore), 'Paging not yet supported. Reduce per processor problem size'
  
  avg_dist = 8.08             # average distance in arrays between accessed
                              #   elements in bytes; assume stride 1 word with
                              #   a little extra cost

  # avg number of unique loads between two consecutive accesses of the same
  #   element (ie use different weights for int and float loads); assume about
  #   10 variables between reuse
  avg_reuse_dist = maxPtsPerCore/ptsPerCore

  L1_hitrate = 1.0 - (1.0/(1.0*avg_reuse_dist))
  L2_hitrate = 1.0 - (1.0/(2.0*avg_reuse_dist))
  L3_hitrate = 1.0 - (1.0/(3.0*avg_reuse_dist))

  total_ops = 200538*nx*ny  # obtained from regression against byfl data
  total_ops_per_chunk = total_ops*chunk_size[rank]/(nx*ny)

  integer_ops = int(0.645822642814*total_ops_per_chunk)
  flops       = int(0.212651963359*total_ops_per_chunk)
  memory_ops  = int(0.114721354566*total_ops_per_chunk)
  vector_ops  = int(0.0956505246563*total_ops_per_chunk)

  fraction_vector_ops = vector_ops/total_ops

  int_alu_ops    = int(integer_ops*(1.0 - fraction_vector_ops))
  int_vector_ops = int(integer_ops*fraction_vector_ops)
  float_alu_ops    = int(flops*(1.0 - fraction_vector_ops))
  float_alu_ops    = int(flops*(1.0 - fraction_vector_ops))
  float_vector_ops = int(flops*fraction_vector_ops)
  index_loads      = memory_ops*int_alu_ops/(int_alu_ops + float_alu_ops)
  float_loads      = memory_ops*float_alu_ops/(int_alu_ops + float_alu_ops)
  float_divisions  = flops*0.08

  num_index_vars = 10
  num_float_vars = 10

  tasklist_per_chunk = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                         ['fDIV', float_divisions],
                         ['INTVEC', int_vector_ops, 1],
                         ['VECTOR', float_vector_ops, 1],
                         ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False] ]
  # TODO: put memory in at some point. Probably at the beginning.

  # Difference in tasklists is the Init_file set to True. In border cells, more
  #   float elements will need to be loaded from main

  print ("\ntime_compute for updating chunk")
  chunk_time, stats_chunk = core.time_compute(tasklist_per_chunk, True)

  # Return
  return chunk_time

###############################################################################

# compute_pack_time(nfields,halo_depth,direction,core,rank):
###############################################################################

def compute_pack_time(nfields,halo_depth,direction,core,rank):
  """
  Put together the task list to send to hardware. Contains all the formulas for
  the flops, iops, and memory ops performed per chunk.
  """

  avg_dist = 8.08

  L1_hitrate = 1.0
  L2_hitrate = 1.0
  L3_hitrate = 1.0

  if(direction == "north"):
    if(on_northern_boundary[rank]):
      total_pack_data = 0
    else:
      total_pack_data = halo_depth*nfields*chunk_lx[rank]
  elif(direction == "south"):
    if(on_southern_boundary[rank]):
      total_pack_data = 0
    else:
      total_pack_data = halo_depth*nfields*chunk_lx[rank]
  elif(direction == "east"):
    if(on_eastern_boundary[rank]):
      total_pack_data = 0
    else:
      total_pack_data = halo_depth*nfields*chunk_ly[rank]
  elif(direction == "west"):
    if(on_western_boundary[rank]):
      total_pack_data = 0
    else:
      total_pack_data = halo_depth*nfields*chunk_ly[rank]
  else:
    total_pack_data = 0

  total_pack_ops = total_pack_data*400

  integer_ops = int(0.6*total_pack_ops)
  flops       = int(0.3*total_pack_ops)
  memory_ops  = int(0.1*total_pack_ops)

  int_alu_ops      = integer_ops
  int_vector_ops   = 0.0
  float_alu_ops    = flops
  float_vector_ops = 0.0
  if(int_alu_ops + float_alu_ops > 0):
    index_loads      = memory_ops*int_alu_ops/(int_alu_ops + float_alu_ops)
    float_loads      = memory_ops*float_alu_ops/(int_alu_ops + float_alu_ops)
  else:
    index_loads = 0.5*memory_ops
    float_loads = 0.5*memory_ops
  float_divisions  = 0.0

  num_index_vars = 10
  num_float_vars = 10

  tasklist_per_pack  = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                         ['fDIV', float_divisions],
                         ['INTVEC', int_vector_ops, 1],
                         ['VECTOR', float_vector_ops, 1],
                         ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False] ]
  # TODO: put memory in at some point. Probably at the beginning.

  # Difference in tasklists is the Init_file set to True. In border cells, more
  #   float elements will need to be loaded from main

  print ("\ntime_compute for updating pack for rank %i" % rank)
  pack_time, stats_pack = core.time_compute(tasklist_per_pack, True)

  # Return
  return pack_time

###############################################################################

# compute_bdry_update_time(nfields,halo_depth,direction,core,rank):
###############################################################################

def compute_bdry_update_time(nfields,halo_depth,direction,core,rank):
  """
  Put together the task list to send to hardware. Contains all the formulas for
  the flops, iops, and memory ops performed per chunk.
  """

  avg_dist = 8.08

  L1_hitrate = 1.0
  L2_hitrate = 1.0
  L3_hitrate = 1.0

  if(direction == "north"):
    if(on_northern_boundary[rank]):
      total_bdry_data = halo_depth*nfields*chunk_lx[rank]
    else:
      total_bdry_data = 0
  elif(direction == "south"):
    if(on_southern_boundary[rank]):
      total_bdry_data = halo_depth*nfields*chunk_lx[rank]
    else:
      total_bdry_data = 0
  elif(direction == "east"):
    if(on_eastern_boundary[rank]):
      total_bdry_data = halo_depth*nfields*chunk_lx[rank]
    else:
      total_bdry_data = 0
  elif(direction == "west"):
    if(on_western_boundary[rank]):
      total_bdry_data = halo_depth*nfields*chunk_ly[rank]
    else:
      total_bdry_data = 0
  else:
    total_bdry_data = 0

  total_bdry_ops = total_bdry_data*137

  integer_ops = int(0.6*total_bdry_ops)
  flops       = int(0.3*total_bdry_ops)
  memory_ops  = int(0.1*total_bdry_ops)

  int_alu_ops      = integer_ops
  int_vector_ops   = 0.0
  float_alu_ops    = flops
  float_vector_ops = 0.0
  if(int_alu_ops + float_alu_ops > 0):
    index_loads      = memory_ops*int_alu_ops/(int_alu_ops + float_alu_ops)
    float_loads      = memory_ops*float_alu_ops/(int_alu_ops + float_alu_ops)
  else:
    index_loads = 0.5*memory_ops
    float_loads = 0.5*memory_ops
  float_divisions  = 0.0

  num_index_vars = 10
  num_float_vars = 10

  tasklist_per_bdry  = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                         ['fDIV', float_divisions],
                         ['INTVEC', int_vector_ops, 1],
                         ['VECTOR', float_vector_ops, 1],
                         ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False] ]

  # Difference in tasklists is the Init_file set to True. In border cells, more
  #   float elements will need to be loaded from main

  print ("\ntime_compute for updating bdry for rank %i" % rank)
  bdry_time, stats_bdry = core.time_compute(tasklist_per_bdry, True)

  # Return
  return bdry_time

###############################################################################

# Halo update
###############################################################################
def CloverLeaf_update_halo(nfields,halo_depth,mpi_comm_world):
  """
  Runs an update of the specified number of fields to a specified halo depth
  Returns the estimated time of the halo update call
  """
  n = mpi_comm_size(mpi_comm_world)                     # total # ranks
  p = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

  host = mpi_ext_host(mpi_comm_world)
  core_id = p % cores_per_host
  core = host.cores[core_id]

  # Communicate halo
  ############################

  # initialize timing
  time_before_halo_update = mpi_wtime(mpi_comm_world)

  # compute the data packing/unpacking times
  north_packing_time_per_field = compute_pack_time(1,halo_depth,"north", core, p)
  south_packing_time_per_field = compute_pack_time(1,halo_depth,"south", core, p)
  east_packing_time_per_field  = compute_pack_time(1,halo_depth,"east", core, p)
  west_packing_time_per_field  = compute_pack_time(1,halo_depth,"west", core, p)

  north_unpacking_time_per_field = north_packing_time_per_field
  south_unpacking_time_per_field = south_packing_time_per_field
  east_unpacking_time_per_field  = east_packing_time_per_field
  west_unpacking_time_per_field  = west_packing_time_per_field

  # compute the boundary update times
  north_bdry_update_time_per_field = compute_bdry_update_time(1,halo_depth,"north", core, p)
  south_bdry_update_time_per_field = compute_bdry_update_time(1,halo_depth,"south", core, p)
  east_bdry_update_time_per_field  = compute_bdry_update_time(1,halo_depth,"east", core, p)
  west_bdry_update_time_per_field  = compute_bdry_update_time(1,halo_depth,"west", core, p)

  for i in range(nfields):
    ##
    # East and West
    ##
    size_of_double = 8
    packing_rate = 6.2e-9
    boundary_rate = 1.5e-9
    halo_bredth = chunk_ly[p]
    sent_data_size = halo_bredth*halo_depth*size_of_double

    # pack buffer to send to the east and the west
    mpi_ext_sleep(west_packing_time_per_field,mpi_comm_world)
    mpi_ext_sleep(east_packing_time_per_field,mpi_comm_world)

    # send/receive data to chunk to/from the east and the west
    requests = []
    if not on_western_boundary[p]:
      mpi_isend(west_neighbor_rank[p],None,sent_data_size,mpi_comm_world)
      req = mpi_irecv(mpi_comm_world,from_rank=west_neighbor_rank[p])
      requests.append(req)
    if not on_eastern_boundary[p]:
      mpi_isend(east_neighbor_rank[p],None,sent_data_size,mpi_comm_world)
      req = mpi_irecv(mpi_comm_world,from_rank=east_neighbor_rank[p])
      requests.append(req)
    mpi_waitall(requests)

    # unpack buffer sent from the east and the west
    mpi_ext_sleep(west_unpacking_time_per_field,mpi_comm_world)
    mpi_ext_sleep(east_unpacking_time_per_field,mpi_comm_world)

    ##
    # North and South
    ##
    halo_bredth = chunk_lx[p]
    sent_data_size = halo_bredth*halo_depth*size_of_double

    # pack buffer to send to the north and south
    mpi_ext_sleep(north_packing_time_per_field,mpi_comm_world)
    mpi_ext_sleep(south_packing_time_per_field,mpi_comm_world)

    # send/receive data to chunk to the north and the south
    requests = []
    if not on_northern_boundary[p]:
      mpi_isend(north_neighbor_rank[p],None,sent_data_size,mpi_comm_world)
      req = mpi_irecv(mpi_comm_world,from_rank=north_neighbor_rank[p])
      requests.append(req)
    if not on_southern_boundary[p]:
      mpi_isend(south_neighbor_rank[p],None,sent_data_size,mpi_comm_world)
      req = mpi_irecv(mpi_comm_world,from_rank=south_neighbor_rank[p])
      requests.append(req)
    mpi_waitall(requests)
    mpi_barrier(mpi_comm_world)

    # unpack buffer to send to the east and the west
    mpi_ext_sleep(north_unpacking_time_per_field,mpi_comm_world)
    mpi_ext_sleep(south_unpacking_time_per_field,mpi_comm_world)

  # This rank is done, print the finished status
  print ("\nCore ", global_core_id, \
        " done at time ", mpi_wtime(mpi_comm_world))

  # update boundary fields
  for i in range(nfields):
    mpi_ext_sleep(north_bdry_update_time_per_field,mpi_comm_world)
    mpi_ext_sleep(south_bdry_update_time_per_field,mpi_comm_world)
    mpi_ext_sleep(east_bdry_update_time_per_field,mpi_comm_world)
    mpi_ext_sleep(west_bdry_update_time_per_field,mpi_comm_world)

  # We synchronize to let all ranks finish inner loops
  mpi_barrier(mpi_comm_world)
  time_after_halo_update = mpi_wtime(mpi_comm_world)
  halo_update_time = time_after_halo_update - time_before_halo_update

  return halo_update_time

###############################################################################

# CloverLeaf_process
###############################################################################

def CloverLeaf_process(mpi_comm_world):
  """
  Driver for the CLOVERLEAF nested loop structure. Computes the time for a work
  chunk for core then calls a halo update.

  Input: mpi_comm_world -- communicator from the MPI model
  Output: nothing returned, but total simulation time printed
  """

  n = mpi_comm_size(mpi_comm_world)                     # total # ranks
  p = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

  host = mpi_ext_host(mpi_comm_world)
  core_id = p % cores_per_host
  core = host.cores[core_id]

  # Compute time to run
  # timestep
  # PdV(predict=true)
  # accelerate
  # PdV(predict=false)
  # flux_calc
  # advection
  chunk_time = compute_chunk_time(core, p)
  # the hardware estimate is WAAY off: rescale
#  chunk_time /= 239.541666050442027700/5.0
  chunk_time /= 19

  # Run calls to update halo
  ############################
  time_per_halo_update = 0.0

  # calls from timestep
  time_per_halo_update += CloverLeaf_update_halo(5,1,mpi_comm_world)
  time_per_halo_update += CloverLeaf_update_halo(1,1,mpi_comm_world)

  # calls from pdv with predict=true
  time_per_halo_update += CloverLeaf_update_halo(1,1,mpi_comm_world)

  # calls from advection
  time_per_halo_update += CloverLeaf_update_halo(4,2,mpi_comm_world)
  time_per_halo_update += CloverLeaf_update_halo(6,2,mpi_comm_world)
  time_per_halo_update += CloverLeaf_update_halo(6,2,mpi_comm_world)

  # calculation of halo update time is off by a large factor
  # adjust  by mulitplying by a large factor to compensate
  #time_per_halo_update *= 127

  # Compute the entire duration (only on one rank)
  if p == 0:
    time = (chunk_time + time_per_halo_update)*nsteps

    # Process 0 has already advanced time by one chunk update and halo update, so
    # we deduct that from time and put mpi to sleep to simulate that time
    time -= chunk_time + time_per_halo_update
    mpi_ext_sleep(time, mpi_comm_world)

    # Print the results to screen
    print ("\nEnd results:")
    print ("============")
    print ("Total time (sec): %1.16e" % (mpi_wtime(mpi_comm_world)))
    print ("Time per chunk update (sec): %1.16e" % chunk_time)
    print ("Time per cell update (sec): %1.16e" % (chunk_time/chunk_size_avg))
    print ("Halo time (sec): %1.16e" % (time_per_halo_update*nsteps))

  # Finalize mpi and the simulation
  mpi_finalize(mpi_comm_world)

###############################################################################


# MAIN
###############################################################################

modeldict = { "model_name"    : "n01",
              "sim_time"      : 1000000,
              #"use_mpi"       : True,
              "use_mpi"       : False,
              #"mpi_path"      : "/projects/opt/centos7/mpich/3.1.3_gcc-4.8.5/lib/libmpi.so.12.0.4",
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              #"mpi_path"      : "/usrsdfsoaf/lib64/libmpich.so.10.0.4",
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
              "intercon_type" : "Fattree",
              "fattree"         : configs.mustang_intercon,
              "host_type"     : "MustangNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.infiniband_mpiopt,
            }

print ("\nCloverSim run with Simian PDES Engine\nVersion = ", version)

# We use __builtin__ to create a truly global variable that the host module
#   can inherit from.
#import __builtin__
#__builtin__.nodeType = "default"
#if "node_type" in modeldict:
#  __builtin__.nodeType = modeldict["node_type"]

cluster = Cluster(modeldict)

total_hosts = cluster.intercon.num_hosts()

cores_per_host = 24
total_cores = total_hosts*cores_per_host

if nproc >= total_cores:
  print ("ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run this \
         job (p=%d)" % total_hosts, cores_per_host, total_cores, nproc)
  sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(nproc)]

cluster.start_mpi(hostmap, CloverLeaf_process)
cluster.run()

###############################################################################
###############################################################################
