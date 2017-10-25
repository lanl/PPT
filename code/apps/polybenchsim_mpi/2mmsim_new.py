"""
 PolyBenchSim: 2mm
"""
#Modified on 23 Aug 2016 by Nandu

version = "2mmsim-mpi-2016.08.23.1"

# Set up path  variables; PPT applications expect it in this fashion.
#from sys import path
#path.append('../..')
#from ppt import *

#path.append('../../simian/simian-master/SimianPie')
#path.append('../../hardware')

#from simian import Simian
#import clusters
#import nodes

#from simple_cielo_config import simple_cielo_intercon
#from simple_gemini_mpiopt import simple_gemini_mpiopt

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

from simple_cielo_config import simple_cielo_intercon
from simple_gemini_mpiopt import simple_gemini_mpiopt

# Global Parameters
N_X = 64 # Size of first input dimension
N_Y = 1 # Size of first input dimension
N_Z = 1 # Size of first input dimension

########################
# 0. Initialization stuff

###############################################################################
#estimated_data_size (eds in bytes)	eds/L1_cache	eds/L2_cache	eds/L3_cache	i) L1_cache_miss	ii) L2_cache_misses	iii) L2_cache_access	iv) L1_cache_access	v) branches_taken	vi) branches missed	vii). L1_cache_line_size	vii). L2_cache_line_size	viii). L1_cache_size	ix). L2_cache_size	x). L1_cm_rate	xi). L2_cm_rate	Runtime (secs)

def TwommSim(mpi_comm_world, *args):
  """
  Simulates 2mm from the PolyBench Suite
  This is the function that gets executes as main process. Only one such
    function exists.
  TODO: Generalize this to simulate all PolyBench apps by making the name a parameter, 
  same for cluster name. Read all values in dicts of dicts from file "fit" and irf
  """
  #### Initialization #####
  arg = args[0]
  core_id, n_x, n_y, n_z =  arg[0], arg[1], arg[2], arg[3]

  host = mpi_ext_host(mpi_comm_world)
  core = host.cores[core_id]

  t = [['iALU', 46.0 * n_x**3],['fALU', 5.0 * n_x**3], 
	['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 300, True],
	['VECTOR', 6, 0]]
  
  #### Compute time and advance time
  time  =  core.time_compute(t, False)
  print "Computed time is ", time
  mpi_ext_sleep(time, mpi_comm_world)
  print "PolyBench app 2mmsim: Input n_x, n_y, n_z: ", n_x, n_y, n_z, \
      " Cluster Cielo, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "2mmsim",
              "sim_time"      : 1.0*10**16,
              "use_mpi"       : False,
              #"use_mpi"       : True,
              #"mpi_path"      : "/projects/opt/centos7/mpich/3.1.3_gcc-4.8.5/lib/libmpi.so.12.0.4",
              "mpi_path"      : "/Users/nsanthi/Work/Lua/PDES/Lua/x86-64-mpich/lib/libmpich.dylib",
              "intercon_type" : "Gemini",
              "torus"         : simple_cielo_intercon,
              "host_type"     : "CieloNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : simple_gemini_mpiopt,
            }

print "\nSNAPSim run with Simian PDES Engine\nVersion = ", version

cluster = Cluster(modeldict)

total_hosts = cluster.intercon.num_hosts()

cores_per_host = 16
total_cores = total_hosts*cores_per_host

total_ranks = 1

if total_ranks >= total_cores:
  print "ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run this \
         job (p=%d)" % total_hosts, cores_per_host, total_cores, total_ranks
  sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(total_ranks)]

args = [0, N_X, N_Y, N_Z] # Node id 0
cluster.start_mpi(hostmap, TwommSim, args)
cluster.run()
