# navie_mm_sim.py -- Tests the new implementation of time_compute2 function

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

########################
# 0. Simian Initialization

simName, startTime, endTime, minDelay, useMPI = \
  "imcsim", 0.0, 100000000000.0, 0.1, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)
                      
def NaivemmSim(mpi_comm_world, *args):

    n_x = 256
    
    host = mpi_ext_host(mpi_comm_world)

#     stack_dist = [115, 22, 23, 144, 215, 216] # stack distances
#     probability_sd = [0.1, 0.5, 0.1, 0.2, 0.05, 0.05] # probability distributions
    stack_dist = [2*n_x]
    probability_sd = [1.0]

    block_size = n_x*8 # in bytes (can be a distribution, but for us ONLY AVERAGE MATTERS)
    total_bytes = 2*(n_x ** 2) * 8
    data_bus_width = 4 #in Bytes
    
    tasklist = [['iALU', 46.0 * n_x**3], ['fALU', 5.0 * n_x**3],
                ['VECTOR', 6, 0],
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, True, stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

#     tasklist = [['iALU', 46.0 * n_x**3], ['fALU', 5.0 * n_x**3], 
#        ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 300, True],
#        ['VECTOR', 6, 0]]
    
    core_id = 0
    core = host.cores[core_id]
    
    time = core.time_compute2(tasklist, False)
    
    print "Computed time is ", time
    mpi_ext_sleep(time, mpi_comm_world)
    print "Naivemm_tests app naivemmsim: Input n_x : ", n_x, \
       " Cluster Mustang, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "naivemmsim",
              "sim_time"      : 1.0*10**16,
              "use_mpi"       : False,
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              "intercon_type" : "Gemini",
              "torus"         : configs.mustang_intercon,
              "host_type"     : "MustangNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
            }

model_dict = { "model_name"    : "naivemmsimCielo",
               "sim_time"      : 1.0*10**16,
               "use_mpi"       : False,
               "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
               "intercon_type" : "Gemini",
               "torus"         : configs.cielo_intercon,
               "mpiopt"        : configs.gemini_mpiopt,
               "host_type"     : "CieloNode",
               "load_libraries": set(["mpi"]),
             }

cluster = Cluster(modeldict)
hostmap = range(1)
cluster.start_mpi(hostmap, NaivemmSim)
cluster.run()