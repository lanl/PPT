# navie_mm_sim.py -- Tests the new implementation of time_compute2 function

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

########################
# 0. Simian Initialization

simName, startTime, endTime, minDelay, useMPI = \
  "blackscholesSim", 0.0, 100000000000.0, 0.1, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

# Read Stack distances and their probabilities from a file
#Note that, these are the pre-processed SD and their probabilities
sd_psd = open(sys.argv[1],'r').read().split('\n')
sd_psd = filter(None,sd_psd)
list_sd_psd = [item.split('\t') for item in sd_psd]


def BlackscholesSim(mpi_comm_world, *args):
    sim = sys.argv[1]
    host = mpi_ext_host(mpi_comm_world)

#     stack_dist = [115, 22, 23, 144, 215, 216] # stack distances
#     probability_sd = [0.1, 0.5, 0.1, 0.2, 0.05, 0.05] # probability distributions
    #stack_dist = [d[0] for d in  list_sd_psd]

    #Column 0 of the sublists of list_sd_psd is SD
    stack_dist = zip(*list_sd_psd)[0]
    stack_dist = map(float,stack_dist) # Convert list strings into floats
    #Column 1 of the sublists of list_sd_psd is prob(SD)
    probability_sd = zip(*list_sd_psd)[2]
    probability_sd = map(float,probability_sd) #Convert the list of strings into floats

    block_size = 8.0 # in bytes (Data bus width -- 64-bit)

    data_bus_width = 4.0 #in Bytes

    total_bytes = 10
    tasklist_example = [['iALU', 8469], ['fALU', 89800], ['fDIV',6400], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, total_bytes, data_bus_width]]
    # MEM_ACCESS values until the boolean are irrelevant as the new model does not use them.

    tasklist_16 = [['iALU', 8471], ['fALU', 89800], ['fDIV',6400], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 1444792, data_bus_width]]

    tasklist_32 = [['iALU', 16615], ['fALU', 180200], ['fDIV',12800], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 2891080, data_bus_width]]

    tasklist_64 = [['iALU', 32903], ['fALU', 365000], ['fDIV',25600], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 5815656, data_bus_width]]

    tasklist_128 = [['iALU', 65479], ['fALU', 732200], ['fDIV',51200], \
                ['MEM_ACCESS', stack_dist, probability_sd, block_size, 11645608, data_bus_width]]
    #total_bytes = 373864232 #Total mem_size in bytes
    #simsmall i/p
    #tasklist = [['iALU', 2085189], ['fALU', 23593800], ['fDIV',1638400], ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

    core_id = 0
    core = host.cores[core_id]

    time = core.time_compute_amm(tasklist_128, False)

    print "Computed time is ", time
    mpi_ext_sleep(time, mpi_comm_world)
    print "parsec 3.0 app ",simName,": Input : ", sim, \
       " Cluster Mustang, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "blackscholesSim",
              "sim_time"      : 1.0*10**16,
              "use_mpi"       : False,
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              "intercon_type" : "Gemini",
              #"torus"         : configs.grizzly_intercon,
              "torus"         : configs.mustang_intercon,
              "host_type"     : "GrizzlyNode",
              #"host_type"     : "I7Node",
              #"host_type"     : "MustangNode",
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
cluster.start_mpi(hostmap, BlackscholesSim)
cluster.run()
