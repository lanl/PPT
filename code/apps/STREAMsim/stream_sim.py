# navie_mm_sim.py -- Tests the new implementation of time_compute2 function

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

########################
# 0. Simian Initialization

simName, startTime, endTime, minDelay, useMPI = \
  "StreamSim", 0.0, 100000000000.0, 0.1, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

# Read Stack distances and their probabilities from a file
#Note that, these are the pre-processed SD and their probabilities
sd_psd = open(sys.argv[1],'r').read().split('\n')
sd_psd = filter(None,sd_psd)
list_sd_psd = [item.split('\t') for item in sd_psd]


def StreamSim(mpi_comm_world, *args):
    '''
    Agner Fog for the ALU instruction values
    '''
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

    #example i/p
    total_bytes = 22575
    tasklist_example = [['iALU', 8469], ['fALU', 89800], ['fDIV',6400], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, total_bytes, data_bus_width]]
    # MEM_ACCESS values until the boolean are irrelevant as the new model does not use them.

    tasklist_10K = [['iALU', 820062], ['fALU', 400000], ['fDIV', 0], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 27921120, data_bus_width]]

    tasklist_20K = [['iALU', 1640062], ['fALU', 800000], ['fDIV', 0], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 55841120, data_bus_width]]

    tasklist_30K = [['iALU', 2460062], ['fALU', 1200000], ['fDIV',0], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 83761120, data_bus_width]]

    tasklist_40K = [['iALU', 3280062], ['fALU', 1600000], ['fDIV', 0], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, \
                 probability_sd, block_size, 111681120, data_bus_width]]
    #total_bytes = 373864232 #Total mem_size in bytes
    #simsmall i/p
    #tasklist = [['iALU', 2085189], ['fALU', 23593800], ['fDIV',1638400], ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

    core_id = 0
    core = host.cores[core_id]

    time = core.time_compute(tasklist_40K, False)

    print "Computed time is ", time
    mpi_ext_sleep(time, mpi_comm_world)
    print "parsec 3.0 app ",simName,": Input : ", sim, \
       " Cluster Mustang, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "StreamSim",
              "sim_time"      : 1.0*10**16,
              "use_mpi"       : False,
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              "intercon_type" : "Gemini",
              "torus"         : configs.mustang_intercon,
              #"host_type"     : "GrizzlyNode",
              "host_type"     : "I7Node",
              #"host_type"     : "MustangNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
            }

model_dict = { "model_name"    : "StreamsimCielo",
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
cluster.start_mpi(hostmap, StreamSim)
cluster.run()
