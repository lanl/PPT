# navie_mm_sim.py -- Tests the new implementation of time_compute2 function

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

########################
# 0. Simian Initialization

simName, startTime, endTime, minDelay, useMPI = \
  "Laplace2DSim", 0.0, 100000000000.0, 0.1, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

# Read Stack distances and their probabilities from a file
#Note that, these are the pre-processed SD and their probabilities
sd_psd = open(sys.argv[1],'r').read().split('\n')
sd_psd = filter(None,sd_psd)
list_sd_psd = [item.split('\t') for item in sd_psd]


def Laplace2DSim(mpi_comm_world, *args):
    sim = sys.argv[1]
    host = mpi_ext_host(mpi_comm_world)

    #Column 0 of the sublists of list_sd_psd is SD
    stack_dist = zip(*list_sd_psd)[0]
    stack_dist = map(float,stack_dist) # Convert list strings into floats
    #Column 1 of the sublists of list_sd_psd is prob(SD)
    probability_sd = zip(*list_sd_psd)[2]
    probability_sd = map(float,probability_sd) #Convert the list of strings into floats

    in_x = 1024.0   # One dimesion of the NxN mesh
    block_size = 8.0 # in bytes (Data bus width -- 64-bit)
    data_bus_width = 4.0 #in Bytes
    #total_bytes = 192000 * in_x**2 - 687960 * in_x + 684236.0 # O0
    total_bytes = 32000 * in_x**2 - 111984 * in_x + 96000.0 #O3
    fALU = 5000.0 * in_x**2 - 20000.0 * in_x + 20000.0
    iALU = 8000.0 * in_x**2 - 25999.0 * in_x + 23000.0
    fDIV = 0

    tasklist_64 = [['iALU', iALU], ['fALU', fALU], ['fDIV', fDIV], \
                ['MEM_ACCESS', stack_dist, probability_sd, block_size, \
                total_bytes, data_bus_width]]

    tasklist_128 = [['iALU', 65479], ['fALU', 732200], ['fDIV',51200], \
                ['MEM_ACCESS', stack_dist, probability_sd, block_size, \
                total_bytes, data_bus_width]]

    tasklist_256 = [['iALU', 65479], ['fALU', 732200], ['fDIV',51200], \
                ['MEM_ACCESS', stack_dist, probability_sd, block_size, \
                total_bytes, data_bus_width]]

    tasklist_512 = [['iALU', 65479], ['fALU', 732200], ['fDIV',51200], \
                ['MEM_ACCESS', stack_dist, probability_sd, block_size, \
                total_bytes, data_bus_width]]
    #total_bytes = 373864232 #Total mem_size in bytes
    #simsmall i/p
    #tasklist = [['iALU', 2085189], ['fALU', 23593800], ['fDIV',1638400], ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

    core_id = 0
    core = host.cores[core_id]

    time = core.time_compute_amm(tasklist_64, False)

    print "Computed time is ", time
    mpi_ext_sleep(time, mpi_comm_world)
    print "Benchmark app ",simName,": Input : ", sim, \
       " Cluster Mustang, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "laplace2DSim",
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


cluster = Cluster(modeldict)
hostmap = range(1)
cluster.start_mpi(hostmap, Laplace2DSim)
cluster.run()
