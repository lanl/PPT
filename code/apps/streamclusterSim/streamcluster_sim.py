# navie_mm_sim.py -- Tests the new implementation of time_compute2 function

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

########################
# 0. Simian Initialization

simName, startTime, endTime, minDelay, useMPI = \
  "streamclustersim", 0.0, 100000000000.0, 0.1, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

# Read Stack distances and their probabilities from a file
#Note that, these are the pre-processed SD and their probabilities
sd_psd = open(sys.argv[1],'r').read().split('\n')
sd_psd = filter(None,sd_psd)
list_sd_psd = [item.split(',') for item in sd_psd]

                      
def StreamclusterSim(mpi_comm_world, *args):
    sim = sys.argv[1]
    host = mpi_ext_host(mpi_comm_world)

#     stack_dist = [115, 22, 23, 144, 215, 216] # stack distances
#     probability_sd = [0.1, 0.5, 0.1, 0.2, 0.05, 0.05] # probability distributions
    #stack_dist = [d[0] for d in  list_sd_psd]
    
    #Column 0 of the sublists of list_sd_psd is SD
    stack_dist = zip(*list_sd_psd)[0]
    stack_dist = map(float,stack_dist) # Convert list strings into floats
    #Column 1 of the sublists of list_sd_psd is prob(SD)
    probability_sd = zip(*list_sd_psd)[1]
    probability_sd = map(float,probability_sd) #Convert the list of strings into floats

    block_size = 8.0 # in bytes (can be a distribution, but for us ONLY AVERAGE MATTERS)
    
    data_bus_width = 4.0 #in Bytes
    
    #total_bytes = 6916753 # simdev
    #simdev i/p
    #tasklist = [['iALU', 193165], ['fALU', 94732], ['fDIV',1269], ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

    total_bytes = 16801040484 #simsmall
    #simsmall i/p
    tasklist = [['iALU', 388404173], ['fALU', 701933366], ['fDIV',179005], ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False, stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

#     tasklist = [['iALU', 46.0 * n_x**3], ['fALU', 5.0 * n_x**3], 
#        ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 300, True],
#        ['VECTOR', 6, 0]]
    
    core_id = 0
    core = host.cores[core_id]
    
    time = core.time_compute(tasklist, False)
    
    print "Computed time is ", time
    mpi_ext_sleep(time, mpi_comm_world)
    print "parsec 3.0 app ",simName,": Input : ", sim, \
       " Cluster Mustang, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "streamclustersim",
              "sim_time"      : 1.0*10**16,
              "use_mpi"       : False,
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              "intercon_type" : "Gemini",
              "torus"         : configs.mustang_intercon,
              "host_type"     : "MustangNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
            }

cluster = Cluster(modeldict)
hostmap = range(1)
cluster.start_mpi(hostmap, StreamclusterSim)
cluster.run()