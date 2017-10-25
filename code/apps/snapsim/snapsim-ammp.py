"""
 SNAPSim, an application simulator of the SNAP radiation transport miniapp.
"""

# To run: python [SNAPSim py file name]
# Output: To screen and to file snapsim-mpi.#.out

version = "snap-ammp"

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *


import math

########################
# 0. Initialization stuff

simName, startTime, endTime, minDelay, useMPI = \
  "SNAPsim_AMMP", 0.0, 1000000.0, 0.000000001, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)


# Problem input
###############################################################################

# Read Stack distances and their probabilities from a file
#Note that, these are the pre-processed SD and their probabilities
sd_psd = open(sys.argv[1],'r').read().split('\n')
sd_psd = filter(None,sd_psd)
list_sd_psd = [item.split('\t') for item in sd_psd]

task_graphlets = open(sys.argv[2],'r').read()

def SNAPsim_AMMP(mpi_comm_world, *args):
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

    total_bytes = 22575
    #simdev i/p
    tasklist_example_time_compute_amm = [['iALU', 8469], ['fALU', 89800], ['fDIV',6400], \
                ['MEM_ACCESS',stack_dist, probability_sd, block_size, total_bytes, data_bus_width]]

    tasklist_example_time_compute = [['iALU', 8469], ['fALU', 89800], ['fDIV',6400], \
                ['MEM_ACCESS', 4, 10, 1, 1, 1, 10, 80, False]]

    # MEM_ACCESS values until the boolean are irrelevant as the new model does not use them.
    tasklist_200 = [['iALU', 16201202], ['fALU', 16000001], ['fDIV', 2], \
                ['MEM_ACCESS', stack_dist, probability_sd, block_size, 611208048, data_bus_width]]

    # Task graphs of each basic block
    taskgraph = eval(task_graphlets)
    #taskgraph = {'BB0': {'V7': {'inst': 'br', 'children': []}}}
    core_id = 0
    core = host.cores[core_id]

    #time = core.time_compute2(tasklist_200, False)
    bbTimesDict = core.time_compute_taskgraph(taskgraph, tasklist_example_time_compute_amm)
    #print bbTimesDict
    bbTimesList = [bbTimesDict]
    with open('predicted_bbTimes.txt', 'w') as f:
        f.write('\n'.join(map(lambda x: str(x), bbTimesList)))
    time = 0.0
    for bb, t in bbTimesDict.iteritems():
        time += t

    print "Computed time for one time execution of each BB: ", time
    mpi_ext_sleep(time, mpi_comm_world)
    print "SNAP ",simName,": Input : ", sim, \
       " Cluster Mustang, serial run, predicted runtime (s): ", mpi_wtime(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################

modeldict = { "model_name"    : "SNAPsim_AMMP",
              "sim_time"      : 1.0*10**16,
              "use_mpi"       : False,
              "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
              "intercon_type" : "Gemini",
              "torus"         : configs.mustang_intercon,
              #"host_type"     : "GrizzlyNode",
              #"host_type"     : "I7Node",
              #"host_type"     : "MustangNode",
              "host_type"     : "GenericCoreNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
            }

cluster = Cluster(modeldict)
hostmap = range(1)
cluster.start_mpi(hostmap, SNAPsim_AMMP)
cluster.run()



###############################################################################
###############################################################################
