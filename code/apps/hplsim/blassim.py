"""
 BLASSim, an application simulator for the DGEMM, DGEMV and DTRSM Blas routines.
 
 Notes:
 rzam: November 7th, 2016 - Starting combination of 3 BLAS routines into single BLASSim code
"""
version = "2016.11.07.1"

# Set up path variables; PPT applications expect it in this fashion.
import sys
from sys import path
path.append('../..')
from ppt import *

# imports for this specific application
import math

#######################
# Constant Parameters
#######################
#U_SEC = 1
#M_SEC = 1000*U_SEC
#SEC = 1000*M_SEC

#######################
# INPUT Parameters
#######################
M_in = 1024 #8192 #4096
N_in = 512 #8192
K_in = 128

total_ranks = 1

global sys_by_cache
global n_doubles

#simName, startTime, endTime, minDelay, useMPI = \
#  "mdsim", 0.0, 100000000000.0, 0.1, False
#simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

###############################################################################
# dgemm_time_kernel
###############################################################################
def dgemm_time_kernel(M, N, K):

    """
    Determine all variables needed to construct task list.
    """
    
    X = 2 * M * N * K # Theoretical FLOPs - Pretty good match to data.
    
    float_alu_ops =     1.0006  * X
    float_div_ops =     0.0
    float_vector_ops =  0.0
    int_alu_ops =       0.5 *8.5036  * X
    int_vector_ops =    0.5 *8.5036  * X
    
    float_loads =       1.5009  * X / 8.0
    index_loads =       6.0032  * X / 4.0
    
    num_float_vars =    3
    num_index_vars =    3
    
    print ""
    print " DGEMM TASK LIST: "
    print " DGEMM FLOPs       - ",float_alu_ops
    print " DGEMM IntOPs      - ",int_alu_ops
    print " DGEMM Float Loads - ",float_loads
    print " DGEMM Int Loads   - ",index_loads
    print ""
    
    ops = [num_index_vars,
           num_float_vars,
           index_loads,
           float_loads,
           int_alu_ops,
           float_alu_ops,
           float_div_ops,
           float_vector_ops,
           int_vector_ops]
         
    return ops

###############################################################################
# compute_dgemm_time
###############################################################################
def compute_dgemm_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: timestep_i_time -- predicted time to solve one timestep. 
    """
    global sys_by_cache
    global n_doubles
    
    M, N, K = timestep_args[0], timestep_args[1], timestep_args[2]
    ops = dgemm_time_kernel(M, N, K)
  
    # Read in force-call time kernel result:
    num_index_vars = ops[0]   # number of index variables
    num_float_vars = ops[1]   # number of float variables
    index_loads = ops[2]      # number of index loads   # all integer loads, ignoring logical
    float_loads = ops[3]      # number of float loads  # float plus float vector loads
    
    avg_dist = 8.08           # average distance in arrays between accessed
                              #   elements in bytes; assume stride 1 word with
                              #   a little extra cost
    avg_reuse_dist = 59 #K		  # avg number of unique loads between two
                                #   consecutive accesses of the same element
                                #   (ie use different weights for int and float
                                #   loads); assume about 10 variables between
                                #   reuse
    stdev_reuse_dist = 24	  # stddev number of unique loads between two
                                #   consecutive accesses of the same element;
                                #   assume our reuse guess above is pretty good
                                
    int_alu_ops = ops[4]	  # number of integer ops  # includes logical ops
    float_alu_ops = ops[5]    # number of float ops    # NOTE: For LU, max directly uses cost of LU solve from literature
    float_div_ops = ops[6]    # number of float-division ops
    float_vector_ops = ops[7] # number of float-vector ops
    int_vector_ops = ops[8]   # number of integer-vector ops
    
    #L1_missrate = float(sys_by_cache*n_doubles)/float(index_loads+float_loads)
    L1_missrate = float(sys_by_cache*n_doubles)/float(index_loads+float_loads)

    L1_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now
    
    print ""
    print " DGEMM float(sys_by_cache*n_doubles): ", float(sys_by_cache*n_doubles)
    print " DGEMM float(index_loads+float_loads): ", float(index_loads+float_loads)
    print " DGEMM USE L1_missrate: ", L1_missrate
    print " DGEMM USE L1_hitrate: ", L1_hitrate
    print ""

    # TASK: Floating point operations (add/multiply)'fALU'
    # TASK: Floating point divisions'fDIV'
    tasklist_per_chunk = [['iALU', int_alu_ops],
                          ['fALU', float_alu_ops],
                          ['fDIV', float_div_ops],
                          ['INTVEC', int_vector_ops, 1],
                          ['VECTOR', float_vector_ops, 1],
                          ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False]]

    # Compute time:
    dgemm_time, stats_ts = core.time_compute(tasklist_per_chunk, True)

    # Return
    return dgemm_time


###############################################################################
# dgemv_time_kernel
###############################################################################
def dgemv_time_kernel(M, N):

    """
    Determine all variables needed to construct task list.
    """
    
    X = 2 * M * N # Theoretical FLOPs - Pretty good match to data.
    
    ## IMPORTANT: Need ByFl Data -- Currently only changed X wrt dgemm_time_kernel
    
    float_alu_ops =     1.0005  * X
    float_div_ops =     0.0
    float_vector_ops =  0.0
    int_alu_ops =       0.3 * 6.5026  * X
    int_vector_ops =    0.7 * 6.5026  * X
    
    float_loads =       1.5008  * X / 8.0
    index_loads =       4.0023  * X / 4.0
    
    num_float_vars =    2
    num_index_vars =    5
    
    print ""
    print " DGEMV TASK LIST: "
    print " DGEMV FLOPs       - ",float_alu_ops
    print " DGEMV IntOPs      - ",int_alu_ops
    print " DGEMV Float Loads - ",float_loads
    print " DGEMV Int Loads   - ",index_loads
    print ""
    
    ops = [num_index_vars,
           num_float_vars,
           index_loads,
           float_loads,
           int_alu_ops,
           float_alu_ops,
           float_div_ops,
           float_vector_ops,
           int_vector_ops]
         
    return ops

###############################################################################
# compute_dgemv_time
###############################################################################
def compute_dgemv_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: timestep_i_time -- predicted time to solve one timestep. 
    """
    global sys_by_cache
    global n_doubles
    
    M, N = timestep_args[0], timestep_args[1]
    ops = dgemv_time_kernel(M, N)
  
    # Read in force-call time kernel result:
    num_index_vars = ops[0]   # number of index variables
    num_float_vars = ops[1]   # number of float variables
    index_loads = ops[2]      # number of index loads   # all integer loads, ignoring logical
    float_loads = ops[3]      # number of float loads  # float plus float vector loads
    
    avg_dist = 8.08           # average distance in arrays between accessed
                              #   elements in bytes; assume stride 1 word with
                              #   a little extra cost
    avg_reuse_dist = 55 #K		  # avg number of unique loads between two
                                #   consecutive accesses of the same element
                                #   (ie use different weights for int and float
                                #   loads); assume about 10 variables between
                                #   reuse
    stdev_reuse_dist = 28	  # stddev number of unique loads between two
                                #   consecutive accesses of the same element;
                                #   assume our reuse guess above is pretty good
                                
    int_alu_ops = ops[4]	  # number of integer ops  # includes logical ops
    float_alu_ops = ops[5]    # number of float ops    # NOTE: For LU, max directly uses cost of LU solve from literature
    float_div_ops = ops[6]    # number of float-division ops
    float_vector_ops = ops[7] # number of float-vector ops
    int_vector_ops = ops[8]   # number of integer-vector ops
    
    #L1_missrate = float(sys_by_cache*n_doubles)/float(index_loads+float_loads)
    L1_missrate = float(sys_by_cache*n_doubles)/float(index_loads+float_loads)

    L1_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now
    
    print ""
    print " DGEMV float(sys_by_cache*n_doubles): ", float(sys_by_cache*n_doubles)
    print " DGEMV float(index_loads+float_loads): ", float(index_loads+float_loads)
    print " DGEMV USE L1_missrate: ", L1_missrate
    print " DGEMV USE L1_hitrate: ", L1_hitrate
    print ""

    # TASK: Floating point operations (add/multiply)'fALU'
    # TASK: Floating point divisions'fDIV'
    tasklist_per_chunk = [['iALU', int_alu_ops],
                          ['fALU', float_alu_ops],
                          ['fDIV', float_div_ops],
                          ['INTVEC', int_vector_ops, 1],
                          ['VECTOR', float_vector_ops, 1],
                          ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False]]

    # Compute time:
    dgemv_time, stats_ts = core.time_compute(tasklist_per_chunk, True)

    # Return
    return dgemv_time

###############################################################################
# dtrsm_time_kernel
###############################################################################
def dtrsm_time_kernel(M, N):

    """
    Determine all variables needed to construct task list.
    """
    
    X = M * N * M # Theoretical FLOPs - Assumes SIDE = 'L'
    
    ## IMPORTANT: Need More ByFl Data -- Currently only based off 1 data point
    
    float_alu_ops =     1.0005  * X
    float_div_ops =     0.0
    float_vector_ops =  0.0
    int_alu_ops =       0.45 *11.006  * X
    int_vector_ops =    0.55 *11.006  * X
    
    float_loads =       1.5007  * X / 8.0
    index_loads =       7.5047  * X / 4.0
    
    num_float_vars =    3
    num_index_vars =    3
    
    print ""
    print " DTRSM TASK LIST: "
    print " DTRSM FLOPs       - ",float_alu_ops
    print " DTRSM IntOPs      - ",int_alu_ops
    print " DTRSM Float Loads - ",float_loads
    print " DTRSM Int Loads   - ",index_loads
    print ""
    
    ops = [num_index_vars,
           num_float_vars,
           index_loads,
           float_loads,
           int_alu_ops,
           float_alu_ops,
           float_div_ops,
           float_vector_ops,
           int_vector_ops]
         
    return ops

###############################################################################
# compute_dtrsm_time
###############################################################################
def compute_dtrsm_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: timestep_i_time -- predicted time to solve one timestep. 
    """
    global sys_by_cache
    global n_doubles
    
    M, N = timestep_args[0], timestep_args[1]
    ops = dtrsm_time_kernel(M, N)
  
    # Read in force-call time kernel result:
    num_index_vars = ops[0]   # number of index variables
    num_float_vars = ops[1]   # number of float variables
    index_loads = ops[2]      # number of index loads   # all integer loads, ignoring logical
    float_loads = ops[3]      # number of float loads  # float plus float vector loads
    
    avg_dist = 8.08           # average distance in arrays between accessed
                              #   elements in bytes; assume stride 1 word with
                              #   a little extra cost
    avg_reuse_dist = 39 #K		  # avg number of unique loads between two
                                #   consecutive accesses of the same element
                                #   (ie use different weights for int and float
                                #   loads); assume about 10 variables between
                                #   reuse
    stdev_reuse_dist = 16	  # stddev number of unique loads between two
                                #   consecutive accesses of the same element;
                                #   assume our reuse guess above is pretty good
                                
    int_alu_ops = ops[4]	  # number of integer ops  # includes logical ops
    float_alu_ops = ops[5]    # number of float ops    # NOTE: For LU, max directly uses cost of LU solve from literature
    float_div_ops = ops[6]    # number of float-division ops
    float_vector_ops = ops[7] # number of float-vector ops
    int_vector_ops = ops[8]   # number of integer-vector ops
    
    #L1_missrate = float(sys_by_cache*n_doubles)/float(index_loads+float_loads)
    L1_missrate = float(sys_by_cache*n_doubles)/float(index_loads+float_loads)

    L1_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now
    
    print ""
    print " DTRSM float(sys_by_cache*n_doubles): ", float(sys_by_cache*n_doubles)
    print " DTRSM float(index_loads+float_loads): ", float(index_loads+float_loads)
    print " DTRSM USE L1_missrate: ", L1_missrate
    print " DTRSM USE L1_hitrate: ", L1_hitrate
    print ""

    # TASK: Floating point operations (add/multiply)'fALU'
    # TASK: Floating point divisions'fDIV'
    tasklist_per_chunk = [['iALU', int_alu_ops],
                          ['fALU', float_alu_ops],
                          ['fDIV', float_div_ops],
                          ['INTVEC', int_vector_ops, 1],
                          ['VECTOR', float_vector_ops, 1],
                          ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False]]

    # Compute time:
    dtrsm_time, stats_ts = core.time_compute(tasklist_per_chunk, True)

    # Return
    return dtrsm_time

###############################################################################
# blas_process
###############################################################################
def blas_process(mpi_comm_world):

    """
    Driver for the blas routine calls.
    Input: mpi_comm_world -- communicator from the MPI model
    Output: nothing returned, but total simulation time printed
    """
    global sys_by_cache
    global n_doubles
    
    n = mpi_comm_size(mpi_comm_world)                     # total # ranks
    p = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

    # Compute time for core/process
    host = mpi_ext_host(mpi_comm_world)
    core_id = p % cores_per_host
    core = host.cores[core_id]
    
    # INPUT Dimensions
    M = M_in
    N = N_in
    K = K_in
    
    # How big is the system relative to cache?
    l1size = core.cache_sizes[0]
    n_doubles = l1size/(64.0/core.kb)
    system_size = (2*M*K + K*N + M*N)
    sys_by_cache = system_size/n_doubles
    print " ~~~~~ "
    print " l1size       = ",l1size
    print " n_doubles    = ",n_doubles
    print " system_size  = ",system_size
    print " sys_by_cache = ",sys_by_cache
    print " ~~~~~ "
    
    ######################################
    # Get DGEMM time for system size M,N,K
    ts_args = [M,N,K]
    process_compute_time_dgemm = compute_dgemm_time(core, ts_args)
    
    ######################################
    # Get DGEMV time for system size M,N
    ts_args = [M,N]
    process_compute_time_dgemv = compute_dgemv_time(core, ts_args)
    
    ######################################
    # Get DTRSM time for system size M,N
    ts_args = [M,N]
    process_compute_time_dtrsm = compute_dtrsm_time(core, ts_args)
        
    ######################################
    # Print the results to screen
    print ""
    print "\nResults for DGEMM: (with M,N,K = ",M,N,K,")"
    print "==========================================="
    print "WCTime for entire process (sec): ", process_compute_time_dgemm
    print ""
    print "\nResults for DGEMV: (with M,N = ",M,N,")"
    print "==========================================="
    print "WCTime for entire process (sec): ", process_compute_time_dgemv
    print ""
    print "\nResults for DTRSM: (with M,N = ",M,N,")"
    print "==========================================="
    print "WCTime for entire process (sec): ", process_compute_time_dtrsm

    print "\nCore ", global_core_id, \
        " done at time ", mpi_wtime(mpi_comm_world)
    
    # Finalize mpi and the simulation
    mpi_finalize(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################
modeldict = { "model_name"    : "blassim",
              "sim_time"      : 1000000,
              "use_mpi"       : False,
              "mpi_path"      : "/projects/opt/centos7/mpich/3.1.3_gcc-4.8.5/lib/libmpi.so.12.0.4",
              "intercon_type" : "Fattree",
              "fattree"       : configs.moonlight_intercon,
              "host_type"     : "MLIntelNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.infiniband_mpiopt,
              "debug_options" : set(["none"]),
            }

print "\nBLASSim run with Simian PDES Engine\nVersion = ", version

cluster = Cluster(modeldict)
total_hosts = cluster.intercon.num_hosts()
cores_per_host = 24
total_cores = total_hosts*cores_per_host

if total_ranks >= total_cores:
  print "ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run this \
         job (p=%d)" % total_hosts, cores_per_host, total_cores, total_ranks
  sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(total_ranks)]

cluster.start_mpi(hostmap, blas_process)
cluster.run()

###############################################################################
###############################################################################
