"""
 HPLSim, hpltimecompute.py
"""

# Set up path variables; PPT applications expect it in this fashion.
from ppt import *

# INTERNAL imports for this specific application

# EXGTERNAL imports for this specific application
import math

# Approximated latency and bandwidth:
#LAT   = 22*1.5e-6 # Darwin/Moonlight (-O0)
LAT   = 25*1.5e-6 # Moonlight (-O0)
LAT   = 5*1.5e-6 # Moonlight (-O0)
#LAT   = 1.5e-6  # MLIntelNode
BDWTH = 1.0e10   #1.0 * 10**10 # MLIntelNode   # If course_grain_pdfact - Bandwidth to Use

#GAMMA23_global = 1.605E-09 # Darwin (-O0)
#GAMMA23_global = 1.125E-10 # Moonlight (-03 MAYBE?)
#GAMMA23_global = 2.02E-09 # Moonlight (-O0)

GAMMA_3  = 2.02E-09 # Moonlight (-O0) - Level 3 BLAS (DGEMM and DTRSM Execution rate)
GAMMA_23 = 2.02E-09 # Moonlight (-O0) - Level 2-3 BLAS (Execution rate per BLAS FLOP in PDFACT)
#GAMMA_3  = 1.40E-09 # Mac (-O0) - Level 3 BLAS (DGEMM and DTRSM Execution rate)
#GAMMA_23 = 1.40E-09 # Mac (-O0) - Level 2-3 BLAS (Execution rate per BLAS FLOP in PDFACT)

###############################################################################
# pdfact_time_kernel
###############################################################################
def pdfact_time_kernel(M,N):

    X = N * N * (M - N/3.0) # Theoretical FLOPs - Pretty good match to data.
    
    float_alu_ops =     0.998  * X
    float_div_ops =     0.0
    float_vector_ops =  0.0
    int_alu_ops =       1.00 *8.6442  * X
    int_vector_ops =    0.00 *8.6442  * X
    
    float_loads =       1.503  * X / 8.0
    index_loads =       6.075  * X / 4.0
    
    num_float_vars =    3
    num_index_vars =    3
    
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
# compute_pdfact_time - For course-grained pdfact time estimate
###############################################################################
def compute_pdfact_time(core, timestep_args):

    use_gamma23 = True
    M, N, p = timestep_args[0], timestep_args[1], timestep_args[2]

    if use_gamma23:
        gam23_pdfact = GAMMA_23
        X = N * N * (M - N/3.0) # Theoretical FLOPs - Pretty good match to data.
        pdfact_time = gam23_pdfact*X
    
    else:
        ops = pdfact_time_kernel(M, N)
      
        # Read in force-call time kernel result:
        num_index_vars = ops[0]   # number of index variables
        num_float_vars = ops[1]   # number of float variables
        index_loads = ops[2]      # number of index loads   # all integer loads, ignoring logical
        float_loads = ops[3]      # number of float loads  # float plus float vector loads
        
        avg_dist = 8.08           # average distance in arrays between accessed
                                  #   elements in bytes; assume stride 1 word with
                                  #   a little extra cost
        avg_reuse_dist = 39		  # avg number of unique loads between two
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

        L1_hitrate = 0.75 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
        L2_hitrate = 0.75 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
        L3_hitrate = 0.75 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now
        
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
    
        # Compute time for Non-Communication Operations:
        pdfact_time, stats_ts = core.time_compute(tasklist_per_chunk, True)
    
    # Add in Approximate Communication time:
    pdfact_time_add = 0.0
    pdfact_swap_lat = math.log(p,2)*LAT
    pdfact_swap_bdwth = math.log(p,2)*(2*N+4)/BDWTH
    pdfact_time_add = N * (pdfact_swap_lat + pdfact_swap_bdwth)

    # Return
    return pdfact_time + pdfact_time_add

###############################################################################
# dgemm_time_kernel
###############################################################################
def dgemm_time_kernel(M, N, K):
    
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
    
    use_gamma23 = True
    M, N, K = timestep_args[0], timestep_args[1], timestep_args[2]
    
    if use_gamma23:
        gam23_dgemm = GAMMA_3
        X = 2 * M * N * K # Theoretical FLOPs - Pretty good match to data.
        dgemm_time = gam23_dgemm*X
    
    else:
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

        L1_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
        L2_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
        L3_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now

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
# dtrsm_time_kernel
###############################################################################
def dtrsm_time_kernel(M, N, Side):
    
    #X = M * N * M # Theoretical FLOPs - Assumes SIDE = 'L'
    if   Side == 'L':
        X = M * N * M # Theoretical FLOPs (Left-Sided Algorithm) - Pretty good match to data.
    elif Side == 'R':
        X = N * M * N # Theoretical FLOPs (Right-sided Algorithm) - Pretty good match to data.
    else:
        print " ERROR -- NEED Side in dtrsm_time_kernel !!!"; sys.exit(1)
    
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
    
    use_gamma23 = True
    M, N, Side = timestep_args[0], timestep_args[1], timestep_args[2]
    
    if use_gamma23:
        gam23_dtrsm = GAMMA_3
        if   Side == 'L':
            X = M * N * M # Theoretical FLOPs (Left-Sided Algorithm) - Pretty good match to data.
        elif Side == 'R':
            X = N * M * N # Theoretical FLOPs (Right-sided Algorithm) - Pretty good match to data.
        else:
            print " ERROR -- NEED Side in compute_dtrsm_time !!!"; sys.exit(1)
        dtrsm_time = gam23_dtrsm*X
    
    else:
        ops = dtrsm_time_kernel(M, N, Side)
      
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

        L1_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
        L2_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
        L3_hitrate = 1.0 #1.0 - 2.0*L1_missrate # 1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now

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
