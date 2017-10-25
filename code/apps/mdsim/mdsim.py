"""
 MDSim, an application simulator for an MD force call.
 
 Notes:
 rzam: September 13, 2016 - Starting MDSim
"""
# To run: python mdsim.py
# Output: To screen and to mdsim.0.out
# MPI Example: mpirun -np 3 python2.7 mdsim.py

version = "2016.09.13.1"

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
#natoms = 500000
#nneighs = 142.688
#nsteps = 500

#natoms = 256
#natoms = 4000
#natoms = 62500
#natoms = 500000
#natoms = 864000
#natoms = 1687500
#natoms = 4000000

#nneighs = 37.9665
#nneighs = 39
#nneighs = 51.4948
#nneighs = 65.31743333
#nneighs = 96.31726667
#nneighs = 142.688

#natoms = 62500
#nneighs = 35.8072

natoms = 256000
nneighs = 37.5544

#nsteps = 10
#nsteps = 500
nsteps = 100

total_ranks = 1

#simName, startTime, endTime, minDelay, useMPI = \
#  "mdsim", 0.0, 100000000000.0, 0.1, False
#simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

###############################################################################
# force_call_time_kernel
###############################################################################
def force_call_time_kernel(timestep_i, natoms, nneighs):

    """
    Determine all variables needed to construct task list.
    """
    
    # ByFl (LAMMPS - pair_lj_cut): Note: cutoff and neighbor list frequency can be important
    #
    # # Sample LAMMPS Input:
    #
    # units           lj
    # atom_style      atomic
    # atom_modify     map array
    # lattice         fcc 0.8442
    # region          box block 0 4 0 4 0 4
    # create_box      1 box
    # create_atoms    1 box
    # mass            1 1.0
    # velocity        all create 0.75 87287 loop geom
    # pair_style      lj/cut 2.5
    # pair_coeff      1 1 1.0 1.0 2.5
    # neighbor            0.3 bin
    # neigh_modify    delay 0 every 20 check no
    # fix             1 all nve
    # run            500
    #
    ## Fixed ~38 neighbors/atom (Avg)
    #float_loads = 412.37 * natoms
    #index_loads = 106.53 * natoms
    #float_alu_ops = 807.832 * natoms
    #float_div_ops = 0.0 * natoms
    #float_vector_ops = 0.0
    #int_alu_ops = 1400.318 * natoms
    #int_vector_ops = 0.0
    #num_float_vars = 16
    #num_index_vars = 15
    
    # Variable neighbor list size (variable cutoff):
    float_loads = 11.067   * (natoms*nneighs)
    index_loads = 2.7971   * (natoms*nneighs)
    float_alu_ops = 21.797 * (natoms*nneighs)
    float_div_ops = 0.0    * (natoms*nneighs)
    float_vector_ops = 0.0
    int_alu_ops = 37.14    * (natoms*nneighs)
    int_vector_ops = 0.0
    num_float_vars = 16
    num_index_vars = 15
    
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
# compute_timestep_time
###############################################################################
def compute_timestep_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: timestep_i_time -- predicted time to solve one timestep. 
    """
    
    timestep_x, natoms, nneighs = timestep_args[0], timestep_args[1], timestep_args[2]
    ops = force_call_time_kernel(timestep_x, natoms, nneighs)
  
    # Read in force-call time kernel result:
    num_index_vars = ops[0]   # number of index variables
    num_float_vars = ops[1]   # number of float variables
    index_loads = ops[2]      # number of index loads
    float_loads = ops[3]      # number of float loads
    avg_dist = 1			  # average distance-guess for now
    avg_reuse_dist = 1		  # avg number of unique loads between two consecutive
                              #	    accesses of the same element (ie use different
                              #	    weights for int and float loads)????
    stdev_reuse_dist = 1	  # stddev number of unique loads between two
                              #     consecutive accesses of the same element????
    int_alu_ops = ops[4]	  # number of integer ops
    float_alu_ops = ops[5]    # number of float ops
    float_div_ops = ops[6]    # number of float-division ops
    float_vector_ops = ops[7] # number of float-vector ops
    int_vector_ops = ops[8]   # number of integer-vector ops

    L1_hitrate = 0.91 #1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 0.93 #1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 0.95 #1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now

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
    timestep_time, stats_ts = core.time_compute(tasklist_per_chunk, True)

    # Return
    return timestep_time

###############################################################################
# md_process
###############################################################################
def md_process(mpi_comm_world):

    """
    Driver for the MD timestep progression.
    Input: mpi_comm_world -- communicator from the MPI model
    Output: nothing returned, but total simulation time printed
    """
    
    n = mpi_comm_size(mpi_comm_world)                     # total # ranks
    p = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

    # Compute time for core/process
    host = mpi_ext_host(mpi_comm_world)
    core_id = p % cores_per_host
    core = host.cores[core_id]

    timestep_i = 0
    process_compute_time = 0
    timestep_i_time = 0

    while (timestep_i < nsteps):
        ts_args = [timestep_i, natoms, nneighs]
        timestep_i_time = compute_timestep_time(core, ts_args)
        process_compute_time += timestep_i_time
        timestep_i += 1

    print "\nCore ", global_core_id, \
        " done at time ", mpi_wtime(mpi_comm_world), \
        " after executing ", timestep_i, " timesteps"
        
    # Print the results to screen
    print "\nResults for ",nsteps," MD steps:"
    print "==========================================="
    print "WCTime for last timestep (sec): ", timestep_i_time
    print "WCTime for entire process (sec): ", process_compute_time

    # Finalize mpi and the simulation
    mpi_finalize(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################
modeldict = { "model_name"    : "mdsim",
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

print "\nMDSim run with Simian PDES Engine\nVersion = ", version

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

cluster.start_mpi(hostmap, md_process)
cluster.run()

###############################################################################
###############################################################################
