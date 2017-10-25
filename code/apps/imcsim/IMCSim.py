"""
 IMCsim, an IMC non-parallel application simulator for Marshak wave.
 
 August 9, 2016: first iteration of IMCSim, which simulates a single process and thread MC. No communications are involved.
 
 Date: August, 2016
"""
# To run: python [IMCSIM py file name]
# Output: To screen and to snapsim.0.out

version = "2016.08.24.1"

# May need to set environment variable PYTHONPATH
# Add to the path .../[Simian], ..../[Simian]/SimianPie
# Need greenlet and may need to add a path
# setenv PYTHONPATH /Users/rzerr/working/ldrd/Simian-v1.4.6:/Users/rzerr/working/ldrd/Simian-v1.4.6/SimianPie:.

# MPI Example: mpirun -np 3 python2.7 SNAPOSSimV7.py 

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *
import time


# imports for this specific application
import math
#import kernel functions containing the operation counts
from imc_fun import *

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

#######################
# Parameters
#######################
total_ranks=1      # this is sequential so far

n_x_cells = 40     # number of elements 1D array
n_photons = 0  # number of photons in the system set it on cluster run
#Do multiple runs with diff number of photons for testing
photon_list = 180000
t_start = 0.0      # time start
t_stop = 0.0001       # time finish
dt_mult = 1.0      # time step size multiplyer
dt = 0.0001       # dt-delta timestep for next timestep

runtime_list = []

########################

###############################################################################
# compute_timestep_time
###############################################################################

def compute_timestep_time(core, timestep_args):
        """
        Put together the task list to send to hardware. Contains all the formulas for
        the flops, iops, and memory ops performed per chunk.

        Incoming: core -- core id, each core will perform same number of timesteps.
        Outgoing: timestep_i_time -- predicted time to solve one timestep. 
        """
        #timestep_args = [timestep_i, n_photons, n_x_cells]
        timestep_x, n_photons, n_x_cells = timestep_args[0], timestep_args[1], timestep_args[2]
        #get the operation counts for the entire timestep
        ops = time_step_kernel(n_photons, timestep_x, n_x_cells)

        # number of index variables 
        num_index_vars = ops[0]        
        # Per timestep basis
        num_float_vars = ops[1]
        # Per timestep basis
        index_loads = ops[2]
        # Per timestep basis
        float_loads = ops[3]
        avg_dist = 1      # average distance-guess for now 
        avg_reuse_dist = 1    # avg number of unique loads between two consecutive
                                                        #  accesses of the same element (ie use different
                                                        #  weights for int and float loads)????
        stdev_reuse_dist = 1  # stddev number of unique loads between two
                                                        #   consecutive accesses of the same element????
        int_alu_ops = ops[4]  # Per timestep basis
        float_alu_ops = ops[5]
        float_div_ops = ops[6]
        float_vector_ops = ops[7]
        int_vector_ops = ops[8]


        L1_hitrate = 0.96  #just a filler for now
        L2_hitrate = 0.96  #just a filler for now
        L3_hitrate = 0.96  #just a filler for now

        # TASK: Floating point operations (add/multiply)'fALU'
        # TASK: Floating point divisions'fDIV'

        tasklist_per_chunk = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                               ['fDIV', float_div_ops],
                               ['INTVEC', int_vector_ops, 1],
                               ['VECTOR', float_vector_ops, 1],
                               ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
                                num_index_vars, num_float_vars, avg_dist,
                                index_loads, float_loads, False] ]
 
        # Loop over the cells, adding time for interior and border cells
        print "\ntime_compute for timestep i: ", tasklist_per_chunk
        timestep_time, stats_ts = core.time_compute(tasklist_per_chunk, True)

        # Return
        return timestep_time


###############################################################################
# imc_process
###############################################################################

def imc_process(mpi_comm_world):
        """
        Driver for the IMC timestep progression untill t_stop. Compute time for all the timesteps until process termination.
        Each timestep takes different amount of compute time. No communication at this time.

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
        process_compute_time=0
        m_time = t_start
        timestep_i_time=0
        print "\n----Start main function:\n"
        
        global dt
        
        while(m_time < t_stop):
                        #this is the compute_time of a timestep
                        #timestep parameters (method parameters)
                        ts_args = [timestep_i, n_photons, n_x_cells]
                        timestep_i_time = compute_timestep_time(core, ts_args)
                        
                        #add individual timestep_compute_time to process compute time
                        process_compute_time += timestep_i_time

                        m_time = m_time+dt
                        dt *= dt_mult
                        timestep_i+=1 #increment timestep

        # We have all requireds, so we can mimic the computation time--i.e., just
        #   sleep for the time it takes to compute over the process
        mpi_ext_sleep(process_compute_time, mpi_comm_world)

                

        # This rank is done, print the finished status
        print "\nCore ", global_core_id, \
                                " done at time ", mpi_wtime(mpi_comm_world), \
                                " after executing ", timestep_i+1, " timesteps"

        # We synchronize to let all ranks finish their timesteps/processes (future work)
        #mpi_barrier(mpi_comm_world)


        runtime_list.append(process_compute_time)
                # Print the results to screen
        print "\nEnd results: #photons:" + str(n_photons) 
        print "============"
        print "Total time (sec):", mpi_wtime(mpi_comm_world)
        print "Time for last timestep (sec): ", timestep_i_time
        print "Time for entire process (sec): ", process_compute_time

        # Finalize mpi and the simulation
        mpi_finalize(mpi_comm_world)

###############################################################################
###############################################################################

# "MAIN"
###############################################################################
# 1. Choose the Cluster that we want to simulate
modeldict = {
  "intercon_type" : "Fattree",
  "fattree" : configs.moonlight_intercon,
  #"intercon_type" : "Bypass",
  "mpi_path"       : "/usr/local/lib/libmpich.dylib",
  "host_type" : "MLIntelNode",
  "load_libraries": set(["mpi"]),
  "mpiopt" : configs.infiniband_mpiopt,
  "debug_options" : set(["none"])
  #"debug_options" : set(["hpcsim", "fattree", "mpi"])
}

print "\nIMCSim run with Simian PDES Engine\nVersion =", version

n_photons = photon_list
cluster = Cluster(modeldict, model_name="imcsim", sim_time=1e6, use_mpi=False)

total_hosts = 1
cores_per_host = 1
total_cores = total_hosts*cores_per_host


  # each compute node has multiple cores; we try to put an mpi process
  # on each core and put neighboring processes on the same node if at
  # all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(total_ranks)]


cluster.start_mpi(hostmap, imc_process)
cluster.run()

print runtime_list #prints out the times from all 4 different photon numbers
print "\n"