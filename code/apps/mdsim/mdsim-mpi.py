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

natoms = 256000
nneighs = 37.5544
r_s = 2.8 # ghost atom cutoff (pot cutoff + skin)
nsteps = 100
delta = 1.0
nsteps_neigh = 20

nx  = 2
ny  = 2
nz  = 2
total_ranks = nx*ny*nz
#taxes = [33.5919,33.5919,33.5919]
taxes = [67.1838,67.1838,67.1838]

# Assume homogenous dist. of atoms:
density = float(nneighs+1.0)/((4.0/3.0)*(math.pi)*math.pow(r_s,3))
dx = taxes[0]/nx
dy = taxes[1]/ny
dz = taxes[2]/nz
SA_avg = 2.0*(dx*dy+dx*dz+dy*dz)/6.0

#simName, startTime, endTime, minDelay, useMPI = \
#  "mdsim", 0.0, 100000000000.0, 0.1, False
#simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

idarray = []
for x in range(0,nx):
    idarray.append([])
    for y in range(0,ny):
        idarray[x].append([])
        for z in range(0,nz):
            idarray[x][y].append(0)
id_cnt = 0
for z in range(0,nz):
    for y in range(0,ny):
        for x in range(0,nx):
            idarray[x][y][z] = id_cnt
            id_cnt += 1

def six_way_comm(data_size,data,east,west,north,south,up,down,mpi_comm):
    data = None # Don't really need to send any data
    if nx > 1 :
        # E-W:
        to_rank = east
        from_rank = west
        r = mpi_sendrecv(to_rank, data, data_size, from_rank, mpi_comm)
        # W-E:
        to_rank = west
        from_rank = east
        r = mpi_sendrecv(to_rank, data, data_size, from_rank, mpi_comm)
    if ny > 1 :
        # N-S:
        to_rank = north
        from_rank = south
        r = mpi_sendrecv(to_rank, data, data_size, from_rank, mpi_comm)
        # S-N:
        to_rank = south
        from_rank = north
        r = mpi_sendrecv(to_rank, data, data_size, from_rank, mpi_comm)
    if nz > 1 :
        # U-D:
        to_rank = up
        from_rank = down
        r = mpi_sendrecv(to_rank, data, data_size, from_rank, mpi_comm)
        # D-U:
        to_rank = down
        from_rank = up
        r = mpi_sendrecv(to_rank, data, data_size, from_rank, mpi_comm)
    return

###############################################################################
# force_call_time_kernel
###############################################################################
def force_call_time_kernel(natoms_p, nneighs):

    """
    Determine all variables needed to construct task list.
    """
    
    # Variable neighbor list size (variable cutoff):
    float_loads = 11.35    * (natoms_p*nneighs)
    index_loads =  2.92    * (natoms_p*nneighs)
    float_alu_ops = 22.30  * (natoms_p*nneighs)
    float_div_ops = 0.0    * (natoms_p*nneighs)
    float_vector_ops = 0.0
    int_alu_ops = 38.29    * (natoms_p*nneighs)
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
# compute_fc_time
###############################################################################
def compute_fc_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: fc_i_time -- predicted time to solve one timestep. 
    """
    
    natoms_p, nneighs = timestep_args[0], timestep_args[1]
    ops = force_call_time_kernel(natoms_p, nneighs)
  
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

    L1_hitrate = 0.915 #1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 0.915 #1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 0.915 #1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now

    # TASK: Floating point operations (add/multiply)'fALU'
    # TASK: Floating point divisions'fDIV'
    tasklist_per_fc = [['iALU', int_alu_ops],
                          ['fALU', float_alu_ops],
                          ['fDIV', float_div_ops],
                          ['INTVEC', int_vector_ops, 1],
                          ['VECTOR', float_vector_ops, 1],
                          ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False]]

    # Compute time:
    timestep_time, stats_ts = core.time_compute(tasklist_per_fc, True)

    # Return
    return timestep_time

###############################################################################
# neighbor_list_time_kernel
###############################################################################
def neighbor_list_time_kernel(natoms_p, nneighs):

    """
    Determine all variables needed to construct task list.
    """
    
    # Task list is linear with system size for a fixed cutoff (fixed nneighs).
    # The slope of this linear relation is a function (taken to be quadratic here) of nneighs.
    # This gives us the task list if neighbor listing is done EVERY time step...
    # Therefore, the operation and load counts must be uniformly scaled by 'alpha',
    # which depends linearly on the number of time steps between neighbor listing...
    
    nn = nneighs
    m_float_loads =   (8.0444E-02)*nn*nn + (2.2600E+01)*nn + (5.1804E+02)
    m_index_loads =   (1.4276E-01)*nn*nn + (4.4424E+01)*nn + (1.7030E+03)
    m_float_alu_ops = (3.5963E-02)*nn*nn + (1.0038E+01)*nn + (2.5166E+02)
    m_int_alu_ops =   (2.6980E-02)*nn*nn + (7.5163E+00)*nn + (3.1807E+02)
    
    na = natoms_p
    #alpha = (7.7737E-05)*nsteps_neigh*nsteps_neigh - (1.0067E-02)*nsteps_neigh + (1.0100E+00)
    alpha = ((9.8878E-03)*nsteps_neigh + (9.9010E-01)) * 0.70 # Note: 0.70 is a fudge factor right now!!!

    # Variable neighbor list size (variable cutoff):
    float_loads =      m_float_loads   * (na) * alpha
    index_loads =      m_index_loads   * (na) * alpha
    float_alu_ops =    m_float_alu_ops * (na) * alpha
    float_div_ops =    0.0             * (na) * alpha
    float_vector_ops = 0.0
    int_alu_ops =      m_int_alu_ops   * (na) * alpha
    int_vector_ops =   0.0
    num_float_vars =   16
    num_index_vars =   15
    
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
# compute_nl_time
###############################################################################
def compute_nl_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: fc_i_time -- predicted time to solve one timestep. 
    """
    
    natoms_p, nneighs = timestep_args[0], timestep_args[1]
    ops = neighbor_list_time_kernel(natoms_p, nneighs)
  
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

    L1_hitrate = 1.0 #1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 1.0 #1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 1.0 #1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now

    # TASK: Floating point operations (add/multiply)'fALU'
    # TASK: Floating point divisions'fDIV'
    tasklist_per_nl = [['iALU', int_alu_ops],
                          ['fALU', float_alu_ops],
                          ['fDIV', float_div_ops],
                          ['INTVEC', int_vector_ops, 1],
                          ['VECTOR', float_vector_ops, 1],
                          ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False]]

    # Compute time:
    timestep_time, stats_ts = core.time_compute(tasklist_per_nl, True)

    # Return
    return timestep_time

###############################################################################
# comm_time_kernel
###############################################################################
def comm_time_kernel(natoms_p, nneighs):

    """
    Determine all variables needed to construct task list.
    """
    
    # Task list is linear with system size for a fixed cutoff (fixed nneighs).
    # The slope of this linear relation is a function (taken to be quadratic here) of nneighs.
    # This gives us the task list if neighbor listing is done EVERY time step...
    # Therefore, the operation and load counts must be uniformly scaled by 'alpha',
    # which depends quadratically on the number of time steps between neighbor listing...
    
    nn = nneighs
    m_float_loads =   (5.5926E-11)*nn*nn + (1.2724E-06)*nn + (1.0020E+01)
    m_index_loads =   (7.1468E-11)*nn*nn + (1.3807E-06)*nn + (1.0298E+00)
    m_float_alu_ops = (1.1307E-10)*nn*nn + (2.4294E-06)*nn + (2.0040E+01)
    m_int_alu_ops =   (8.0838E-10)*nn*nn + (1.6755E-05)*nn + (8.5303E+01)
    
    na = natoms #natoms_p
    alpha = (9.9025E-03)*nsteps_neigh + (9.9010E-01)
    
    # Note: MISSING DEPENDENCE ON MPI PROCS!

    # Variable neighbor list size (variable cutoff):
    float_loads =      m_float_loads   * (na) * alpha
    index_loads =      m_index_loads   * (na) * alpha
    float_alu_ops =    m_float_alu_ops * (na) * alpha
    float_div_ops =    0.0             * (na) * alpha
    float_vector_ops = 0.0
    int_alu_ops =      m_int_alu_ops   * (na) * alpha
    int_vector_ops =   0.0
    num_float_vars =   16
    num_index_vars =   15
    
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
# compute_comm_time
###############################################################################
def compute_comm_time(core, timestep_args):

    """
    Put together the task list to send to hardware.

    Incoming: core -- core id, each core will perform same number of timesteps.
    Outgoing: fc_i_time -- predicted time to solve one timestep. 
    """
    
    natoms_p, nneighs = timestep_args[0], timestep_args[1]
    ops = comm_time_kernel(natoms_p, nneighs)
  
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

    L1_hitrate = 0.55 #1.0 - (1.0/(1.0*avg_reuse_dist))  # just a filler for now
    L2_hitrate = 0.55 #1.0 - (1.0/(2.0*avg_reuse_dist))  # just a filler for now
    L3_hitrate = 0.55 #1.0 - (1.0/(3.0*avg_reuse_dist))  # just a filler for now

    # TASK: Floating point operations (add/multiply)'fALU'
    # TASK: Floating point divisions'fDIV'
    tasklist_per_nl = [['iALU', int_alu_ops],
                          ['fALU', float_alu_ops],
                          ['fDIV', float_div_ops],
                          ['INTVEC', int_vector_ops, 1],
                          ['VECTOR', float_vector_ops, 1],
                          ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                          num_index_vars, num_float_vars, avg_dist,
                          index_loads, float_loads, False]]

    # Compute time:
    timestep_time, stats_ts = core.time_compute(tasklist_per_nl, True)

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
    
    # Define MPI neighbor ranks: (use xy row major and z is third index)
    #    x - east-west direction (Rows)
    #    y - north-south direction (Columns)
    #    z - up-down direction (Levels)
    z = int( p / (nx*ny) )
    y = int( ( p % (nx*ny) ) / nx )
    x = int( ( p % (nx*ny) ) % nx )
    
    # Z - Levels:
    if nz > 1:
        if z == 0:
            down = idarray[x][y][nz-1]
            up = idarray[x][y][z+1]
        elif z == (nz - 1):
            down = idarray[x][y][z-1]
            up = idarray[x][y][0]
        else:
            down = idarray[x][y][z-1]
            up = idarray[x][y][z+1]
    else:
        down = idarray[x][y][z]
        up = idarray[x][y][z]

    # Y - Columns:
    if ny > 1:
        if y == 0:
            south = idarray[x][ny-1][z]
            north = idarray[x][y+1][z]
        elif y == (ny - 1):
            south = idarray[x][y-1][z]
            north = idarray[x][0][z]
        else:
            south = idarray[x][y-1][z]
            north = idarray[x][y+1][z]
    else:
        north = idarray[x][y][z]
        south = idarray[x][y][z]

    # X - Rows:
    if nx > 1:
        if x == 0:
            west = idarray[nx-1][y][z]
            east = idarray[x+1][y][z]
        elif x == (nx - 1):
            west = idarray[x-1][y][z]
            east = idarray[0][y][z]
        else:
            west = idarray[x-1][y][z]
            east = idarray[x+1][y][z]
    else:
        east = idarray[x][y][z]
        west = idarray[x][y][z]

    print "\nCore ",p," has x,y,z,id = ",x,y,z,idarray[x][y][z]

    # Initialize:
    timestep_i = 0
    tot_fc_time = 0
    tot_nl_time = 0
    tot_comm_time = 0
    fc_i_time = 0

    # Data Size:
    #    "delta" is a scaling parameter
    natoms_p = natoms/n
    #Delta = int(delta*float(natoms_p)*SA_avg) # Assumes d > r_s
    Delta = int(delta*SA_avg*density*r_s) # Assumes d >= r_s
    data = None
    
    # Expected Force Call and Neighbor list compute times per step:
    ts_args = [natoms_p, nneighs]
    fc_i_time = compute_fc_time(core, ts_args)
    nl_i_time = compute_nl_time(core, ts_args)
    comm_i_time = compute_comm_time(core, ts_args)
    
    # Initial Sync
    mpi_barrier(mpi_comm_world)
    #print "Core ", global_core_id, " did init sync "

    while (timestep_i < nsteps):
    
        # Do neighborlist MPI work if necessary:
        if ((timestep_i % nsteps_neigh) == 0):
    
            # Update This Box's atoms (1a)
            six_way_comm(Delta,data,east,west,north,south,up,down,mpi_comm_world)
            #print "Core ", global_core_id, " did (1a) "
            
            # Share Ghost Atoms (1b)
            six_way_comm(Delta,data,east,west,north,south,up,down,mpi_comm_world)
            #print "Core ", global_core_id, " did (1b) "
            
            # Neighbor Listing:
            mpi_ext_sleep(nl_i_time, mpi_comm_world)
            tot_nl_time += nl_i_time
            
            # LAMMPS COMM:
            mpi_ext_sleep(comm_i_time, mpi_comm_world)
            tot_comm_time += comm_i_time
        
        # Force Call:
        mpi_ext_sleep(fc_i_time, mpi_comm_world)
        tot_fc_time += fc_i_time
        
        # Update atom positions across proc boundaries (5):
        six_way_comm(Delta,data,east,west,north,south,up,down,mpi_comm_world)
        #print "Core ", global_core_id, " did (5)"
        
        # Increment step:
        timestep_i += 1

    # Final Sync
    mpi_barrier(mpi_comm_world)
    #print "Core ", global_core_id, " did final sync "

    print "Core ", global_core_id, \
        " done at time ", mpi_wtime(mpi_comm_world), \
        " after executing ", timestep_i, " timesteps"
        
    # Print the results to screen
    if p == 0:
        print "\nResults for ",nsteps," MD steps:"
        print "==========================================="
        print "WCTime for entire run (sec): ", mpi_wtime(mpi_comm_world)
        print "WCTime for FC (sec): ", tot_fc_time
        print "WCTime for NL (sec): ", tot_nl_time
        print "WCTime for COMM (sec): ", tot_comm_time + (mpi_wtime(mpi_comm_world) - (tot_fc_time+tot_nl_time+tot_comm_time))
        print "WCTime for OTHER (sec): ", mpi_wtime(mpi_comm_world) - (tot_fc_time+tot_nl_time+tot_comm_time)

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
cores_per_host = 16
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
