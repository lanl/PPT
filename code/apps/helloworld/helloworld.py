# helloworld.py :- use mpi as simple as possible
# test omnipath architecture

# IMPORTANT: To make it to work, PYTHONAPTH must be set to include
# PPT's 'code' directory, where the 'ppt.py' file is located
from sys import path
path.append('../..')
from ppt import *

def main(mpi_comm_world, msg):
    n = mpi_comm_size(mpi_comm_world) # total num of ranks
    p = mpi_comm_rank(mpi_comm_world) # rank of this process

    # mpi_ext_host is a good way for the mpi process to access the
    # underneath compute node; here, we simply print out its name
    print("%d/%d main('%s') starts on %s" %
          (p, n, msg, mpi_ext_host(mpi_comm_world)))
    print("%f: myapp: rank %d sends msg to rank %d" %
          (mpi_wtime(mpi_comm_world), p, (p+1)%n))
    succ = mpi_send((p+1)%n, "hello", 10, mpi_comm_world)
    print("%f: myapp: rank %d done sent: %s" %
          (mpi_wtime(mpi_comm_world), p, "success" if succ else "failed"))

    # it is required an mpi process ends with mpi_finalize
    mpi_finalize(mpi_comm_world)

model_dict = {
    # simian parameters (these are required unless explicitly
    # specified when instantiating Cluster)
    "model_name" : "helloworld", # name of the model
    "sim_time" : 1e6, # simulation time
    "use_mpi" : False, # whether using mpi for parallel

    "intercon_type" : "Fattree",         # IMPORTANT: type is case sensitive
    #"torus" : configs.cielo_intercon,   # use standardized config for Cielo
    "fattree" : configs.omnipath_intercon,   # use sample Omnipath config

    # compute node parameters; IMPORTANT: type is case sensitive
    #"host_type" : "CieloNode",  # cielo with or without mpi installed
    "host_type" : "I7Node",       # generic compute node with or without mpi installed
    "mpi_path"      : "/ccs/opt/intel/impi/5.1.3.210/lib64/libmpi.so.12",
    # each host type can have a distinct configuration like
    # interconnect (not yet implemented)

    # optional libraries/modules to be loaded onto compute nodes
    #"load_libraries": set([]),
    "load_libraries": set(["mpi"]),     # IMPORANT: lib names are case sensitie

    # mpi configurations (necessary if mpi is loaded)
    "mpiopt" : configs.infiniband_mpiopt,  # standard mpi config for Infiniband (TODO: check and may be change later)
    "debug_options": set(["fattree"]),
}

cluster = Cluster(model_dict)
hostmap = range(10) # 10 mpi processes on separate hosts
cluster.start_mpi(hostmap, main, "hello world")
cluster.run()
