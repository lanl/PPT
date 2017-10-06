# helloworld.py :- use mpi as simple as possible

# IMPORTANT: To make it to work, PATHONAPTH must be set to include
# PPT's 'code' directory, where the 'ppt.py' file is located
from ppt import *

def main(mpi_comm_world, msg):
    n = mpi_comm_size(mpi_comm_world) # total num of ranks
    p = mpi_comm_rank(mpi_comm_world) # rank of this process

    # mpi_ext_host is a good way for the mpi process to access the
    # underneath compute node; here, we simply print out its name
    print("%d/%d main('%s') starts on %s" % 
          (p, n, msg, mpi_ext_host(mpi_comm_world)))

    # it is required an mpi process ends with mpi_finalize
    mpi_finalize(mpi_comm_world)

model_dict = {
    # simian parameters (these are required unless explicitly
    # specified when instantiating Cluster)
    "model_name" : "helloworld", # name of the model
    "sim_time" : 1e6, # simulation time
    "use_mpi" : True, # whether using mpi for parallel
    "mpi_path" : "/opt/local/lib/mpich-mp/libmpi.dylib", # mpi library path needed for SimianPie
                                                         # (rather than SiianPie.MPI4Py)
    
    # interconnection network parameters
    #"intercon_type" : "Crossbar",       # IMPORTANT: type is case sensitive
    #"crossbar" : {                      # configuration specific to crossbar
    #    "nhosts" : 101,                 # the number of hosts (compute nodes) connected by crossbar
    #},
    
    "intercon_type" : "BlueGeneQ",         # IMPORTANT: type is case sensitive
    #"torus" : configs.sequoia_intercon,   # use standardized config for Sequoia
    "torus" : configs.mira_intercon,   # use standardized config for Mira
    
    # compute node parameters; IMPORTANT: type is case sensitive
    #"host_type" : "CieloNode",  # cielo with or without mpi installed
    "host_type" : "Host",       # generic compute node with or without mpi installed
    
    # each host type can have a distinct configuration like
    # interconnect (not yet implemented)
    
    # optional libraries/modules to be loaded onto compute nodes
    #"load_libraries": set([]),
    "load_libraries": set(["mpi"]),     # IMPORANT: lib names are case sensitie

    "debug_options": set(["torus"]),

    # mpi configurations (necessary if mpi is loaded)
    "mpiopt" : configs.bluegeneq_mpiopt,  # standard mpi config for BlueGene/Q
    #"mpiopt" : {
    #    "min_pktsz" : 2048,    # max packet size
    #    "max_pktsz" : 10000,   # min packet size
    #    "resend_intv" : 0.01,  # time between resends
    #    "resend_trials" : 10,  # num of trials until giving up
    #    "call_time" : 1e-7,    # async calls may still cost time
    #    "max_injection" : 1e9  # max mpi injection rate into network
    #},
}

cluster = Cluster(model_dict)
hostmap = range(10) # 10 mpi processes on separate hosts
cluster.start_mpi(hostmap, main, "hello world")
cluster.run()
