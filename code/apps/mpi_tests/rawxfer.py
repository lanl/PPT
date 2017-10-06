#
# rawxfer.py :- two data transfers on hopper interconnect
#

from ppt import *

model_dict = {
    "model_name" : "rawxfer", 
    "sim_time" : 100, 
    "use_mpi" : True,
    "mpi_path" : "/opt/local/lib/mpich-mp/libmpi.dylib",
    "intercon_type" : "Gemini",
    "torus" : configs.hopper_intercon,
    "host_type" : "Host",
    "debug_options" : set(["hpcsim", "torus", "host"])
}

cluster = Cluster(model_dict)
cluster.sched_raw_xfer(10, 1, 1000, 1000)
cluster.sched_raw_xfer(20, 3, 400, 1000000)
cluster.run()
