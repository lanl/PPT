#
# test_split.py :- test split communicators
#

import sys
from ppt import *

def test_split(mpi_comm_world):
    n = mpi_comm_size(mpi_comm_world) 
    p = mpi_comm_rank(mpi_comm_world)
    total = mpi_allreduce(p, mpi_comm_world)

    new_comm = mpi_comm_split(mpi_comm_world, int(p/3), p)
    nn = mpi_comm_size(new_comm) 
    np = mpi_comm_rank(new_comm)
    x = mpi_allreduce(p, new_comm)

    new_new_comm = mpi_comm_split(new_comm, int(np/2), nn-np)
    nnn = mpi_comm_size(new_new_comm) 
    nnp = mpi_comm_rank(new_new_comm)
    y = mpi_allreduce(p, new_new_comm)

    print("%d/%d on %s (%r) total=%d => %d/%d (%r) x=%d => %d/%d (%r) y=%d" % 
          (p, n, mpi_ext_host(mpi_comm_world), mpi_comm_world['hostmap'], total,
           np, nn, new_comm['hostmap'], x, nnp, nnn, new_new_comm['hostmap'], y))

    mpi_finalize(mpi_comm_world)

if len(sys.argv) != 2:
    print("Usage: test_split.py total_ranks ")
    sys.exit(1)
n = int(sys.argv[1])
#print("test_split.py %d" % n)

modeldict = {
    "model_name" : "test_split",
    "sim_time" : 1e9,
    "use_mpi" : True,
    "intercon_type" : "Gemini",
    "torus" : configs.hopper_intercon,
    "host_type" : "Host",
    "load_libraries": set(["mpi"]),
    "mpiopt" : configs.gemini_mpiopt,
}
cluster = Cluster(modeldict)

total_hosts = cluster.num_hosts()
hostmap = [i%total_hosts for i in range(n)]
cluster.start_mpi(hostmap, test_split)

cluster.run()
