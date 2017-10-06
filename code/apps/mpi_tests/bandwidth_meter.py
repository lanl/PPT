#
# bandwidth_meter.py :- measure throughput of between a pair of MPI
# ranks with a given message size
#

import sys
from ppt import *

x = 5

def measure(mpi_comm_world, sz):
    n = mpi_comm_size(mpi_comm_world) 
    p = mpi_comm_rank(mpi_comm_world)
    print("rank %d/%d on %s" % (p, n, mpi_ext_host(mpi_comm_world)))

    if p%2 == 0: 
        # the even ranks are senders, which is to send x packets of
        # the given size to the corresponding receiver
        t0 = t1 = mpi_wtime(mpi_comm_world)
        for i in xrange(x):
            succ = mpi_send(p+1, t1, sz, mpi_comm_world)
            if not succ: raise Exception("send failed");
            t1 = mpi_wtime(mpi_comm_world)
            sofar = (i+1)*sz
            print("%f: %d sent %d, throughput=%g" % (t1, p/2, sofar, 8*sofar/(t1-t0)))
        else: 
            r = mpi_reduce(0, 8*sofar/(t1-t0), mpi_comm_world)
            if p == 0:
                print("summary: ps=%d chunksz: %d throughput: %g" % (n/2, sz, r))
    else:
        for i in xrange(x):
            succ = mpi_recv(mpi_comm_world)
            if succ is None: raise Exception("recv failed");
            t2 = mpi_wtime(mpi_comm_world)
            print("%f: %d delay=%g" % (t2, p/2, t2-succ["data"]))
        else:
            mpi_reduce(0, 0, mpi_comm_world)

    # required to end mpi process with finalize
    mpi_finalize(mpi_comm_world)

if len(sys.argv) != 5:
    print("Usage: python bandwidth_meter.py host0 host1 parallel_sessions data_size")
    sys.exit(1)

h0 = int(sys.argv[1])
h1 = int(sys.argv[2]) 
ps = int(sys.argv[3])
sz = int(sys.argv[4])
print("python bandwidth_meter.py %d %d %d %d" % (h0, h1, ps, sz))

model_dict = {
    "intercon_type" : "Gemini",
    "torus" : configs.hopper_intercon,
    "host_type" : "Host",
    "load_libraries": set(["mpi"]),
    "mpiopt" : configs.gemini_mpiopt,
}
cluster = Cluster(model_dict, model_name="bandwidth_meter", sim_time=1e9, use_mpi=False)
cluster.start_mpi([h0, h1]*ps, measure, sz)
cluster.run()
