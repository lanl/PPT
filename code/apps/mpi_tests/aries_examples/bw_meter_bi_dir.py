#
# bw_meter_bi_dir.py :- measure throughput of between a pair of MPI
# ranks with a given message size
# the traffic is of bi-directional pattern

import sys
from ppt import *

x = 5

def measure(mpi_comm_world, sz):
    n = mpi_comm_size(mpi_comm_world) 
    p = mpi_comm_rank(mpi_comm_world)
    print("rank %d/%d on %s" % (p, n, mpi_ext_host(mpi_comm_world)))

    wait_reqs = dict()
    if p%2 == 1:
        # the odd ranks are senders, which is to send x packets of
        # the given size to the corresponding receiver
        t0 = mpi_wtime(mpi_comm_world)
        
        # Note: id for recv request is 0. Other ids are from 1 till x+1 
        #recv
        req_id = 0
        wait_reqs[req_id] = mpi_irecv(mpi_comm_world)
        
        #send
        for i in xrange(x):
            req_id = i+1 
            wait_reqs[req_id] = mpi_isend(p-1, None, sz, mpi_comm_world)
            sofar = (i+1)*sz
        reqids = [wait_reqs[i] for i in xrange(x+1)]
        mpi_waitall(reqids)
        t1 = mpi_wtime(mpi_comm_world)
        #print("%f: %d sent %d, throughput=%g" % (t1, p/2, sofar, 8*sofar/(t1-t0)))
        print("summary: ps=%d chunksz: %d, throughput: %g" % (n/2, sz, 8*sofar/(t1-t0)))
    else:
        # the even ranks are receivers
        t2 = mpi_wtime(mpi_comm_world)
        
        #recv
        req_id = 0
        wait_reqs[req_id] = mpi_irecv(mpi_comm_world)
        
        #send
        for i in xrange(x):
            req_id = i+1 
            wait_reqs[req_id] = mpi_isend(p+1, None, sz, mpi_comm_world)
            sofar = (i+1)*sz
        reqids = [wait_reqs[i] for i in xrange(x+1)]
        mpi_waitall(reqids)
        t3 = mpi_wtime(mpi_comm_world)
        #print("%f: %d sent %d, throughput=%g" % (t3, p/2, sofar, 8*sofar/(t3-t2)))
        print("summary: ps=%d chunksz: %d, throughput: %g" % (n/2, sz, 8*sofar/(t3-t2)))

    # required to end mpi process with finalize
    mpi_finalize(mpi_comm_world)

if len(sys.argv) != 5:
    print("Usage: python bw_meter_bi_dir.py host0 host1 parallel_sessions data_size")
    sys.exit(1)

h0 = int(sys.argv[1])
h1 = int(sys.argv[2]) 
ps = int(sys.argv[3])
sz = int(sys.argv[4])
print("python bw_meter_bi_dir.py %d %d %d %d" % (h0, h1, ps, sz))

model_dict = {
    "intercon_type" : "Aries",
    "dragonfly" : configs.edison_intercon,
    "host_type" : "Host",
    "load_libraries": set(["mpi"]),
    "mpiopt" : configs.aries_mpiopt,
    "debug_options" : set([]),
}
cluster = Cluster(model_dict, model_name="bw_meter_bi_dir", sim_time=1e9, use_mpi=False)
cluster.start_mpi([h0, h1]*ps, measure, sz)
cluster.run()
