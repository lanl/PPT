# 
# test_mpicalls.py :- test (almost) all mpi functions implemented
#

from ppt import *

# user application main function requires only one argument:
# mpi_comm_world is the communicator representing all mpi processes;
# here, x, y, z are only to test the simple passing of parameters from
# the start_mpi function
def myapp(mpi_comm_world, x, y, z):
    n = mpi_comm_size(mpi_comm_world) 
    p = mpi_comm_rank(mpi_comm_world)
    print("myapp(x=%d y=%d z=%d) rank %d/%d runs on %s" % 
          (x, y, z, p, n, mpi_ext_host(mpi_comm_world)))

    d = (n-p)*10
    #print("%f: myapp: rank %d about to sleep for %f" %
    #      (mpi_wtime(mpi_comm_world), p, d))
    mpi_ext_sleep(d, mpi_comm_world)

    print("%f: myapp: rank %d sends msg to rank %d" %
          (mpi_wtime(mpi_comm_world), p, (p+1)%n))
    succ = mpi_send((p+1)%n, "hello", 100, mpi_comm_world)
    print("%f: myapp: rank %d done sent: %s" %
          (mpi_wtime(mpi_comm_world), p, "success" if succ else "failed"))

    r = mpi_recv(mpi_comm_world)
    print("%f: myapp: rank %d receive: %s" %
          (mpi_wtime(mpi_comm_world), p, "success" if r is not None else "failed"))
    if r is not None:
        print("  => from rank=%d type=%r data=%r sz=%d" %
              (r["from_rank"], r["type"], r["data"], r["data_size"]))
    
    r = mpi_reduce(5, p, mpi_comm_world, op="sum")
    print("%f: myapp: rank %d reduce(root=5, data=%d) %s" %
          (mpi_wtime(mpi_comm_world), p, p, "=> %d"%r if r is not None else ": failed"))

    if p == 3: mpi_ext_sleep(100, mpi_comm_world)
    r = mpi_bcast(3, 100+p, mpi_comm_world)
    print("%f: myapp: rank %d bcast(root=3, data=%d) %s" %
          (mpi_wtime(mpi_comm_world), p, p+100, "=> %d"%r if r is not None else ": failed"))

    if p == 1: mpi_ext_sleep(100, mpi_comm_world)
    mpi_barrier(mpi_comm_world)

    if p == 2: mpi_ext_sleep(100, mpi_comm_world)
    r = mpi_allreduce(p, mpi_comm_world, op="max")
    print("%f: myapp: rank %d allreduce(data=%d, op='max') %s" %
          (mpi_wtime(mpi_comm_world), p, p, "=> %d"%r if r is not None else ": failed"))

    if p == 3: mpi_ext_sleep(100, mpi_comm_world)
    r = mpi_allgather(p+10, mpi_comm_world)
    print("%f: myapp: rank %d allgather(root=1, data=%d) %s" %
          (mpi_wtime(mpi_comm_world), p, p+10, "=> %r"%r if r is not None else ": failed"))

    data = range(p, p+n)
    r = mpi_scatter(1, data, mpi_comm_world)
    print("%f: myapp: rank %d scatter(root=1, data=%r) %s" %
          (mpi_wtime(mpi_comm_world), p, data, "=> %r"%r if r is not None else ": failed"))
    
    r = mpi_alltoallv(data, mpi_comm_world)
    print("%f: myapp: rank %d alltoallv(data=%r) %s" %
          (mpi_wtime(mpi_comm_world), p, data, "=> %r"%r if r is not None else ": failed"))
    
    # test immediate send and recv between rank 2 and 3
    if p == 2: 
        print("## %f: rank 2 irecv" % mpi_wtime(mpi_comm_world))
        rreq = mpi_irecv(mpi_comm_world, 3)
        print("## %f: rank 2 irecv returns" % mpi_wtime(mpi_comm_world))
        mpi_ext_sleep(10, mpi_comm_world)
        print("## %f: rank 2 isend" % mpi_wtime(mpi_comm_world))
        sreq = mpi_isend(3, "hello from 2", 100, mpi_comm_world)
        print("## %f: rank 2 isend returns" % mpi_wtime(mpi_comm_world))
        mpi_ext_sleep(10, mpi_comm_world)
        if not mpi_test(sreq): 
            print("## %f: rank 2 test(s) failed, wait(s)" % mpi_wtime(mpi_comm_world))
            mpi_wait(sreq)
            print("## %f: rank 2 wait(s) returns" % mpi_wtime(mpi_comm_world))
        else:
            print("## %f: rank 2 test(s) succeeded" % mpi_wtime(mpi_comm_world))
        print("## %f: rank 2 waitall(r)" % mpi_wtime(mpi_comm_world))
        mpi_waitall([rreq])
        print("## %f: rank 2 waitall([r]) done" % mpi_wtime(mpi_comm_world))
    elif p == 3:
        print("## %f: rank 3 recv" % mpi_wtime(mpi_comm_world))
        r = mpi_recv(mpi_comm_world)
        print("## %f: rank 3 recv returns: %r" % (mpi_wtime(mpi_comm_world), r))
        mpi_ext_sleep(100, mpi_comm_world)
        print("## %f: rank 3 send" % mpi_wtime(mpi_comm_world))
        mpi_send(2, "hello from 3", 500, mpi_comm_world)
        print("## %f: rank 3 send returns" % mpi_wtime(mpi_comm_world))

    mpi_barrier(mpi_comm_world)
    print("%f: rank %d final barrier" % (mpi_wtime(mpi_comm_world), p))

    # BUG-AND-FIX: simian process when terminates will not context
    # switch back to another running process; to avoid this problem,
    # we require the user mpi process to call mpi_finalize (which
    # actually hibernates the process forever)
    mpi_finalize(mpi_comm_world)


modeldict = {
    "model_name" : "test_mpicalls",
    "sim_time" : 10000,
    "use_mpi" : False,
    "host_type" : "Host",
    #
    #"intercon_type" : "Gemini",
    #"torus" : configs.gemini_anydim(3,3,3),
    #
    "intercon_type" : "Crossbar",
    "crossbar" : {
        "nhosts" : 100,
    },
    #
    #"torus" : configs.hopper_intercon,
    #mpiopt" : configs.hopper.mpiopt,
    #
    "load_libraries": set(["mpi"]),
    "mpiopt" : {
        "min_pktsz" : 2048,
        "max_pktsz" : 10000,
        "resend_intv" : 0.01, 
        "resend_trials" : 10,
        "call_time" : 1e-7
    }
}

cluster = Cluster(modeldict)
hostmap = range(13)
cluster.start_mpi(hostmap, myapp, 1, 2, 3)
cluster.run()
