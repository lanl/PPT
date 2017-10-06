#
# getlat.py :- end-to-end latency test
#

from ppt import *

def getlat(mpi_comm_world):
    n = mpi_comm_size(mpi_comm_world) 
    p = mpi_comm_rank(mpi_comm_world)
    #print("%d/%d on %s" % (p, n, mpi_ext_host(mpi_comm_world)))

    if p>0: 
        mpi_ext_sleep(p, mpi_comm_world)
        mpi_recv(mpi_comm_world)
        t = mpi_wtime(mpi_comm_world)-p
        print("%d %f (nanosecs)" % (p, t*1e9))
    else:
        for i in range(1, n):
            t = mpi_wtime(mpi_comm_world)
            print("%f: myapp: rank %d sends msg to rank %d"%
                    (mpi_wtime(mpi_comm_world), p, i))
            mpi_ext_sleep(i-t, mpi_comm_world)
            mpi_send(i, None, 4, mpi_comm_world)

    mpi_finalize(mpi_comm_world)

modeldict = {
    "model_name" : "getlat",
    "sim_time" : 1e9,
    "use_mpi" : False,
    "intercon_type" : "Dragonfly",
    "dragonfly" : configs.dragonfly_intercon,
    "host_type" : "Host",
    "load_libraries": set(['mpi']),
    "mpiopt" : configs.aries_mpiopt,
    "debug_options" : set(["dragonfly"])
}

cluster = Cluster(modeldict)
total_hosts = cluster.num_hosts()
#cores_per_host = 24
#total_cores = total_hosts*cores_per_host
cluster.start_mpi(range(total_hosts), getlat)
cluster.run()
