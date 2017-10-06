# cannon's algorithm for matrix multiplication

#
# Measure allreduce time with respect to the number of mpi processes
# and data size (both input as command-line argument)
#
# To run, we need to add simian1/SimianPie and PPT to PYTHONPATH.
#

import sys, math
from ppt import *

modeldict = {
    "model_name" : "cannon",
    "sim_time" : 1e9,
    "use_mpi" : False,
    "mpi_path" : "/opt/local/lib/mpich-mp/libmpi.dylib",
    "intercon_type" : "Gemini",
    "torus" : configs.hopper_intercon,
    "host_type" : "Host",
    "load_libraries": set(["mpi"]),
}

# n is matrix dimension, debug_look_id is the rank of the process printing stuff
def cannon(mpi_comm_world, n, utime, debug_look_id = -1):
    p = mpi_comm_size(mpi_comm_world) 
    id = mpi_comm_rank(mpi_comm_world)
    print("%d/%d on %s" % (id, p, mpi_ext_host(mpi_comm_world)))

    # p must be a square number; px = sqrt(p)
    px = int(math.sqrt(p))
    if px*px != p: 
        raise Exception("p=%d is not a square number" % p)

    # each submatrix should be m*m
    m = int(n/px)
    if m*px != n:
        raise Exception("n=%d cannot be divided evenly by sqrt(p=%d)" % (n, p))
    m2 = m*m

    # (i,j) coordinate
    cart_comm = mpi_cart_create(mpi_comm_world, (px,px))
    i, j = mpi_cart_coords(cart_comm, id)
    if id == debug_look_id:
        print("%f: rank %d coords (%d,%d): submatrix %dx%d" % 
              (mpi_wtime(cart_comm), id, i, j, m, m))
    
    # read submatrices A(i,j) and B(i,j), initialize submatrix C(i,j)
    mpi_ext_sleep(m2*utime, cart_comm)

    # initial x shift: shift submatrix A(i,j) left by i column positions
    right, left = mpi_cart_shift(cart_comm, 1, -i)
    if id == debug_look_id:
        print("%f: rank %d: shift left by %d positions: %d <- %d" % 
              (mpi_wtime(cart_comm), id, i, left, right))
    mpi_sendrecv(left, None, m2*8, right, cart_comm)

    # initial y shift: shift submatrix A(i,j) up by j row positions
    down, up = mpi_cart_shift(cart_comm, 0, -j)
    if id == debug_look_id:
        print("%f: rank %d: shift up by %d positions: %d <- %d" % 
              (mpi_wtime(cart_comm), id, j, up, down))
    mpi_sendrecv(up, None, m2*8, down, cart_comm)

    # remember neighbor ranks
    left, right = mpi_cart_shift(cart_comm, 1, 1)
    up, down = mpi_cart_shift(cart_comm, 0, 1)
    
    for round in range(px-1):
        # first multiply the submatrices
        mpi_ext_sleep(m2*m*utime, cart_comm)

        # shift A to the left and B upward one position
        mpi_sendrecv(left, None, m2*8, right, cart_comm)
        mpi_sendrecv(up, None, m2*8, down, cart_comm)
        if id == debug_look_id:
            print("%f: rank %d: round #%d shifts: left=%d right=%d up=%d down=%d" % 
                  (mpi_wtime(cart_comm), id, round, left, right, up, down))

    if id == debug_look_id:
        print("%f: rank %d: done" % (mpi_wtime(cart_comm), id))

    # write the result submatrix C(i,j)
    # ...

    t = mpi_reduce(0, mpi_wtime(cart_comm), cart_comm, op="max")
    if id == 0: print("run time : %f" % t)

    # BUG-AND-FIX: simian process when terminates will not context
    # switch back to another running process; to avoid this problem,
    # we require the user mpi process to call mpi_finalize (which
    # actually hibernates the process forever)
    mpi_finalize(mpi_comm_world)

def cannon_async(mpi_comm_world, n, utime, debug_look_id = -1):
    p = mpi_comm_size(mpi_comm_world) 
    id = mpi_comm_rank(mpi_comm_world)

    # p must be a square number; px = sqrt(p)
    px = int(math.sqrt(p))
    if px*px != p: 
        raise Exception("p=%d is not a square number" % p)

    # each submatrix should be m*m
    m = int(n/px)
    if m*px != n:
        raise Exception("n=%d cannot be divided evenly by sqrt(p=%d)" % (n, p))
    m2 = m*m

    # (i,j) coordinate
    cart_comm = mpi_cart_create(mpi_comm_world, (px,px))
    i, j = mpi_cart_coords(cart_comm, id)
    if id == debug_look_id:
        print("%f: rank %d coords (%d,%d): submatrix %dx%d" % 
              (mpi_wtime(cart_comm), id, i, j, m, m))
    
    # read submatrices A(i,j) and B(i,j), initialize submatrix C(i,j)
    mpi_ext_sleep(m2*utime, cart_comm)

    # initial x shift: shift submatrix A(i,j) left by i column positions
    right, left = mpi_cart_shift(cart_comm, 1, -i)
    if id == debug_look_id:
        print("%f: rank %d: shift left by %d positions: %d <- %d" % 
              (mpi_wtime(cart_comm), id, i, left, right))
    mpi_sendrecv(left, None, m2*8, right, cart_comm)

    # initial y shift: shift submatrix A(i,j) up by j row positions
    down, up = mpi_cart_shift(cart_comm, 0, -j)
    if id == debug_look_id:
        print("%f: rank %d: shift up by %d positions: %d <- %d" % 
              (mpi_wtime(cart_comm), id, j, up, down))
    mpi_sendrecv(up, None, m2*8, down, cart_comm)

    # remember neighbor ranks
    left, right = mpi_cart_shift(cart_comm, 1, 1)
    up, down = mpi_cart_shift(cart_comm, 0, 1)
    
    for round in range(px):
        if round < px-1:
            # unless it's the last iteration, shift A left and B up by
            # one position; initiate asynchronous sends/receives
            r0 = mpi_isend(left, None, m2*8, cart_comm)
            r1 = mpi_isend(up, None, m2*8, cart_comm)
            r2 = mpi_irecv(cart_comm, right)
            r3 = mpi_irecv(cart_comm, down)
            if id == debug_look_id:
                print("%f: rank %d: round #%d async with left=%d right=%d up=%d down=%d" % 
                      (mpi_wtime(cart_comm), id, round, left, right, up, down))

        # first multiply the submatrices
        mpi_ext_sleep(m2*m*utime, cart_comm)
 
        if round < px-1:
            # unless it's the last iteration, finishes all transactions
            mpi_waitall([r0, r1, r2, r3])
            if id == debug_look_id:
                print("%f: rank %d: round #%d waitall" % 
                      (mpi_wtime(cart_comm), id, round))

    if id == debug_look_id:
        print("%f: rank %d: done" % (mpi_wtime(cart_comm), id))

    # write the result submatrix C(i,j)
    # ...

    t = mpi_reduce(0, mpi_wtime(cart_comm), cart_comm, op="max")
    if id == 0: print("run time: %f" % t)

    # BUG-AND-FIX: simian process when terminates will not context
    # switch back to another running process; to avoid this problem,
    # we require the user mpi process to call mpi_finalize (which
    # actually hibernates the process forever)
    mpi_finalize(mpi_comm_world)


if len(sys.argv) != 4:
    print("Usage: cannon.py total_ranks matrix_dim method(sync/async)")
    sys.exit(1)

p = int(sys.argv[1])
n = int(sys.argv[2])
method = sys.argv[3]
if method != 'sync' and method != 'async':
    print("ERROR: invalid method (%s), must be either 'sync' or 'async'" % method)
    sys.exit(3)

print("cannon.py %d %d %s" % (p, n, method))

#cluster = Cluster(modeldict, debug_options={"host", "mpi"})
cluster = Cluster(modeldict)

total_hosts = cluster.num_hosts()
cores_per_host = 24
total_cores = total_hosts*cores_per_host
if p > total_cores:
    print("ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run this job (p=%d)" %
          (total_hosts, cores_per_host, total_cores, p))
    sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(p)]

# the time in seconds needed to update a matrix cell (add multiply)
utime = 1e-8 

if method == 'sync':
    cluster.start_mpi(hostmap, cannon, n, utime, 15)
else:
    cluster.start_mpi(hostmap, cannon_async, n, utime, 15)

cluster.run()
