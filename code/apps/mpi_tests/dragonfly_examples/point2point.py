#
# point2point.py :- test traffic (random and 1/2/3-d nearest neighbor)
#

import sys, math, random
from ppt import *

def exponential(mean):
    return -math.log(random.random())*mean

def equilikely(low, high):
    return low+int((high-low+1)*random.random())

def point2point(mpi_comm_world, iat, data_size, pattern, sample_intv=1):
    n = mpi_comm_size(mpi_comm_world)
    p = mpi_comm_rank(mpi_comm_world)
    #print("%d/%d on %s: iat=%f sz=%d pattern=%s" % 
    #      (p, n, mpi_ext_host(mpi_comm_world), iat, data_size, pattern))

    if p%2 == 0:
        cull = 0 # we use this to reduce samples
        # even processes are receivers
        while True:
            r = mpi_recv(mpi_comm_world)
            if r is None:
                raise Exception("recv failed at rank %d" % p)
            now = mpi_wtime(mpi_comm_world)
            cull += 1
            if cull >= sample_intv:
                cull = 0
                print("%f %f (nanosecs) from=%d to=%d" % 
                      (now, (now-r['data'])*1e9, int(r['from_rank']/2), int(p/2)))
    else:
        # odd processes are senders
        nn = n/2; pp = int(p/2)
        if pattern == '1dnn':
            choices = [ (pp-1+nn)%nn, (pp+1)%nn ]
        elif pattern == '2dnn':
            n1 = int(math.sqrt(nn))
            i = int(pp/n1); j = pp%n1
            jleft = (j-1+n1)%n1; left = i*n1+jleft
            jright = (j+1)%n1; right = i*n1+jright
            iup = (i-1+n1)%n1; up = iup*n1+j
            idown = (i+1)%n1; down = idown*n1+j
            choices = [ left, right, up, down ]
        elif pattern == '3dnn': 
            n1 = int(round(math.pow(nn, 1./3)))
            i = int(pp/(n1*n1)); j = int((pp%(n1*n1))/n1); k = (pp%(n1*n1))%n1
            kleft = (k-1+n1)%n1; left = i*n1*n1+j*n1+kleft
            kright = (k+1)%n1; right = i*n1*n1+j*n1+kright
            jup = (j-1+n1)%n1; up = i*n1*n1+jup*n1+k
            jdown = (j+1)%n1; down = i*n1*n1+jdown*n1+k
            ifront = (i-1+n1)%n1; front = ifront*n1*n1+j*n1+k
            iback = (i+1)%n1; back = iback*n1*n1+j*n1+k
            choices = [ left, right, up, down, front, back ]
        else: # random
            choices = range(0, nn)

        while True:
            t = exponential(iat)
            mpi_ext_sleep(t, mpi_comm_world)
            d = equilikely(0, len(choices)-1)
            t = mpi_wtime(mpi_comm_world)
            r = mpi_send(choices[d]*2, t, data_size, mpi_comm_world)
            if r is None:
                raise Exception("send failed at rank %d" % p)

    # BUG-AND-FIX: simian process when terminates will not context
    # switch back to another running process; to avoid this problem,
    # we require the user mpi process to call mpi_finalize (which
    # actually hibernates the process forever)
    mpi_finalize(mpi_comm_world)

if len(sys.argv) != 6 and len(sys.argv) != 7:
    print("Usage: point2point.py gemini_dims total_ranks ranks_per_host iat data_size pattern [sample_interval]")
    sys.exit(1)

dims = int(sys.argv[1]) 
n = int(sys.argv[2]) 
x = int(sys.argv[3]) 
iat = float(sys.argv[4])
sz = int(sys.argv[5])
pattern = sys.argv[6].lower()
if pattern != 'random' and \
   pattern != '1dnn' and \
   pattern != '2dnn' and \
   pattern != '3dnn':
    print("ERROR: pattern (%s) is not recognized" % sys.argv[6])
    sys.exit(1)
if len(sys.argv) == 8: samp = int(sys.argv[7])
else: samp = 1

if pattern == '2dnn':
    n1 = int(math.sqrt(n))
    if n1*n1 != n:
        print("ERROR: n=%d is not a square number for 2DNN" % n)
        sys.exit(1)
elif pattern == '3dnn':
    n1 = int(round(math.pow(n, 1./3)))
    if n1*n1*n1 != n:
        print("ERROR: n=%d is not a cubic number for 3DNN" % n)
        sys.exit(1)
print("point2point.py dims=%d n=%d x=%d iat=%d sz=%d pattern=%s samp=%d" % 
      (dims, n, x, iat, sz, pattern, samp))

modeldict = {
    "model_name" : "point2point",
    "sim_time" : 0.2,
    "use_mpi" : False,
    "intercon_type" : "Dragonfly",
    "host_type" : "Host",
    "dragonfly" : configs.dragonfly_intercon,
    "load_libraries": set(["mpi"]),
    "mpiopt" : configs.aries_mpiopt,
    "debug_options" : set(["dragonfly"]),
}

cluster = Cluster(modeldict)

# we actually launch 2n processes; odd ones as senders and even ones
# as receivers; the pairs are surely on the same host
total_hosts = cluster.num_hosts()
hostmap = [(i/2/x)%total_hosts for i in range(n*2)]

cluster.start_mpi(hostmap, point2point, iat*n, sz, pattern, samp)
cluster.run()
