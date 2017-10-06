#
# group.py :- test sub-communicators using groups
#

import sys
from ppt import *

def test_group(mpi_comm_world):
    world_size = mpi_comm_size(mpi_comm_world) 
    world_rank = mpi_comm_rank(mpi_comm_world)

    world_group = mpi_comm_group(mpi_comm_world)
    prime_group = mpi_group_incl(world_group, [1, 2, 3, 5, 7, 11, 13])
    non_prime_group = mpi_group_excl(world_group, [1, 2, 3, 5, 7, 11, 13])
    
    prime_comm = mpi_comm_create(mpi_comm_world, prime_group)
    prime_size = mpi_comm_size(prime_comm) 
    prime_rank = mpi_comm_rank(prime_comm)

    non_prime_comm = mpi_comm_create_group(mpi_comm_world, non_prime_group)
    non_prime_size = mpi_comm_size(non_prime_comm) 
    non_prime_rank = mpi_comm_rank(non_prime_comm)

    print("world: %d/%d\tprime: %d/%d\tnon-prime:%d/%d" %
          (world_rank, world_size, prime_rank, prime_size, 
           non_prime_rank, non_prime_size))

    mpi_group_free(world_group);
    mpi_group_free(prime_group);
    mpi_group_free(non_prime_group);
    mpi_comm_free(prime_comm);
    mpi_comm_free(non_prime_comm);

    mpi_finalize(mpi_comm_world)

modeldict = {
    "model_name" : "test_group",
    "sim_time" : 1e9,
    "use_mpi" : True,
    "intercon_type" : "Gemini",
    "torus" : configs.hopper_intercon,
    "host_type" : "Host",
    "load_libraries": set(["mpi"]),
    "mpiopt" : configs.gemini_mpiopt,
}
cluster = Cluster(modeldict)
cluster.start_mpi(range(16), test_group)
cluster.run()
