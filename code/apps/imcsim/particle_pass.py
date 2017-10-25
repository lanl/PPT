
'''
returns the total time of transport_particle_pass function communication per timestep
'''
def transport_particle_pass(mpi_comm_world):
  #get global photon count
  n_local = 20000
  time_before = mpi_wtime(mpi_comm_world)
  mpi_allreduce(n_local, mpi_comm_world, data_size=8, op="sum")


  return mpi_wtime(mpi_comm_world)-time_before

'''
return processor adjacency list
'''
def get_proc_adj_list():


'''
Post receives for photons from adjacent sub-domains
'''
def recv_photons(adjacent_procs):
  mpi_ircv_reqs_list = []
  for i in adjacent_procs:
    i[0] = adj_rank
    i[1] = i_b
    req = mpi_irecv(mpi_comm, from_rank=adj_rank, type=None)
    mpi_ircv_reqs_list.append(req)
  return mpi_ircv_reqs_list

'''
process photon send and receives
'''
def pht_send_rcv(adjacent_procs, mpi_comm_world):
  #test completion of send buffer
  pht_send_reqs = []
  pht_recv_reqs = []
  for i in adjacent_procs:
    i[0] = adj_rank
    i[1] = i_b
    #fill in type in bytes?
    req = mpi_isend(mpi_comm, to_rank=adj_rank, type=None)
    pht_send_reqs.append(req)
    #deal with receive buffer
    req = mpi_irecv(mpi_comm, from_rank=adj_rank, type=None)
    pht_recv_reqs.append(req)
  #wait for all ranks to finish
  mpi_barrier(mpi_comm_world)
  return pht_send_reqs
