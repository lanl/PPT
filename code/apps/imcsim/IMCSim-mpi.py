"""
IMCsim: an IMC application simulator for Marshak wave.
"""

from sys import path
path.append('../..')
from ppt import *
from math import ceil

# mpi ranks (in 3D-grid arrangement)128 ranks
p_x = 8  # number of ranks in x direction
p_y = 8  # number of ranks in y direction
p_z = 2 # number of ranks in z direction

# total number of photons for the entire system, assuming they are
# uniformly distributed
total_photons = 102400001

# cells are arranged in three dimensions
x_cells = 40 # number of cells in x direction
y_cells = 40 # number of cells in y direction
z_cells = 40 # number of cells in z direction

# derived parameters
total_ranks = p_x*p_y*p_z
total_cells = x_cells*y_cells*z_cells
photons_per_rank = ceil(total_photons/total_ranks)
cells_per_rank   = int(total_cells/total_ranks)
x_cells_per_rank = int(x_cells/p_x)
y_cells_per_rank = int(y_cells/p_y)
z_cells_per_rank = int(z_cells/p_z)

'''not really needed now (round up) so that ranks might have slightly more photons that total
# make sure photons and cells are divisible among the ranks
if photons_per_rank*total_ranks != total_photons:
    print("ERROR: photons (%d) cannot be divided evenly among the ranks (%d)" % \
          (total_photons, total_ranks))
    sys.exit(-1)
'''
if x_cells_per_rank*p_x != x_cells or \
   y_cells_per_rank*p_y != y_cells or \
   z_cells_per_rank*p_z != z_cells:
    print("ERROR: cells (%d x %d x %d) cannot be divided evenly among the ranks (%d x %d x %d)" % \
          (x_cells, y_cells, z_cells, p_x, p_y, p_z))
    sys.exit(-1)

# model parameters are constants
t_start = 0.0    # time to start
t_stop  = 0.0001 # time to finish
dt = 0.0001      # delta_t to next timestep
dt_mult = 1.0    # timestep size multiplier

# mpi communication parameters
batch_size = 50000   # number of particles to process at each batch
sendbuf_size = 20000  # number of particles to be sent in one message
particle_size = 80    # in bytes

########scale_test_1 input file monte carlo event counts###########
event_dict = {
"pass_particles":0.1, #percentage of particles to pass to other ranks
"total_loops":2.91,  #batch_photons=num_photons
"exit":0,
"kill":0,
"census":0.9,
"cell_crossings":0.71,
"scatter":1.17,
"b_vacuum":0,
"b_reflect":0.023,
"tail_photons":0.16,#% of photons in the tail computation phase, 84% time is in the main_photon compute-0.24 #bad avg is 0.
"tail_time":0.115, #% of time- tail computation takes of the batch compute time 0.115-0.175
}
#############################################################


# all the operation counts are condensed in this function.
# returns task list, # of particles to send to neighbor ranks
def batch_compute(num_photons):
    census = event_dict["census"]*num_photons
    cell_crossings = event_dict["cell_crossings"]*num_photons
    scatter =event_dict["scatter"]*num_photons
    b_reflect=event_dict["b_reflect"]*num_photons
    b_vacuum = event_dict["b_vacuum"]*num_photons
    kill = event_dict["kill"]*num_photons

    # put together the task list
    int_alu_ops = 4*b_reflect + 11*b_vacuum + 4*cell_crossings + 48142*cells_per_rank + 7*census + 19840091*event_dict["total_loops"]*num_photons/10000 + 10*kill + 273.67*num_photons + 13*scatter + 300005210.5
    float_alu_ops = b_reflect + b_vacuum + cell_crossings + 1971.9*cells_per_rank + 24*event_dict["total_loops"]*num_photons + kill + 11.434*num_photons + 2*scatter + 10000193.49
    float_div_ops = 10*event_dict["total_loops"]*num_photons
    int_vector_ops = 0
    float_vector_ops = 0
    num_index_vars = 2*b_vacuum + 22*event_dict["total_loops"]*num_photons + 2*kill + 3
    num_float_vars = 32*event_dict["total_loops"]*num_photons + 12
    avg_dist = 1
    index_loads = 2*b_vacuum + 807501*event_dict["total_loops"]*num_photons/5000 + 2*kill + 3
    float_loads = 3*b_reflect + 5*b_vacuum + 3*cell_crossings + 25084*cells_per_rank + 5*census + 2407501*event_dict["total_loops"]*num_photons/5000 + 4*kill + 124.59*num_photons + 6*scatter + 100002342.5

    L1_hitrate = 0.96  # figure it out later
    L2_hitrate = 0.96  # figure it out later
    L3_hitrate = 0.96  # figure it out later

    task_list = [ \
        ['iALU', int_alu_ops], \
        ['fALU', float_alu_ops], \
        ['fDIV', float_div_ops], \
        ['INTVEC', int_vector_ops, 1], \
        ['VECTOR', float_vector_ops, 1], \
        ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate, 
         num_index_vars, num_float_vars, avg_dist,
         index_loads, float_loads, False] \
    ]
    print task_list

    # two values to return from this function: 1) the task list which
    # will be used by the caller function (transport_particle_pass) to
    # simulate time passing for handling this batch of photons; and 2)
    # the number of photons escaping from this rank, which will be
    # used by the calling function to send messages to the neighboring
    # ranks
    return task_list, int(num_photons*event_dict["pass_particles"]) # assuming x% particles move to other ranks


###############################################################################
# transport_particle_pass: process all particles for one pass (timestep)
###############################################################################
def transport_particle_pass(cart_comm, host, core, p, neighbors, mpi_comm_world):
    # initialize send buffers (each of which contains the number of
    # leaving photons to the corresponding neighbor); also 
    # post receive for photons from adjacent sub-domains
    send_buf = {}
    #recv_req = {}
    for nx in neighbors:
        send_buf[nx] = 0
        #recv_req[nx] = mpi_irecv(cart_comm, from_rank=nx, type='nphotons')
    recv_req = mpi_irecv(cart_comm, type='nphotons')

    # the main loop (interate through the batches)
  
    # the number of photons to be processed in this rank; the number
    # may go up as more photons arrive from other ranks
    photons_to_process = photons_per_rank*(1-event_dict["tail_photons"])
    total_photons_main_loop = photons_to_process*total_ranks

    # the number of photons that have been processed by this rank;
    # this term will go up (in batches), all the way to
    # photons_to_process
    photons_processed = 0

    # the numer of photons that have finished by this rank during this
    # iteration; when the total among ranks reaches all photons, the
    # iteration terminates
    photons_finished = 0

    ######speed up the simulation##########
    #compute time for a regular batch, and multiply by the number of total batches
    # to get total_compute_time
    task_list, leaving_photons = batch_compute(batch_size)
    if mpi_comm_rank(mpi_comm_world)==0:
      print ("###########################TASK_LIST")
      print task_list
      print ("**********************************")
    compute_time = core.time_compute(task_list)
    total_compute_time = compute_time*int(ceil(photons_to_process/batch_size))

    #######end of speed up#################

    # the following loop break out when the timestep finishes
    while True: 
        # receive particles from neighbors
        while mpi_test(recv_req):
            # if returns true, recv_req['from_rank', 'data',
            # 'data_size', 'type'] have already been filled;
            # 'data' will contain the number of particles
            photons_to_process += recv_req['data']
            #print("%d: incoming %d from %d" % (p, recv_req['data'], recv_req['from_rank']))
            # post more receives
            recv_req = mpi_irecv(cart_comm, type='nphotons')

        #print("%d: processed=%d, to_process=%d, finished=%d" % \
        #      (p, photons_processed, photons_to_process, photons_finished))

        # while there are still photons to process in the main compute phase
        while photons_processed < photons_to_process:
            # the number of photons to be processed in the next batch
            #batchsz = batch_size
            #if photons_processed+batchsz > photons_to_process:
            #    batchsz = photons_to_process - photons_processed

            # process the batch of photons (returns task list and the
            # number of photons that will be leaving the rank)
            #task_list, leaving_photons = batch_compute(batchsz) # already computed
            photons_processed += batch_size
            photons_finished += batch_size-leaving_photons
            #print("%d: batch_compute(bsz=%d) lv=%d, processed=%d, to_process=%d, finished=%d" % \
            #     (p, batchsz, leaving_photons, photons_processed, photons_to_process, photons_finished))

            # simulate the compute time
            #compute_time = core.time_compute(task_list)
            mpi_ext_sleep(compute_time, cart_comm)
            #total_compute_time += compute_time

            # send messages to neighbors when send buffers are filled
            idx = 0
            all = leaving_photons
            for nx in neighbors:
                # evenly distribute the leaving photons (consider odd ends)
                my_share = int((idx+1)*leaving_photons/len(neighbors)) - \
                           int(idx*leaving_photons/len(neighbors))
                send_buf[nx] += my_share
                idx += 1
                all -= my_share
                while send_buf[nx] >= sendbuf_size:
                    send_buf[nx] -= sendbuf_size
                    mpi_send(nx, sendbuf_size, sendbuf_size*particle_size, cart_comm, type='nphotons')
                    #print("%d: send %d photons to %d, %d remain in buffer" % \
                          #(p, sendbuf_size, nx, send_buf[nx]))
            if all > 0: print("ERROR: all=%d" % all)
                    
            # receive particles from neighbors
            while mpi_test(recv_req):
                # if returns true, recv_req['from_rank', 'data',
                # 'data_size', 'type'] have already been filled;
                # 'data' will contain the number of particles
                photons_to_process += recv_req['data']
                #print("%d: incoming %d, to_process=%d" % \
                      #(p, recv_req['data'], photons_to_process))
                # post more receives
                recv_req = mpi_irecv(cart_comm, type='nphotons')

        # after all particles have been processed, we send the
        # remaining leaving particles in the send buffer (this is tail_compute time
        for nx in neighbors: 
          if send_buf[nx] > 0:
            mpi_send(nx, send_buf[nx], send_buf[nx]*particle_size, cart_comm, type='nphotons')
            #print("%d: send all %d photons to %d" % (p, send_buf[nx], nx))
            send_buf[nx] = 0

        # receive particles from neighbors-finish
        while mpi_test(recv_req):
          # if returns true, recv_req['from_rank', 'data',
          # 'data_size', 'type'] have already been filled;
          # 'data' will contain the number of particles
          photons_finished += recv_req['data']
          #print("1********%d: incoming %d, to_process=%d" % (p, recv_req['data'], photons_to_process))
          recv_req = mpi_irecv(cart_comm, type='nphotons')
        

        # check terminating condition
        x = mpi_allreduce(photons_finished, mpi_comm_world, data_size=8, op="sum")
        if x >= total_photons_main_loop:
          print("%d terminate? finished=(%d:%d<%d) (total=%d)" % (p, p, photons_finished, x, total_photons_main_loop))      
          ####--add the tail_compute time to the main compute time
          total_compute_time += total_compute_time*event_dict["tail_time"] #add the tail compute time
          mpi_ext_sleep(total_compute_time*event_dict["tail_time"], cart_comm)
          return total_compute_time

    


###############################################################################
# imc_main: the mpi main function at each rank
###############################################################################
def imc_main(mpi_comm_world):
  # make sure total number of ranks is consistent
  n = mpi_comm_size(mpi_comm_world)
  if n != total_ranks:
    print("ERROR: inconsistent number of mpi ranks specified (%d)" % total_ranks)
    sys.exit(-1)

  # create the corresponding cartesian topology communicator (we set
  # 'periodic' to be false; i.e., no wrap around)
  cart_comm = mpi_cart_create(mpi_comm_world, (p_x, p_y, p_z), (False,)*3)

  # find this rank and neighboring ranks
  p = mpi_comm_rank(cart_comm)
  #print("%d (%r) ->" % (p, mpi_cart_coords(cart_comm, p)))
  neighbors = set()
  left_r, right_r = mpi_cart_shift(cart_comm, 0, 1)
  #print("%d: %r,%r (left,right)" % (p, left_r, right_r))
  if left_r is not None:
      neighbors.add(left_r)
      #print("left ->%d (%r)" % (left_r, mpi_cart_coords(cart_comm, left_r)))
  if right_r is not None:
      neighbors.add(right_r)
      #print("right ->%d (%r)" % (right_r, mpi_cart_coords(cart_comm, right_r)))
  up_r, down_r = mpi_cart_shift(cart_comm, 1, 1)
  #print("%d: %r,%r (up,down)" % (p, up_r, down_r))
  if up_r is not None:
      neighbors.add(up_r)
      #print("up ->%d (%r)" % (up_r, mpi_cart_coords(cart_comm, up_r)))
  if down_r is not None:
      neighbors.add(down_r)
      #print("down ->%d (%r)" % (down_r, mpi_cart_coords(cart_comm, down_r)))
  front_r, back_r = mpi_cart_shift(cart_comm, 2, 1)
  #print("%d: %r,%r (front,back)" % (p, front_r, back_r))
  if front_r is not None:
      neighbors.add(front_r)
      #print("front ->%d (%r)" % (front_r, mpi_cart_coords(cart_comm, front_r)))
  if back_r is not None:
      neighbors.add(back_r)
      #print("back ->%d (%r)" % (back_r, mpi_cart_coords(cart_comm, back_r)))
  print("%d: neighbors: %r" % (p, neighbors))

  # access the machine and the core on which this rank is running;
  # according to the hostmap, the ranks are mapped consecutively
  # (packed configuration)
  host = mpi_ext_host(cart_comm)
  core_id = p % host.num_cores
  core = host.cores[core_id]

  # iterate through the time steps (a.k.a. photon passes)
  total_compute_time = 0  # total time used for computing
  timestep_i = 0   # timestep index
  m_time = t_start # model time
  dt_local = dt    # keep a local copy of dt (global is constant)
  while m_time < t_stop:
    # for each photon pass, we get the total photon energy using a
    # global all-reduction (we only simulate the effect here) not the exact number
    mpi_allreduce(450.0, cart_comm, data_size=8, op="sum")

    # doing the real work for each timestep
    t = host.engine.now
    # not all these parameters are necessary...
    timestep_i_compute_time = transport_particle_pass(cart_comm, host, core, p, neighbors, mpi_comm_world)
    timestep_i_time = host.engine.now - t
    total_compute_time += timestep_i_compute_time

    timestep_i += 1
    m_time = m_time + dt_local
    dt_local *= dt_mult

    print("rank %d finished %d timesteps at time %f (last iteration: total_time=%f, compute_time=%f)" %
          (p, timestep_i, host.engine.now, timestep_i_time, timestep_i_compute_time))

  # print the final results
  print("rank %d processed %d photons in %d timesteps: total_time=%f, compute_time=%f" % \
        (p, photons_per_rank, timestep_i, host.engine.now, total_compute_time))

  # finalize mpi and the simulation
  mpi_finalize(mpi_comm_world)


###############################################################################
###############################################################################

# model parmeters
modeldict = {
#  "intercon_type" : "Fattree",
#  "fattree" : configs.moonlight_intercon,
  "intercon_type" : "Bypass",
  "mpi_path"       : "/usr/local/lib/libmpich.dylib",
  "bypass" : {
      "nhosts" : int(ceil((total_ranks+15)/16)),
      "bdw" : 1e10,
      "link_delay" : 1e-6
  },
  "host_type" : "MLIntelNode",
  "load_libraries": set(["mpi"]),
  "mpiopt" : configs.infiniband_mpiopt,
  "debug_options" : set(["none"])
  #"debug_options" : set(["hpcsim", "fattree", "mpi"])
}

# create the cluster and run simulation
cluster = Cluster(modeldict, model_name="imcsim", sim_time=1e6, use_mpi=True)

total_hosts = cluster.intercon.num_hosts()
cores_per_host = 16 # supposedly accessible from cluster
total_cores = total_hosts*cores_per_host
if total_ranks > total_cores:
  print("ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run application (p=%d)" % \
        (total_hosts, cores_per_host, total_cores, total_ranks))
  sys.exit(-1)

hostmap = [(y/cores_per_host)%total_hosts for y in range(total_ranks)]
cluster.start_mpi(hostmap, imc_main)
cluster.run()
