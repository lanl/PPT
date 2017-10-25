"""
IMCsim: an IMC application simulator for Hot Box wave.
Version 17 includes warp speed up for mpi communications
and also adds the mpi barrier st only one rank computes after warp has been entered
V19 adds the analytical rank crossing speed up
"""


from sys import path
path.append('../..')
from ppt import *
from math import ceil
from sys import argv

# total number of photons for the entire system, assuming they are
# uniformly distributed
total_photons = int(argv[2])#102400001
# mpi ranks (in 3D-grid arrangement)128 ranks
p_x = int(argv[3]) # number of ranks in x direction
p_y = int(argv[4])	# number of ranks in y direction
p_z = int(argv[5]) # number of ranks in z direction

# cells are arranged in three dimensions
x_cells = 40 # number of cells in x direction
y_cells = 40 # number of cells in y direction
z_cells = 40 # number of cells in z direction

# derived parameters
total_ranks = p_x*p_y*p_z
total_cells = x_cells*y_cells*z_cells
photons_per_rank = ceil(total_photons/total_ranks)
cells_per_rank	 = int(total_cells/total_ranks)
x_cells_per_rank = int(x_cells/p_x)
y_cells_per_rank = int(y_cells/p_y)
z_cells_per_rank = int(z_cells/p_z)

num_boundary_faces = total_cells*6/2 + x_cells*y_cells + x_cells*z_cells + y_cells*z_cells 
# each cell has 6 boundary faces, doublecounting, then add global boundary faces back in
'''
rank_crossing_boundaries = (x_cells_per_rank*y_cells_per_rank + \
  x_cells_per_rank*z_cells_per_rank + y_cells_per_rank*z_cells_per_rank) * total_ranks/2 \
  - x_cells*y_cells + x_cells*z_cells + y_cells*z_cells
'''
rank_crossing_boundaries = (2*total_ranks*(x_cells_per_rank*y_cells_per_rank + \
  x_cells_per_rank*z_cells_per_rank + y_cells_per_rank*z_cells_per_rank) - \
   (x_cells*y_cells + x_cells*z_cells + y_cells*z_cells))/2
# think about it again
alpha = rank_crossing_boundaries / float(num_boundary_faces) 
# alpha is the fraction of rank crossing boundaries
#alpha = 0.02
#print "Alpha = ", alpha


# model parameters are constants
t_start = 0.0  # time to start
t_stop	= 0.0001 # time to finish
dt = 0.0001	   # delta_t to next timestep
dt_mult = 1.0  # timestep size multiplier

receive_bias = 0.99

# mpi communication parameters
batch_size = int(argv[6]) # number of particles to process at each batch
sendbuf_size = float(argv[7])  # number of particles to be sent in one message
particle_size = 80	  # in bytes

########scale_test_1 input file monte carlo event counts###########
event_dict = {
#"cell_crossings_frac":[.05,.05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, 
#    .05, .05, .043, .038, .031, .024],
"cell_crossings_frac":[.00478037,
0.0529839, 
0.0557233,
0.05572,
0.0555224,
0.0556875,
0.0553682,
0.0554887,
0.0554733,
0.0553934,
0.0552296,
0.0546867,
0.0534541,
0.0514539,
0.048065,
0.0435708,
0.0378876,
0.0312187,
0.0247883,
0.0186094,
0.0131898,
0.00895363,
0.00572732,
0.00350756,
0.00207787,
0.0011628,
0.000610887,
0.000322833,
0.000171119,
7.98891e-05,
3.75504e-05,
1.66331e-05,
8.31653e-06,
3.02419e-06,
1.00806e-06,
1.00806e-06,
2.52016e-07],
    
    
#"cell_crossings_frac":[.05,.05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, 
#    .05, .05, .05,.05,.05,.05,.05],
# fraction of photons 
# that experience exactly list position of cell crossings
"total_loops":100963,  #batch_photons=num_photons
"exit":0,
"kill":0.001,
"census":0.999,
"scatter":0.493,
"b_vacuum":0,
"b_reflect":0.0216,
}
#############################################################
# compute probability that a photon will not pass off the rank

prob = {} # p[k][j] Probability that a photon with k crossings left, will pass off the rank
# after exactly j local crossings
for k in range(len(event_dict["cell_crossings_frac"])):
  prob[k] = {}
for k in range(len(event_dict["cell_crossings_frac"])):
  sum = 0.0 # sum of probabilities
  for j in range(k):
	prob[k][j] = (1- alpha)**j * alpha
	sum += prob[k][j]
  prob[k][k] = 1- sum # probability that photon will finish on current rank
#print "P :", prob




###############################################################################
# transport_particle_pass: process all particles for one pass (timestep)
###############################################################################
def transport_particle_pass(cart_comm, host, core, p, neighbors, mpi_comm_world):
  # Initializations
  ph_init = photons_per_rank # local photons
  ph_finished = 0 # number of photons finished on this rank
  ph = {} # List of number of local photons remaining to be sent with i cross overs left for ph[i]
  ph_r = {} # Same but for photons received from other ranks
  maxcrossings = len(event_dict["cell_crossings_frac"])
  for i in range(maxcrossings):
	ph[i] = int(event_dict["cell_crossings_frac"][i] * ph_init)
	ph_r[i] = 0
	
  send_buf, send_bins = {}, {}
  for nx in neighbors:
	send_buf[nx], send_bins[nx] = {}, {}
	send_bins[nx]['sum'] = 0
	for k in range(maxcrossings):
		send_bins[nx][k] = 0
  
  recv_req = mpi_irecv(cart_comm, type='nphotons')
  
  compute_time_per_photon_per_crossing = float(argv[8]) #9.70e-07, changed definition _per_crossing

  # Monitoring speed up Inits
  warp_done = False
  #if len(argv)>9:
  #	print "No Warp Mode"
  #	warp_done = True
  #warp_time, last_batch = 0, -1
  send_times =[]
  hist_ph, hist_ph_r = [], []
  #	warp_cycles[nx] = 0
  allreducecount,batchcount, innercount = 0, 0, 0 # history parameters
  batchcounts = []
  finished = False
  

  # MAIN LOOP  
  while not finished: # main loop over batches
	#print p, " entered main loop with ph, ph_r", ph, ph_r
	# (1) Treat batch
	
	# Populate b proportionally from ph and ph_l
	ph_left, ph_r_left, sum_b = 0,0,0
	for i in range(maxcrossings):
		ph_left += ph[i]
		ph_r_left += ph_r[i]
	#print " before inner while ", ph_left, ph_r_left, sum_b, batch_size
	while ph_left >= 1 or ph_r_left >= 1:
		b ={} # List of batch photons with i crossings left
		for i in range(maxcrossings):
			b[i] = 0
		#first deal with remotely received photons, which we treat preferentially
		r_desired = receive_bias*batch_size
		l_desired = batch_size - r_desired
		batch_remain = batch_size
		#print p,": ph_r_left, ph_left before building b : ", ph_r_left, ph_left
		if ph_r_left > 0:
			r_frac = min(1.0, r_desired/float(ph_r_left))
			for i in range(maxcrossings):
				#print "in rec loop  ", r_frac
				add = ph_r[i]*r_frac
				b[i] += add
				batch_remain -= add
				ph_r_left -= add
				ph_r[i] -= add
		if ph_left > 0:
			l_frac = min(1.0, batch_remain/float(ph_left))
			for i in range(maxcrossings):
				add = ph[i]*l_frac
				b[i] += add
				batch_remain -= add
				ph_left -= add
				ph[i] -= add
	
		batch_actual = batch_size - batch_remain
		# (2) Numbers of passing photons into passing bins mimicking the census and other endstates
		passing = {} # number of photons that have k crossings left before census
		num_crossings = b[0] # need to count the photons that don't cross
		for k in range(maxcrossings):
			passing[k] = 0
		for k in range(maxcrossings):
			for j in range(k+1):
				passing[k-j] += b[k] * prob[k][j] # this many photons added to pass count 
				num_crossings += b[k] * prob[k][j] * j
				#print "in passing, k, j, b[k], prob[k][j] ", k, j, b[k], prob[k][j]
		# (2a)  Determine census and other non-passing photon count: passing[0]
		ph_finished += passing[0]
		tmp1, tmp2 = 0,0
		for k in range(maxcrossings):
			tmp1 += b[k]
			tmp2 += passing[k]
		#print p, ": Size of Passing and b, and finished:", tmp2, tmp1, passing[0]
		num_crossings += passing[0]
		passing[0] = 0 # these photons do not pass, but we still count them as crossing
		
		#We need to count the rank passing as a crossing, so deduct one more crossing
		for k in range(maxcrossings-1):
			num_crossings += passing[k+1]# for the computational work
			passing[k] = passing[k+1]
		passing[maxcrossings-1] = 0
		#print p, " built passing structure ", passing
	
		# (3) Process elements in b the compute time, changed from earlier versions
		mpi_ext_sleep(num_crossings * compute_time_per_photon_per_crossing, cart_comm)
	
		# (4) Distribute pass bins among neighbors
		for k in range(maxcrossings):
			for nx in neighbors:
				# evenly distribute the leaving photons
				send_bins[nx][k] += passing[k]/float(len(neighbors))
			#print p, " send_bins ", send_bins
		
		# (5) Prepare to send and send emptying high crossing bins first (should not matter much)
		for nx in neighbors:
			send_bins[nx]['sum'] = 0
			for k in range(maxcrossings):
				send_bins[nx]['sum'] += send_bins[nx][k]
	
		# (5.2) Warp logic 
		if (send_bins[list(neighbors)[0]]['sum'] >= sendbuf_size) and not warp_done:
			# warp cycle is between two sends to neighbor in different batches
			# update state monitoring
			send_times.append(mpi_wtime(cart_comm))
			batchcounts.append(batchcount)
			j = len(send_times) - 1
			hist_ph.append(0)
			hist_ph_r.append(0)
			for k in range(maxcrossings):
				hist_ph[j] += ph[k]
				hist_ph_r[j] += ph_r[k]
			# Decide if we have observed enough to start warping
			if j > 2: # perhaps play with this number to see when stability is achieved
				warp_time = send_times[j] - send_times[j-1]
				warp_time_per_batch = warp_time /float(batchcounts[j] - batchcounts[j-1])
				if p == 0:
					print p, " at time ",  mpi_wtime(cart_comm), " started warping", alpha, send_times
				# Warp the warp here: essence of V19 add
				n_total_rank_crossings = 0
				for k in range(maxcrossings):
					n_total_rank_crossings += alpha * k * (ph_r[k] + ph[k])
				# I would have sent these to myself, so really, will need to process ph_left + this number	
				#n_rank_crossings_per_cycle = hist_ph[j-1] - hist_ph[j] + hist_ph_r[j-1] - hist_ph_r[j]
				#print "hist_ph, hist_ph_r, n_total_rank_crossings", hist_ph, hist_ph_r, n_total_rank_crossings
				warp_batches = math.floor((n_total_rank_crossings + ph_left) / float(batch_size)) #projected no of batches
				#warp_cycles = warp_cycles / float(batchcounts[j] - batchcounts[j-1])
				#crossings_left = warp_cycles[nx] * int(n_rank_crossings_per_cycle)
				# batchsize sendbuf_size * len(neighbors)
				r_f = (n_total_rank_crossings / float(ph_left + n_total_rank_crossings)) * batch_size # no r photons per pbatch
				l_f = batch_size - r_f
				
				lpw  = warp_batches * l_f
				rpw  = warp_batches * r_f
				#lpw  = warp_batches * (hist_ph[j-1] - hist_ph[j])/ float(batchcounts[j] - batchcounts[j-1]) # n of local particles warped	
				#rpw	 = warp_batches * (hist_ph_r[j-1] - hist_ph_r[j])/ float(batchcounts[j] - batchcounts[j-1]) # n of remote particles warped					
				l_frac = min(1.0, lpw/float(ph_left))
				if ph_r_left == 0: ph_r_left = 1.0
				r_frac = min(1.0, rpw/float(ph_r_left))
				for i in range(maxcrossings):
					addl = ph[i]*l_frac
					addr = ph_r[i] * r_frac
					ph_left -= addl
					ph_r_left -= addr
					ph[i] -= addl
					ph_r[i] -= addr
				# We can do the warp_sleep
				if p == 0: 
					print p, " before warping at time ", mpi_wtime(cart_comm)," with warp_cycles, time, ph, ph_r", warp_batches, warp_time, warp_time_per_batch
				mpi_ext_sleep(warp_time_per_batch * warp_batches, cart_comm)
				warp_done = True
				if p == 0:
					print p, " after warping at time ", mpi_wtime(cart_comm)," with ph, ph_r", ph_left, ph_r_left
	
		# (5.3) Back to the sending logic
		for nx in neighbors:
			sum = 0					
			while (send_bins[nx]['sum'] >= sendbuf_size) or (ph_left <= 1 and send_bins[nx]['sum'] > 0):
				#print " in while with nx, sumsendbin, sendbuf_size, ph_left ", nx, send_bins[nx]['sum'], sendbuf_size, ph_left
				sum = 0
				for k in reversed(range(maxcrossings)):
					send_buf[nx][k] = 0 # init so we have it to send afterwards
					if (sendbuf_size - sum >= send_bins[nx][k]):
						send_buf[nx][k] = send_bins[nx][k]
						sum += send_bins[nx][k]
						send_bins[nx][k] = 0
					else:
						send_buf[nx][k] = sendbuf_size - sum
						sum = sendbuf_size
						send_bins[nx][k] -= send_buf[nx][k]
					
					# we have a full buffer or nothing left in the send bins
				send_bins[nx]['sum'] = 0 
				for k in range(maxcrossings):
					send_bins[nx]['sum'] += send_bins[nx][k]
				mpi_send(nx, [sum, send_buf[nx]], sum*particle_size, cart_comm, type='nphotons')
				
		# 6. Receive particles from neighbors
		while mpi_test(recv_req):
			for k in range(maxcrossings):
				ph_r[k] += recv_req['data'][1][k]
				ph_r_left += recv_req['data'][1][k]
			#print p, " received new photons, left: ", recv_req['data'][0], ph_r_left
			recv_req = mpi_irecv(cart_comm, type='nphotons')
					
		#if p == 1:
		#	print p, ": time, batchcount, batchsize, r_left, left, finished, send_bins  ", mpi_wtime(cart_comm), batchcount, batch_actual, ph_r_left, ph_left, ph_finished #, send_bins[n]['sum']
		batchcount += 1
	# end of while "local or received particles still exist"
	

	# 8. send again	as part of Control Termination
	for nx in neighbors:
		send_bins[nx]['sum'] = 0
		for k in range(maxcrossings):
			send_bins[nx]['sum'] += send_bins[nx][k]
	for nx in neighbors:
		sum = 0
		while (send_bins[nx]['sum'] >= sendbuf_size) or (ph_left <= 1 and send_bins[nx]['sum'] > 0):
			#print " in while with nx, sumsendbin, sendbuf_size, ph_left ", nx, send_bins[nx]['sum'], sendbuf_size, ph_left
			sum = 0
			for k in reversed(range(maxcrossings)):
				send_buf[nx][k] = 0 # init so we have it to send afterwards
				if (sendbuf_size - sum >= send_bins[nx][k]):
					send_buf[nx][k] = send_bins[nx][k]
					sum += send_bins[nx][k]
					send_bins[nx][k] = 0
				else:
					send_buf[nx][k] = sendbuf_size - sum
					sum = sendbuf_size
					send_bins[nx][k] -= send_buf[nx][k]
			
				# we have a full buffer or nothing left in the send bins
			send_bins[nx]['sum'] = 0 
			for k in range(maxcrossings):
				send_bins[nx]['sum'] += send_bins[nx][k]
			mpi_send(nx, [sum, send_buf[nx]], sum*particle_size, cart_comm, type='nphotons')
			#print p, " sent ", sum, " photons to neighbor ", nx#, " with send_buf ", send_buf[nx] ," remaining in send_bins ", send_bins[nx]
	# 9. receive again	as part of Control Termination
	while mpi_test(recv_req):
			for k in range(maxcrossings):
				ph_r[k] += recv_req['data'][1][k]
				ph_r_left += recv_req['data'][1][k]
			#print p, " outer received new photons: ", recv_req['data'][0]#, ph_r_left
			recv_req = mpi_irecv(cart_comm, type='nphotons')
			
	# 10. Call all-reduce to synchronize 
	#print p,": end of round ", count, " with photons_finished:", photons_finished
	#x = mpi_allreduce(photons_finished, mpi_comm_world, data_size=8, op="sum")
	left1 = max(ph.values())
	left2 = max(ph_r.values())
	maxsum = 0
	for nx in neighbors:
		send_bins[nx]['sum'] = 0
		for k in range(maxcrossings):
			send_bins[nx]['sum'] += send_bins[nx][k]
	for nx in neighbors: 
		maxsum = max(maxsum, send_bins[nx]['sum'])
	left3 = maxsum
	left = max(left1, left2, left3)
	#print p, left1, left2,left3
	# All reduce to determine if anyone still has has 
	x = mpi_allreduce(left, cart_comm, data_size=8, op="max")
	allreducecount += 1
	#if p == 0:
	#	print p, ": outer allreducecount, batchsize, r_left, left, finished, sum_finished ", allreducecount, batch_actual, ph_r_left, ph_left, ph_finished, x#, send_bins
	#raw_input()
	if x <= 1: # to allow for rounding errors 
		return mpi_wtime(cart_comm)
	

  
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
  # 'periodic' to be false; i.e., no wrap around) NO in V17 we are trying wrap around to 
  #make the warp speedup easier
  #cart_comm = mpi_cart_create(mpi_comm_world, (p_x, p_y, p_z), (False,)*3)
  cart_comm = mpi_cart_create(mpi_comm_world, (p_x, p_y, p_z), (True,)*3)

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
  #print("%d: neighbors: %r" % (p, neighbors))

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
  dt_local = dt	   # keep a local copy of dt (global is constant)
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

	#print("rank %d finished %d timesteps at time %f (last iteration: total_time=%f, compute_time=%f)" %
	#	  (p, timestep_i, host.engine.now, timestep_i_time, timestep_i_compute_time))

  # print the final results
  #print("rank %d processed %d photons in %d timesteps: total_time=%f, compute_time=%f" % \
  #		(p, photons_per_rank, timestep_i, host.engine.now, total_compute_time))
  final_time = host.hpcsim_dict["results"]["time"]
  if host.engine.now > final_time:
	host.hpcsim_dict["results"]["time"] = host.engine.now
	#print p, ": Setting a new final time . . ."
  # finalize mpi and the simulation
  mpi_finalize(mpi_comm_world)




###############################################################################
###############################################################################
# argv[1] can be one of the following:
# cielo, hopper, titan (gemini)
# sequoia, mira, vulcan (bluegeneq)
# darter, edison (aries)
# stampede, moonlight, mustang (fattree)
# crossbar, bypass
mdict = {}
# gemini
mdict['cielo'] = { "intercon_type" : "Gemini", "torus" : configs.cielo_intercon, "mpiopt" : configs.gemini_mpiopt, "cores_per_node" : 16}
mdict['hopper'] = { "intercon_type" : "Gemini", "torus" : configs.hopper_intercon, "mpiopt" : configs.gemini_mpiopt, "cores_per_node" : 24 }
mdict['titan'] = { "intercon_type" : "Gemini", "torus" : configs.titan_intercon, "mpiopt" : configs.gemini_mpiopt, "cores_per_node" : 16}
# bluegeneq
mdict['sequoia'] = { "intercon_type" : "BlueGeneQ", "torus" : configs.sequoia_intercon, "mpiopt" : configs.bluegeneq_mpiopt, "cores_per_node" : 16 }
mdict['mira'] = { "intercon_type" : "BlueGeneQ", "torus" : configs.mira_intercon, "mpiopt" : configs.bluegeneq_mpiopt, "cores_per_node" : 16 }
mdict['vulcan'] = { "intercon_type" : "BlueGeneQ", "torus" : configs.vulcan_intercon, "mpiopt" : configs.bluegeneq_mpiopt, "cores_per_node" : 16 }
# aries
mdict['darter'] = { "intercon_type" : "Aries", "dragonfly" : configs.darter_intercon, "mpiopt" : configs.aries_mpiopt, "cores_per_node" : 16 }
mdict['edison'] = { "intercon_type" : "Aries", "dragonfly" : configs.edison_intercon, "mpiopt" : configs.aries_mpiopt, "cores_per_node" : 24 }
# fattree
mdict['stampede'] = { "intercon_type" : "Fattree", "fattree" : configs.stampede_intercon, "mpiopt" : configs.infiniband_mpiopt, "cores_per_node" : 16 }
mdict['moonlight'] = { "intercon_type" : "Fattree", "fattree" : configs.moonlight_intercon, "mpiopt" : configs.infiniband_mpiopt, "cores_per_node" : 16 }
mdict['mustang'] = { "intercon_type" : "Fattree", "fattree" : configs.mustang_intercon, "mpiopt" : configs.infiniband_mpiopt, "cores_per_node" : 24 }
# crossbar
mdict['crossbar'] = { "intercon_type" : "Crossbar", "crossbar" : { "nhosts" : int((total_ranks+15)/16) }, "cores_per_node" : 16 }
# bypass
mdict['bypass'] = { "intercon_type" : "Bypass", "bypass" : { "nhosts" : int((total_ranks+15)/16) }, "cores_per_node" : 16 }

resultdict = {"time": 0.0}
modeldict = {
  "load_libraries": set(["mpi"]),
  #"mpi_path"   : "/usr/local/lib/libmpich.dylib",
  "host_type" : "MLIntelNode",
  #"debug_options" : set([]),
  "debug_options" : set(["hpcsim"]),
  #"debug_options" : set(["hpcsim", "fattree", "mpi"])
  "results" : resultdict
}

modeldict['default_configs'] = {}
#modeldict['default_configs']['mpi_call_time'] = float(argv[9])
#modeldict['default_configs']['intercon_link_delay'] = float(argv[10]) # default 0.7e-6
#modeldict['default_configs']['intercon_bandwidth'] = float(argv[11]) # default 40e9
#modeldict['default_configs']['mem_delay'] = float(argv[12]) # default 1e-7
#modeldict['default_configs']['mem_bandwidth'] = float(argv[13]) # default 5e11
#modeldict['default_configs']['mpi_max_pktsz'] = 0 # use this if no mpi segmentation 
modeldict.update(mdict[argv[1]])

# create the cluster and run simulation
cluster = Cluster(modeldict, model_name="imcsim", sim_time=1e12, use_mpi=False)

total_hosts = cluster.intercon.num_hosts()
cores_per_host = modeldict['cores_per_node'] # supposedly accessible from cluster
total_cores = total_hosts*cores_per_host
if total_ranks > total_cores:
  print("ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run application (p=%d)" % \
  (total_hosts, cores_per_host, total_cores, total_ranks))
  sys.exit(-1)

hostmap = [(y/cores_per_host)%total_hosts for y in range(total_ranks)]
cluster.start_mpi(hostmap, imc_main)
cluster.run()
print "RUN JUST COMPLETED WITH PARAMETERS:", argv, resultdict
exit([argv, resultdict["time"]])
