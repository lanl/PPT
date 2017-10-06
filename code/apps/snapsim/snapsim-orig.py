"""
 SNAPSim, an application simulator of the SNAP radiation transport miniapp.
 
 Date: November 12, 2015
"""
# To run: python [SNAPSIM py file name]
# Output: To screen and to snapsim.0.out

version = "orig-2015.11.20.1"

# May need to set environment variable PYTHONPATH
# Add to the path .../[Simian], ..../[Simian]/SimianPie
# Need greenlet and may need to add a path
# setenv PYTHONPATH /Users/rzerr/working/ldrd/Simian-v1.4.6:/Users/rzerr/working/ldrd/Simian-v1.4.6/SimianPie:.

# MPI Example: mpirun -np 3 python2.7 SNAPOSSimV7.py 

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../../simian/simian-master/SimianPie')
path.append('../../hardware')
path.append('../../hardware/interconnect')

sys.dont_write_bytecode = True

#import simian
from simian import Simian 
import clusters
import nodes

# imports for this specific application
import math
from copy import deepcopy

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

############ Additional variables ###########################
SYNC_INTERVAL = 1       # number of timesteps a chunk process waits before 
                        # starting to get the thread efficiency correct, set to
                        #   zero if not worried about threads
PSI_SIZE_PER_CELL = 1   # Number of data bytes per cell, determines size of MPI
                        #   message

STAT_FREQUENCY = 1      # How often (in number of chunk processes completed)
                        #   should statistics be kept

# TODO: Determine if all const variables should be defined here as there is
#   only one master.


########################
# 0. Initialization stuff

simName, startTime, endTime, minDelay, useMPI = \
  "snapsim", 0.0, 1000000.0, 0.000000001, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)


###############################################################################
###############################################################################


# SNAPSim
###############################################################################

def SNAPSim(this):
  """
  Simulates SNAP
  This is the function that gets executes as main process. Only one such
    function exists.
  """

  # Parameters
  #######################

  nsteps = 5   #10   # number of time-steps to be simulated
  nmom = 4     #4    # anisotropic scattering order
  nx = 2             # discretization units (aka cells) in x-dimension
  ny = 2             # discretization units (aka cells) in y-dimension
  nz = 2             # discretization units (aka cells) in z-dimension
  ichunk = 2         # number of cells per chunk in x direction
  jchunk = 2         # number of cells per chunk in y direction
  kchunk = 2         # number of cells per chunk in z direction
  ng = 30            # number of energy groups
  nang = 24          # number of discrete ordinates per octant
  iitm = 2           # number of inner iterations before convergence
  oitm = 2           # number of outer iterations before convergence

  # Initializations
  #######################

  entity = this.entity
  entity.out.write("SNAPSim Master at Initialization\n"
                   "Version = " + version + "\n\n")

  # assume 3D problem always
  noct = 8           # number of octants
  ncor = 4           # number of starting corners
  cmom = nmom**2     # number of flux moments

  # number of spatial chunks in each dimension
  # note in SNAP nychunks-->npey and nzchunks--npez
  nxchunks = int(math.ceil(float(nx)/ichunk))
  nychunks = int(math.ceil(float(ny)/jchunk))
  nzchunks = int(math.ceil(float(nz)/kchunk))

  # determine number of cores on each node and the total number of cores
  # use cores[node_id] = number of cores on node_id
  cores = {}
  total_cores = 0

  for node_id in xrange(cluster.num_nodes):
    # Send our requests
    entity.reqService(minDelay, "SNAP_CN_Handler", ['send_num_cores'], "Node",
                      node_id)
  response_count = 0
  while response_count < cluster.num_nodes:
    this.hibernate()
    while entity.msg_q != []:
      msg = entity.msg_q.pop()
      if msg[0] == 'num_cores':
        response_count += 1
        node_id, num_cores =  msg[1], msg[2]
        cores[node_id] = num_cores
        total_cores += num_cores
  entity.update_hw_stats({'node_count': cluster.num_nodes,
                          'core_count': total_cores})

  # print the computational resources used
  print "\nNodes =", cluster.num_nodes, "\nCores =", total_cores, "\n" \
        "Cores per node =", cores, "\n\n", \
        "Begin SNAPSim loops\n--------------------------------"

  #################
  # Main loop
  #################
  for timestep_i in range(nsteps):
    entity.update_sw_stats({'timestep_i':timestep_i})
    print "Timestep", timestep_i
    for outer_i in range(oitm):
      entity.update_sw_stats({'outer_i':outer_i})
      print "  Outer", outer_i
      for inner_i in range(iitm):
        entity.update_sw_stats({'inner_i':inner_i})
        print "    Inner", inner_i
        # print "in inner loop, timestep, outer, inner, octants, :", \
        # timestep_i, outer_i, inner_i, noct

        status = {'wait':[], 'started':[], 'done':[], 'intranode_sent':[],
                  'internode_sent':[]}
        # status['wait', 'started', 'done', 'intranode_send_done',
        #        'internode_send_done'] = a list of chunk_id s

        assignment = {}   # assignment[chunk_id] = (node_id, core_id)
        dependents = {}   # dependents[chunk_id] = [chunk_1, chunk_2, ...]
                          #   which chunks depends on me
        required = {}     # required[chunk_id] = [chunk_1, chunk_2, ...]
                          #   which chunks do I depend on

        ####
        #### 1. We first fill the creating chunk processes list over all
        ####    octants, groups, spatial chunks
        ####
        for cor_i in range(ncor):
          for group_i in range(ng):

            global_core_id = -1
            for z_i in range(nzchunks):
              for y_i in range(nychunks):

                global_core_id += 1
                if global_core_id > total_cores:
                  print "Simulated ranks outnumber simulated cores!!"
                  exit()

                for id_i in range(2):
                  # The corner helps compute the octant. This works for now
                  # but will need changed with simultaneous corner
                  octant_i = 2*cor_i + id_i

                  for x_i in range(nxchunks):
                    # compute node and core ids
                    lo_core_id = 0
                    hi_core_id = -1
                    for node_id in xrange(cluster.num_nodes):
                      lo_core_id = hi_core_id + 1
                      hi_core_id += cores[node_id]
                      if hi_core_id >= global_core_id:
                        break
                    local_core_id = global_core_id - lo_core_id

                    # print group_i, octant_i, z_i, y_i, x_i, node_id, \
                    #   local_core_id
                    chunk_id = (group_i, octant_i, z_i, y_i, x_i)

                    assignment[chunk_id] = (node_id, local_core_id)
                    status['wait'].append(chunk_id)
                    #print "      chunk_id", chunk_id, \
                    #      "assignment", assignment[chunk_id]

                    required[chunk_id] = compute_dependencies(chunk_id, True,
                                         nzchunks, nychunks, nxchunks, ng,
                                         id_i)
                    #print "      chunk_id", chunk_id, \
                    #      "required", required[chunk_id]

                    dependents[chunk_id] = compute_dependencies(chunk_id, False,
                                             nzchunks, nychunks, nxchunks, ng,
                                             id_i)
                    #print "      chunk_id", chunk_id, \
                    #      "dependents", dependents[chunk_id]

        #tot_req=0
        #tot_dep=0
        #for chunk_id in required:
        #  tot_req = tot_req + len(required[chunk_id])
        #for chunk_id in dependents:
        #  tot_dep = tot_dep + len(dependents[chunk_id])
        #print "total requirements length =", tot_req
        #print "total dependents length =", tot_dep

        #exit()

        #print " Waiting\n", status['wait']
        entity.out.write("Time: " + str(simianEngine.now) + ":  " +
                         entity.name + " " + str(entity.num) + " SNAPSim: " +
                         "number of waiting chunk processes after init: " +
                         str(len(status['wait']))+"\n")

        entity.out.write("Reminder re chunk_id:\t (group_i, octant_i, z_i, " +
                         "y_i, x_i)  \n")

        #####
        ##### 2. Schedule all chunk processes
        #####

        #####   2.1 First create chunk processes that have no dependencies
        chunk_stats = {}
        for item in status:
          chunk_stats[item] = len(status[item])      # how many chunk processes
                                                     #   in different status
        entity.update_sw_stats(chunk_stats)

        for chunk_id in required:
          # chunk does not depend on any other:
          if len(required[chunk_id]) == 0:
            (target_node_id, target_core_id) = assignment[chunk_id]
            msg = ['start', chunk_id, target_core_id, ichunk, jchunk, kchunk,
                   nang, cmom]
#            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
#                             entity.name + " " + str(entity.num) + 
#                             " SNAPSim: Starting chunk process \t " + 
#                             str(chunk_id) + " on (node, core)=(" + 
#                             str(assignment[chunk_id]) + ")\n")
            status['wait'].remove(chunk_id)
            status['started'].append(chunk_id)
            # print "Initial Chunk Process started: ", chunk_id
            entity.reqService(
                              minDelay,            # add min-delay
                              "SNAP_CN_Handler",   # Handler/Service name
                              msg,                 # this is the data we send
                              "Node",              # Target entity type
                              target_node_id       # Target entity id
                             )

        chunk_stats = {}
        for item in status:
          # how many chunk processes in different status
          chunk_stats[item] = len(status[item])
        entity.update_sw_stats(chunk_stats)

        #####   2.2 We wait for all chunks to be executed
        # some chunks exist that are still waiting or haven't finished yet
        while status['wait'] or status['started']:

          # We update stats
          chunk_stats = {}
          for item in status:
            chunk_stats[item] = len(status[item])
          entity.update_sw_stats(chunk_stats)

          # We hibernate until woken up by the master handler who will have put
          #   something into the q
          # entity.out.write("Time: " + str(simianEngine.now) +
          #                  ", before hibernation with scp " +
          #                  str(started_chunk_processes) +
          #                  ", and wcp " + str(waiting_chunk_processes) + "\n")
          this.hibernate()
          # entity.out.write("Time: " + str(simianEngine.now) +
          #                  ", after hibernation with msg_q " +
          #                  str(entity.msg_q) + "\n")

          # We check the q
          while entity.msg_q != []:
            # print simianEngine.now, ": Msg q ", entity.msg_q
            msg = entity.msg_q.pop()
            # Message looks as follows: ['chunk_done_intranode_send', chunk_id]
            msg_type = msg[0]
            chunk_id = msg[1]
            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                             entity.name + " " + str(entity.num) +
                             " SNAPSim: received msg \t\t" + str(msg[0]) +
                             str(msg[1]) + "\n")

            # Update status
            if msg_type == 'chunk_done':
              # print simianEngine.now, ": chunk_done received: ", chunk_id, \
              #   entity.msg_q
              status['started'].remove(chunk_id)
              status['done'].append(chunk_id)
              entity.update_hw_stats(msg[2])    #msg[2] contains the stats info
            else:
              bytes_sent = msg[2]
              if msg_type == 'chunk_done_intranode_send':
                status['intranode_sent'].append(chunk_id)
              if msg_type == 'chunk_done_internode_send':
                status['internode_sent'].append(chunk_id)

            # Check if dependents can be run and do so
            for c_chunk in dependents[chunk_id]:
              # Set the candidate chunk attributes
              (c_node, c_core) = assignment[c_chunk]
              (chunk_node, chunk_core) = assignment[chunk_id]
              # Check if the candidate is ready 
              if  (msg_type == 'chunk_done' and \
                    (c_node, c_core) == (chunk_node, chunk_core)) \
                 or \
                  (msg_type == 'chunk_done_intranode_send' and \
                    c_node == chunk_node) \
                 or \
                  (msg_type == 'chunk_done_internode_send'):
                # Is the candidate waiting
                if c_chunk in status['wait']:
                  # Look at the candidates requirements, start with all good
                  chunk_ok = True
                  for r_chunk in required[c_chunk]:
                    # Set attribute for each requirment
                    (r_node, r_core) = assignment[r_chunk]
                    # Check ALL requirements are met
                    if  (r_chunk in status['wait']) \
                       or \
                        (r_chunk in status['started']) \
                       or \
                        not( (r_chunk in status['done'] and \
                               (c_node, c_core) == (r_node, r_core)) \
                            or \
                             (r_chunk in status['intranode_sent'] and \
                               c_node == r_node) \
                            or \
                             (r_chunk in status['internode_sent']) ):
                      # Not all other required chunks done for c_chunk
                      chunk_ok = False

                  # We can start candidate chunk c_chunk
                  if chunk_ok:
                    if msg_type =='chunk_done_internode_send':
                      entity.update_hw_stats({'inter_bytes': bytes_sent})
                    if msg_type =='chunk_done_intranode_send':
                      entity.update_hw_stats({'intra_bytes': bytes_sent})
                    (target_node_id, target_core_id) = assignment[c_chunk]
                    msg = ['start', c_chunk, target_core_id, ichunk, jchunk,
                           kchunk, nang, cmom]
                    #entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                    #                 entity.name + " " + str(entity.num) +
                    #                 " SNAPSim: Starting chunk process \t " +
                    #                 str(c_chunk) + " on (node, core)=(" +
                    #                 str(assignment[c_chunk]) + ")\n")
                    #print simianEngine.now, ":Starting chunk", c_chunk
                    status['wait'].remove(c_chunk)
                    status['started'].append(c_chunk) 
                    entity.reqService(minDelay, "SNAP_CN_Handler", msg,
                                      "Node", target_node_id)

        # end inner for
        entity.update_sw_stats({'inner_i':inner_i})
        # print "End of inner_i", inner_i
        entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                         entity.name + " " + str(entity.num) +
                         " SNAPSim: End of inner loop " + str(inner_i) + "\n")

      # end outer for
      entity.update_sw_stats({'outer_i':outer_i})
      entity.out.write("Time: " + str(simianEngine.now) + ":\t " + entity.name +
                       " " + str(entity.num) + " SNAPSim: End of outer loop " +
                       str(outer_i) + "\n")

      ########
      # Compute cross-group interactions
      #print "Computing cross groups with outer_i: ", outer_i
      ########
      for node_id in xrange(cluster.num_nodes):
        # cells_count[core_id] = cells_on_core
        cells_count = {}
        for core_id in xrange(cores[node_id]):
          # How many chunks were computed on this core
          chunk_count = 0
          for chunk_id in assignment:
            # inefficient, but rare, so let's keep it for now
            if (node_id, core_id) == assignment[chunk_id]:
              chunk_count += 1
          cells_on_core = chunk_count*ichunk*jchunk*kchunk
          cells_count[core_id] = cells_on_core
        #print "Sending cross compute request to node,core ", cells_count
        entity.reqService(minDelay, "SNAP_CN_Handler",
                          ['cross_compute', cells_count, ng],
                          "Node", node_id)
      response_count = 0
      while response_count < cluster.num_nodes:
        this.hibernate()
        while entity.msg_q != []:
          msg = entity.msg_q.pop()
          if msg[0] == 'cross_compute_done':
            response_count += 1
            # don't know what to do with this, will use for stats later
            node_id =  msg[1]
            # msg[2] are stats
            entity.update_hw_stats(msg[2])
      #print "Finished cross_compute"

    # end of timestep
    entity.out.write("Time: " + str(simianEngine.now) + ":\t " + entity.name +
                     " " + str(entity.num) + " SNAPSim: End of timestep " +
                     str(timestep_i) + "\n")

  entity.out.write("End of computation, now waiting for potential " +
                   "internode_send late arrival wake_up notifications to " +
                   "ignore \n")

  count1 =0
  print "Almost done"
  entity.print_stats()
  print "Almost done2"
  while True:
    this.hibernate()
    count1 +=1
    entity.out.write("Hibernate count: " + str(count1) + "\n")


###############################################################################
###############################################################################


# compute_dependencies
###############################################################################

def compute_dependencies(chunk_id, requiredFlag, nzc, nyc, nxc, ng, idir):
  """
   If requiredFlag==true, returns the chunks required before specified
     chunk_id can be processed.
   If requiredFlag==false, returns the chunks that immediately depend
     on specified chunk_id.
  
   Incoming: chunk_id=(group, octant, z, y, x)
             requiredFlag = true/false
             nxc, nyc, nzc = number of chunks in x, y, and z, respectively
   Outgoing: dep_list and req_list
  """

  (group, octant, z, y, x) = chunk_id
  dep_list = []
  req_list = []

  # The group stuff isn't set up correctly for threads; needs fixed

  # Set the requirements/dependents at x==0, where sweep returns in
  #   positive x-direction for same group
  if x == 0:
    if idir == 1:
      req_list.append((group, octant-1, z, y, x))
    if idir == 0:
      dep_list.append((group, octant+1, z, y, x))

  # Set up the requirements/dependents at x==nxc-1, where sweep starts in
  #   negative x-direction for next group
  if x == nxc-1:
    if idir == 0 and group > 0:
      req_list.append((group-1, octant+1, z, y, x))
    if idir == 1 and group < ng-1:
      dep_list.append((group+1, octant-1, z, y, x))

  # Set up the requirements/dependents for switching starting corners, where
  #   sweeps start anew from group 0
  # Corner 0: excluded: no requirements/dependents in terms of corner switch
  # Corner 1
  if z == nzc-1 and y == 0 and x == nxc-1:
    if group == 0 and octant == 2:
      req_list.append((ng-1, octant-1, z, y, x))
    if group == ng-1 and octant == 1:
      dep_list.append((0, octant+1, z, y, x))
  # Corner 2
  if z == 0 and y == nyc-1 and x == nxc-1:
    if group == 0 and octant == 4:
      req_list.append((ng-1, octant-1, z, y, x))
    if group == ng-1 and octant == 3:
      dep_list.append((0, octant+1, z, y, x))
  # Corner 2
  if z == 0 and y == 0 and x == nxc-1:
    if group == 0 and octant == 6:
      req_list.append((ng-1, octant-1, z, y, x))
    if group == ng-1 and octant == 5:
      dep_list.append((0, octant+1, z, y, x))

  # Set spatial chunk requirements/dependents for a given octant and group
  if octant == 0:
    if x < nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y < nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z < nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 1:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y < nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z < nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 2:
    if x < nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z < nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 3:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z < nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z > 0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 4:
    if x < nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y < nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  if octant == 5:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y < nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y > 0:
      dep_list.append((group, octant, z, y-1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  if octant == 6:
    if x < nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x > 0:
      dep_list.append((group, octant, z, y, x-1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  if octant == 7:
    if x > 0:
      req_list.append((group, octant, z, y, x-1))
    if x < nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y > 0:
      req_list.append((group, octant, z, y-1, x))
    if y < nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z > 0:
      req_list.append((group, octant, z-1, y, x))
    if z < nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  # Return the requested list type
  if requiredFlag:
    return req_list
  else:
    return dep_list


###############################################################################
###############################################################################


# ChunkProcess
###############################################################################

def ChunkProcess(this, arg, *args):
  # Again args is not used, but Simian attempts to send it more arguments
  """
  Individual process for an update of a chunk of cells
  """
  # 0. Initializations
  # Here's what is in arg msg = ['start', chunk_id, local_core_id, ichunk,
  #                              jchunk, kchunk, nang, cmom]
  (chunk_id, core_id, ichunk, jchunk, kchunk, nang, cmom) = \
    (arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7])

  node = this.entity
  entity = this.entity
  core = node.cores[core_id]

  # node.out.write("Time: " + str(node.engine.now) + 
  #                " Chunk Process strated with chunk_id " + str(chunk_id) +
  #                " on core id " + str(core_id) + "\n")
	
  # 1. Advance a small waiting time so other synchronized threads get on board
  #    and don't screw up the thread_efficiency
  this.sleep(core.time_compute([['CPU', SYNC_INTERVAL]]))

  # 2.Put together task list to send to hardware 
  # TODO: verify with Joe, also, memory usage 
  num_index_vars = 10              # number of index variables
                                   # decent enough guess for now (12/23 rjz)

  # Add up all the float variables. Guess for now.
#  num_float_vars = 6 + \
#                   (9 + cmom + jchunk*kchunk + 3*ichunk*kchunk + \
#                    3*ichunk*jchunk + 3*ichunk*jchunk*kchunk)*nang + \
#                   (5 + 2*cmom)*ichunk*jchunk*kchunk
#  num_float_vars = 10

#  Per cell basis
  num_float_vars = 11 + 2*cmom + (19 + cmom)*nang


  # i32 loads + 2x i64 loads
#  index_loads = 15 + 7*ichunk*jchunk + 6*ichunk*kchunk + jchunk*kchunk + \
#                11*ichunk*jchunk*kchunk + 16*ichunk*jchunk*kchunk
#  index_loads = 10

# Per cell basis
  index_loads = 55

  # Normal float loads plus vector loads
#  float_loads = (7*nang + 9 + 2*(cmom-1)*nang + 2*(cmom-1)) * \
#                ichunk*jchunk*kchunk + \
#                (19 + 2*(cmom-1))*nang*ichunk*jchunk*kchunk
#  float_loads = num_float_vars

# Per cell basis
  float_loads = 26*nang + 9 + 4*(cmom-1)*nang + 2*(cmom-1)

  avg_dist = 1           # average distance in arrays between accessed elements 
  avg_reuse_dist = 1    # avg number of unique loads between two consecutive
                         #   accesses of the same element (ie use different
                         #   weights for int and float loads)
  stdev_reuse_dist = 1    # stddev number of unique loads between two
                           #   consecutive accesses of the same element

#  int_alu_ops = ((89*nang + 35*(nang%2) + 205 + \
#                (15*nang + 4*(nang%2) + 23)*(cmom-1))*ichunk*jchunk*kchunk + \
#                15*ichunk*kchunk + 47*jchunk*kchunk + 13*ichunk*jchunk + 41)/2
#  int_alu_ops = 5
# Per cell basis
  int_alu_ops = 89*nang + 35*(nang%2) + 321 + \
                (15*nang + 4*(nang%2) + 23)*(cmom-1)


#  float_alu_ops = ((7*nang + 2*(cmom-1)*nang - (cmom-1) - 2) * \
#                  ichunk*jchunk*kchunk + 4)
#  float_alu_ops=5
# Per cell basis
  float_alu_ops = 7*nang + 2*(cmom-1)*nang - (cmom-1) + 2


#  float_vector_ops = (((21 + 2*(cmom-1))*nang + 2*(cmom-1) + 10) * \
#                     ichunk*jchunk*kchunk)      # Total vec_ops. Ignore VL
#  float_vector_ops=5
# Per cell basis
  float_vector_ops = (21 + 2*(cmom-1))*nang + 2*(cmom-1) + 10

  tasklist_per_interior_cell = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                                 ['VECTOR', float_vector_ops, nang],
                                 ['MEM_ACCESS', num_index_vars, num_float_vars,
                                  avg_dist, avg_reuse_dist, stdev_reuse_dist,
                                  index_loads, float_loads, False] ]
  # ['alloc', 1024]\ TODO: put memory in at some point. Probably at the very
  #   very beginning

  tasklist_per_border_cell = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                               ['VECTOR', float_vector_ops, nang],
                               ['MEM_ACCESS', num_index_vars, num_float_vars,
                                avg_dist, avg_reuse_dist, stdev_reuse_dist,
                                index_loads, float_loads, True] ]
  # Difference in tasklists is the Init_file set to True. In border cells, more
  #   float elements will need to be loaded from main

  # 3. Loop over cells in chunk to compute the time that is spent for the
  #    entire chunk, then perform a sleep (premature optimization ...)
  time = 0

  time_per_border_cell, stats_border = \
    core.time_compute(tasklist_per_border_cell, True)

  time_per_interior_cell, stats_interior = \
    core.time_compute(tasklist_per_interior_cell, True)

  # We only want to put out full stats once
  if chunk_id == (0,0,0,0,0):
    entity.out.write("Time: " + str(simianEngine.now) + ":\t " + entity.name +
                     " " + str(entity.num) +
                     " ChunkProcess: Stats from single border cell call: \n")
    for statitem in stats_interior:
      entity.out.write(str(statitem) + " \t= " +
                       str(stats_interior[statitem]) + "\n")

  num_border = 0
  num_interior = 0
  for z in range(kchunk):
    for y in range(jchunk):
      for x in range(ichunk):
        if (z==0) or (y==0) or (x==0):
          num_border += 1
          time += time_per_border_cell
        else:
          time += time_per_interior_cell
          num_interior += 1

  # Update stats from cell level to chunk into stats dictionary
  # TODO: note this is wrong for thread-efficiency, will deal with this later
  threadeff = stats_interior['Thread Efficiency']

  for statsitem in stats_border:
    stats_border[statsitem] *= num_border

  for statsitem in stats_interior:
    stats_interior[statsitem] *= num_interior

  stats = {}

  for statitem in stats_interior and stats_border:
    stats[statitem] = stats_interior[statitem] + stats_border[statitem]

  stats['Thread Efficiency'] = threadeff          # bad hack around this issue
  this.sleep(time)

  # 4. Send done info to let SNAP_CN_Handler know that this chunk is done
  bytes_sent = ichunk*math.sqrt(jchunk*kchunk)*PSI_SIZE_PER_CELL 
  internode_time = max(minDelay, core.time_compute([['internode', bytes_sent]]))
  intranode_time = max(minDelay, core.time_compute([['intranode', bytes_sent]]))
  # print ("Time: " + str(simianEngine.now) + ":\t " + entity.name + " " +
  #        str(entity.num) + " ChunkProcess: \t " + str(chunk_id) +
  #        " completed on core id " + str(core_id) + "; execution time: " +
  #        str(time) + "\n")
  node.reqService(
                  0,                                    # zero delay
                  "SNAP_CN_Handler",                    # Handler/Service name
                  ['chunk_done', chunk_id, core_id,
                   bytes_sent, internode_time,
                   intranode_time, stats]       # this is the data that we send
                 )


###############################################################################
###############################################################################


# SNAP_CN_HANDLER
###############################################################################

def SNAP_CN_Handler(self, msg, *args):
  # *args is only given because Simian complains otherwise
  # This method will be attached to the Node entity
  #   self thus refers to the node. A bit more trickery than we like to have,
  #     but it does allow us to keep hardware and software separated.
  #
  # This handler also manages the active threads, perhaps in slight violation
  #   of the coveted hw/sw separation, to be corrected at some later point,
  #   here the old os from simx is missing a little
  #
  # What was sent is this:
  #  msg = ['start' chunk_id, local_core_id, self.ichunk, self.jchunk,
  #         self.kchunk, self.nang, self.cmom]
  # We just start a process on the resource local_core_id == msg[2]
  # OR it is the kill message
  #

  # print("SNAP_CN_Handler Time: " + str(self.engine.now) + ", Msg " +
  #       str(msg) + " received")
  # print "local core id, active threads:", local_core_id, core.activethreads

  # Type 1: Counting cores
  if msg[0] == 'send_num_cores':
    self.reqService(minDelay, "Master_Handler",
                    ['num_cores', self.num, self.num_cores], "Master", 1)

  # Type 2: Start a process
  elif msg[0] == 'start':
    chunk_id = msg[1]
    local_core_id = msg[2]
    core = self.cores[local_core_id]
    if core.activethreads < core.maxthreads:
      core.activethreads+= 1
      self.createProcess("chunk_proc"+str(chunk_id), ChunkProcess)
      self.startProcess("chunk_proc"+str(chunk_id), msg, msg[2])
    else:
      core.waiting_processes.insert(0, (chunk_id, msg, local_core_id))

  # Type 3: Process done, pass on to next
  elif msg[0] == 'chunk_done':
    chunk_id, local_core_id, bytes_sent = msg[1], msg[2], msg[3]
    internode_time, intranode_time, stats = msg[4], msg[5], msg[6]
    # print self.engine.now, ":A CN_Handler with chunk_done msg ", chunk_id
    core = self.cores[local_core_id]
    # Update thread count, kill process and check if processes are still
    #   waiting to be executed on that core
    core.activethreads -= 1
    self.killProcess("chunk_proc"+str(chunk_id))
    # self.out.write("Time: " + str(self.engine.now) + ", killed proc " +
    #                str(chunk_id) + " " + str(core.waiting_processes) + "\n")
    core.waiting_processes
    if core.waiting_processes != []:
      # print self.engine.now, ":A1 CN_Handler with wait procs ", \
      #   core.waiting_processes
      (new_chunk, msg, local_core_id) = core.waiting_processes.pop()
      core.activethreads += 1
      self.createProcess("chunk_proc"+str(new_chunk), ChunkProcess) 
      self.startProcess("chunk_proc"+str(new_chunk), msg, local_core_id)
    # Forward done info to SNAP MasterHandler with min delay
    self.reqService(
                    simianEngine.minDelay,               # minDelay
                    "Master_Handler",                    # Handler/Service name
                    ['chunk_done', chunk_id, stats],     # data that we send
                    'Master', 1                          # destination entity
                   )
    # Forward done info to SNAP MasterHandler after communication delay (to be
    #   replaced with call to MPI model in the future)
    self.reqService(intranode_time, "Master_Handler",
                    ['chunk_done_intranode_send', chunk_id, bytes_sent],
                    "Master", 1)
    self.reqService(internode_time, "Master_Handler",
                    ['chunk_done_internode_send', chunk_id, bytes_sent],
                    "Master", 1)

  # Type 4: Cross compute (scatter source computation)
  elif msg[0] == 'cross_compute':
    cells_count = msg[1]
    ng = msg[2]
    maxtime = 0.0
    nodestats = {}

    for core_id in cells_count:
      num_elements = 5
      core = self.cores[core_id]

      cross_compute_tasklist = \
        [['fALU', cells_count[core_id]*num_elements*ng**2]]

      (time, stats) = core.time_compute(cross_compute_tasklist, True)
      maxtime = max(maxtime, time)

      #Sum up statistics across all cores on node
      for item in stats:
        if item in nodestats:
          nodestats[item] += stats[item]
        else:
          nodestats[item] = stats[item]

    time = max(time, minDelay)
    # Thus only slowest core determines the elapse time, we report to master
    #   with delay time
    self.reqService(time, "Master_Handler",
                    ['cross_compute_done', self.num, stats], "Master", 1)


###############################################################################
###############################################################################


# MasterNode
###############################################################################

class MasterNode(simianEngine.Entity):

  def __init__(self, baseInfo, *args):
    super(MasterNode, self).__init__(baseInfo)
    self.msg_q = []
    self.hw_stats = {}           # Dictionary with stats at current sim time
    self.hw_stats_history = {}   # Dictionary of a dictionary of stats entries:
                                 #   eg hw_stats[time]['fALU'] = 100
    self.sw_stats_history = {}
    self.counter = 0             # Counter to steer how often statistics get
                                 #   put into history
    self.max_sw_stats_time = 0.0

    self.createProcess("master_proc", SNAPSim)
    self.startProcess("master_proc")                #no arguments

  def Master_Handler(self, msg, *args):
    # args is artificial
    # Put message into q and wake the master process up again
    self.msg_q.insert(0, msg)
    # print simianEngine.now, ": inserted msg", msg
    self.wakeProcess("master_proc")

  def update_hw_stats(self, stats, *args):
    self.counter += 1
    for item in stats:
      if item in self.hw_stats:
        self.hw_stats[item] += stats[item]
      else:
        self.hw_stats[item] = stats[item]
    if self.counter%STAT_FREQUENCY == 0:
      new_dict = deepcopy(self.hw_stats)
      self.hw_stats_history[self.engine.now] = new_dict

  def update_sw_stats(self, stats, *args):
    # Unlike hw, sw stats are absolute (not cumulative) and get printed out
    #   every time
    if self.engine.now not in self.sw_stats_history:
      self.sw_stats_history[self.engine.now] = {}
      self.max_sw_stats_time = self.engine.now
    for item in stats:
      self.sw_stats_history[self.engine.now][item] = stats[item] 

  def print_stats(self, *args):
    print "Hardware statistics:"
    for item in self.hw_stats:
      print item, "\t:\t", self.hw_stats[item]

    print "Software statistics:"
    for item in self.sw_stats_history[self.max_sw_stats_time]:
      print item, "\t\t:\t", self.sw_stats_history[self.max_sw_stats_time][item]


###############################################################################
###############################################################################


# "MAIN"
###############################################################################

# 1. Choose and instantiate the Cluster that we want to simulate 

print "\nSNAPSim run with Simian PDES Engine\nVersion =", version
#cluster = clusters.MiniTrinity(simianEngine)
cluster = clusters.SingleCielo(simianEngine)


# 2. Create snap sim master node, this automatically creates the master_proc
#     and Master_Handler

simianEngine.addEntity("Master", MasterNode, 1)


# 3. Create a Snap Handler Service on each node, note the handler just acts as
#    a pass-through for spawning chunk processes

simianEngine.attachService(nodes.Node, "SNAP_CN_Handler" , SNAP_CN_Handler)
	
# 4. Run simx

simianEngine.run()
simianEngine.exit()


###############################################################################
###############################################################################


# END
###############################################################################
