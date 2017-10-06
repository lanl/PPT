"""
 CWBJSim, an application simulator of the CWBJ radiation transport method.
 
 Date: February 19, 2016 Max Rosa (MR)
"""
# To run: python [CWBJSIM py file name]
# Output: To screen and to cwbjsim-orig.0.out

version = "orig-2016.06.06.1"

# May need to set environment variable PYTHONPATH
# Add to the path .../[Simian], ..../[Simian]/SimianPie
# Need greenlet and may need to add a path
# setenv PYTHONPATH /Users/rzerr/working/ldrd/Simian-v1.4.6:/Users/rzerr/working/ldrd/Simian-v1.4.6/SimianPie:.

# MPI Example: mpirun -np 3 python2.7 cwbjsim.py 

# Set up path  variables; PPT applications expect it in this fashion.
import sys
from sys import path
path.append('../../simian/simian-master/SimianPie')
path.append('../../hardware')
path.append('../../hardware/interconnect')
path.append('../../middleware/threading')

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
SYNC_INTERVAL = 1  #10 # number of timesteps a chunk process waits before 
                       # starting to get the thread efficiency correct, set to
                       #   zero if not worried about threads
PSI_SIZE = 8  # Number of data bytes per angular flux component (double), determines
              # size of MPI message #MR

STAT_FREQUENCY = 1  # How often (in number of chunk processes completed)
                    #   should statistics be kept

# TODO: Determine if all const variables should be defined here as there is
#   only one master.


########################
# 0. Initialization stuff

simName, startTime, endTime, minDelay, useMPI = \
  "cwbjsim-orig", 0.0, 1000000.0, 0.000000001, False #MR

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)


###############################################################################
###############################################################################


# CWBJSim #MR
###############################################################################

def CWBJSim(this): #MR
  """
  Simulates CWBJ #MR
  This is the function that gets executed as main process. Only one such
    function exists.
  """

#TODO: For the moment left in outer/inner logic loops
#  Might need to modify to reflect actual CWBJ logic

  # Parameters
  #######################

  nsteps = 1      # number of time-steps to be simulated
  nmom = 1        # anisotropic scattering order
  nx = 8          # discretization units (aka cells) in x-dimension >0
  ny = 8          # discretization units (aka cells) in y-dimension >=0
  nz = 0          # discretization units (aka cells) in z-dimension >=0
                  #   set nz=0 for 2-D; set nz=ny=0 for 1-D
  ichunk = 8      # number of cells per chunk in x direction >=1 <=nx
  jchunk = 8      # number of cells per chunk in y direction >=1 <=ny
  kchunk = 1      # number of cells per chunk in z direction >=1 <=nz
  ng = 1          # number of energy groups
  ng_bndls = 1    # number of energy group bumdles #MR: add  
  nang = 25      # number of discrete ordinates per octant(3D), quadrant(2D) 
                  # or hemisphere(1D). #MR: If q is the sn quadrature order:
                  # nang = q*(q+2)/8 for 2D/3D Level Symmetric quadrature
                  # nang = q/2 for 1D Level Symmetric quadrature
                  # nang = q*q/4 for 2D Square Chebychev-Legendre quadrature  
  iitm = 4        # number of inner iterations before convergence
  oitm = 1        # number of outer iterations before convergence
  method = 'CWBJ' # method flag ('CWBJ', 'CWBGS' and 'IPBJ') #MR: add

  # Initializations
  #######################

  entity = this.entity
  entity.out.write("CWBJSim Master at Initialization\n" #MR
                   "Version = " + version + "\n\n")

  # set 1D-3D problem subcells, octants and moments
  # MR: Need 3D potentially for comapring with SNAPSim
  # MR: Need 2D for comapring with Capsaicin
  # MR: No need for 1D at present  
  noct = 8           # number of octants in 3D
  cmom = nmom**2     # number of flux moments
  subcells = 1       # 1 (Hex), 2 (Prisms), 5, 6, 24 (Tets)
  if nz==0:
    noct = 4         # number of quadrants in 2D
    cmom = nmom*(nmom+1)/2
    subcells = 4     # 1 (Quad), 2, 4 (Triangles)
    if ny==0:
      #noct = 2      # number of hemispheres in 1D
      #cmom = nmom
      #subcells = 1  # 1 (Segment)
      print "Warning: At present 1D problems are not supported!"
      exit()

  # number of spatial chunks in each dimension
  # note in SNAP nychunks-->npey and nzchunks--npez
  nxchunks = int(math.ceil(float(nx)/ichunk))
  nychunks = int(math.ceil(float(ny)/jchunk))
  if nz > 0:
    nzchunks = int(math.ceil(float(nz)/kchunk))
  else:
    nzchunks = 1

  # determine number of cores on each node and the total number of cores
  # use cores[node_id] = number of cores on node_id
  cores = {}
  total_cores = 0

  for node_id in xrange(cluster.num_nodes):
    # Send our requests
    entity.reqService(minDelay, "CWBJ_CN_Handler", ['send_num_cores'], "Node",
                      node_id) #MR
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

  #MR: Add this check since at present we do not overload cores with chunks
  total_chunks = nxchunks*nychunks*nzchunks
  if total_chunks > total_cores:
    print "Warning: total_chunks > total_cores!\n" \
          "At present we do not overload cores with chunks,\n" \
          "hence total_chunks should not exceed total_cores."
    exit()

  # print the computational resources used
  print "\nNodes =", cluster.num_nodes, "\nCores =", total_cores, "\n" \
        "Cores per node =", cores, "\n\n", \
        "Begin CWBJSim loops\n--------------------------------"

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
#MR        dependents = {}   # dependents[chunk_id] = [chunk_1, chunk_2, ...]
#MR                          #   which chunks depends on me
#MR        required = {}     # required[chunk_id] = [chunk_1, chunk_2, ...]
#MR                          #   which chunks do I depend on

        chunk_list = [] #MR: in place of deps and reqd

        ####
        #### 1. We first fill the creating chunk processes list over all
        ####    spatial chunks
        ####
        # print "group_i, octant_i, z_i, y_i, x_i, node_id, local_core_id"

#MR        for group_i in range(ng):
#MR          for octant_i in range(noct):

#MR            global_core_id = -1
        for z_i in range(nzchunks):
          for y_i in range(nychunks):

#MR                global_core_id += 1
#MR                if global_core_id > total_cores:
#MR                  print "Simulated ranks outnumber simulated cores!!"
#MR                  exit()

            for x_i in range(nxchunks):
              # compute node and core ids
              global_core_id = x_i+nxchunks*(y_i+nychunks*z_i) % total_cores
              #MR: modify global_core_id formula but keep utilizing
              #MR: original machinery used to compute local_core_id
              #MR: under the assumption total_cores >= total_chunks (no overload) 
              cum_cores = 0
              prev_cum_cores = 0
              for node_id in xrange(cluster.num_nodes):
                prev_cum_cores = cum_cores
                cum_cores += cores[node_id]
                if cum_cores > global_core_id:
                  break
              local_core_id = global_core_id - prev_cum_cores

                       #MR: Machinery presently used in SNAPSim
#MR                    lo_core_id = 0
#MR                    hi_core_id = -1
#MR                    for node_id in xrange(cluster.num_nodes):
#MR                      lo_core_id = hi_core_id + 1
#MR                      hi_core_id += cores[node_id]
#MR                      if hi_core_id >= global_core_id:
#MR                        break
#MR                    local_core_id = global_core_id - lo_core_id

                  # print group_i, octant_i, z_i, y_i, x_i, node_id, \
                  #   local_core_id
#MR                  chunk_id = (group_i, octant_i, z_i, y_i, x_i)
              chunk_id = (z_i, y_i, x_i) #MR

              assignment[chunk_id] = (node_id, local_core_id)
              status['wait'].append(chunk_id)
              chunk_list.append(chunk_id) #MR: add
              #print "      chunk_id", chunk_id, \
              #      "assignment", assignment[chunk_id]

#MR                  required[chunk_id] = compute_dependencies(chunk_id, True,
#MR                                         nxchunks, nychunks, nzchunks, ng)

#MR                  dependents[chunk_id] = compute_dependencies(chunk_id, False,
#MR                                           nxchunks, nychunks, nzchunks, ng)

        #print " Waiting\n", status['wait']
        entity.out.write("Time: " + str(simianEngine.now) + ":  " +
                         entity.name + " " + str(entity.num) + " CWBJSim: " +
                         "number of waiting chunk processes after init: " +
                         str(len(status['wait']))+"\n") #MR

#MR        entity.out.write("Reminder re chunk_id:\t (group_i, octant_i, z_i, " +
#MR                         "y_i, x_i)  \n")

        entity.out.write("Reminder re chunk_id:\t (z_i, y_i, x_i)  \n") #MR

        #####
        ##### 2. Schedule all chunk processes
        #####

        #####   2.1 First we start chunk processes
        chunk_stats = {}
        for item in status:
          chunk_stats[item] = len(status[item])      # how many chunk processes
                                                     #   in different status
        entity.update_sw_stats(chunk_stats)

        for chunk_id in chunk_list:
          (target_node_id, target_core_id) = assignment[chunk_id]
          msg = ['start', chunk_id, target_core_id, ichunk, jchunk, kchunk,
                 nang, cmom, ng, ng_bndls, noct, subcells, method] #MR
          entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                           entity.name + " " + str(entity.num) +
                           " CWBJSim: Starting chunk process \t " +
                           str(chunk_id) + " on (node, core)=(" +
                           str(assignment[chunk_id]) + ")\n") #MR
          status['wait'].remove(chunk_id)
          status['started'].append(chunk_id)
          # print "Initial Chunk Process started: ", chunk_id
          entity.reqService(
                            minDelay,            # add min-delay
                            "CWBJ_CN_Handler",   # Handler/Service name #MR
                            msg,                 # this is the data we send
                            "Node",              # Target entity type
                            target_node_id       # Target entity id
                           )

        chunk_stats = {}
        for item in status:
          # how many chunk processes in different status
          chunk_stats[item] = len(status[item])
        entity.update_sw_stats(chunk_stats)

        #####   2.1 First create chunk processes that have no dependencies
#MR        chunk_stats = {}
#MR        for item in status:
#MR          chunk_stats[item] = len(status[item])      # how many chunk processes
                                                     #   in different status
#MR        entity.update_sw_stats(chunk_stats)

#MR        for chunk_id in required:
#MR          # chunk does not depend on any other:
#MR          if len(required[chunk_id]) == 0:
#MR            (target_node_id, target_core_id) = assignment[chunk_id]
#MR            msg = ['start', chunk_id, target_core_id, ichunk, jchunk, kchunk,
#MR                   nang, cmom]
#MR            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
#MR                             entity.name + " " + str(entity.num) + 
#MR                             " CWBJSim: Starting chunk process \t " + 
#MR                             str(chunk_id) + " on (node, core)=(" + 
#MR                             str(assignment[chunk_id]) + ")\n") #MR
#MR            status['wait'].remove(chunk_id)
#MR            status['started'].append(chunk_id)
#MR            # print "Initial Chunk Process started: ", chunk_id
#MR            entity.reqService(
#MR                              minDelay,            # add min-delay
#MR                              "CWBJ_CN_Handler",   # Handler/Service name #MR
#MR                              msg,                 # this is the data we send
#MR                              "Node",              # Target entity type
#MR                              target_node_id       # Target entity id
#MR                             )

#MR        chunk_stats = {}
#MR        for item in status:
#MR          # how many chunk processes in different status
#MR          chunk_stats[item] = len(status[item])
#MR        entity.update_sw_stats(chunk_stats)

        #####   2.2 Then we wait for all chunks to be executed
        # some chunks exist that haven't finished yet (but none is waiting)
        while status['started']:

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
            #print simianEngine.now, ": Msg q ", entity.msg_q #MR: debug
            msg = entity.msg_q.pop()
            # Message looks as follows: ['chunk_done_intranode_send', chunk_id]
            msg_type = msg[0]
            chunk_id = msg[1]
            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                             entity.name + " " + str(entity.num) +
                             " CWBJSim: received msg \t\t" + str(msg) + "\n") #MR

#            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
#                             entity.name + " " + str(entity.num) +
#                             " CWBJSim: received msg \t\t" + str(msg[0]) +
#                             str(msg[1]) + "\n") #MR: alternative print in SNAPSim

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
                entity.update_hw_stats({'intra_bytes': bytes_sent})
              if msg_type == 'chunk_done_internode_send':
                status['internode_sent'].append(chunk_id)
                entity.update_hw_stats({'inter_bytes': bytes_sent})

        #MR: For very small problems chunk_done messages can be so fast
        #MR: that intranode_sent and internode_sent messages are not collected!
        #MR: Add following loop to capture any outstanding send messages
        while len(status['intranode_sent'])*len(status['internode_sent']) != total_chunks*total_chunks:

          this.hibernate()

          # We check the q
          while entity.msg_q != []:
            #print simianEngine.now, ": Msg q ", entity.msg_q #MR: debug
            msg = entity.msg_q.pop()
            # Message looks as follows: ['chunk_done_intranode_send', chunk_id]
            msg_type = msg[0]
            chunk_id = msg[1]
            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                             entity.name + " " + str(entity.num) +
                             " CWBJSim: received msg \t\t" + str(msg) + "\n") #MR

#            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
#                             entity.name + " " + str(entity.num) +
#                             " CWBJSim: received msg \t\t" + str(msg[0]) +
#                             str(msg[1]) + "\n")

            bytes_sent = msg[2]
            if msg_type == 'chunk_done_intranode_send':
              status['intranode_sent'].append(chunk_id)
              entity.update_hw_stats({'intra_bytes': bytes_sent})
            if msg_type == 'chunk_done_internode_send':
              status['internode_sent'].append(chunk_id)
              entity.update_hw_stats({'inter_bytes': bytes_sent})

        #####   2.2 We wait for all chunks to be executed
        # some chunks exist that are still waiting or haven't finished yet
#MR        while status['wait'] or status['started']:

#MR          # We update stats
#MR          chunk_stats = {}
#MR          for item in status:
#MR            chunk_stats[item] = len(status[item])
#MR          entity.update_sw_stats(chunk_stats)

#MR          # We hibernate until woken up by the master handler who will have put
#MR          #   something into the q
#MR          # entity.out.write("Time: " + str(simianEngine.now) +
#MR          #                  ", before hibernation with scp " +
#MR          #                  str(started_chunk_processes) +
#MR          #                  ", and wcp " + str(waiting_chunk_processes) + "\n")
#MR          this.hibernate()
#MR          # entity.out.write("Time: " + str(simianEngine.now) +
#MR          #                  ", after hibernation with msg_q " +
#MR          #                  str(entity.msg_q) + "\n")

#MR          # We check the q
#MR          while entity.msg_q != []:
#MR            # print simianEngine.now, ": Msg q ", entity.msg_q
#MR            msg = entity.msg_q.pop()
#MR            # Message looks as follows: ['chunk_done_intranode_send', chunk_id]
#MR            msg_type = msg[0]
#MR            chunk_id = msg[1]
#MR            entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
#MR                             entity.name + " " + str(entity.num) +
#MR                             " CWBJSim: received msg \t\t" + str(msg) + "\n") #MR

#MR            # Update status
#MR            if msg_type == 'chunk_done':
#MR              # print simianEngine.now, ": chunk_done received: ", chunk_id, \
#MR              #   entity.msg_q
#MR              status['started'].remove(chunk_id)
#MR              status['done'].append(chunk_id)
#MR              entity.update_hw_stats(msg[2])    #msg[2] contains the stats info
#MR            else:
#MR              bytes_sent = msg[2]
#MR              if msg_type == 'chunk_done_intranode_send':
#MR                status['intranode_sent'].append(chunk_id)
#MR              if msg_type == 'chunk_done_internode_send':
#MR                status['internode_sent'].append(chunk_id)

#MR            # Check if dependents can be run and do so
#MR            for c_chunk in dependents[chunk_id]:
#MR              # Set the candidate chunk attributes
#MR              (c_node, c_core) = assignment[c_chunk]
#MR              (chunk_node, chunk_core) = assignment[chunk_id]
#MR              # Check if the candidate is ready 
#MR              if  (msg_type == 'chunk_done' and \
#MR                    (c_node, c_core) == (chunk_node, chunk_core)) \
#MR                 or \
#MR                  (msg_type == 'chunk_done_intranode_send' and \
#MR                    c_node == chunk_node) \
#MR                 or \
#MR                  (msg_type == 'chunk_done_internode_send'):
#MR                # Is the candidate waiting
#MR                if c_chunk in status['wait']:
#MR                  # Look at the candidates requirements, start with all good
#MR                  chunk_ok = True
#MR                  for r_chunk in required[c_chunk]:
#MR                    # Set attribute for each requirment
#MR                    (r_node, r_core) = assignment[r_chunk]
#MR                    # Check ALL requirements are met
#MR                    if  (r_chunk in status['wait']) \
#MR                       or \
#MR                        (r_chunk in status['started']) \
#MR                       or \
#MR                        not( (r_chunk in status['done'] and \
#MR                               (c_node, c_core) == (r_node, r_core)) \
#MR                            or \
#MR                             (r_chunk in status['intranode_sent'] and \
#MR                               c_node == r_node) \
#MR                            or \
#MR                             (r_chunk in status['internode_sent']) ):
#MR                      # Not all other required chunks done for c_chunk
#MR                      chunk_ok = False

#MR                  # We can start candidate chunk c_chunk
#MR                  if chunk_ok:
#MR                    if msg_type =='chunk_done_internode_send':
#MR                      entity.update_hw_stats({'inter_bytes': bytes_sent})
#MR                    if msg_type =='chunk_done_intranode_send':
#MR                      entity.update_hw_stats({'intra_bytes': bytes_sent})
#MR                    (target_node_id, target_core_id) = assignment[c_chunk]
#MR                    msg = ['start', c_chunk, target_core_id, ichunk, jchunk,
#MR                           kchunk, nang, cmom]
#MR                    entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
#MR                                     entity.name + " " + str(entity.num) +
#MR                                     " CWBJSim: Starting chunk process \t " +
#MR                                     str(c_chunk) + " on (node, core)=(" +
#MR                                     str(assignment[c_chunk]) + ")\n") #MR
#MR                    #print simianEngine.now, ":Starting chunk", c_chunk
#MR                    status['wait'].remove(c_chunk)
#MR                    status['started'].append(c_chunk) 
#MR                    entity.reqService(minDelay, "CWBJ_CN_Handler", msg,
#MR                                      "Node", target_node_id) #MR

        # end inner for
        entity.update_sw_stats({'inner_i':inner_i})
        # print "End of inner_i", inner_i
        entity.out.write("Time: " + str(simianEngine.now) + ":\t " +
                         entity.name + " " + str(entity.num) +
                         " CWBJSim: End of inner loop " + str(inner_i) + "\n") #MR

      # end outer for
      entity.update_sw_stats({'outer_i':outer_i})
      entity.out.write("Time: " + str(simianEngine.now) + ":\t " + entity.name +
                       " " + str(entity.num) + " CWBJSim: End of outer loop " +
                       str(outer_i) + "\n") #MR

      ########
      # Compute cross-group interactions
      #print "Computing cross groups with outer_i: ", outer_i
      ########
#      for node_id in xrange(cluster.num_nodes):
#        # cells_count[core_id] = cells_on_core
#        cells_count = {}
#        for core_id in xrange(cores[node_id]):
#          # How many chunks were computed on this core
#          chunk_count = 0
#          for chunk_id in assignment:
#            # inefficient, but rare, so let's keep it for now
#            if (node_id, core_id) == assignment[chunk_id]:
#              chunk_count += 1
#          cells_on_core = chunk_count*ichunk*jchunk*kchunk
#          cells_count[core_id] = cells_on_core
#        #print "Sending cross compute request to node,core ", cells_count
#        entity.reqService(minDelay, "CWBJ_CN_Handler",
#                          ['cross_compute', cells_count, ng],
#                          "Node", node_id) #MR
#      response_count = 0
#      while response_count < cluster.num_nodes:
#        this.hibernate()
#        while entity.msg_q != []:
#          msg = entity.msg_q.pop()
#          if msg[0] == 'cross_compute_done':
#            response_count += 1
#            # don't know what to do with this, will use for stats later
#            node_id =  msg[1]
#            # msg[2] are stats
#            entity.update_hw_stats(msg[2])
#      #print "Finished cross_compute"

    # end of timestep
    entity.out.write("Time: " + str(simianEngine.now) + ":\t " + entity.name +
                     " " + str(entity.num) + " CWBJSim: End of timestep " +
                     str(timestep_i) + "\n") #MR

  entity.out.write("End of computation, now waiting for potential " +
                   "late arrival wake_up notifications to " +
                   "ignore \n") #MR

  count1 =0
  print "Almost done 1"
  entity.print_stats()
  print "Almost done 2"
  while True:
    this.hibernate()
    count1 +=1
    entity.out.write("Hibernate count: " + str(count1) + "\n")


###############################################################################
###############################################################################


# compute_dependencies #MR: CWBJ should not need this function. Comment it out.
###############################################################################
"""
def compute_dependencies(chunk_id, requiredFlag, nxc, nyc, nzc, ng):
#   If requiredFlag==true, returns the chunks required before specified
#     chunk_id can be processed.
#   If requiredFlag==false, returns the chunks that immediately depend
#     on specified chunk_id.
#  
#   Incoming: chunk_id=(group, octant, z, y, x)
#             requiredFlag = true/false
#             nxc, nyc, nzc = number of chunks in x, y, and z, respectively
#   Outgoing: dep_list and req_list

  (group, octant, z, y, x) = chunk_id
  dep_list = []
  req_list = []

  # Set general octant requirements/dependents
  if octant > 0:
    req_list.append((group, octant-1, z, y, x))
  if octant < 7:
    dep_list.append((group, octant+1, z, y, x))

  # Set general group requirements/dependents
  # Not sure if this is set up correctly for threads; may need re-worked
  if group > 0:
    req_list.append((group-1, octant, z, y, x))
  if group < ng:
    dep_list.append((group+1, octant, z, y, x))

  # Set spatial chunk requirements/dependents; ordering accounts for 1D/2D/3D
  if octant == 0:
    if x<nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x>0:
      dep_list.append((group, octant, z, y, x-1))
    if y<nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y>0:
      dep_list.append((group, octant, z, y-1, x))
    if z<nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z>0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 1:
    if x>0:
      req_list.append((group, octant, z, y, x-1))
    if x<nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y<nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y>0:
      dep_list.append((group, octant, z, y-1, x))
    if z<nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z>0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 2:
    if x<nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x>0:
      dep_list.append((group, octant, z, y, x-1))
    if y>0:
      req_list.append((group, octant, z, y-1, x))
    if y<nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z<nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z>0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 3:
    if x>0:
      req_list.append((group, octant, z, y, x-1))
    if x<nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y>0:
      req_list.append((group, octant, z, y-1, x))
    if y<nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z<nzc-1:
      req_list.append((group, octant, z+1, y, x))
    if z>0:
      dep_list.append((group, octant, z-1, y, x))

  if octant == 4:
    if x<nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x>0:
      dep_list.append((group, octant, z, y, x-1))
    if y<nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y>0:
      dep_list.append((group, octant, z, y-1, x))
    if z>0:
      req_list.append((group, octant, z-1, y, x))
    if z<nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  if octant == 5:
    if x>0:
      req_list.append((group, octant, z, y, x-1))
    if x<nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y<nyc-1:
      req_list.append((group, octant, z, y+1, x))
    if y>0:
      dep_list.append((group, octant, z, y-1, x))
    if z>0:
      req_list.append((group, octant, z-1, y, x))
    if z<nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  if octant == 6:
    if x<nxc-1:
      req_list.append((group, octant, z, y, x+1))
    if x>0:
      dep_list.append((group, octant, z, y, x-1))
    if y>0:
      req_list.append((group, octant, z, y-1, x))
    if y<nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z>0:
      req_list.append((group, octant, z-1, y, x))
    if z<nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  if octant == 7:
    if x>0:
      req_list.append((group, octant, z, y, x-1))
    if x<nxc-1:
      dep_list.append((group, octant, z, y, x+1))
    if y>0:
      req_list.append((group, octant, z, y-1, x))
    if y<nyc-1:
      dep_list.append((group, octant, z, y+1, x))
    if z>0:
      req_list.append((group, octant, z-1, y, x))
    if z<nzc-1:
      dep_list.append((group, octant, z+1, y, x))

  # Return the requested list type
  if requiredFlag:
    return req_list
  else:
    return dep_list
"""

###############################################################################
###############################################################################


# ChunkProcess #MR
###############################################################################

def ChunkProcess(this, arg, *args):
  # Again args is not used, but Simian attempts to send it more arguments
  """
  Individual process for an update of a chunk of cells
  """
  # 0. Initializations
  # Here's what is in arg msg = ['start', chunk_id, local_core_id, ichunk,
  #                              jchunk, kchunk, nang, cmom, ng, ng_bndls,
  #                              noct, subcells, method]
  (chunk_id, core_id, ichunk, jchunk, kchunk, nang, cmom, ng, ng_bndls, 
    noct, subcells, method) = \
     (arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7], arg[8], 
       arg[9], arg[10], arg[11], arg[12])

  total_angles = noct * nang # total number of discrete ordinates

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
  # TODO: verify guestimates, also, memory usage 
  if method == 'CWBJ':
    if noct == 4: # 2D
      if subcells == 1: # Quad
        vertices = 4; 
      elif (subcells == 2 or subcells == 4): # Triangles
        vertices = 3;
      else:
        print "Warning: Unknown number of 2D subcells!"
        exit()
    elif noct == 8: # 3D
      if subcells == 1: # Hex
        vertices = 8;
      elif subcells == 2: # Prisms
        vertices = 6;
      elif (subcells == 5 or subcells == 6 or subcells == 24): # Tets
        vertices = 4;
      else:
        print "Warning: Unknown number of 3D subcells!"
        exit()
    else:
      print "Warning: Supported dimensions are 2 and 3!"
      exit()

    if (ng % ng_bndls) != 0:
      print "Warning: For ease of implementation assume ng multiple of ng_bndls!"
      exit()

    gbndl = ng / ng_bndls # number of groups per energy group bundle

    N = vertices * total_angles * gbndl # Size of linear system per 
                                         # subcell and energy group bundle

#    print "N =", N, "\n" #MR: Comment out to debug

    #MR: Tasklist parameters
    #MR: At present very rough order of magnitude guestimates of operations required
    #MR: to solve each NxN Ax=b!
    #MR: Also ignoring operations involved in building A and b which also explains fake
    #MR: independence from cmom.

    num_index_vars = subcells*ng_bndls*N           # number of index variables
    num_float_vars = subcells*ng_bndls*N*(N + 1) # number of float variables
    index_loads = subcells*ng_bndls*(12.0/13.0)*(1.005/2.0)*N**3 # all integer loads, ignoring logical
    float_loads = subcells*ng_bndls*(1.0/13.0)*(1.005/2.0)*N**3 # float plus float vector loads

    avg_dist = 8.08           # average distance in arrays between accessed elements 
    avg_reuse_dist = 150*N    # avg number of unique loads between two consecutive
                              # accesses of the same element (ie use different
                              # weights for int and float loads)
    stdev_reuse_dist = 135*N  # stddev number of unique loads between two
                              # consecutive accesses of the same element

    int_alu_ops = subcells*ng_bndls*8.0*N**3  # includes logical ops

    float_alu_ops = subcells*ng_bndls*(2.0/3.0)*N**3
                                          # cost of LU solve from literature

    float_DIV_ops    = subcells*ng_bndls*(1.0/2.0)*N**2

    int_vector_ops = 0

    float_vector_ops = 0

    tasklist_per_interior_cell = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                                   ['fDIV', float_DIV_ops],
                                   ['INTVEC', int_vector_ops, 1],
                                   ['VECTOR', float_vector_ops, 1],
                                   ['MEM_ACCESS', num_index_vars, num_float_vars,
                                    avg_dist, avg_reuse_dist, stdev_reuse_dist,
                                    index_loads, float_loads, False] ]
    # ['alloc', 1024]\ TODO: put memory in at some point. Probably at the very
    #   very beginning

    tasklist_per_border_cell = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                               ['VECTOR', float_vector_ops, N], #MR
                               ['MEM_ACCESS', num_index_vars, num_float_vars,
                                avg_dist, avg_reuse_dist, stdev_reuse_dist,
                                index_loads, float_loads, True] ]
    # Difference in tasklists is the Init_file set to True. In border cells, more
    #   float elements will need to be loaded from main
  elif method == 'CWBGS':
    print "Warning: CWBGS method not implemented yet!"
    exit()
  elif method == 'IPBJ':
    print "Warning: IPBJ method not implemented yet!"
    exit()
  else:
    print "Warning: Unknown method!" 
    exit()

  # 3. Loop over cells in chunk to compute the time that is spent for the
  #    entire chunk, then perform a sleep (premature optimization ...)
  time = 0

  time_per_border_cell, stats_border = \
    core.time_compute(tasklist_per_border_cell, True)

#  print "time_per_border_cell =", time_per_border_cell, "\n" #MR: Comment out to debug

  time_per_interior_cell, stats_interior = \
    core.time_compute(tasklist_per_interior_cell, True)

#  print "time_per_interior_cell =", time_per_interior_cell, "\n" #MR: Comment out to debug

  # We only want to put out full stats once
#MR  if chunk_id == (0,0,0,0,0): #MR: Three components only for CWBJ
  if chunk_id == (0,0,0):
    entity.out.write("Time: " + str(simianEngine.now) + ":\t " + entity.name +
                     " " + str(entity.num) +
                     " ChunkProcess: Stats from single interior cell call: \n") #MR
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

#  print "time =", time, "\n" #MR: Comment out to debug

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

  # 4. Send done info to let CWBJ_CN_Handler know that this chunk is done
  # MR: adapt bytes_sent calculation to CWBJ 
  if noct == 4: # 2D
    bytes_sent = 2 * (2*ichunk+2*jchunk) * ng * (total_angles/2) * PSI_SIZE
  elif noct == 8: # 3D
    Area_bt = 2 * ichunk * jchunk # bottom - top
    Area_fk = 2 * ichunk * kchunk # front - back
    Area_lr = 2 * jchunk * kchunk # left - right
    Area_v = 0  
    if subcells == 1: # Hex
      fvertices = 4; # All cell's faces
      Area_v = fvertices * (Area_bt + Area_fk + Area_lr)
    elif subcells == 2: # Prisms
      fvertices = 4; # Cell's fklr faces
      fvertices_bt = 6; # Cell's bt faces
      Area_v = fvertices_bt * Area_bt + fvertices * (Area_fk + Area_lr)
    elif (subcells == 5 or subcells == 6): # Tets
      fvertices = 6; # All cell's faces
      Area_v = fvertices * (Area_bt + Area_fk + Area_lr)
    elif subcells == 24: # Tets
      fvertices = 12; # All cell's faces
      Area_v = fvertices * (Area_bt + Area_fk + Area_lr)
    else:
      print "Warning: Unknown number of 3D subcells!"
      exit()
    bytes_sent = Area_v * ng * (total_angles/2) * PSI_SIZE
  else:
    print "Warning: Supported dimensions are 2 and 3!"
    exit()

  internode_time = max(minDelay, core.time_compute([['internode', bytes_sent]]))
  intranode_time = max(minDelay, core.time_compute([['intranode', bytes_sent]]))
  # print ("Time: " + str(simianEngine.now) + ":\t " + entity.name + " " +
  #        str(entity.num) + " ChunkProcess: \t " + str(chunk_id) +
  #        " completed on core id " + str(core_id) + "; execution time: " +
  #        str(time) + "\n")
  node.reqService(
                  0,                                    # zero delay
                  "CWBJ_CN_Handler",                    # Handler/Service name
                  ['chunk_done', chunk_id, core_id,
                   bytes_sent, internode_time,
                   intranode_time, stats]       # this is the data that we send
                 ) # MR


###############################################################################
###############################################################################


# CWBJ_CN_HANDLER #MR
###############################################################################

def CWBJ_CN_Handler(self, msg, *args): #MR
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
  #  msg = ['start', chunk_id, local_core_id, self.ichunk, self.jchunk,
  #         self.kchunk, self.nang, self.cmom, 
  #         self.ng, self.ng_bndls, self.noct, self.subcells, self.method] #MR
  # We just start a process on the resource local_core_id == msg[2]
  # OR it is the kill message
  #

  # print("CWBJ_CN_Handler Time: " + str(self.engine.now) + ", Msg " +
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
    # Forward done info to CWBJ MasterHandler with min delay
    self.reqService(
                    simianEngine.minDelay,               # minDelay
                    "Master_Handler",                    # Handler/Service name
                    ['chunk_done', chunk_id, stats],     # data that we send
                    'Master', 1                          # destination entity
                   )
    # Forward done info to CWBJ MasterHandler after communication delay (to be
    #   replaced with call to MPI model in the future)
    self.reqService(intranode_time, "Master_Handler",
                    ['chunk_done_intranode_send', chunk_id, bytes_sent],
                    "Master", 1)
    self.reqService(internode_time, "Master_Handler",
                    ['chunk_done_internode_send', chunk_id, bytes_sent],
                    "Master", 1)

  # Type 4: Cross compute (scatter source computation)
#  elif msg[0] == 'cross_compute':
#    cells_count = msg[1]
#    ng = msg[2]
#    maxtime = 0.0
#    nodestats = {}
#
#    for core_id in cells_count:
#      num_elements = 5
#      core = self.cores[core_id]
#
#      cross_compute_tasklist = \
#        [['fALU', cells_count[core_id]*num_elements*ng**2]]
#
#      (time, stats) = core.time_compute(cross_compute_tasklist, True)
#      maxtime = max(maxtime, time)
#
#      #Sum up statistics across all cores on node
#      for item in stats:
#        if item in nodestats:
#          nodestats[item] += stats[item]
#        else:
#          nodestats[item] = stats[item]
#
#    time = max(time, minDelay)
#    # Thus only slowest core determines the elapse time, we report to master
#    #   with delay time
#    self.reqService(time, "Master_Handler",
#                    ['cross_compute_done', self.num, stats], "Master", 1)


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

    self.createProcess("master_proc", CWBJSim) #MR
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

print "\nCWBJSim run with Simian PDES Engine\nVersion =", version
#cluster = clusters.MiniTrinity(simianEngine)
#cluster = clusters.SingleCielo(simianEngine)
cluster = clusters.Moonlight(simianEngine)
#cluster = clusters.HalfTrinity(simianEngine, 1)


# 2. Create cwbj sim master node, this automatically creates the master_proc
#     and Master_Handler

simianEngine.addEntity("Master", MasterNode, 1) #MR


# 3. Create a CWBJ Handler Service on each node, note the handler just acts as
#    a pass-through for spawning chunk processes

simianEngine.attachService(nodes.Node, "CWBJ_CN_Handler" , CWBJ_CN_Handler) #MR
	
# 4. Run simx

simianEngine.run()
simianEngine.exit()


###############################################################################
###############################################################################


# END
###############################################################################
