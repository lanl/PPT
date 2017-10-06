"""
 PolyBenchSim: 2mm
"""

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../../simian/simian-master/SimianPie')
path.append('../../hardware')

#import simian
from simian import Simian 
import clusters
import nodes

# Global Parameters

N_X = 64 # Size of first input dimension
N_Y = 1 # Size of first input dimension
N_Z = 1 # Size of first input dimension

########################
# 0. Initialization stuff

simName, startTime, endTime, minDelay, useMPI = \
  "2mmsim", 0.0, 1.0*10**16, 1, False
simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

###############################################################################

def TwommSim(this, *args):
  """
  Simulates 2mm from the PolyBench Suite
  This is the function that gets executes as main process. Only one such
    function exists.
  TODO: Generalize this to simulate all PolyBench apps by making the name a parameter, 
  same for cluster name. Read all values in dicts of dicts from file "fit" and irf
  """
  #### Initialization #####
  arg = args[1] # args[0]is function pointer, args[1] is list of passed arguments
  core_id, n_x, n_y, n_z =  arg[0], arg[1], arg[2], arg[3]
  node = this.entity
  core = node.cores[core_id]
  t = {} # tasklist
  
  ####  Operations Parameters  (ByFl counters) #######
  t['integer_ops'] 			=	46.0 * n_x**3	
  t['FAdd'] 				=	2.0 * n_x**3
  t['FMul'] 				=	3.0 * n_x**3
  t['loads'] 				=	31.0 * n_x**3
  t['stores'] 				=	4.1 * n_x**3
  t['uncond_brach_ops'] 	=	4.1 * n_x**3
  t['cond_brach_ops'] 		=	2.1 * n_x**3
  t['comparison'] 			=	2.1 * n_x**3
  t['cpu_ops'] 				=	12.0 * n_x**3

  ####  Dependency Graph Parameters  (irf counters) #######
  t['num_loops']			=	6
  t['num_vec_loops']		=	0
  t['num_par_loops']		=	0
  t['num_par_dep_loops']	=	6
  t['num_vac_dep_loops']	=	6 
  t['raw_dep']				=	4
  t['war_dep']				=	4
  t['waw_dep']				=	4
  t['num_bbs']				=	25
  t['num_edges']			=	30

  ####  Expected Data Size weight parameter #######
  t['eds']					=	40.0 * n_x**2 # 
  
  #### Compute time and advance time
  time  =  core.time_compute(t, False)
  print "Computed time is ", time
  this.sleep(time)
  print "PolyBench app", simName, ", Input n_x, n_y, n_z: ", n_x, n_y, n_z, \
      " Cluster Mustang, serial run, predicted runtime (s): ", simianEngine.now 

###############################################################################
def os_handler(self, arg, *args):
	self.createProcess("Twomm", TwommSim)
	self.startProcess("Twomm", TwommSim, arg)

###############################################################################
###############################################################################
# "MAIN"
###############################################################################

# 1. Choose and instantiate the Cluster that we want to simulate 
cluster = clusters.Mustang(simianEngine, 1) # Single node Mustang

# 2. Create a OS Service Service on each node
simianEngine.attachService(nodes.Node, "os_handler" , os_handler)

# 3. Schedule OS Handler with arguments for PolyBench app
arg = [0, N_X, N_Y, N_Z] # Node id 0
simianEngine.schedService(0.0, "os_handler", arg, "Node", 0)

# 4. Run simx
simianEngine.run()
simianEngine.exit()
