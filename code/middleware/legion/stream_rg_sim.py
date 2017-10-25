"""
Simulate stream.rg  a simple Regent program in PPT

"""
# To run: python [stream py filename] [optional command line arguments use -h for help]

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

from sys import dont_write_bytecode
dont_write_bytecode = True

import math

#import simian
from simian import Simian 
import clusters
import nodes

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

#
# Parameters for this application
#

import argparse
parser = argparse.ArgumentParser('PPT simulation of stream.rg')

parser.add_argument("-a","--array_size", 
										type=int, default=1000000,
                    help="Set the stream array size, default is 1000000.")
parser.add_argument("-t","--num_tasks",
										type=int, default=2,
                    help="Set the number of tasks, default is 2.")
parser.add_argument("-i","--num_iterations",
										type=int, default=1000,
                    help="Set the number of iterations, default is 1000.")
params = vars(parser.parse_args())
array_size = params['array_size']
num_tasks = params['num_tasks']
num_iterations = params['num_iterations']
size_double=8 # assume 8 byte double
print("\nSimulating Regent stream benchmark with %d iterations, %d tasks, and %d triples.\n"
	"Stream Array Size is %f MBytes" % (num_iterations, num_tasks, array_size, array_size*3*size_double*1e-6))

########################
# 0. Initialization stuff

simName, startTime, endTime, minDelay, useMPI = \
  "regent_stream", 0.0, 1000000.0, 0.000000001, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)


###############################################################################
###############################################################################


# Regent_stream_sim 
###############################################################################

def RgStreamSim(this):
  """
  Simulates Regent stream benchmark 
  This is the function that executes as main process. 
    Only one such function exists.
  """


########################
# 0. Initialization for  

simName, startTime, endTime, minDelay, useMPI = \
  "regent_stream_sim", 0.0, 1000000.0, 0.000000001, False

simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)


###############################################################################
###############################################################################


