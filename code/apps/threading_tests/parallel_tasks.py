#
# omp_for_loop_simple.py :- test simple parallel for loop construct
#    Models the parallel execution of independent tasks or code sections
#
from sys import path
path.append('../..')
from ppt import *

def callback(node, args):
  print("Tasks finished executing at time: "+str(node.engine.now))

def run_tasklist(mpi_comm_world):

  host = mpi_ext_host(mpi_comm_world)
  
  # Each of the following tasklists corresponds to an independent task
  task1 = ["task" ,['iALU', 10], ['fALU', 800],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 1600, True]]
  task2 = ["task" ,['iALU', 10], ['fALU', 2400],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 4800, True]]
  task3 = ["task" ,['iALU', 10], ['fALU', 400],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 200, True]]
  task4 = ["task" ,['iALU', 10], ['fALU', 1200],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 2400, True]]
  task5 = ["task" ,['iALU', 10], ['fALU', 6000],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 12000, True]]
  task6 = ["task" ,['iALU', 10], ['fALU', 8000],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 16000, True]]
  tasklist = [task1,task2,task3,task4,task5,task6]
  tasklist += tasklist
  tasklist += tasklist
  tasklist += tasklist
  tasklist += tasklist
  tasklist += tasklist
  tasklist += tasklist
  tasklist += tasklist
  print("Starting computations of "+str(len(tasklist))+" tasks")
  """
    Syntaxe of omp wrapper: 
    [["parallel_tasks", nthreads, tasklist, callback_function, callback_args]]
    
    nthreads         : Number of threads to simulate
    tasklist         : Array of tasklists describing the different tasks
    callback_function: Function to be called at the end of the parallel tasks 
                       simulation
    callback_args    : Arguments to be passed to the callback function
  """
  wrapper = [["parallel_tasks", 16, tasklist, callback, None]]
  time = host.time_compute(wrapper)
  
model_dict = { "model_name"    : "for_loop_sim",
               "sim_time"      : 1e6,
               "use_mpi"       : False,
               "intercon_type" : "Gemini",
               "torus"         : configs.cielo_intercon,
               "mpiopt"        : configs.gemini_mpiopt,
               "host_type"     : "CieloNode",
               "load_libraries": set(["mpi"]),
             }

cluster = Cluster(model_dict)
hostmap = range(1)
cluster.start_mpi(hostmap, run_tasklist)
cluster.run()