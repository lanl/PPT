#
# omp_for_loop.py :- test parallel for loop construct
#    Models an openMP style parallel loop with critical sections
#
from sys import path
path.append('../..')
from ppt import *

  
def callback(node, args):
  print("Loop finished executing at time: "+str(node.engine.now))

def run_tasklist(mpi_comm_world):

  host = mpi_ext_host(mpi_comm_world)
  
  """
    The modeled code could be written as follows:
    
    for niter iterations do in parallel over nthreads threads:
    
      execute regular code section tasklist1
      
      execute critical code section tasklist2 
      (only a single thread executing this section at any given time)
      
      execute regular code section tasklist3 
      
      execute critical code section tasklist4 
      (only a single thread executing this section at any given time)
      
    end for
  """
  
  tasklist1 = ['regular', ['iALU', 10], ['fALU', 40],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 80, True]]
  critical1 = ['critical',['iALU', 1], ['fALU', 3],
                ['MEM_ACCESS', 1, 2,
                1, 1, 1,
                1, 6, True]]
  tasklist2 = ['regular', ['iALU', 10], ['fALU', 40],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 80, True]]
  critical2 = ['critical',['iALU', 1], ['fALU', 3],
                ['MEM_ACCESS', 1, 2,
                1, 1, 1,
                1, 6, True]]
  tasklist = [tasklist1, critical1,tasklist2, critical2]
  """
    Syntaxe of omp wrapper: 
    [["parallel_for", nthreads, niter, scheduling_type, use_processes, tasklist,
    callback_function, callback_args]]
    
    nthreads         : Number of threads to simulate
    niter            : Number of iterations to simulate
    scheduling_type  : As defined in openmp
    use_processes    : Whether to use processes (required for critical sections)
    tasklist         : Array of tasklists describing a single loop iteration
    callback_function: Function to be called at the end of the loop simulation
                       This argument can be omitted if not using processes
    callback_args    : Arguments to be passed to the callback function
                       This argument can be omitted if not using processes
  """
  omp_wrapper = [["parallel_for", 16, 10000000000, "static", True, tasklist,
                  callback, None]]
  time = host.time_compute(omp_wrapper)
  

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
