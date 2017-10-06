#
# omp_for_loop_simple.py :- test simple parallel for loop construct
#    Models an openMP style parallel loop
#
from sys import path
path.append('../..')
from ppt import *

def run_tasklist(mpi_comm_world):

  host = mpi_ext_host(mpi_comm_world)

  # The following tasklist corresponds to a single loop iteration 
  tasklist = [['iALU', 10], ['fALU', 80],
                ['VECTOR', 20, 24],
                ['MEM_ACCESS', 4, 10,
                1, 1, 1,
                10, 300, True]]
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
  omp_wrapper = [["parallel_for", 32, 100000000, "static", False, tasklist]]
  time = host.time_compute(omp_wrapper)
  print 'Computations finished at time',time
 

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
