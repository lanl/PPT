import nodes
import math

def exec_loop_simple(node, num_threads, num_iter, scheduling, tasklist):
  time_iter = node.cores[0].time_compute(tasklist)
  time = node.spawn_threads(num_threads)
  num_cores = len(node.cores)
  if scheduling == "static":
    iter_per_thread = math.ceil(float(num_iter)/float(num_threads))
    efficiency = node.thread_efficiency()
    time+=iter_per_thread*time_iter/efficiency
  elif scheduling == "dynamic":
    print('Warning: unsupported scheduling policy')
  else:
    print('Warning: unsupported scheduling policy:', scheduling,
          '. Ignoring construct.')
  node.unspawn_threads(num_threads)
  return time

def process_chunk(this, data):
  time     = data[0]
  num_iter = data[1]
  tasklist = data[2]
  timings  = data[3]
  mutexes  = data[4]
  node     = data[5]
  callback = data[6]
  args     = data[7]
  node.refcount += 1
  this.sleep(time)
  idle = 0
  start_hibernate = 0
  start = node.engine.now
  #print("Process", this.name, "starting", num_iter, "iterations at time", time, "with task list", tasklist)
  # Only compute a sample of the total number of iterations and extrapolate
  num_sample_iter = min(100, num_iter)
  for iter in range(num_sample_iter):
    for task in range(len(tasklist)):
      if tasklist[task][0] == 'regular':
        this.sleep(timings[task])
      elif tasklist[task][0] == 'critical':
        start_hibernate = node.engine.now
        if mutexes[task] == []: # mutex is available
          mutexes[task].append(this.name)
        else:
          mutexes[task].append(this.name)
          this.hibernate()
        idle += node.engine.now - start_hibernate
        this.sleep(timings[task])
        mutexes[task].pop(0)
        if mutexes[task] != []:
          node.wakeProcess(mutexes[task][0])
      else:
        print("Warning: unknown task type",  tasklist[task][0], "ignoring...")
  sample_time = node.engine.now - start
  total_time = sample_time*(num_iter/num_sample_iter)
  this.sleep(total_time-sample_time)
  print(this.name+" was idle "+str(idle*100/(sample_time))+"% of the time")
  print("Process "+this.name+" finished at time "+str(node.engine.now))
  node.refcount -= 1
  if node.refcount==0 and callback:
    callback(node, args)
  
def exec_loop(node, num_threads, num_iter, scheduling, tasklist, callback, 
              cb_args):
  init_time = node.spawn_threads(num_threads)
  efficiency = node.thread_efficiency()
  timings = []
  mutexes = []
  node.refcount = 0
  for task in tasklist:
    timings.append(node.cores[0].time_compute(task[1:])/efficiency)
    mutexes.append([])
  num_cores = len(node.cores)
  processes = []
  for i in range(num_threads):
    name = "thread" + str(i)
    node.createProcess(name, process_chunk)
    
  if scheduling == "static":
    iter_per_thread = math.ceil(float(num_iter)/float(num_threads))
    args = [init_time, int(iter_per_thread), tasklist, timings, mutexes, node, 
            callback, cb_args]
    for i in range(num_threads):
      name = "thread" + str(i)
      node.startProcess(name,args)
    
  elif scheduling == "dynamic":
    print('Warning: unsupported scheduling policy')
  else:
    print('Warning: unsupported scheduling policy: '+scheduling+
          '. Ignoring construct.')
  node.unspawn_threads(num_threads)
  return init_time

def process_tasks(this, data):
  time       = data[0]
  node       = data[1]
  tasklist   = data[2]
  efficiency = data[3]
  callback   = data[4]
  args       = data[5]
  node.refcount += 1
  this.sleep(time)
  while tasklist != []:
    task = tasklist.pop(0)
    timing = node.cores[0].time_compute(task[1:])/efficiency
    this.sleep(timing)
  node.refcount -= 1
  print(this.name+" finished computations at time "+str(node.engine.now))
  if node.refcount == 0 and callback:
    callback(node, args)
  
def exec_tasks(node, num_threads, tasklist, callback, cb_args):
  time = node.spawn_threads(num_threads)
  efficiency = node.thread_efficiency()
  for i in range(num_threads):
    name = "thread" + str(i)
    node.createProcess(name, process_tasks)
    args = [time, node, tasklist, efficiency, callback, cb_args]
    node.startProcess(name, args)
  
  node.unspawn_threads(num_threads)
  return time