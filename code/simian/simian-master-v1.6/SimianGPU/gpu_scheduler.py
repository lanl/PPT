#Author: Guillaume CHapuis
#Date: 27 February, 2015
#Copyright: Open source, must acknowledge original author
#Purpose: PDES Engine in Python, mirroring a subset of the Simian JIT-PDES
#  GPU scheduler class, batches jobs for available GPUs

#NOTE: There are some user-transparent differences in SimianPie
#Unlike Simian, in SimianPie:
#   1. heapq API is different from heap.lua API
#       We push tuples (time, event) to the heapq heap for easy sorting.
#       This means events do not need a "time" attribute; however it is
#       still present for compatibility with Simian JIT.
#   2. mpi4py API is different from the MPI.lua API
#   3. hashlib API is diferent from hash.lua API
import weakref
from utils import SimianError
from collections import namedtuple
class GPU_scheduler(object):
    # Simple structure containing information to process results of a simple kernel
    Result = namedtuple("Result", "event, result, callback")
    def __init__(self, engine):
        self.engine = weakref.ref(engine)
	self.event_queue = []
	self.current_gpu = None
	self.current_stream = (0,0)
	self.process_list = []
	#Initialize CUDA
	try:
	    import pycuda.driver as cuda
	    from pycuda.compiler import SourceModule
	    self.driver = cuda
	    self.driver.init()
	except:
	    raise SimianError("Please install pyCuda before using Simian for CUDA based simulation")
	self.num_devices = cuda.Device.count()
	print "Found ", self.num_devices, " cuda devices on this node"
	if self.num_devices == 0:
	    raise SimianError("No cuda capable device found on this node")
	self.device_queues = []
	self.contexts = []
	for gpu in range(self.num_devices):
	    self.contexts.append(cuda.Device(gpu).make_context())
	    streams = []
	    num_mp = cuda.Device(gpu).get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
	    for mp in range(num_mp):
		streams.append(cuda.Stream())
	    self.device_queues.append(streams)
	    print "Device ", gpu, " contains ", num_mp, " multiprocessors"
	self.contexts[0].push()
	self.current_gpu = 0

    def next_stream(self):
	self.current_stream = (self.current_stream[0], self.current_stream[1]+1)
	if self.current_stream[1] == len(self.device_queues[self.current_stream[0]]):
	    self.current_stream = (self.current_stream[0]+1, 0)
	    if self.current_stream[0] == len(self.device_queues):
		self.current_stream = (0, self.current_stream[1])

    def add_simple_job(self, data_in, data_out, gpu_code, name, call_back, block = (32,1,1), grid = (1,1)):
	if self.current_gpu != self.current_stream[0]:
	    # Change context to current gpu
	    self.contexts[self.current_gpu].pop()
	    self.contexts[self.current_stream[0]].push()
	    self.current_gpu = self.current_stream[0]
	kernel = gpu_code.get_function(name)
	kernel(self.driver.In(data_in), self.driver.Out(data_out), block=block, grid=grid, stream=self.device_queues[self.current_stream[0]][self.current_stream[1]])
	# Enqueue end of computations event
	event = self.driver.Event()
	event.record(self.device_queues[self.current_stream[0]][self.current_stream[1]])
	# Store information to process end of event when results are available
	self.process_list.append(GPU_scheduler.Result(event, data_out, call_back))
	self.next_stream()

    def process_jobs(self):
	while not len(self.process_list)==0:
	    result = self.process_list.pop(0)
	    result.event.synchronize()
	    result.callback(result.result)

    def exit(self):
	# Pop current contexts
	self.contexts[self.current_gpu].pop()
	for ctxt in self.contexts:
	    # Pop contexts for each available GPUs
	    ctxt.pop()
