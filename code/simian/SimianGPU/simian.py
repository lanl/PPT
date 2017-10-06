#Author: Nandakishore Santhi
#Date: 23 November, 2014
#Copyright: Open source, must acknowledge original author
#Purpose: PDES Engine in Python, mirroring a subset of the Simian JIT-PDES
#  Main simumation engine class

#NOTE: There are some user-transparent differences in SimianPie
#Unlike Simian, in SimianPie:
#   1. heapq API is different from heap.lua API
#       We push tuples (time, event) to the heapq heap for easy sorting.
#       This means events do not need a "time" attribute; however it is
#       still present for compatibility with Simian JIT.
#   2. mpi4py API is different from the MPI.lua API
#   3. hashlib API is diferent from hash.lua API
MPI = None
CUDA = None

import hashlib, heapq

import time as timeLib

from utils import SimianError
from entity import Entity

class Simian(object):
    def __init__(self, simName, startTime, endTime, minDelay=1, useMPI=False, useCUDA=False):
        self.Entity = Entity #Include in the top Simian namespace

        self.name = simName
        self.startTime = startTime
        self.endTime = endTime
        self.minDelay = minDelay
        self.now = startTime

        #If simulation is running
        self.running = False

        #Stores the entities available on this LP
        self.entities = {}

        #Events are stored in a priority-queue or heap, in increasing
        #order of time field. Heap top can be accessed using self.eventQueue[0]
        #event = {time, name, data, tx, txId, rx, rxId}.
        self.eventQueue = []

        #Stores the minimum time of any event sent by this process,
        #which is used in the global reduce to ensure global time is set to
        #the correct minimum.
        self.infTime = endTime + 2*minDelay
        self.minSent = self.infTime

        #[[Base rank is an integer hash of entity's name]]
        self.baseRanks = {}

        #Make things work correctly with and without MPI
        if useMPI:
            #Initialize MPI
            global MPI
            try:
                from mpi4py import MPI
                self.useMPI = True
                self.comm = MPI.COMM_WORLD
                self.rank = self.comm.Get_rank()
                self.size = self.comm.Get_size()
            except:
                raise SimianError("Please install mpi4py before using Simian for MPI based simulation")
        else:
            self.useMPI = False
            self.comm = None
            self.rank = 0
            self.size = 1

	if useCUDA:
	    from gpu_scheduler import GPU_scheduler
            self.gpu_scheduler = GPU_scheduler(self)
	    self.useCUDA = True
    	else:
	    self.gpu_scheduler = None
	    self.useCUDA = False

        #One output file per rank
        self.out = open(self.name + "." + str(self.rank) + ".out", "w")

    def exit(self):
	if self.useCUDA:
	    self.gpu_scheduler.exit()
        self.out.close()
        del self.out

    def run(self): #Run the simulation
        startTime = timeLib.clock()
        if self.rank == 0:
            print("===========================================")
            print("----------SIMIAN-PIE PDES ENGINE-----------")
            print("===========================================")
            if self.useMPI:
                print("MPI: ON")
            else:
                print("MPI: OFF")
        numEvents = 0

        self.running = True
        baseTime = self.startTime

        while baseTime < self.endTime:
            self.minSent = self.infTime

            while len(self.eventQueue) > 0 \
                    and self.eventQueue[0][0] < baseTime + self.minDelay \
                    and self.eventQueue[0][0] < self.endTime:
                (time, event) = heapq.heappop(self.eventQueue) #Next event
                self.now = time #Advance time

                #Simulate event
                entity = self.entities[event["rx"]][event["rxId"]]
                service = getattr(entity, event["name"])
                service(event["data"], event["tx"], event["txId"]) #Receive

                numEvents = numEvents + 1

	    if self.useCUDA:
	        self.gpu_scheduler.process_jobs()
            minLeft = self.endTime
            if len(self.eventQueue) > 0:
                minLeft = self.eventQueue[0][0]

            if self.size > 1:
                baseTime = self.comm.allreduce(min(self.minSent, minLeft), op=MPI.MIN)
                while self.comm.Iprobe(source=MPI.ANY_SOURCE): #As long as there are messages waiting
                    remoteEvent = self.comm.recv(source=MPI.ANY_SOURCE)
                    heapq.heappush(self.eventQueue, (remoteEvent["time"], remoteEvent))
            else:
                baseTime = min(self.minSent, minLeft)

        if self.size > 1:
            totalEvents = self.comm.allreduce(numEvents, op=MPI.SUM)
        else:
            totalEvents = numEvents

        if self.rank == 0:
            elapsedTime = timeLib.clock() - startTime
            print "SIMULATION COMPLETED IN: " + str(elapsedTime) + " SECONDS"
            print "SIMULATED EVENTS: " + str(totalEvents)
            print "EVENTS PER SECOND: " + str(totalEvents/elapsedTime)
            print "==========================================="

    def schedService(self, time, eventName, data, rx, rxId):
        #Purpose: Add an event to the event-queue.
        #For kicking off simulation and waking processes after a timeout
        recvRank = self.getOffsetRank(rx, rxId)

        if recvRank == self.rank:
            e = {
                    "tx": None, #String (Implictly self.name)
                    "txId": None, #Number (Implictly self.num)
                    "rx": rx, #String
                    "rxId": rxId, #Number
                    "name": eventName, #String
                    "data": data, #Object
                    "time": time, #Number
                }

            heapq.heappush(self.eventQueue, (time, e))

    def getBaseRank(self, name):
        #Can be overridden for more complex Entity placement on ranks
        return int(hashlib.md5(name).hexdigest(), 16) % self.size

    def getOffsetRank(self, name, num):
        #Can be overridden for more complex Entity placement on ranks
        val = (self.baseRanks[name] + num) % self.size
        return (self.baseRanks[name] + num) % self.size

    def getEntity(self, name, num):
        #Returns a reference to a named entity of given serial number
        if name in self.entities:
            entity = self.entities[name]
            if num in entity:
                return entity[num]

    def attachService(self, klass, name, fun):
        #Attaches a service at runtime to an entity klass type
        setattr(klass, name, fun)

    def addEntity(self, name, entityClass, num, *args):
        #Purpose: Add an entity to the entity-list if Simian is idle
        #This function takes a pointer to a class from which the entities can
        #be constructed, a name, and a number for the instance.
        if self.running: raise SimianError("Adding entity when Simian is running!")

        if not (name in self.entities):
            self.entities[name] = {} #To hold entities of this "name"
        entity = self.entities[name]

        self.baseRanks[name] = self.getBaseRank(name) #Register base-ranks
        computedRank = self.getOffsetRank(name, num)

        if computedRank == self.rank: #This entity resides on this engine
            #Output log file for this Entity
            self.out.write(name + "[" + str(num) + "]: Running on rank " + str(computedRank) + "\n")

            entity[num] = entityClass({
                "name": name,
                "out": self.out,
                "engine": self,
                "num": num,
                }, *args) #Entity is instantiated
