#Copyright (c) 2015, Los Alamos National Security, LLC
#All rights reserved.
#
#Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
#Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
#	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
#	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
#THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
#   2. hashlib API is diferent from hash.lua API
MPI = None
import hashlib, heapq

import time as timeLib

from utils import SimianError
from entity import Entity

import os
defaultMpichLibName = os.path.join(os.path.dirname(__file__), ".", "libmpich.dylib")
#print defaultMpichLibName

class Simian(object):
    def __init__(self, simName, startTime, endTime, minDelay=1, useMPI=False, mpiLibName=defaultMpichLibName):
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
            try:
                global MPI
                from MPILib import MPI
                self.useMPI = True
                self.MPI = MPI(mpiLibName)
                self.rank = self.MPI.rank()
                self.size = self.MPI.size()
            except:
                raise SimianError("Please ensure libmpich is available to ctypes before using Simian for MPI based simulations.\nTry passing absolute path to libmpich.[dylib/so/dll] to Simian.")
        else:
            self.useMPI = False
            self.MPI = None
            self.rank = 0
            self.size = 1

        #One output file per rank
        self.out = open(self.name + "." + str(self.rank) + ".out", "w")

        #Write some header information for each output file
        self.out.write("===========================================\n")
        self.out.write("----------SIMIAN-PIE PDES ENGINE-----------\n")
        self.out.write("===========================================\n")
        if self.useMPI:
            self.out.write("MPI: ON\n\n")
        else:
            self.out.write("MPI: OFF\n\n")

    def exit(self):
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
        globalMinLeft = self.startTime
        while globalMinLeft < self.endTime:
            epoch = globalMinLeft + self.minDelay

            self.minSent = self.infTime
            while len(self.eventQueue) > 0 and self.eventQueue[0][0] < epoch:
                (time, event) = heapq.heappop(self.eventQueue) #Next event
                self.now = time #Advance time

                #Simulate event
                entity = self.entities[event["rx"]][event["rxId"]]
                service = getattr(entity, event["name"])
                service(event["data"], event["tx"], event["txId"]) #Receive

                numEvents = numEvents + 1

            if self.size > 1:
                globalMinSent = self.MPI.allreduce(self.minSent, self.MPI.MIN) #Synchronize minSent
                while True: #Busy wait for incoming messages; synchronize
                    while self.MPI.iprobe(): #Outer repeat loop needed since per standard, MPI_Iprobe can give false negatives!!
                        remoteEvent = self.MPI.recvAnySize()
                        heapq.heappush(self.eventQueue, (remoteEvent["time"], remoteEvent))
                    minLeft = self.infTime
                    if len(self.eventQueue) > 0: minLeft = self.eventQueue[0][0]
                    globalMinLeft = self.MPI.allreduce(minLeft, self.MPI.MIN) #Synchronize minLeft
                    if globalMinLeft <= globalMinSent: break #Global queue is not ahead in time to global minsent
            else:
                minLeft = self.infTime
                if len(self.eventQueue) > 0: minLeft = self.eventQueue[0][0]
                globalMinLeft = min(self.minSent, minLeft)

        if self.size > 1:
            self.MPI.barrier()
            totalEvents = self.MPI.allreduce(numEvents, self.MPI.SUM)
        else:
            totalEvents = numEvents

        if self.rank == 0:
            elapsedTime = timeLib.clock() - startTime
            print "SIMULATION COMPLETED IN: " + str(elapsedTime) + " SECONDS"
            print "SIMULATED EVENTS: " + str(totalEvents)
            if elapsedTime > 10.0**(-12):
                print "EVENTS PER SECOND: " + str(totalEvents/elapsedTime)
            else:
                print "EVENTS PER SECOND: Inf"
            print "==========================================="

    def schedService(self, time, eventName, data, rx, rxId):
        #Purpose: Add an event to the event-queue.
        #For kicking off simulation and waking processes after a timeout
        if time > self.endTime: #No need to push this event
            return

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
