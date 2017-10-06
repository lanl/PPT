--[[
Copyright (c) 2015, Los Alamos National Security, LLC
All rights reserved.

Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
]]

--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Main engine script
--]]
local eventQ = require "eventQ"
local priorityQ = require "priorityQ"
local hash = require "hash"
local Entity = require "entity"

local Simian = {
    Entity = Entity
}

function Simian.init(self, simName, startTime, endTime, minDelay, useMPI)
    self.name = simName
    self.startTime = startTime
    self.endTime = endTime or 1e100
    self.minDelay = minDelay or 1
    self.useMPI = useMPI and true or false

    self.now = startTime

    --Status of JITing
    self.jitStatus = jit and jit.status() or false

    --If simulation is running
    self.running = false

    --Stores the entities available on this LP
    self.entities = {}

    --[[Events are stored in a priority-queue or heap, in increasing
    order of time field. Heap top can be accessed using self.eventQueue[1]
    event = {time, name, data, tx, txId, rx, rxId}.]]
    self.eventQueue = {}

    --Contains minSent in time priority
    self.timeQueueTx = {}

    --Contains number of messages sent during each epoch, including to self
    self.txEpochs = {}

    --[[Stores the minimum time of any event sent by this process,
    which is used in the global reduce to ensure global time is set to
    the correct minimum.]]
    self.infTime = endTime + 2*minDelay

    --Base rank is an integer hash of entity's name
    self.baseRanks = {}

    --Make things work correctly with and without MPI
    if self.useMPI then
        --Initialize MPI
        self.MPI = require "MPI"
        self.MPI:init()
        self.rank = self.MPI:rank()
        self.size = self.MPI:size()
    else
        self.rank = 0
        self.size = 1
    end

    --One output file per rank
    self.out = io.open(self.name .. "." .. self.rank .. ".out", "w")
    self.out:setvbuf("no")
end

function Simian.exit(self)
    if self.useMPI then --Exit only when all engines are ready
        self.out:flush()
        self.out:close()
        self.out = nil
        self.MPI:finalize()
    end
end

function Simian.run(self) --Run the simulation
    local startClock = os.clock()
    if self.rank == 0 then
        print("===========================================")
        print("----------SIMIAN JIT-PDES ENGINE-----------")
        print("===========================================")
        if self.jitStatus then
            print("JIT: ON")
        else
            print("JIT: OFF")
        end
        if self.useMPI then
            print("MPI: ON")
        else
            print("MPI: OFF")
        end
    end

    local infTime, startTime, endTime, minDelay, rank, size, eventQueue, timeQueueTx, txEpochs, min
        = self.infTime, self.startTime, self.endTime, self.minDelay, self.rank, self.size, self.eventQueue,
            self.timeQueueTx, self.txEpochs, math.min

    self.running = true
    local MPI = self.MPI --Cache MPI locally

    local eventsDone, numEvents = 0, 0
    --Initialize globalEpochStart
    local globalEpochStart
    if size > 1 then
        local minSent = timeQueueTx[1] or infTime
        globalEpochStart = MPI:allreduce(minSent, MPI.MIN) --Synchronize, advance time
        if minSent <= globalEpochStart then
            minSent = priorityQ.pop(timeQueueTx) --Prune the Tx time Queue
        end
    else
        globalEpochStart = (#eventQueue > 0) and eventQueue[1].time or infTime
    end
    while globalEpochStart <= endTime do --Exit loop only when global epochStart is past endTime
        local epochEnd = globalEpochStart + minDelay
        while #eventQueue > 0 and eventQueue[1].time < epochEnd do
            event = eventQ.pop(eventQueue) --Next event
            self.now = event.time --Advance time

            --Simulate event
            local entity = self.entities[event.rx][event.rxId]
            local service = entity[event.name]
            service(entity, event.data, event.tx, event.txId) --Receive

            eventsDone = eventsDone + 1
        end

        --Advance globalEpochStart
        if size > 1 then --Synchronize across ranks
            -- Synchronize by MPI.SUM, how many remaining messages to simulate globally in this epoch
            local globalRemainder = MPI:allreduce((txEpochs[globalEpochStart] or 0)-eventsDone, MPI.SUM)
            if globalRemainder > 0 then --Busy wait for incoming messagese
                while MPI:iprobe() do --Per standard, MPI_Iprobe can give false negatives!!
                    eventQ.push(eventQueue, MPI:recvAnySize())
                end
            else -- Advance time to globalMinSent if no more messages to simulate in this epoch
                local minSent = timeQueueTx[1] or infTime
                globalEpochStart = MPI:allreduce(minSent, MPI.MIN) --Synchronize, advance time
                if minSent <= globalEpochStart then
                    priorityQ.pop(timeQueueTx) --Prune the Tx time Queue
                end
                numEvents = numEvents + eventsDone
                eventsDone = 0
            end
        else
            globalEpochStart = (#eventQueue > 0) and eventQueue[1].time or infTime
            numEvents = numEvents + eventsDone
            eventsDone = 0
        end
    end

    local totalEvents
    if self.size > 1 then
        MPI:barrier() --Forcibly synchronize all ranks before counting total events
        totalEvents = MPI:allreduce(numEvents, MPI.SUM)
    else
        totalEvents = numEvents
    end
    if rank == 0 then
        local elapsedClock = os.clock() - startClock
        print("SIMULATION COMPLETED IN: " .. elapsedClock .. " SECONDS")
        print("SIMULATED EVENTS: " .. totalEvents)
        print("EVENTS PER SECOND: " .. totalEvents/elapsedClock)
        print("===========================================")
    end
end

function Simian.schedService(self, time, eventName, data, rx, rxId)
    --[[Purpose: Add an event to the event-queue.
    --For kicking off simulation and waking processes after a timeout]]
    if time > self.endTime then --No need to push this event
        return
    end

    local recvRank = self:getOffsetRank(rx, rxId)

    if recvRank == self.rank then
        local e = {
            tx = nil, --String (Implictly self.name)
            txId = nil, --Number (Implictly self.num)
            rx = rx, --String
            rxId = rxId, --Number
            name = eventName, --String
            data = data, --Object
            time = time, --Number
        }

        if self.size > 1 then
            local epochStart = math.floor(time/self.minDelay)*self.minDelay
            local txEpochs = self.txEpochs
            if txEpochs[epochStart] then
                txEpochs[epochStart] = txEpochs[epochStart] + 1
            else
                txEpochs[epochStart] = 1
                priorityQ.push(self.timeQueueTx, epochStart)
            end
        end
        eventQ.push(self.eventQueue, e)
    end
end

function Simian.getBaseRank(self, name)
    --Can be overridden for more complex Entity placement on ranks
    return hash(name) % self.size
end

function Simian.getOffsetRank(self, name, num)
    --Can be overridden for more complex Entity placement on ranks
    return (self.baseRanks[name] + num) % self.size
end

function Simian.getEntity(self, name, num)
    --Returns a reference to a named entity of given serial number
    if self.entities[name] then
        local entity = self.entities[name]
        return entity[num]
    end
end

function Simian.attachService(self, klass, name, fun)
    --Attaches a service at runtime to an entity klass type
    rawset(klass, name, fun)
end

function Simian.addEntity(self, name, entityClass, num, ...)
    --[[Purpose: Add an entity to the entity-list if Simian is idle
    This function takes a pointer to a class from which the entities can
    be constructed, a name, and a number for the instance.]]
    if self.running then
        error("Adding entity when Simian is running!")
    end

    if not self.entities[name] then
        self.entities[name] = {} --To hold entities of this "name"
    end
    local entity = self.entities[name]

    self.baseRanks[name] = self:getBaseRank(name) --Register base-ranks
    local computedRank = self:getOffsetRank(name, num)

    if computedRank == self.rank then --This entity resides on this engine
        --Output log file for this Entity
        self.out:write(name .. "[" .. num .. "]: Running on rank " .. computedRank .. "\n")

        entity[num] = entityClass(name, self.out, self, num, ...) --Entity is instantiated
    end
end

return Simian
