--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Main engine script
--]]
local eventQ = require "eventQ"
local hash = require "hash"
local Entity = require "entity"

local Simian = {
    Entity = Entity
}

function Simian.init(self, simName, startTime, endTime, minDelay, useMPI)
    self.name = simName
    self.startTime = startTime
    self.endTime = endTime
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

    --[[Stores the minimum time of any event sent by this process,
    which is used in the global reduce to ensure global time is set to
    the correct minimum.]]
    self.infTime = endTime + 2*minDelay
    self.minSent = self.infTime

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
end

function Simian.exit(self)
    if self.useMPI then
        self.MPI:finalize()
        self.out:close()
        self.out = nil
    end
end

function Simian.run(self) --Run the simulation
    local startTime = os.clock()
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
    local numEvents, totalEvents = 0, 0

    self.running = true
    local baseTime, min = self.startTime, math.min

    local MPI = self.MPI --Cache MPI locally
    while baseTime < self.endTime do
        self.minSent = self.infTime

        while #self.eventQueue > 0
                and self.eventQueue[1].time < baseTime + self.minDelay
                and self.eventQueue[1].time < self.endTime do
            event = eventQ.pop(self.eventQueue) --Next event
            self.now = event.time --Advance time

            --Simulate event
            local entity = self.entities[event.rx][event.rxId]
            local service = entity[event.name]
            service(entity, event.data, event.tx, event.txId) --Receive

            numEvents = numEvents + 1
        end

        local minLeft = self.endTime
        if #self.eventQueue > 0 then
            minLeft = self.eventQueue[1].time
        end

        if self.size > 1 then
            baseTime = MPI:allreduce(min(self.minSent, minLeft), MPI.MIN)
            while MPI:iprobe() do --As long as there are messages waiting
                eventQ.push(self.eventQueue, MPI:recvAnySize())
            end
        else
            baseTime = min(self.minSent, minLeft)
        end
    end

    if self.size > 1 then
        totalEvents = MPI:allreduce(numEvents, MPI.SUM)
    else
        totalEvents = numEvents
    end
    if self.rank == 0 then
        local elapsedTime = os.clock() - startTime
        print("SIMULATION COMPLETED IN: " .. elapsedTime .. " SECONDS")
        print("SIMULATED EVENTS: " .. totalEvents)
        print("EVENTS PER SECOND: " .. totalEvents/elapsedTime)
        print("===========================================")
    end
end

function Simian.schedService(self, time, eventName, data, rx, rxId)
    --[[Purpose: Add an event to the event-queue.
    --For kicking off simulation and waking processes after a timeout]]
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

function Simian.addEntity(self, entityClass, num, ...)
    --[[Purpose: Add an entity to the entity-list if Simian is idle
    This function takes a pointer to a class from which the entities can
    be constructed, a name, and a number for the instance.]]
    if self.running then
        error("Adding entity when Simian is running!")
    end

    local name = entityClass.name
    if not self.entities[name] then
        self.entities[name] = {} --To hold entities of this "name"
    end
    local entity = self.entities[name]

    self.baseRanks[name] = self:getBaseRank(name) --Register base-ranks
    local computedRank = self:getOffsetRank(name, num)

    if computedRank == self.rank then --This entity resides on this engine
        --Output log file for this Entity
        self.out:write(name .. "[" .. num .. "]: Running on rank "
                        .. computedRank .. "\n")

        entity[num] = entityClass(self.out, self, num, ...) --Entity is instantiated
    end
end

return Simian
