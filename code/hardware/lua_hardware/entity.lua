--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Named entity class with single inheritence and processes
--]]
local eventQ = require "eventQ"
local processFactory = require "process"

local function Entity(name, base)
    local c = {name = name} --A named entity class

    if type(base) == 'table' then --New entity is a shallow copy of base
        for k,v in pairs(base) do
            c[k] = v
        end
        c._base = base
    end

    c.is_a = function(thisEntity, entityClass)
        local m = getmetatable(thisEntity)
        while m do
            if m == entityClass then return true end
            m = m._base
        end
        return false
    end

    c.reqService = function(thisEntity, offset, eventName, data, rx, rxId)
        --Purpose: Send an event if Simian is running
        local engine = thisEntity.engine --Get the engine for this entity

        if rx and offset < engine.minDelay then
            if not engine.running then
                error("Sending event when Simian is idle!")
            end
            --If sending to thisEntity, then do not check against min-delay
            error(thisEntity.name .. "[" .. thisEntity.num .. "]"
                .. " attempted to send with too little delay")
        end

        local rx = rx or thisEntity.name
        local rxId = rxId or thisEntity.num
        local time = engine.now + offset

        local e = {
            tx = thisEntity.name, --String
            txId = thisEntity.num, --Number
            rx = rx, --String
            rxId = rxId, --Number
            name = eventName, --String
            data = data, --Object
            time = time, --Number
        }

        local recvRank = engine:getOffsetRank(rx, rxId)

        if recvRank == engine.rank then --Send to thisEntity
            eventQ.push(engine.eventQueue, e)
        else
            if time < engine.minSent then
                engine.minSent = time
            end
            engine.MPI:send(e, recvRank) --Send to others
        end
    end

    c.attachService = function (thisEntity, name, fun)
        --Attaches a service to instance at runtime
        thisEntity[name] = fun
    end

    --Following code is to support coroutine processes on entities:
    --Entity methods to interact with processes
    c.createProcess = function(thisEntity, name, fun, kind) --Creates a named process
        if name == "*" then
            error("Reserved name to represent all child processes: " .. name)
        end
        local proc = processFactory(name, fun, thisEntity, nil)
        if not proc then
            error("Could not create a valid process named: " .. name)
        end
        thisEntity._procList[name] = proc
        if kind then
            thisEntity:categorizeProcess(kind, name) --Categorize
        end
    end

    c.startProcess = function(thisEntity, name, ...) --Starts a named process
        local proc = thisEntity._procList[name]
        if proc then
            if not proc.started then
                proc.started = true
                --When starting, pass process instance as first arg
                --This first arg is in turn passed to the process function
                --(which is the factory for the co-routine - as "this" in examples)
                --Through this argument all the members of the process table below become
                --accessible to all the process-instance co-routine functions.
                proc:wake(proc, ...)
            else
                error("Starting an already started process: " .. proc.name)
            end
        end
    end

    c._wakeProcess = function(thisEntity, name) --Hidden, implicit wake a named process without arguments
        local proc = thisEntity._procList[name]
        if proc then --If existing and not been killed asynchronously
            proc:wake()
        end
    end

    c.wakeProcess = function(thisEntity, name, ...) --Wake a named process with arguments
        local proc = thisEntity._procList[name]
        if not proc then
            error("Attempted to wake a non existant process: " .. name)
        else
            proc:wake(...)
        end
    end

    c.killProcess = function(thisEntity, name) --Kills named process or all entity-processes
        if name then --Kills named child-process
            local proc = thisEntity._procList[name]
            proc:kill() --Kill itself and all subprocesses
        else --Kills all subprocesses
            for _,proc in pairs(thisEntity._procList) do
                proc:kill() --Kill itself and all subprocesses
            end
            thisEntity._procList = {} --A new process table
        end
    end

    c.killProcessKind = function(thisEntity, kind) --Kills all @kind processes on entity
        local nameSet = thisEntity._category[kind]
        if not nameSet then
            error("killProcessKind: No category of processes on this entity called " .. tostring(kind))
        end
        for name,_ in pairs(nameSet) do
            local proc = thisEntity._procList[name]
            if proc then
                proc:kill() --Kill itself and all subprocesses
            end
        end
    end

    c.statusProcess = function(thisEntity, name)
        local proc = thisEntity._procList[name]
        if proc then
            return proc:status()
        else
            return "NonExistent"
        end
    end

    c.categorizeProcess = function(thisEntity, kind, name)
        --Check for existing process and then categorize
        local proc = thisEntity._procList[name]
        if proc then --Categorize both ways for easy lookup
            if not proc._kindSet then
                proc._kindSet = {}
            end
            proc._kindSet[kind] = true --Indicate to proc that it is of this kind to its entity
            --Indicate to entity that proc is of this kind
            local kindList = thisEntity._category[kind]
            if not kindList then --New kind
                thisEntity._category[kind] = {[name] = true}
            else --Existing kind
                kindList[name] = true
            end
        else
            error("categorize: Expects a proper child to categorize")
        end
    end

    c.unCategorizeProcess = function(thisEntity, kind, name)
        --Check for existing process and then unCategorize
        local proc = thisEntity._procList[name]
        if proc then --unCategorize both ways for easy lookup
            if not proc._kindSet then
                proc._kindSet[kind] = nil --Indicate to proc that it is not of this kind to its entity
            end
            --Indicate to entity that proc is not of this kind
            local kindList = thisEntity._category[kind]
            if kindList then --Existing kind
                kindList[name] = nil
            end
        else
            error("unCategorize: Expects a proper child to un-categorize")
        end
    end

    c.isProcess = function(thisEntity, name, kind)
        local proc = thisEntity._procList[name]
        if proc then
            return proc:is_a(kind)
        end
    end

    c.getProcess = function(thisEntity, name)
        --A reference to a named process is returned if it exists
        --NOTE: User should delete it by setting to nil to free its small memory if no longer needed
        local proc = thisEntity._procList[name]
        if proc then
            return proc
        else
            return nil
        end
    end

    c.getCategoryNames = function(thisEntity)
        local kindSet = {}
        local n = 1
        for k,_ in pairs(thisEntity._category) do
            kindSet[n] = k
            n = n + 1
        end
        return kindSet
    end

    c.getProcessNames = function(thisEntity)
        local nameSet = {}
        local n = 1
        for k,_ in pairs(thisEntity._procList) do
            nameSet[n] = k
            n = n + 1
        end
        return nameSet
    end

    c.__index = c --Class-methods are looked up in this table by entity-instances

    local mt = {
        __call = function(entityTable, out, engine, num, ...)
            --Constructor to be called as <entityName>(<args>)
            local obj = {
                out = out, --Log file for this instance
                engine = engine, --Engine
                num = num, --Serial Number
                _procList = {}, --A separate process table for each instance
                _category = {}, --A map of sets for each kind of process
            }
            setmetatable(obj, c) --The table "c" will be the metatable for all entity instances

            if entityTable.__init then --init this entity
                entityTable.__init(obj, ...)
            elseif base and base.__init then --init the base entity
                base.__init(obj, ...)
            end

            return obj --An instance of the entity
        end
    }

    return setmetatable(c, mt) --Return Entity class
end

return Entity
