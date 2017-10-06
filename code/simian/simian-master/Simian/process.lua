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

--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JIT-PDES Engine written in Lua
--  Named processes
--Process class:
--
--ent:createProcess/proc:hibernate <=> proc:wake/ent:wakeProcess
--proc:sleep/proc:compute <=> ent:wakeProcess
--
local wake = function(thisProcess, ...)
    --Arguments "..." to __call => function-body
    --Arguments "..." to wake => LHS of hibernate
    local co = thisProcess.co
    if co and coroutine.status(co) == "suspended" then
        local ok, res = coroutine.resume(co, ...)
        if ok then
            return res
        else
            error("Could not explicitly wake the process: " .. thisProcess.name .. ":\n\t" .. res)
        end
    else
        error("Attempted to wake a process which was either killed or not suspended: " .. thisProcess.name)
    end
end

local hibernate = function(thisProcess, ...)
    --Arguments "..." to hibernate => LHS of wake
    --Processes to be woken explicitly by events may return values
    if thisProcess.co then
        return coroutine.yield(...)
    else
        error("Cannot hibernate killed process")
    end
end

local sleep = function(thisProcess, x, ...)
    --Processes which are to implicitly wake at set timeouts
    --All return values are passed to __call/wake
    if type(x) ~= "number" or x < 0 then
        error("Sleep not given non-negative number argument!")
    end

    --Schedule a local event after x timesteps to wakeup
    if thisProcess.co then
        local entity = thisProcess.entity
        entity.engine:schedService(entity.engine.now + x, "_wakeProcess",
                    thisProcess.name, entity.name, entity.num)
        return coroutine.yield(...)
    else
        error("Cannot sleep killed process")
    end
end

local categorize = function(thisProcess, kind, name)
    local entity = thisProcess.entity
    entity:categorize(kind, name) --Also categorize as @kind on entity
end

local unCategorize = function(thisProcess, kind, name)
    local entity = thisProcess.entity
    entity:unCategorize(kind, name) --Also categorize as @kind on entity
end

local spawn = function(thisProcess, name, fun, kind)
    --Create a new named processes as child or @kind
    local entity = thisProcess.entity
    if entity._procList[name] then
        error("spawn: Process by name '" .. name .. "' already exists in entity " .. entity.name .. "[" .. entity.num .. "]")
    end
    entity:createProcess(name, fun, kind) --Creates a named process of kind type
    --Make this a child of thisProcess
    --NOTE: This is the difference between process:spawn and entity:createProcess
    entity._procList[name].parent = thisProcess
    if thisProcess._childList then
        thisProcess._childList[name] = true
    else
        thisProcess._childList = {[name] = true}
    end
end

local killallChildren = function(thisProcess) --Hidden function to kill all children
    local entity = thisProcess.entity
    if thisProcess._childList then
        for name,_ in pairs(thisProcess._childList) do
            local proc = entity._procList[name] --Get actual process
            if proc then
                proc:kill() --Kill child and all its subprocesses
            end
        end
        thisProcess._childList = nil --Empty child table
    end
end

local kill = function(thisProcess, name) --Kills itself, or named child-process
    --name: One of nil or process-name
    local entity = thisProcess.entity
    local parent = thisProcess.parent
    if not name then --Kill self
        killallChildren(thisProcess) --Killall children recursively
        thisProcess.co = nil
        --Parent process is guaranteed to be alive
        if parent then --Remove from child-list of parent
            parent._childList[thisProcess.name] = nil
        end
        --Parent entity is always guaranteed to be alive
        --Remove references from entity category and process lists
        if thisProcess._kindSet then
            for k,_ in pairs(thisProcess._kindSet) do
                entity._category[k][thisProcess.name] = nil
            end
        end
        entity._procList[thisProcess.name] = nil --Remove all references to this process
    elseif name == "*" then --Kill every chid-process
        killallChildren(thisProcess)
    elseif thisProcess._childList and thisProcess._childList[name] then --Is this a child process?
        entity._procList[name]:kill() --Kill it
    end
end

local is_a = function(thisProcess, kind)
    local name = thisProcess.name
    local entity = thisProcess.entity
    if entity._category[kind] and entity._category[kind][name] and entity._procList[name] then --Is indeed a @kind?
        return true
    end
    return false
end

local getCategoryNames = function(thisProcess)
    local kindSet = {}
    if thisProcess._kindSet then
        local n = 1
        for k,_ in pairs(thisProcess._kindSet) do
            kindSet[n] = k
            n = n + 1
        end
    end
    return kindSet
end

local getChildNames = function(thisProcess)
    local nameSet = {}
    if thisProcess._childList then
        local n = 1
        for k,_ in pairs(thisProcess._childList) do
            nameSet[n] = k
            n = n + 1
        end
    end
    return nameSet
end

local status = function(thisProcess)
    if thisProcess.started then
        return coroutine.status(thisProcess.co)
    else
        return "NotStarted"
    end
end

--Helper stuff
local process_mt = {
    --Processes lookup missing attributes in this metatable, this reduces memory use
    wake = wake,
    hibernate = hibernate,
    sleep = sleep,
    spawn = spawn,
    kill = kill,
    is_a = is_a,
    status = status,
    categorize = categorize,
    unCategorize = unCategorize,
    getCategoryNames = getCategoryNames,
    getChildNames = getChildNames,
}
process_mt.__index = process_mt

local processFactory = function(name, fun, thisEntity, thisParent)
    --Creates and returns a function which resumes a coroutine
    local process = {
        name = name,
        co = coroutine.create(fun),
        started = false,
        entity = thisEntity,
        parent = thisParent, --Parent is nil if created by entity

        --These optional members will be created as needed at runtime to conserve memory:
        --At present, on OSX, per process memory need is about 2842 bytes for simple processes
        --_kindSet = {}, --Set of kinds that it belongs to on its entity
        --_childList = {}, --Set of kinds that it belongs to on its entity
    }

    return setmetatable(process, process_mt)
end

return processFactory
