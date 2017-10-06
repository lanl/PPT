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
--  Simple example simulation script for PHOLD with application process
--]]
package.path = "Simian/?.lua;" .. package.path

local Simian = require "simian"
local ln, random = math.log, math.random

local simName, startTime, endTime, minDelay, useMPI = "PROC-NOMPI", 0, 10000, 0.0001, false

local count = 10
local lookahead = minDelay

local function exponential(mean)
    return -ln(random())/mean
end

--Example of a process on an entity
local function appProcess(this, data1, data2) --Here arg(1) "this" is current process
    local entity = this.entity
    entity.out:write("Process App started with data: " .. tostring(data1) .. ", " .. tostring(data2) .. "\n")
    while true do
        local x = random(100)
        --Shows how to log outputs
        entity.out:write("Time " .. entity.engine.now
            .. ": Process App is sleeping for " .. x .. "\n")
        this:sleep(x) --Shows how to compute/sleep
        entity.out:write("Time " .. entity.engine.now
            .. ": Waking up Process App\n")
    end
end

local Node = Simian.Entity("Node")
function Node:__init(...)
    self:createProcess("App", appProcess) --Shows how to create "App"
    self:startProcess("App", 78783, {two = 2}) --Shows how to start "App" process with arbitrary number of data
end

function Node:generate(...)
    local targetId = random(count)
    local offset = exponential(1) + lookahead

    --Shows how to log outputs
    self.out:write("Time "
                .. self.engine.now
                .. ": Waking " .. targetId
                .. " at " .. offset .. " from now\n")

    self:reqService(offset, "generate", nil, "Node", targetId)
end

--Initialize Simian
Simian:init(simName, startTime, endTime, minDelay, useMPI)

for i=1,count do
    Simian:addEntity("Node", Node, i)
end

for i=1,count do
    Simian:schedService(0, "generate", nil, "Node", i)
end

Simian:run()
Simian:exit()
