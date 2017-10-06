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
--  Simple example simulation scipt, showing how to attach services at runtime
--]]
package.path = "Simian/?.lua;" .. package.path
local Simian = require "simian"

local simName, startTime, endTime, minDelay, useMPI = "HELLO-ATTACH", 0, 100000, 0.0001, true

local count = 4

local Alice = Simian.Entity("Alice")
do
    function Alice:__init()
        --All agents have signature: Agent:__init(...)
    end

    function Alice:generate(...)
        local targets = {{entity = "Alice", service = "square"},
                         {entity = "Bob", service = "sqrt"}}

        local target = targets[math.random(#targets)]
        local targetId = math.random(0, count-1)

        local data = math.random(100)

        self.out:write("Time "
                    .. self.engine.now
                    .. ": Sending " .. data .. " to "
                    .. target.entity .. "[" .. targetId .. "]\n")

        self:reqService(10, target.service, data, target.entity, targetId)
        self:reqService(25, "generate")
    end

    function Alice:result(data, tx, txId)
        self.out:write("Time "
                    .. self.engine.now
                    .. ": Got " .. data
                    .. " from " .. tx .. "[" .. txId .. "]\n")
    end
end

function square(self, data, tx, txId)
    self:reqService(10, "result", data^2, tx, txId)
end

local Bob = Simian.Entity("Bob")
do
    function Bob:__init()
        --All agents have signature: Agent:__init(...)
    end

    function Bob:generate(...)
        local targets = {{entity = "Alice", service = "square"},
                         {entity = "Bob", service = "sqrt"}}

        local target = targets[math.random(#targets)]
        local targetId = math.random(0, count-1)

        local data = math.random(100)

        self.out:write("Time "
                    .. self.engine.now
                    .. ": Sending " .. data .. " to "
                    .. target.entity .. "[" .. targetId .. "]\n")

        self:reqService(10, target.service, data, target.entity, targetId)
        self:reqService(25, "generate")
    end

    function Bob:result(data, tx, txId)
        self.out:write("Time "
                    .. self.engine.now
                    .. ": Got " .. data
                    .. " from " .. tx .. "[" .. txId .. "]\n")
    end
end

function sqrt(self, data, tx, txId)
    self:reqService(10, "result", math.sqrt(data), tx, txId)
end

--Initialize Simian
Simian:init(simName, startTime, endTime, minDelay, useMPI)

for i=0,count-1 do
    Simian:addEntity("Alice", Alice, i)
    Simian:addEntity("Bob", Bob, i)
end

Simian:attachService(Alice, "square", square) --Attaches square service at runtime to klass Alice
for i=0,count-1 do --Attach sqrt service at runtime to all Bob instances
    local entity = Simian:getEntity("Bob", i)
    if entity then
        entity:attachService("sqrt", sqrt)
    end
end

for i=0,count-1 do
    Simian:schedService(0, "generate", nil, "Alice", i)
    Simian:schedService(50, "generate", nil, "Bob", i)
end

Simian:run()
Simian:exit()
