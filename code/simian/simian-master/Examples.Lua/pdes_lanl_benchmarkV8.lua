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
PDES LANL BENCHMARK is a benchmark to test parallel discrete event simulation performance
through a combination of communication loads, memory requirements, and computational loads

Overview
==========
Each entity A sends a "request computation" message to another entity B; upon message receipt,
B performs randomly weighted subset sum calculations on its local list data structure.
Each entity A also sends "timer" messages to itself with some delay before it sends another
"request computation" message. The main parameters are as follows:

Communication Parameters
========================
n_ent:        Number of entities
s_ent:        Average number of send events per entity
            Individual entities determine how many events they need to send
            based on p_send and their index and then adjust their local intersend_delay
            using an exponential distribution.
endTime:    Duration of simulation. Note that minDelay = 1.0 always, so
            setting endTime to n_ent*s_ent will result in one event per minDelay
            epoch when running in parallel mode
q_avg:        Average number of events in the event queue per entity
            For individual entities this is made proportional
            the number of total events that the entity needs to send.
            Default value is 1. Higher values will stress-test the event queue
            mechanism of the DES engine
p_receive: Parameter for geometric distribution of destination entities indexed by entity index.
            Entity i receives a fraction of p_receive*(1-p_receive)**(i-1) of all request messages
            Lower-indexed entities receive larger shares
            p_receive = 0: uniform distribution; p_receive = 1: only entity 1 receives messages
p_send:        Parameter for geometric distribution of source entities indexed by entity index
            See p_receive for more details
invert:        Flag to indicate whether receive and sent distribution should be inverted
            If set to True: highest-index entity sends most messages

Memory Parameters
==========================
m_ent:         Average memory footprint per entity,
            modeled as the average linear list size (8 byte units).
            Each entity has a local list as a data structure that  uses up memory
p_list:        Parameter for geometric distribution of linear list sizes
            Set to 0 for uniform distribution
            Set to 1.0 to make entity 0 the only entity with a list

Computation Parameters
==========================
ops_ent:     Average operations per handler per entity.
            Computational cycle use is implemented as a weighted subset sum calculation
            of the first k elements of the list with randomly drawn weights (to eliminate
            the possibility that the calculation gets optimized away).
            Each entity linearly scales down the number of operations based on its local
            list size as determined by p_list.
ops_sigma:     Variance of numer of operations per handler per entity, as a fraction of ops_ent

cache_friendliness:
            Determines how many different list elements are accessed during operations
            traded off with more operations per list element
            Set to p to access the first p fraction of list elements
            Set to 0.0 to access only first list element
            Set to 1.0 to access all list elements
            Set to 0.5 if no other value is known

PDES Parameters
========================
time_bins:    Purely for reporting purposes, this parameter gives the number of equal-size
            time bins in which send events are sent
init_seed:    Initial seed value for random number generation. Built-in Python random number
            generator is used. Seed values are passed along to entities upon creation and
            also as parameters for graph/matrix generation

Output statistics are written into the output file of entity 0.

POC: Stephan Eidenbenz, eidenben@lanl.gov

Ported to Lua: Nandakishore Santhi, nsanthi@lanl.gov
Date: April 16, 2015
]]
package.path = "Simian/?.lua;" .. package.path
jit.opt.start(3, 'hotloop=3', 'hotexit=8', 'instunroll=10', 'loopunroll=10', 'callunroll=10', 'recunroll=10', 'tryside=30')

local Simian = require "simian"

-------------------------- Variables ------------------------------------------------------
if #arg ~= 17 then
    print("Usage: luajit-2.1 " .. arg[0] .. " n_ent s_ent q_avg p_receive p_send invert m_ent p_list ops_ent ops_sigma cache_friendliness init_seed time_bins endTime minDelay useMPI logName")
    os.exit()
end

local n_ent = tonumber(arg[1])
local s_ent = tonumber(arg[2])
local q_avg = tonumber(arg[3])
local p_receive = tonumber(arg[4])
local p_send = tonumber(arg[5])
local invert = (tostring(arg[6]):lower() == "true")

local m_ent = tonumber(arg[7])
local p_list = tonumber(arg[8])
local ops_ent = tonumber(arg[9])
local ops_sigma = tonumber(arg[10])
local cache_friendliness = tonumber(arg[11])
local init_seed = tonumber(arg[12])
local time_bins = tonumber(arg[13])

local endTime
if tostring(arg[14]):lower() == "compute" then
    endTime = n_ent*s_ent
else
    endTime = tonumber(arg[14])
end
local minDelay = tonumber(arg[15])
local useMPI = (tostring(arg[16]):lower() == "true")

local logName = arg[17]

print(n_ent, s_ent, q_avg, p_receive, p_send, invert, m_ent, p_list, ops_ent, ops_sigma, cache_friendliness, init_seed, time_bins, endTime, minDelay, useMPI, logName)

--[[
-------------------------- Variables ------------------------------------------------------
local n_ent = 10        -- Number of entities 10
local s_ent = 10000        -- Average number of send events per entity 100
                -- Individual entities determine how many events they need to send
                -- based on p_send and their index and then adjust their local intersend_delay
                -- using an exponential distribution.
--local endTime = n_ent*s_ent        -- Duration of simulation. Note that minDelay = 1.0 always, so
local endTime = 5        -- Duration of simulation. Note that minDelay = 1.0 always, so
                -- setting endTime to n_ent*s_ent will result in one event per minDelay
                -- epoch when running in parallel mode
local q_avg = 1 -- Average number of events in the event queue per entity
                -- For individual entities this is made proportional
                -- the number of total events that the entity needs to send.
                -- Default value is 1. Higher values will stress-test the event queue
                -- mechanism of the DES engine
                -- try from 1(default), 0.2*s_ent, 0.5*s_ent, 0.8*s_ent,s_ent
local p_receive = 0            -- Parameter to geometric distribution for choosing destination entities
                -- Set to 0 for uniform distribution
                -- Set to 1.0 to make entity 0 the only destination
                -- Lower index entities receive more messages
local p_send = 0            -- Parameter for geometric distribution of source entities

                -- Set to 0 for uniform distribution
                -- Set to 1.0 to make entity 0 the only source
local invert = false    -- Flag to indicate whether receive and sent distribution should be inverted
                -- If True: entity n_ent sends most  messages

local m_ent = 1000    -- Average memory footprint per entity,
                -- modeled as the average linear list size (8 byte units)
local p_list    = 0    -- Parameter for geometric distribution of linear list sizes
                -- Set to 0 for uniform distribution
                -- Set to 1.0 to make entity 0 the only entity with a list
local ops_ent = 1000    -- Average operations per handler per entity.
local ops_sigma = 0    -- Variance of numer of operations per handler per entity, as a fraction of ops_ent
                -- drawn from a Gaussian
local cache_friendliness = 0.5
                -- Determines how many different list elements are accessed during operations
                -- traded off with more operations per list element
                -- Set to p to access the first p fraction of list elements
                -- Set to 0.0 to access only first list element (cache-friendly)
                -- Set to 1.0 to access all list elements (cache-unfriendly)
                -- Set to 0.5 if no other value is known

local init_seed = 1    -- Initial random seed to be passed around
local time_bins = 10     -- Number of bins for time and event reporting (Stats only)
local useMPI = true
]]

------------------------------------------------
--  Initialization stuff
local max, min, int, inf, ceil, log, sqrt, random, randomseed = math.max, math.min, math.floor, 1e100, math.ceil, math.log, math.sqrt, math.random, math.randomseed

local function exponential(lambda)
    --Lambda := 1/Mean
    return -log(random())/lambda
end

local function gauss(mu, sigma)
    local x1, x2, w, y1
 
    w = 2.0
    while (w >= 1.0) do
        x1 = 2.0 * random() - 1.0
        x2 = 2.0 * random() - 1.0
        w = (x1^2) + (x2^2)
    end

    w = sqrt( (-2.0 * log( w ) ) / w )
    y1 = x1 * w

    return (y1 * sigma) + mu
end

local minDelay = 1.0    -- Minimum Delay value for synchronization between MPI ranks (if applicable)
local endTime = max(endTime, 2)

-- Compute the min value for geometric distribution function
local r_min = (1-p_receive)^n_ent
-- Compute target number of send events
local target_global_sends = n_ent * s_ent

local simName = "PDES_LANL_Benchmark_" .. logName
local startTime =  0.0

Simian:init(simName, startTime, endTime+3*minDelay, minDelay, useMPI)
-- Note little trick with endTime setting, as we need to collect statistics in the end

------------------------------------------------
local PDES_LANL_Node = Simian.Entity("PDES_LANL_Node")
do
    function PDES_LANL_Node:__init(...)
        seed = self.num + init_seed -- initialize random seed with own id
        -- 1. Compute number of events that the entity will send out
        local prob
        if p_send == 0 then -- uniform case
            prob =  1.0/n_ent
        else
            if invert then
                prob =  p_send*(1-p_send)^(n_ent - self.num) -- Probability that an event gets generated on this entity
            else
                prob =  p_send*(1-p_send)^self.num
            end
        end
        local target_sends = int(prob * target_global_sends)
        if target_sends > 0 then
            self.local_intersend_delay = endTime/target_sends
        else
            self.local_intersend_delay = 10*endTime
            -- if the entity sends zero events, we let it create one that will most likely be
            -- after the sim ends
        end

        -- 2. Allocate appropriate memory space through list size, and number of ops
        if p_list == 0 then -- uniform case
            prob =  1.0/n_ent
        else
            prob =  p_list*(1-p_list)^self.num
        end
        self.list_size = int(prob * n_ent * m_ent)+1 -- there are n_ent*m_ent list elements in total
        self.ops = int(prob * n_ent * ops_ent)+1 -- there are n_ent*m_ent list elements in total
        self.active_elements = int(cache_friendliness * self.list_size) -- only this many list elements will be accessed
        self.list = {}
        for i=1,self.list_size do -- create a list of random elements of length list_size
            self.list[i] = random()
        end

        -- 3. Set up queue size
        self.q_target = q_avg/s_ent * target_sends
        self.q_size = 1 -- number of send events scheduled ahead of time by this entity
        self.last_scheduled = self.engine.now -- time of last scheduled event

        -- 4. Set up statistics
        self.send_count, self.receive_count, self.opsCount =  0, 0, 0 -- for stats
        self.ops_max, self.ops_min, self.ops_mean = 0, inf, 0.0 -- for stats
        self.time_sends = {} -- for time reporting
        for i=1,time_bins do
            self.time_sends[i] = 0
        end
        if self.num == 0 then -- only for the global statistics entity
            self.stats_received, self.gsend_count, self.greceive_count, self.gopsCount = 0, 0, 0, 0
            self.gops_max, self.gops_min, self.gops_mean = 0, inf, 0.0
            self.gtime_sends = {} -- for time reporting
            for i=1,time_bins do
                self.gtime_sends[i] = 0
            end
        end

        -- 5. Schedule FinishUp at end of time
        self:reqService(endTime - self.engine.now, "FinishHandler", nil)
        self:SendHandler(seed)
    end

    function PDES_LANL_Node:SendHandler(seed, ...) -- varargs ... is artificial
        randomseed(seed)
        self.send_count = self.send_count + 1
        self.q_size = self.q_size - 1
        local key = int(self.engine.now/(endTime+0.0001)*time_bins) + 1 --TODO: Check!
        self.time_sends[key] = self.time_sends[key] + 1
        -- Generate next event for myself
        -- Reschedule myself until q is full or time has run out
        while (self.q_size < self.q_target) and not (self.last_scheduled > endTime) do
            local own_delay = exponential(1.0/self.local_intersend_delay)
            self.last_scheduled =  self.last_scheduled + own_delay
            if self.last_scheduled < endTime then
                self.q_size = self.q_size + 1
                self:reqService(self.last_scheduled-self.engine.now, "SendHandler", random())
            end
        end
        -- Generate computation request event to destination entity
        local DestIndex
        if p_receive == 1.0 then -- If p is exactly 1.0, then the only entity 0 is only destination
            DestIndex = 0
        elseif p_receive == 0 then -- by convention, p == 0 means we want uniform distribution
            DestIndex = int(random()*n_ent)
        else
            local U = (1.0 - r_min)*random() + r_min -- We computed r_min such that the we only get indices less than num_ent
            DestIndex = int(ceil(log(U) / log(1.0 - p_receive))) - 1 --TODO: Check!
        end
        local new_seed = random()
        -- Send event to destination ReceiveHandler (only if not past reporting time)
        if self.engine.now+minDelay < endTime then
            self:reqService(minDelay, "ReceiveHandler", new_seed, "PDES_LANL_Node", DestIndex)
        end
    end

    function PDES_LANL_Node:ReceiveHandler(seed, ...) -- varargs ... is artificial
        randomseed(seed)
        local r_ops = max(1, int(gauss(self.ops, self.ops*ops_sigma))) -- number of operations
        local r_active_elements = int(self.active_elements * (r_ops/self.ops)) -- only this many list elements will be accessed
        local r_active_elements = min(r_active_elements, self.list_size) -- cannot be more than list size
        local r_active_elements = max(1, r_active_elements) -- cannot be less than 1
        local r_ops_per_element = int(r_ops/r_active_elements)
        -- Update stats
        self.receive_count = self.receive_count + 1
        self.ops_max = max(self.ops_max, r_ops)
        self.ops_min = min(self.ops_min, r_ops)
        self.ops_mean = (self.ops_mean*(self.receive_count-1) +  r_ops)/self.receive_count
        -- Compute loop
        local value = 0.0
        for i=1,r_active_elements do
            for j=1,r_ops_per_element do
                value = value + self.list[i] * random()
                --value = value + (self.list[i] * self.list[i])
                self.opsCount = self.opsCount + 1
            end
        end
        return value
    end

    function PDES_LANL_Node:FinishHandler(...) -- varargs ... is artificial
        -- Send stats to entity 0 for outputting of global stats
        local msg = {self.num, self.send_count, self.receive_count, self.ops_min, self.ops_mean, self.ops_max, self.time_sends, self.opsCount}
        self:reqService(minDelay, "OutputHandler", msg, "PDES_LANL_Node", 0)
    end

    function PDES_LANL_Node:OutputHandler(msg, ...) -- varargs ... is artificial
        local header = string.format("%10s %10s %10s %10s %10s %10s    %q\n", "#EntityID", "#Sends", "#Receives", "Ops(Min", "Avg", "Max)", "Time Bin Sends")
        -- Write out Stats, only invoked on entity 0
        if self.stats_received == 0 then -- Only write header line a single time
            self.out:write(header)
        end

        self.stats_received = self.stats_received + 1
        self.gops_mean = (msg[3]*msg[5] + self.gops_mean*self.greceive_count)/(self.greceive_count+msg[3])
        self.gsend_count = self.gsend_count + msg[2]
        self.greceive_count = self.greceive_count + msg[3]
        self.gopsCount = self.gopsCount + msg[8]
        self.gops_min = min(self.gops_min, msg[4])
        self.gops_max = max(self.gops_max, msg[6])
        for i=1,time_bins do
            self.gtime_sends[i] = self.gtime_sends[i] + msg[7][i]
        end

        local str_out = string.format("%10d %10d %10d %10d %10.5g %10d    %s\n", msg[1], msg[2], msg[3], msg[4], msg[5], msg[6], "[" .. table.concat(msg[7], ", ") .. "]")
        self.out:write(str_out)

        if self.stats_received == n_ent then -- We can write out global stats
            self.out:write("===================== LANL PDES BENCHMARK  Collected Stats from All Ranks =======================\n")
            header = string.format("%10s %10s %10s %10s %10s %10s %10s    %q\n", "#Entities", "#Sends", "#Receives", "OpsCount", "Ops(Min", "Avg", "Max)", "Time Bin Sends")
            self.out:write(header)
            str_out = string.format("%10d %10d %10d %10d %10d %10.5g %10d    %s\n", n_ent, self.gsend_count, self.greceive_count, self.gopsCount, self.gops_min, self.gops_mean, self.gops_max, "[" .. table.concat(self.gtime_sends, ", ") .. "]")
            self.out:write(str_out)
            self.out:write("=================================================================================================\n")
        end
    end
end

----------------------------------------------------------------
-- "MAIN"
----------------------------------------------------------------
for i=0,n_ent-1 do
    Simian:addEntity("PDES_LANL_Node", PDES_LANL_Node, i)
end

-- 5. Run Simian
Simian:run()

Simian:exit()
