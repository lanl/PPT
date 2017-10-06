-- Copyright (c) 2014. Los Alamos National Security, LLC. 

-- This material was produced under U.S. Government contract DE-AC52-06NA25396
-- for Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
-- National Security, LLC for the U.S. Department of Energy. The U.S. Government 
-- has rights to use, reproduce, and distribute this software.  

-- NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, 
-- EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
-- If software is modified to produce derivative works, such modified software should
-- be clearly marked, so as not to confuse it with the version available from LANL.

-- Additionally, this library is free software; you can redistribute it and/or modify
-- it under the terms of the GNU Lesser General Public License v 2.1 as published by the 
-- Free Software Foundation. Accordingly, this library is distributed in the hope that 
-- it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE.txt for more details.

local Simian = require "simian"
local processors = require "processors"

local module = {}
--[[
--Base class for Node. Actual nodes should derive from this. The application can add handlers to this base class.
--]]
module.Node =  Simian.Entity("Node")

--[[
--Class that represents a KnightsLanding compute node
--]]
module.KNLNode = Simian.Entity("KNLNodes", module.Node)
module.KNLNode.memory_footprint =  0
---
-- PARAMETERS
---      
module.KNLNode.num_cores =  64  -- number of cores on the node
module.KNLNode.memorysize = 512 * module.KNLNode.num_cores  -- total memory on node  in MB
module.KNLNode.filesystem_access_time =  1.0      -- filesystem access time in seconds
module.KNLNode.interconnect_bandwidth = 1.0 * 10^8 -- speed of interconnect in bits/sec
module.KNLNode.interconnect_latency = 10^-9      -- Delay in getting packet ready in sec      
--module.KNLNode.out:write("KNL node generated at time "..module.KNLNode.engine.now.."\n")
-- So let's generate the cores
module.KNLNode.cores = {}
for i =1,module.KNLNode.num_cores do
  module.KNLNode.cores[i] = processors.KNLCore(module.KNLNode)
  --module.KNLNode.out:write(node.num_cores.." KNL cores generated at time "..node.engine.now.."\n")
end  
--[[
--Allocates or deallocates memory of size 
--]]
function module.KNLNode:mem_alloc(size)
  self.memory_footprint = self.memory_footprint + size
  if self.memory_footprint > self.memorysize then
    if size < 0 then -- still too high, but at least in unalloc
      print("Warning: KNLNode"..self.." still out of memory at time "..self.engine.now)
      return true
    else
      --print "Warning: KNLNode", self, " ran out of memory at time ", simx.get_now()
      return false
    end
  else
    if self.memory_footprint - size > self.memorysize then
      -- We are back to normal memory use
      print("Warning: KNLNode" .. self .. " has recovered from swap mode " .. self.engine.now)
      return true
    end
  end   
end
--[[
-- Class that represents a Cielo compute node
--]]
module.CieloNode = Simian.Entity("CieloNode", module.Node)
module.CieloNode.memory_footprint =  0 -- Number of bytes allocated in memory
  
--[[
-- PARAMETERS
--]]
module.CieloNode.num_cores = 16 -- number of cores on the node
module.CieloNode.memorysize = 32000 -- total memory on node  in MB

module.CieloNode.interconnect_bandwidth = 6.8 * 10^10      -- speed of interconnect in bits/sec
module.CieloNode.interconnect_latency = 10^-6              -- Delay in getting packet ready in sec
module.CieloNode.interconnect_latency_mpi = 1.5 * 10^-6    -- Delay in getting MPI packet ready in sec

-- This number - needs to look more
module.CieloNode.filesystem_access_time =  1.0      -- filesystem access time in seconds
        
--module.CieloNode.out:write("Cielo node generated at time "..node.engine.now.."\n")
-- So let's generate the cores
module.CieloNode.cores = {}
for i =1, module.CieloNode.num_cores do
  module.CieloNode.cores[i] = processors.CieloCore(module.CieloNode)
  --module.CieloNode.out:write(module.CieloNode.num_cores.." Cielo cores generated at time "..node.engine.now.."\n")
end
--[[
-- Allocates or deallocates memory of size 
--]]
function module.CieloNode:mem_alloc(size)
  self.memory_footprint = self.memory_footprint + size
  if self.memory_footprint > self.memorysize then
    if size < 0 then -- still too high, but at least in unalloc
      print("Warning: CieloNode"..self.." still out of memory at time "..self.engine.now)
      return true
    else
      --print "Warning: KNLNode", self, " ran out of memory at time ", simx.get_now()
      return false
    end
  else
    if self.memory_footprint - size > self.memorysize then
      -- We are back to normal memory use
      print("Warning: CieloNode"..self.." has recovered from swap mode "..self.engine.now)
      return true
    end
  end   
end

--[[
--Class that represents a MacPro compute node
--]]
module.MBPNode = Simian.Entity("MBPNode", module.Node)
module.MBPNode.memory_footprint =  0 -- Number of bytes allocated in memory
  
--[[
-- PARAMETERS
--]]
        
module.MBPNode.num_cores = 12                     -- number of cores on the node
module.MBPNode.memorysize = 64000     -- total memory on node  in MB

module.MBPNode.interconnect_bandwidth = 6.8 * 10^10      -- speed of interconnect in bits/sec
module.MBPNode.interconnect_latency = 10^-6              -- Delay in getting packet ready in sec
module.MBPNode.interconnect_latency_mpi = 1.5 * 10^-6    -- Delay in getting MPI packet ready in sec

-- This number - needs to look more
module.MBPNode.filesystem_access_time =  1.0      -- filesystem access time in seconds
        
--module.MBPNode.out:write("MacPro node generated at time "..module.MBPNode.engine.now.."\n")
-- So let's generate the cores
module.MBPNode.cores = {}
for i = 1, module.MBPNode.num_cores do
  module.MBPNode.cores[i] = processors.MacProCore(module.MBPNode)
  --module.MBPNode.out:write(module.MBPNode.num_cores.." MacPro cores generated at time "..module.MBPNode.engine.now.."\n")
end
--[[
-- Allocates or deallocates memory of size 
]]--     
function module.MBPNode:mem_alloc(size)
  self.memory_footprint = self.memory_footprint + size
  if self.memory_footprint > self.memorysize then
    if size < 0 then -- still too high, but at least in unalloc
      print ("Warning: MacProNode"..self.." still out of memory at time "..self.engine.now)
      return true
    else
      --print "Warning: MacProNode", self, " ran out of memory at time ", simx.get_now()
      return false
    end
  else
    if self.memory_footprint - size > self.memorysize then
      -- We are back to normal memory use
      print ("Warning: MacProNode"..self.." has recovered from swap mode "..self.engine.now)
      return true
    end
  end
end

return module