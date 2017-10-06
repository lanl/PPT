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
local nodes  = require "nodes"

local module = {}
---
-- Class that represents a supercomputer cluster. It consists of a number of nodes
---
function module.Cluster(simianEngine)
  local cluster = {}
  cluster.num_nodes = 100 -- number of compute nodes
  for i =1,cluster.num_nodes do
    simianEngine:addEntity(nodes.Node, i)
  end
  return cluster
end  


---
-- Class that represents a supercomputer cluster. It consists of a number of nodes
---
function module.HalfTrinity(simianEngine, n)
  local trin = {} 
  trin.num_nodes = n -- number of compute nodes
  for i = 1,trin.num_nodes do
    simianEngine:addEntity(nodes.KNLNode, i)
  end
  return trin
end
   
---
-- Class that represents a supercomputer cluster. It consists of a number of nodes
---
function module.MiniTrinity(simianEngine)
  local trin = {} 
  trin.num_nodes = 50 -- number of compute nodes  
  for i = 1,trin.num_nodes do
    simianEngine:addEntity(nodes.KNLNode, i)
  end
  return trin
end         

---
-- Class that represents a supercomputer scluster. It consists of a number of nodes
---
function module.SingleCielo(simianEngine)
  local ciel = {}
  ciel.num_nodes = 50 -- number of compute nodes
  for i = 1, ciel.num_nodes do
    simianEngine:addEntity(nodes.CieloNode, i)
  end
  return ciel
end

---
-- Class that represents a single MBP, my own machine
---
function module.SingleMBP(simianEngine)
  local mbp = {}
  mbp.num_nodes = 50 -- number of compute nodes
  for i = 1, mbp.num_nodes do
    simianEngine:addEntity(nodes.MBPNode, i)
  end
  return mbp
end

return module
