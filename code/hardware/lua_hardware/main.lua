---
-- Test for performance prediction in LUA.
 
-- Date: January 29, 2015
---

local math = require "math"

local simianEngine = require "simian"
local clusters = require "clusters"
local nodes = require "nodes"

--########################
    
local function LUA_Test(node, arg, args)
  local num_index_vars    = 10 -- number of index variables
  local num_float_vars    = 5 +  38*2 -- number of float variables
  local index_loads       = 141+8*2 --  all integer loads, ignoring logical
  local float_loads       = 25 + 38*2 + 4*2 + 5*4 -- all from arrays
      
  local avg_dist          = 1   -- average distance in arrays between accessed elements 
  local avg_reuse_dist    = 100 -- avg number of unique loads between two consecutive accesses of the same element (ie use different weights for int and float loads)
  local stdev_reuse_dist  = 100 -- stddev number of unique loads between two consecutive accesses of the same element
  
  local int_alu_ops       = 51+12+6*2 -- includes logical ops
  local float_alu_ops     = 11 + 1*2
  local float_vector_ops  = 39 + 6*2
  
  local regcount      = 8
  local core          = node.cores[1]
  local nb_iter       = 1000
  local vector_size   = 2048*14*128
  local CPU_tasklist  = {{'iALU', int_alu_ops}, {'fALU', float_alu_ops}, 
                        {'VECTOR', float_vector_ops, 2}, {'MEM_ACCESS', 
                        num_index_vars, num_float_vars, avg_dist, 
                        avg_reuse_dist, stdev_reuse_dist, index_loads, 
                        float_loads, false}} 
  -- Compute time for a single iteration
  local time,stats = core:time_compute(CPU_tasklist, true)
  time = time*nb_iter
  --this.sleep(time)
  local te = stats["Thread Efficiency"]
  node.out:write("Time: "..simianEngine.now..":\t "..node.name.." "..node.num..
                  " computations completed on core id ".. 0 ..
                  "; execution time: "..time.."; Thread Efficiency: "..te.."\n")
end   
local function LUATest_Handler(self, msg, args)
    --self.createProcess("LUA_Test", LUA_Test) 
    --self.startProcess("LUA_Test", self)
    LUA_Test(self, msg, args)   
end
--[[################################
--# "MAIN"
]]--################################
local function main()
  local U_SEC = 1
  local M_SEC = 1000*U_SEC
  local SEC   = 1000*M_SEC

  --########################
  --# 0. Initialization stuff 
  local simName, startTime, endTime, minDelay = "LUATest", 0.0, 1000000.0, 0.000001
  simianEngine:init(simName, startTime, endTime, minDelay)
  -- 1. Choose and instantiate the Cluster that we want to simulate 
  local cluster = clusters.SingleMBP(simianEngine)
  -- cluster = clusters.HalfTrinity(simianEngine)

  -- 2. Create a GPUtest Service on the node
  simianEngine:attachService(nodes.MBPNode, "LUATest_Handler" , LUATest_Handler)

  simianEngine:schedService(0, "LUATest_Handler", nil, "Node", 2)
    
  -- 3. Run simx
  simianEngine:run()
  simianEngine:exit()
end
main()
