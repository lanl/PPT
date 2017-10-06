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

local math = require "math"
math.erf = function(x)
    -- constants
    local a1 =  0.254829592
    local a2 = -0.284496736
    local a3 =  1.421413741
    local a4 = -1.453152027
    local a5 =  1.061405429
    local p  =  0.3275911
    -- Save the sign of x
    local sign = 1
    if x < 0 then
        sign = -1
    end
    x = math.abs(x)
    -- A&S formula 7.1.26
    local t = 1.0/(1.0 + p*x)
    local y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*math.exp(-x*x)
    return sign*y
end
---
-- A SimX resource  that represents a computer CPU (or CPU core) with hardware threads
---
local ThreadedProcessor = function(node) -- defined as base class as other bases seem useless in SNAPSim context (1.6.15)
  local proc = {}
  proc.activethreads = 0 -- number of active threads or processes
  proc.maxthreads = 100000 -- upper bound on number active threads
  proc.node = node -- needed so processor can access node memory parameters
  proc.waiting_processes = {} -- list of processes waiting to be executed (only non-empty if maxthreads is exceeded)
  return proc
end

---
-- A SimX resource  that represents KnightsLanding core
-- It has 2 cache levels and a vector unit
---
local KNLCore = function(node)
  local core = ThreadedProcessor(node)
  ---
  --  PARAMETERS
  ---
  core.maxthreads = 8       -- upper bound on number active threads
  core.clockspeed = 1.24*10^9  -- Hertz 
  core.cycles_per_CPU_ops = 5   -- Number of clock cycles per CPU operation (avg)
  core.cycles_per_iALU = 2    -- Number of clock cycles per integer ALU operation
  core.cycles_per_fALU = 3    -- Number of clock cycles per float ALU operation
    
  core.cycles_per_vector_ops = 50 -- Number of clock cycles per vector operation 
    
  core.hwthreads = 2        -- number of hardware threads
  core.vector_width = 64      -- width of vector unit in bytes
  core.cache_levels = 2     -- number of cache levels
  core.cache_sizes = {32*10^3, 256*10^3}        --   list of cache sizes
  core.cache_page_sizes = {8, 1024} -- list of page sizes for the different cache levels[bytes]
  core.num_registers = 64     -- number of registers (single precision, holds an int) [4 bytes]
  core.register_cycles = 3    -- cycles per register access
    
  core.cache_cycles = {3, 21} -- list of cache access cycles per level
  core.ram_cycles = 330     -- number of cycles to load from main memory
  core.ram_page_size = 1024   -- page size for main memory access [bytes]

  ---
  -- Computes the cycles that 
  -- the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
  -- take 
  ---
  function core:time_compute(tasklist, statsFlag)
    local cycles = 0.0
    local time = 0.0
    local stats = {}
    stats['L1_float_hits'] =0
    stats['L2_float_hits'] =0
    stats['L1_int_hits'] =0
    stats['L2_int_hits'] =0
    stats['L1_int_misses'] =0
    stats['L2_int_misses'] =0
    stats['L1_float_misses'] =0
    stats['L2_float_misses'] =0
    stats['RAM accesses'] =0
    stats['L1 cycles'] =0
    stats['L2 cycles'] =0
    stats['RAM cycles'] =0
    stats['CPU cycles'] =0
    stats['iALU cycles'] =0
    stats['fALU cycles'] =0
    stats['VECTOR ops'] =0
    stats['VECTOR cycles'] =0
    stats['internode comm time'] =0
    stats['intranode comm time'] =0
    
    for _, item in ipairs(tasklist) do
      --print "Item is:", item
      local num_accesses, num_ops
      --#####  Memory Access #########
      if item[1] == 'MEM_ACCESS' then
        -- memory signature access, followed by 
        local num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[2], item[3], item[4], item[5], item[6]
        local index_loads, float_loads, init_flag = item[7], item[8], item[9]
        --print "Task list received", item
        
        --TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
        -- This is a V0.1 model for realistic caching behavior
        -- The arguments to MEM_ACCESS are architecture independent
        --
        -- We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
        -- Assume index variables are in registers as much as possible
        local avg_index_loads = index_loads / num_index_vars
        local num_reg_accesses = self.num_registers * avg_index_loads
        local nonreg_index_loads =  math.max(0, index_loads - num_reg_accesses) -- number of index loads that are not handled by register
            
        local num_vars_per_page = self.ram_page_size / avg_dist -- avg number of variables per ram page 
        local initial_ram_pages
        if init_flag then -- We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
          initial_ram_pages = num_float_vars / num_vars_per_page -- number of ram pages to be loaded at least once    
          float_loads = float_loads - num_float_vars -- we treat the first time that a float is loaded separately 
        end
        -- Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
        --L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
        -- (avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
        local L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[1]/self.cache_page_sizes[1] - 
          (avg_reuse_dist*avg_dist/self.cache_page_sizes[1]))/math.sqrt(2 * (stdev_reuse_dist/self.cache_page_sizes[1])^2)))
        local L1_float_hits = L1_hitrate*float_loads
        local L1_int_hits = L1_hitrate*nonreg_index_loads
        
        local L1_int_misses = nonreg_index_loads - L1_int_hits
        local L1_float_misses = float_loads - L1_float_hits
        
        local L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[2]/self.cache_page_sizes[2] - 
          (avg_reuse_dist/self.cache_page_sizes[2]))/math.sqrt(2 * (stdev_reuse_dist/self.cache_page_sizes[2])^2)))

        local L2_float_hits = L2_hitrate*L1_float_misses
        local L2_int_hits = L2_hitrate*L1_int_misses    
        
        local L2_int_misses = L1_int_misses - L2_int_hits
        local L2_float_misses = L1_float_misses - L2_float_hits
        
        -- Update the cycles number
        cycles = cycles + num_reg_accesses*self.register_cycles
        cycles = cycles + self.cache_cycles[1] * (2*L1_float_hits + L1_int_hits) -- float accesses are twice as expensive
        cycles = cycles + self.cache_cycles[2] * (2*L2_float_hits + L2_int_hits) -- float accesses are twice as expensive
        --cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
        cycles = cycles + self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) -- forget initial_ram_pages for now. 1.6.15
        
        
        stats['L1_float_hits'] = stats['L1_float_hits'] + L1_float_hits
        stats['L2_float_hits'] = stats['L2_float_hits'] + L2_float_hits
        stats['L1_int_hits'] = stats['L1_int_hits'] + L1_int_hits
        stats['L2_int_hits'] = stats['L2_int_hits'] + L2_int_hits
        stats['L1_int_misses'] = stats['L1_int_misses'] + L1_int_misses
        stats['L2_int_misses'] = stats['L2_int_misses'] + L2_int_misses
        stats['L1_float_misses'] = stats['L1_float_misses'] + L1_float_misses
        stats['L2_float_misses'] = stats['L2_float_misses'] + L2_float_misses
        stats['RAM accesses'] = stats['RAM accesses'] + ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
        stats['L1 cycles'] = stats['L1 cycles'] + self.cache_cycles[1] * (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles'] = stats['L2 cycles'] + self.cache_cycles[2] * (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles'] = stats['RAM cycles'] + self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
        
      elseif item[1] == 'L1' then  -- L1 accesses, followed by number
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.cache_cycles[1]
      elseif item[1] == 'L2' then  -- L2 accesses, followed by number
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.cache_cycles[2]          
      elseif item[1] == 'L3' or item[1] == 'L4' or item[1] == 'L5' or item[1] == 'RAM' or item[1] == 'mem' then  -- Higher cache access defaults to memory
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.ram_cycles
        
      --##### CPU access  ###############
      elseif item[1] == 'CPU' then -- CPU ops
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_CPU_ops
        stats['CPU cycles'] = stats['CPU cycles'] + num_ops*self.cycles_per_CPU_ops
      elseif item[1] == 'iALU' then -- Integer additions
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_iALU
        stats['iALU cycles'] = stats['iALU cycles'] +  num_ops*self.cycles_per_iALU
      elseif item[1] == 'fALU' then -- Integer additions
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_fALU          
        stats['fALU cycles'] = stats['fALU cycles'] + num_ops*self.cycles_per_fALU
      elseif item[1] == 'VECTOR' then -- ['vector', n_ops, width ]
        num_ops = item[2]
        local vec_width = item[3]
        -- vec_ops is the actual number of operations necessary,  
        -- even if required width was too high.
        -- the // operator is floor integer division
        local vec_ops = (1+math.floor(vec_width/self.vector_width))*num_ops
        cycles = cycles + vec_ops*self.cycles_per_vector_ops
        stats['VECTOR ops'] = stats['VECTOR ops'] + vec_ops
        stats['VECTOR cycles'] = stats['VECTOR cycles'] + vec_ops*self.cycles_per_vector_ops
      --#####  Inter-process communication #########
      elseif item[1] == 'internode' then -- communication across node
        local msg_size = item[2] -- number of bytes to be sent
        time = time + msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
        stats['internode comm time'] = stats['internode comm time'] + msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
      elseif item[1] == 'intranode' then   -- communication across cores on same node
        num_accesses = item[2]/self.ram_page_size -- number of bytes to be sent
        -- We treat this case  as memory access
        cycles = cycles + num_accesses*self.ram_cycles
        stats['intranode comm time'] = stats['intranode comm time'] + num_accesses*self.ram_cycles
      --#####  Memory Management #########
      elseif item[1] == 'alloc' then  -- ['alloc', n_bytes]
        local mem_size = item[2]
        if mem_size < 0 then
          mem_size = - mem_size
        end
        local mem_alloc_success = self.node:mem_alloc(mem_size)
        if mem_alloc_success then
          -- we will count this as a single memory access for timing purposes
          -- TODO: check validity of above assumption with Nandu
          cycles = cycles + self.ram_cycles
        else
          -- well, that would be a file system access then or just an exception
          -- we add to time, not cycles
          time = time + self.node.filesystem_access_time
          -- print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
          -- " attempted allocated memory that is not available, thus causing ", \
          -- " a filesystem action. You are in swapping mode now. "
        end
      elseif item[1] == 'unalloc' then -- ['unalloc', n_bytes]
        local mem_size = item[2]
        if mem_size > 0 then
          mem_size = - mem_size
        end
        local mem_alloc_success = self.node:mem_alloc(mem_size)
        -- Unalloc just changes the memory footprint, has no impact on time   
      -- ################
      else
        print('Warning: task list item'..item..' cannot be parsed, ignoring it') 
      end
      time = time + cycles * 1/self.clockspeed * self:thread_efficiency()
      stats['Thread Efficiency'] = self:thread_efficiency()
    end
    if statsFlag then  
      return time, stats
    else
      return time
    end
  end  
    
  ---
  -- A SimX resource  that represents Cielo core.
  -- It has 3 cache levels and a vector units. 
  ---
  function core:thread_efficiency()
    local efficiency = 0.0
    --print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
    if self.activethreads <=self.hwthreads then
      efficiency = 1.0
    else
      efficiency = self.hwthreads/self.activethreads
    --print "efficiency = ", efficiency
    end
    return efficiency
  end
  return core
end

local CieloCore = function(node)
  local core = ThreadedProcessor(node)
  ---
  --  PARAMETERS
  ---
  core.maxthreads = 16       -- upper bound on number active threads
  core.clockspeed = 2.4*10^9  -- Hertz 
    
  core.hwthreads = 16        -- number of hardware threads
  core.vector_width = 16      -- width of vector unit in bytes
  core.cache_levels = 3     -- number of cache levels
  core.cache_sizes = {64*10^3, 512*10^3, 12*10^6}        --   list of cache sizes
  
   
  core.cycles_per_CPU_ops = 6.4 -- Number of clock cycles per CPU operation (avg)
  core.cycles_per_iALU = 3.9    -- Number of clock cycles per integer ALU operation
  core.cycles_per_fALU = 6.6    -- Number of clock cycles per float ALU operation
    
  core.cycles_per_vector_ops = 4.8 -- Number of clock cycles per vector operation
  core.cache_page_sizes = {4, 2*10^3,10^9} -- list of page sizes for the different cache levels[bytes]
  core.num_registers = 16     -- number of registers (single precision, holds an int) [4 bytes]
  core.register_cycles = 1    -- cycles per register access
    
  core.cache_cycles = {3, 9, 30} -- list of cache access cycles per level
  core.ram_cycles = 330     -- number of cycles to load from main memory
  core.ram_page_size = 1024   -- page size for main memory access [bytes]
    
  ---
  -- Computes the cycles that 
  -- the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
  -- take 
  ---
  function core:time_compute(tasklist, statsFlag)
    if not statsFlag then
      statsFlag = false
    end    
    local cycles = 0.0
    local time = 0.0
    local stats = {}
    stats['L1_float_hits'] =0
    stats['L2_float_hits'] =0
    stats['L1_int_hits'] =0
    stats['L2_int_hits'] =0
    stats['L1_int_misses'] =0
    stats['L2_int_misses'] =0
    stats['L1_float_misses'] =0
    stats['L2_float_misses'] =0
    stats['RAM accesses'] =0
    stats['L1 cycles'] =0
    stats['L2 cycles'] =0
    stats['RAM cycles'] =0
    stats['CPU cycles'] =0
    stats['iALU cycles'] =0
    stats['fALU cycles'] =0
    stats['VECTOR ops'] =0
    stats['VECTOR cycles'] =0
    stats['internode comm time'] =0
    stats['intranode comm time'] =0
    
    -- print tasklist
    for _, item in ipairs(tasklist) do
      --print "Item is:", item
      local num_accesses, num_ops
      --#####  Memory Access #########
      if item[1] == 'MEM_ACCESS' then  
        -- memory signature access, followed by 
        local num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[2], item[3], item[4], item[5], item[6]
        local index_loads, float_loads, init_flag = item[7], item[8], item[9]
        --print "Task list received", item
        
        --TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
        -- This is a V0.1 model for realistic caching behavior
        -- The arguments to MEM_ACCESS are architecture independent
        --
        -- We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
        -- Assume index variables are in registers as much as possible
        local avg_index_loads = index_loads / num_index_vars
        local num_reg_accesses = self.num_registers * avg_index_loads
        local nonreg_index_loads =  math.max(0, index_loads - num_reg_accesses) -- number of index loads that are not handled by register
            
        local num_vars_per_page = self.ram_page_size / avg_dist -- avg number of variables per ram page 
        local initial_ram_pages
        if init_flag then -- We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
          initial_ram_pages = num_float_vars / num_vars_per_page -- number of ram pages to be loaded at least once    
          float_loads = float_loads - num_float_vars -- we treat the first time that a float is loaded separately 
        end
        -- Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
        --L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
        -- (avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
        local L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[1]/self.cache_page_sizes[1] - 
          (avg_reuse_dist*avg_dist/self.cache_page_sizes[1]))/math.sqrt(2 * (stdev_reuse_dist/self.cache_page_sizes[1])^2)))
        local L1_float_hits = L1_hitrate*float_loads
        local L1_int_hits = L1_hitrate*nonreg_index_loads
        
        local L1_int_misses = nonreg_index_loads - L1_int_hits
        local L1_float_misses = float_loads - L1_float_hits
        
        local L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[2]/self.cache_page_sizes[2] - 
          (avg_reuse_dist/self.cache_page_sizes[2]))/math.sqrt(2 * (stdev_reuse_dist/self.cache_page_sizes[2])^2)))

        local L2_float_hits = L2_hitrate*L1_float_misses
        local L2_int_hits = L2_hitrate*L1_int_misses    
        
        local L2_int_misses = L1_int_misses - L2_int_hits
        local L2_float_misses = L1_float_misses - L2_float_hits
        
        -- Update the cycles number
        cycles = cycles + num_reg_accesses*self.register_cycles
        cycles = cycles + self.cache_cycles[1] * (2*L1_float_hits + L1_int_hits) -- float accesses are twice as expensive
        cycles = cycles + self.cache_cycles[2] * (2*L2_float_hits + L2_int_hits) -- float accesses are twice as expensive
        --cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
        cycles = cycles + self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) -- forget initial_ram_pages for now. 1.6.15
        
        
        stats['L1_float_hits'] = stats['L1_float_hits'] + L1_float_hits
        stats['L2_float_hits'] = stats['L2_float_hits'] + L2_float_hits
        stats['L1_int_hits'] = stats['L1_int_hits'] + L1_int_hits
        stats['L2_int_hits'] = stats['L2_int_hits'] + L2_int_hits
        stats['L1_int_misses'] = stats['L1_int_misses'] + L1_int_misses
        stats['L2_int_misses'] = stats['L2_int_misses'] + L2_int_misses
        stats['L1_float_misses'] = stats['L1_float_misses'] + L1_float_misses
        stats['L2_float_misses'] = stats['L2_float_misses'] + L2_float_misses
        stats['RAM accesses'] = stats['RAM accesses'] + ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
        stats['L1 cycles'] = stats['L1 cycles'] + self.cache_cycles[1] * (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles'] = stats['L2 cycles'] + self.cache_cycles[2] * (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles'] = stats['RAM cycles'] + self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
        
      elseif item[1] == 'L1' then  -- L1 accesses, followed by number
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.cache_cycles[1]
      elseif item[1] == 'L2' then  -- L2 accesses, followed by number
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.cache_cycles[2]          
      elseif item[1] == 'L3' or item[1] == 'L4' or item[1] == 'L5' or item[1] == 'RAM' or item[1] == 'mem' then  -- Higher cache access defaults to memory
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.ram_cycles
        
      --##### CPU access  ###############
      elseif item[1] == 'CPU' then -- CPU ops
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_CPU_ops
        stats['CPU cycles'] = stats['CPU cycles'] + num_ops*self.cycles_per_CPU_ops
      elseif item[1] == 'iALU' then -- Integer additions
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_iALU
        stats['iALU cycles'] = stats['iALU cycles'] +  num_ops*self.cycles_per_iALU
      elseif item[1] == 'fALU' then -- Integer additions
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_fALU          
        stats['fALU cycles'] = stats['fALU cycles'] + num_ops*self.cycles_per_fALU
      elseif item[1] == 'VECTOR' then -- ['vector', n_ops, width ]
        num_ops = item[2]
        local vec_width = item[3]
        -- vec_ops is the actual number of operations necessary,  
        -- even if required width was too high.
        -- the // operator is floor integer division
        local vec_ops = (1+math.floor(vec_width/self.vector_width))*num_ops
        cycles = cycles + vec_ops*self.cycles_per_vector_ops
        stats['VECTOR ops'] = stats['VECTOR ops'] + vec_ops
        stats['VECTOR cycles'] = stats['VECTOR cycles'] + vec_ops*self.cycles_per_vector_ops
      --#####  Inter-process communication #########
      elseif item[1] == 'internode' then -- communication across node
        local msg_size = item[2] -- number of bytes to be sent
        time = time + msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
        stats['internode comm time'] = stats['internode comm time'] + msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
      elseif item[1] == 'intranode' then   -- communication across cores on same node
        num_accesses = item[2]/self.ram_page_size -- number of bytes to be sent
        -- We treat this case  as memory access
        cycles = cycles + num_accesses*self.ram_cycles
        stats['intranode comm time'] = stats['intranode comm time'] + num_accesses*self.ram_cycles
      --#####  Memory Management #########
      elseif item[1] == 'alloc' then  -- ['alloc', n_bytes]
        local mem_size = item[2]
        if mem_size < 0 then
          mem_size = - mem_size
        end
        local mem_alloc_success = self.node:mem_alloc(mem_size)
        if mem_alloc_success then
          -- we will count this as a single memory access for timing purposes
          -- TODO: check validity of above assumption with Nandu
          cycles = cycles + self.ram_cycles
        else
          -- well, that would be a file system access then or just an exception
          -- we add to time, not cycles
          time = time + self.node.filesystem_access_time
          -- print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
          -- " attempted allocated memory that is not available, thus causing ", \
          -- " a filesystem action. You are in swapping mode now. "
        end
      elseif item[1] == 'unalloc' then -- ['unalloc', n_bytes]
        local mem_size = item[2]
        if mem_size > 0 then
          mem_size = - mem_size
        end
        local mem_alloc_success = self.node:mem_alloc(mem_size)
        -- Unalloc just changes the memory footprint, has no impact on time   
      -- ################
      else
        print('Warning: task list item'..item..' cannot be parsed, ignoring it') 
      end
      time = time + cycles * 1/self.clockspeed * self:thread_efficiency()
      stats['Thread Efficiency'] = self:thread_efficiency()
    end
    if statsFlag then  
      return time, stats
    else
      return time
    end
  end  
    
  ---
  -- A SimX resource  that represents Cielo core.
  -- It has 3 cache levels and a vector units. 
  ---
  function core:thread_efficiency()
    local efficiency = 0.0
    --print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
    if self.activethreads <=self.hwthreads then
      efficiency = 1.0
    else
      efficiency = self.hwthreads/self.activethreads
    --print "efficiency = ", efficiency
    end
    return efficiency
  end
  return core
end

---
-- A SimX resource  that represents MacPro core.
-- It has 3 cache levels and a vector units.
---
local MacProCore = function(node)
  local core = ThreadedProcessor(node)
  ---
  --  PARAMETERS
  ---
  core.maxthreads = 24       -- upper bound on number active threads
  core.clockspeed = 2.7*10^9  -- Hertz 
    
  core.hwthreads = 12      -- number of hardware threads
  core.vector_width = 32      -- width of vector unit in bytes
  core.cache_levels = 3     -- number of cache levels
  core.cache_sizes = {12*10^3, 256*10^3, 30*10^6}        --   list of cache sizes
  
   
  core.cycles_per_CPU_ops = 1 -- Number of clock cycles per CPU operation (avg)
  core.cycles_per_iALU = 1    -- Number of clock cycles per integer ALU operation
  core.cycles_per_fALU = 1    -- Number of clock cycles per float ALU operation
    
  core.cycles_per_vector_ops = 1 -- Number of clock cycles per vector operation
  core.cache_page_sizes = {4, 2*10^3,10^9} -- list of page sizes for the different cache levels[bytes]
  core.num_registers = 16     -- number of registers (single precision, holds an int) [4 bytes]
  core.register_cycles = 1    -- cycles per register access
    
  core.cache_cycles = {3, 9, 30} -- list of cache access cycles per level
  core.ram_cycles = 330     -- number of cycles to load from main memory
  core.ram_page_size = 1024   -- page size for main memory access [bytes]
      
  ---
  -- Computes the cycles that 
  -- the items in the tasklist (CPU ops, data access, vector ops, memory alloc)
  -- take 
  ---
  function core:time_compute(tasklist, statsFlag)
    if not statsFlag then
      statsFlag = false
    end    
    local cycles = 0.0
    local time = 0.0
    local stats = {}
    stats['L1_float_hits'] =0
    stats['L2_float_hits'] =0
    stats['L1_int_hits'] =0
    stats['L2_int_hits'] =0
    stats['L1_int_misses'] =0
    stats['L2_int_misses'] =0
    stats['L1_float_misses'] =0
    stats['L2_float_misses'] =0
    stats['RAM accesses'] =0
    stats['L1 cycles'] =0
    stats['L2 cycles'] =0
    stats['RAM cycles'] =0
    stats['CPU cycles'] =0
    stats['iALU cycles'] =0
    stats['fALU cycles'] =0
    stats['VECTOR ops'] =0
    stats['VECTOR cycles'] =0
    stats['internode comm time'] =0
    stats['intranode comm time'] =0
    
    -- print tasklist
    for _, item in ipairs(tasklist) do
      --print "Item is:", item
      local num_accesses, num_ops
      --#####  Memory Access #########
      if item[1] == 'MEM_ACCESS' then  
        -- memory signature access, followed by 
        local num_index_vars, num_float_vars, avg_dist, avg_reuse_dist, stdev_reuse_dist = item[2], item[3], item[4], item[5], item[6]
        local index_loads, float_loads, init_flag = item[7], item[8], item[9]
        --print "Task list received", item
        
        --TODO: Insert formula that turns these variables into actual L1, L2, ram accesses 
        -- This is a V0.1 model for realistic caching behavior
        -- The arguments to MEM_ACCESS are architecture independent
        --
        -- We use index vars and integers vars interchageably, ie all int variables are treated as array indices, all floats are array elements
        -- Assume index variables are in registers as much as possible
        local avg_index_loads = index_loads / num_index_vars
        local num_reg_accesses = self.num_registers * avg_index_loads
        local nonreg_index_loads =  math.max(0, index_loads - num_reg_accesses) -- number of index loads that are not handled by register
            
        local num_vars_per_page = self.ram_page_size / avg_dist -- avg number of variables per ram page 
        local initial_ram_pages
        if init_flag then -- We treat this like a new function call, being sure that all floats need to be loaded from ram (to ensure a min ram load count)
          initial_ram_pages = num_float_vars / num_vars_per_page -- number of ram pages to be loaded at least once    
          float_loads = float_loads - num_float_vars -- we treat the first time that a float is loaded separately 
        end
        -- Compute probability that reload is required, assume normal distribution with reuse dist larger than cache page size
        --L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[0]/float(self.cache_page_sizes[0]) - \
        -- (avg_reuse_dist/float(self.cache_page_sizes[0])))/math.sqrt(2 * (stdev_reuse_dist/float(self.cache_page_sizes[0]))**2)))
        local L1_hitrate =  0.5 * (1 + math.erf((self.cache_sizes[1]/self.cache_page_sizes[1] - 
          (avg_reuse_dist*avg_dist/self.cache_page_sizes[1]))/math.sqrt(2 * (stdev_reuse_dist/self.cache_page_sizes[1])^2)))
        local L1_float_hits = L1_hitrate*float_loads 
        local L1_int_hits = L1_hitrate*nonreg_index_loads
        
        local L1_int_misses = nonreg_index_loads - L1_int_hits
        local L1_float_misses = float_loads - L1_float_hits
        
        local L2_hitrate =   0.5 * (1 + math.erf((self.cache_sizes[2]/self.cache_page_sizes[2] - 
          (avg_reuse_dist/self.cache_page_sizes[2]))/math.sqrt(2 * (stdev_reuse_dist/self.cache_page_sizes[2])^2)))

        local L2_float_hits = L2_hitrate*L1_float_misses
        local L2_int_hits = L2_hitrate*L1_int_misses 
        
        local L2_int_misses = L1_int_misses - L2_int_hits
        local L2_float_misses = L1_float_misses - L2_float_hits
        
        -- Update the cycles number
        cycles = cycles + num_reg_accesses*self.register_cycles
        cycles = cycles + self.cache_cycles[1] * (2*L1_float_hits + L1_int_hits) -- float accesses are twice as expensive
        cycles = cycles + self.cache_cycles[2] * (2*L2_float_hits + L2_int_hits) -- float accesses are twice as expensive
        --cycles += self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page + initial_ram_pages) # take into account page size again in main memory
        cycles = cycles + self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page) -- forget initial_ram_pages for now. 1.6.15
        
        
        stats['L1_float_hits'] = stats['L1_float_hits'] + L1_float_hits
        stats['L2_float_hits'] = stats['L2_float_hits'] + L2_float_hits
        stats['L1_int_hits'] = stats['L1_int_hits'] + L1_int_hits
        stats['L2_int_hits'] = stats['L2_int_hits'] + L2_int_hits
        stats['L1_int_misses'] = stats['L1_int_misses'] + L1_int_misses
        stats['L2_int_misses'] = stats['L2_int_misses'] + L2_int_misses
        stats['L1_float_misses'] = stats['L1_float_misses'] + L1_float_misses
        stats['L2_float_misses'] = stats['L2_float_misses'] + L2_float_misses
        stats['RAM accesses'] = stats['RAM accesses'] + ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
        stats['L1 cycles'] = stats['L1 cycles'] + self.cache_cycles[1] * (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles'] = stats['L2 cycles'] + self.cache_cycles[2] * (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles'] = stats['RAM cycles'] + self.ram_cycles * ((2*L2_float_misses + L2_int_misses)/num_vars_per_page)
        
      elseif item[1] == 'L1' then  -- L1 accesses, followed by number
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.cache_cycles[1]
      elseif item[1] == 'L2' then  -- L2 accesses, followed by number
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.cache_cycles[2]          
      elseif item[1] == 'L3' or item[1] == 'L4' or item[1] == 'L5' or item[1] == 'RAM' or item[1] == 'mem' then  -- Higher cache access defaults to memory
        num_accesses = item[2]
        cycles = cycles + num_accesses*self.ram_cycles
        
      --##### CPU access  ###############
      elseif item[1] == 'CPU' then -- CPU ops
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_CPU_ops
        stats['CPU cycles'] = stats['CPU cycles'] + num_ops*self.cycles_per_CPU_ops
      elseif item[1] == 'iALU' then -- Integer additions
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_iALU
        stats['iALU cycles'] = stats['iALU cycles'] +  num_ops*self.cycles_per_iALU
      elseif item[1] == 'fALU' then -- Integer additions
        num_ops = item[2]
        cycles = cycles + num_ops*self.cycles_per_fALU          
        stats['fALU cycles'] = stats['fALU cycles'] + num_ops*self.cycles_per_fALU
      elseif item[1] == 'VECTOR' then -- ['vector', n_ops, width ]
        num_ops = item[2]
        local vec_width = item[3]
        -- vec_ops is the actual number of operations necessary,  
        -- even if required width was too high.
        -- the // operator is floor integer division
        local vec_ops = (1+math.floor(vec_width/self.vector_width))*num_ops
        cycles = cycles + vec_ops*self.cycles_per_vector_ops
        stats['VECTOR ops'] = stats['VECTOR ops'] + vec_ops
        stats['VECTOR cycles'] = stats['VECTOR cycles'] + vec_ops*self.cycles_per_vector_ops
      --#####  Inter-process communication #########
      elseif item[1] == 'internode' then -- communication across node
        local msg_size = item[2] -- number of bytes to be sent
        time = time + msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
        stats['internode comm time'] = stats['internode comm time'] + msg_size/self.node.interconnect_bandwidth + self.node.interconnect_latency
      elseif item[1] == 'intranode' then   -- communication across cores on same node
        num_accesses = item[2]/self.ram_page_size -- number of bytes to be sent
        -- We treat this case  as memory access
        cycles = cycles + num_accesses*self.ram_cycles
        stats['intranode comm time'] = stats['intranode comm time'] + num_accesses*self.ram_cycles
      --#####  Memory Management #########
      elseif item[1] == 'alloc' then  -- ['alloc', n_bytes]
        local mem_size = item[2]
        if mem_size < 0 then
          mem_size = - mem_size
        end
        local mem_alloc_success = self.node:mem_alloc(mem_size)
        if mem_alloc_success then
          -- we will count this as a single memory access for timing purposes
          -- TODO: check validity of above assumption with Nandu
          cycles = cycles + self.ram_cycles
        else
          -- well, that would be a file system access then or just an exception
          -- we add to time, not cycles
          time = time + self.node.filesystem_access_time
          -- print "Warning: KNLCore", id(self), " on KNLNode ", id(self.node), \
          -- " attempted allocated memory that is not available, thus causing ", \
          -- " a filesystem action. You are in swapping mode now. "
        end
      elseif item[1] == 'unalloc' then -- ['unalloc', n_bytes]
        local mem_size = item[2]
        if mem_size > 0 then
          mem_size = - mem_size
        end
        local mem_alloc_success = self.node:mem_alloc(mem_size)
        -- Unalloc just changes the memory footprint, has no impact on time   
      -- ################
      else
        print('Warning: task list item'..item..' cannot be parsed, ignoring it') 
      end
      time = time + cycles * 1/self.clockspeed * self:thread_efficiency()
      stats['Thread Efficiency'] = self:thread_efficiency()
    end
    if statsFlag then  
      return time, stats
    else
      return time
    end
  end  
    
  ---
  -- A SimX resource  that represents Cielo core.
  -- It has 3 cache levels and a vector units. 
  ---
  function core:thread_efficiency()
    local efficiency = 0.0
    --print "Computing thread efficiency: active threads, hwtreads", self.activethreads, self.hwthreads
    if self.activethreads <=self.hwthreads then
      efficiency = 1.0
    else
      efficiency = self.hwthreads/self.activethreads
    --print "efficiency = ", efficiency
    end
    return efficiency
  end
  return core
end

return {KNLCore=KNLCore,
        CieloCore= CieloCore,
        MacProCore= MacProCore}