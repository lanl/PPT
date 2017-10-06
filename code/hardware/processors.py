# Copyright (c) 2014. Los Alamos National Security, LLC.

# This material was produced under U.S. Government contract DE-AC52-06NA25396
# for Los Alamos National Laboratory (LANL), which is operated by Los Alamos
# National Security, LLC for the U.S. Department of Energy. The U.S. Government
# has rights to use, reproduce, and distribute this software.

# NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
# WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
# SOFTWARE.

# If software is modified to produce derivative works, such modified software
# should be clearly marked, so as not to confuse it with the version available
# from LANL.

# Additionally, this library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License v 2.1 as
# published by the Free Software Foundation. Accordingly, this library is
# distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE.txt for more details.


"""
*********** Performance Prediction Toolkit PPT *********

File: processors.py
Description main library of hardware core definitions.

Comments:
 2016-04-26: included into repository, contains
  1. ThreadedProcessor
  3. CieloCore
  4. MLIntelCore
"""

import math

class ThreadedProcessor(object):
  """
  Defined as base class as other bases seem useless in SNAPSim context. A
  Simian resource that represents a computer CPU (or CPU core) with hardware
  threads.
  """

  def __init__(self, node):
    super(ThreadedProcessor,self).__init__()
    self.activethreads = 0
    self.maxthreads = 100000
    self.node = node               # needed so processor can access node memory
                                   #   parameters
    self.waiting_processes = []    # list of processes waiting to be executed
                                   #   (only non-empty if maxthreads is
                                   #   exceeded)


###########################################################
###########################################################
###########################################################
class CieloCore(ThreadedProcessor):
  """
  A Simian resource that represents Cielo core. It has 3 cache levels and
  vector units.
  """


  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(CieloCore, self).__init__(node)

    # Units
    self.ns = 1.0*10**(-9)    # nanoseconds
    self.kb = 1024.0          # Kilobytes
    self.mb = 1024.0**2       # Megabytes
    self.isize = 4.0          # bytes per word
    self.fsize = 8.0          # bytes per word

    # Threads and vectors
    self.maxthreads = 16            # upper bound on number active threads
    self.clockspeed = 2.40*10**9    # Hertz
    self.hwthreads = 1              # number of hardware threads
    self.vector_width = 16          # width of vector unit in bytes, 128 bit?

    # Registers
    self.num_registers = 16.0       # number of registers [bytes]
    self.register_cycles = 1        # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [64.0*self.kb, 512.0*self.kb, 1.5*self.mb]   # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]                      # bytes
    self.cache_latency = [1.0*self.ns, 5.0*self.ns, 25.0*self.ns]   # seconds
    self.cache_cycles = [self.cache_latency[0]*self.clockspeed,
                         self.cache_latency[1]*self.clockspeed,
                         self.cache_latency[2]*self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0       # bytes
    self.ram_latency = 60.0*self.ns   # seconds
    self.ram_cycles = self.ram_latency*self.clockspeed

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.  
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.5
    self.cycles_per_int_vec = 0.1
    self.cycles_per_fALU = 1.0
    self.cycles_per_vector_ops = 0.5
    self.cycles_per_division = 10.0


  def time_compute(self, tasklist, statsFlag=False):
    """
    Computes the cycles that the items in the tasklist (CPU ops, data access,
    vector ops, memory alloc) take.
    """

    # Initialize
    cycles = 0.0
    time = 0.0
    stats = {}
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
    stats['fDIV cycles'] =0
    stats['INTVEC ops'] =0
    stats['INTVEC cycles'] =0
    stats['VECTOR ops'] =0
    stats['VECTOR cycles'] =0
    stats['internode comm time'] =0
    stats['intranode comm time'] =0

    print "cycles = ", cycles

    for item in tasklist:

      # TASK: Cycles associated with moving data through memory and cache
      if item[0] == 'MEM_ACCESS':

        num_index_vars   = item[1]
        num_float_vars   = item[2]
        avg_dist         = item[3]
        avg_reuse_dist   = item[4]
        stdev_reuse_dist = item[5]
        index_loads      = item[6]
        float_loads      = item[7]
        init_flag        = item[8]
        
        # Insert formula that turns these variables into actual L1, L2, ram
        #   accesses. This is a V0.1 model for realistic caching behavior. The
        #   arguments to MEM_ACCESS are architecture independent. We use index
        #   vars and integers vars interchageably, ie all int variables are
        #   treated as array indices, all floats are array elements.

        # Assume index variables are in registers as much as possible. Compute
        #   number of index loads that are not handled by registers
        avg_index_loads = index_loads / num_index_vars
        num_reg_accesses = int(self.num_registers * avg_index_loads)
        nonreg_index_loads = max(0, index_loads - num_reg_accesses)

        # Compute number of variables on a typical page in memory
        num_vars_per_page = self.ram_page_size / avg_dist

	# Check memory layout to begin
        if init_flag:
          # New function call (true); floats need loaded from main memory and
          #   fewer float_loads from cache.
          initial_ram_pages = num_float_vars / num_vars_per_page
          float_loads -= num_float_vars
        else:
          # float data already in cache
          initial_ram_pages = 0.0

        # Main memory: initial_ram_pages moving through the cache
        cycles += self.ram_cycles*initial_ram_pages
        #cycles += self.cache_cycles[2]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[2]
        #cycles += self.cache_cycles[1]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[1]
        #cycles += self.cache_cycles[0]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[1]
        print "inital_ram_pages, cycles=", cycles, initial_ram_pages

        # Compute probability that reload is required, assume normal
        #   distribution with reuse dist larger than cache page size

        # L1
        nlines = self.cache_sizes[0]/self.cache_line_sizes[0]
        nvars  = avg_reuse_dist/self.cache_line_sizes[0]
        vvars  = math.sqrt( 2.0*(stdev_reuse_dist/self.cache_line_sizes[0])**2 )
        L1_hitrate = 0.5 * ( 1.0 + math.erf( (nlines-nvars)/vvars ) )

        L1_int_hits   = int(L1_hitrate*nonreg_index_loads)
        L1_float_hits = int(L1_hitrate*float_loads)

        L1_int_misses   = nonreg_index_loads - L1_int_hits
        L1_float_misses = float_loads - L1_float_hits

        print "l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses,\
                        L1_float_misses

        # L2
        nlines = self.cache_sizes[1]/self.cache_line_sizes[1]
        nvars  = avg_reuse_dist/self.cache_line_sizes[1]
        vvars  = math.sqrt( 2.0*(stdev_reuse_dist/self.cache_line_sizes[1])**2 )
        L2_hitrate = 0.5 * ( 1.0 + math.erf( (nlines-nvars)/vvars ) )

        L2_int_hits   = int(L2_hitrate*L1_int_misses)
        L2_float_hits = int(L2_hitrate*L1_float_misses)
        
        L2_int_misses   = L1_int_misses - L2_int_hits
        L2_float_misses = L1_float_misses - L2_float_hits

        print "l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses,\
                        L2_float_misses

        # L3
        nlines = self.cache_sizes[2]/self.cache_line_sizes[2]
        nvars  = avg_reuse_dist/self.cache_line_sizes[2]
        vvars  = math.sqrt( 2.0*(stdev_reuse_dist/self.cache_line_sizes[2])**2 )
        L3_hitrate = 0.5 * ( 1.0 + math.erf( (nlines-nvars)/vvars ) )

        L3_int_hits   = int(L3_hitrate*L2_int_misses)
        L3_float_hits = int(L3_hitrate*L2_float_misses)

        L3_int_misses = L2_int_misses - L3_int_hits
        L3_float_misses = L2_float_misses - L3_float_hits

        print "l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses,\
                        L3_float_misses

        # Update the cycles count for cache hits
        # Registers
        cycles += num_reg_accesses*self.register_cycles
        print "registers, cycles= ", cycles

        # L1, data moves together in cache lines
        cycles += self.cache_cycles[0] * \
                  (L1_float_hits*self.fsize + L1_int_hits*self.isize) / \
                  self.cache_line_sizes[0]
        print "l1 ", cycles

        # L2
        cycles += self.cache_cycles[1] * \
                  (L2_float_hits*self.fsize + L2_int_hits*self.isize) / \
                  self.cache_line_sizes[1]
        print "l2 ", cycles

        # L3
        cycles += self.cache_cycles[2] * \
                  (L3_float_hits*self.fsize + L3_int_hits*self.isize) / \
                  self.cache_line_sizes[2]
        print "l3 ", cycles

        # L3 misses accumulate time from main memory
        cycles += self.ram_cycles * \
                  (L3_float_misses + L3_int_misses) / num_vars_per_page
        print "l3 misses", cycles

        # L2 misses accumulate time from L3
        #cycles += self.cache_cycles[2] * \
        #          (L2_float_misses*self.fsize + L2_int_misses*self.isize) / \
        #          self.cache_line_sizes[2]
        #print "l2 misses", cycles

        # L1 misses accumulate time from L2
        #cycles += self.cache_cycles[1] * \
        #          (L1_float_misses*self.fsize + L1_int_misses*self.isize) / \
        #          self.cache_line_sizes[2]
        #print "l1 misses", cycles

        print "memaccess cycles= ", cycles

        stats['L1_float_hits']   += L1_float_hits
        stats['L2_float_hits']   += L2_float_hits
        stats['L1_int_hits']     += L1_int_hits
        stats['L2_int_hits']     += L2_int_hits
        stats['L1_int_misses']   += L1_int_misses
        stats['L2_int_misses']   += L2_int_misses
        stats['L1_float_misses'] += L1_float_misses
        stats['L2_float_misses'] += L2_float_misses
        stats['RAM accesses']    += (2*L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page
        stats['L1 cycles']       += self.cache_cycles[0] * \
                                    (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles']       += self.cache_cycles[1] * \
                                    (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles']      += self.ram_cycles * \
                                    (2*L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page

      # TASK: Direct input of cache level hitrates
      elif item[0] == 'HITRATES':

        L1_hitrate       = item[1]
        L2_hitrate       = item[2]
        L3_hitrate       = item[3]
        num_index_vars   = item[4]
        num_float_vars   = item[5]
        avg_dist         = item[6]
        index_loads      = item[7]
        float_loads      = item[8]
        init_flag        = item[9]
        
        # Insert formula that turns these variables into actual L1, L2, ram
        #   accesses. This is a V0.1 model for realistic caching behavior. The
        #   arguments to MEM_ACCESS are architecture independent. We use index
        #   vars and integers vars interchageably, ie all int variables are
        #   treated as array indices, all floats are array elements.

        # Assume index variables are in registers as much as possible. Compute
        #   number of index loads that are not handled by registers
        avg_index_loads = index_loads / num_index_vars
        num_reg_accesses = int(self.num_registers * avg_index_loads)
        nonreg_index_loads = max(0, index_loads - num_reg_accesses)

        # Compute number of variables on a typical page in memory
        num_vars_per_page = self.ram_page_size / avg_dist

	# Check memory layout to begin
        if init_flag:
          # New function call (true); floats need loaded from main memory and
          #   fewer float_loads from cache.
          initial_ram_pages = num_float_vars / num_vars_per_page
          float_loads -= num_float_vars
        else:
          # float data already in cache
          initial_ram_pages = 0.0

        # Main memory: initial_ram_pages moving through the cache
        cycles += self.ram_cycles*initial_ram_pages

        print "inital_ram_pages, cycles=", cycles, initial_ram_pages

        # Compute probability that reload is required, assume normal
        #   distribution with reuse dist larger than cache page size

        # L1
        L1_int_hits   = int(L1_hitrate*nonreg_index_loads)
        L1_float_hits = int(L1_hitrate*float_loads)

        L1_int_misses   = nonreg_index_loads - L1_int_hits
        L1_float_misses = float_loads - L1_float_hits

        print "l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses,\
                        L1_float_misses

        # L2
        L2_int_hits   = int(L2_hitrate*L1_int_misses)
        L2_float_hits = int(L2_hitrate*L1_float_misses)
        
        L2_int_misses   = L1_int_misses - L2_int_hits
        L2_float_misses = L1_float_misses - L2_float_hits

        print "l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses,\
                        L2_float_misses

        # L3
        L3_int_hits   = int(L3_hitrate*L2_int_misses)
        L3_float_hits = int(L3_hitrate*L2_float_misses)

        L3_int_misses = L2_int_misses - L3_int_hits
        L3_float_misses = L2_float_misses - L3_float_hits

        print "l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses,\
                        L3_float_misses

        # Update the cycles count for cache hits
        # Registers
        cycles += num_reg_accesses*self.register_cycles
        print "registers, cycles= ", cycles

        # L1, data moves together in cache lines
        cycles += self.cache_cycles[0] * \
                  (L1_float_hits*self.fsize + L1_int_hits*self.isize) / \
                  self.cache_line_sizes[0]
        print "l1 ", cycles

        # L2
        cycles += self.cache_cycles[1] * \
                  (L2_float_hits*self.fsize + L2_int_hits*self.isize) / \
                  self.cache_line_sizes[1]
        print "l2 ", cycles

        # L3
        cycles += self.cache_cycles[2] * \
                  (L3_float_hits*self.fsize + L3_int_hits*self.isize) / \
                  self.cache_line_sizes[2]
        print "l3 ", cycles

        # L3 misses accumulate time from main memory
        cycles += self.ram_cycles * \
                  (L3_float_misses + L3_int_misses) / num_vars_per_page
        print "l3 misses", cycles

        print "memaccess cycles= ", cycles

        stats['L1_float_hits']   += L1_float_hits
        stats['L2_float_hits']   += L2_float_hits
        stats['L1_int_hits']     += L1_int_hits
        stats['L2_int_hits']     += L2_int_hits
        stats['L1_int_misses']   += L1_int_misses
        stats['L2_int_misses']   += L2_int_misses
        stats['L1_float_misses'] += L1_float_misses
        stats['L2_float_misses'] += L2_float_misses
        stats['RAM accesses']    += (2*L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page
        stats['L1 cycles']       += self.cache_cycles[0] * \
                                    (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles']       += self.cache_cycles[1] * \
                                    (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles']      += self.ram_cycles * \
                                    num_vars_per_page

      # TASK: Direct input of L1 accesses
      elif item[0] == 'L1':

        num_accesses = item[1]
        cycles += num_accesses*self.cache_cycles[0]
        print "l1 ", cycles

      # TASK: Direct input of L2 accesses
      elif item[0] == 'L2':

        num_accesses = item[1]
        cycles += num_accesses*self.cache_cycles[1]           
        print "l2 ", cycles

      # TASK: Direct input of higher cache and memory accesses
      elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:

        num_accesses = item[1]
        cycles += num_accesses*self.ram_cycles
        print "l3 ", cycles

      # TASK: CPU ops
      elif item[0] == 'CPU':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_CPU_ops
        stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
        print "cpu ", cycles

      # TASK: Integer operations
      elif item[0] == 'iALU':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_iALU
        stats['iALU cycles'] +=   num_ops*self.cycles_per_iALU
        print "ialu ", cycles

      # TASK: Floating point operations (add/multiply)
      elif item[0] == 'fALU':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_fALU
        stats['fALU cycles'] +=   num_ops*self.cycles_per_fALU
        print "falu ", cycles, num_ops

      # TASK: Floating point divisions
      elif item[0] == 'fDIV':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_division
        stats['fDIV cycles'] +=   num_ops*self.cycles_per_division
        print "fDIV ", cycles, num_ops

      # TASK: Integer vector operations
      elif item[0] == 'INTVEC':

        num_ops   = item[1]
        vec_width = item[2]

        # vec_ops is the actual number of operations necessary, even if
        #   required width was too high. The // operator is floor integer
        #   division.

        vec_ops = (1+vec_width//self.vector_width)*num_ops
        cycles += vec_ops*self.cycles_per_int_vec
        stats['INTVEC ops'] +=  vec_ops
        stats['INTVEC cycles'] += vec_ops*self.cycles_per_int_vec
        print "intvect ", cycles, num_ops, vec_ops

      # TASK: Vector operations
      elif item[0] == 'VECTOR':

        num_ops   = item[1]
        vec_width = item[2]

        # vec_ops is the actual number of operations necessary, even if
        #   required width was too high. The // operator is floor integer
        #   division.

        vec_ops = (1+vec_width//self.vector_width)*num_ops
        cycles += vec_ops*self.cycles_per_vector_ops
        stats['VECTOR ops'] +=  vec_ops
        stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
        print "vector ", cycles, num_ops, vec_ops

      # TASK: communication across nodes, update time directly instead of cycles
      elif item[0] == 'internode':

        msg_size = item[1]
        tmp = msg_size / self.node.interconnect_bandwidth + \
               self.node.interconnect_latency
        time += tmp
        stats['internode comm time'] += tmp
        print "inter ", cycles

      # TASK: communication within a node treated as memory access
      elif item[0] == 'intranode':

        num_accesses = float(item[1])/self.ram_page_size
        cycles += num_accesses*self.ram_cycles
        stats['intranode comm time'] += num_accesses*self.ram_cycles
        print "intra ", cycles

      # TASK: memory management allocation
      elif item[0] == 'alloc':

        mem_size = item[1]
        if mem_size < 0:
          mem_size = - mem_size
        mem_alloc_success = self.node.mem_alloc(mem_size)
        if mem_alloc_success:
           # Count this as a single memory access for timing purposes
           cycles += self.ram_cycles
        else:
          # File system access or just an exception, add to time not cycles
          time += self.node.filesystem_access_time

        print "alloc ", cycles

      # TASK: memory management deallocation; changes memory footprint, not time
      elif item[0] == 'unalloc':

        mem_size = item[1]
        if mem_size > 0:
          mem_size = - mem_size
        mem_alloc_success = self.node.mem_alloc(mem_size)
        print "unalloc ", cycles

      # Invalid task
      else:

        print 'Warning: task list item', item,' cannot be parsed, ignoring it'

    # Divide cumulative cycles by clockspeed to get time; apply thread eff
    time += cycles / self.clockspeed * self.thread_efficiency()
    stats['Thread Efficiency'] = self.thread_efficiency()

    if statsFlag:
      return time, stats
    else:
      return time


  def thread_efficiency(self):
    """
    Gives the efficiency back as a function of the number of active threads.
    Function chosen as inverse of active threads. This is a cheap way of
    mimicing time slicing.
    """

    efficiency = 0.0
    if self.activethreads <=self.hwthreads:
      efficiency = 1.0
    else:
      efficiency = float(self.hwthreads)/float(self.activethreads)

    return efficiency


###########################################################
class MLIntelCore(ThreadedProcessor):
  """
  A Simian resource that represents ML Intel Sandy Bridge core. It has
  3 cache levels and vector units.
  """


  def __init__(self, node):
    """
    Initialize machine-specific values.
    """
    super(MLIntelCore, self).__init__(node)

    # Units
    self.ns = 1.0*10**(-9)    # nanoseconds
    self.kb = 1024.0          # Kilobytes
    self.mb = 1024.0**2       # Megabytes
    self.isize = 4.0          # bytes per word
    self.fsize = 8.0          # bytes per word

    # Threads and vectors
    self.maxthreads = 32            # upper bound on number active threads
    self.clockspeed = 2.60*10**9    # Hertz
    self.hwthreads = 32             # number of hardware threads
    self.vector_width = 32          # width of vector unit in bytes, 256 bit?

    # Registers
    self.num_registers = 16.0       # number of registers [bytes]
    self.register_cycles = 1.0      # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [32.0*self.kb, 256.0*self.kb, 2.5*self.mb]   # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]                      # bytes
    self.cache_latency = [0.3*self.ns, 4.0*self.ns, 16.0*self.ns]   # seconds
    self.cache_cycles = [self.cache_latency[0]*self.clockspeed,
                         self.cache_latency[1]*self.clockspeed,
                         self.cache_latency[2]*self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0       # bytes
    self.ram_latency = 50.0*self.ns   # seconds
    self.ram_cycles = self.ram_latency*self.clockspeed

    # Operations
    # Each operation takes 1 cycle but can execute more than 1 instruction per
    #   cycle - microarchitecture; e.g., a lot of multiplication, alawys need
    #   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do
    #   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some
    #   assumptions about not perfect efficiency.  
    self.cycles_per_CPU_ops = 1.0
    self.cycles_per_iALU = 0.3
    self.cycles_per_int_vec = 0.075
    self.cycles_per_fALU = 0.3
    self.cycles_per_vector_ops = 0.075
    self.cycles_per_division = 3.0


  def time_compute(self, tasklist, statsFlag=False):
    """
    Computes the cycles that the items in the tasklist (CPU ops, data access,
    vector ops, memory alloc) take.
    """

    # Initialize
    cycles = 0.0
    time = 0.0
    stats = {}
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
    stats['fDIV cycles'] =0
    stats['INTVEC ops'] =0
    stats['INTVEC cycles'] =0
    stats['VECTOR ops'] =0
    stats['VECTOR cycles'] =0
    stats['internode comm time'] =0
    stats['intranode comm time'] =0

    print "cycles = ", cycles

    for item in tasklist:

      # TASK: Cycles associated with moving data through memory and cache
      if item[0] == 'MEM_ACCESS':

        num_index_vars   = item[1]
        num_float_vars   = item[2]
        avg_dist         = item[3]
        avg_reuse_dist   = item[4]
        stdev_reuse_dist = item[5]
        index_loads      = item[6]
        float_loads      = item[7]
        init_flag        = item[8]
        
        # Insert formula that turns these variables into actual L1, L2, ram
        #   accesses. This is a V0.1 model for realistic caching behavior. The
        #   arguments to MEM_ACCESS are architecture independent. We use index
        #   vars and integers vars interchageably, ie all int variables are
        #   treated as array indices, all floats are array elements.

        # Assume index variables are in registers as much as possible. Compute
        #   number of index loads that are not handled by registers
        avg_index_loads = index_loads / num_index_vars
        num_reg_accesses = int(self.num_registers * avg_index_loads)
        nonreg_index_loads = max(0, index_loads - num_reg_accesses)

        # Compute number of variables on a typical page in memory
        num_vars_per_page = self.ram_page_size / avg_dist

	# Check memory layout to begin
        if init_flag:
          # New function call (true); floats need loaded from main memory and
          #   fewer float_loads from cache.
          initial_ram_pages = num_float_vars / num_vars_per_page
          float_loads -= num_float_vars
        else:
          # float data already in cache
          initial_ram_pages = 0.0

        # Main memory: initial_ram_pages moving through the cache
        cycles += self.ram_cycles*initial_ram_pages
        #cycles += self.cache_cycles[2]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[2]
        #cycles += self.cache_cycles[1]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[1]
        #cycles += self.cache_cycles[0]*initial_ram_pages*self.ram_page_size/ \
        #          self.cache_line_sizes[1]
        print "inital_ram_pages, cycles=", cycles, initial_ram_pages

        # Compute probability that reload is required, assume normal
        #   distribution with reuse dist larger than cache page size

        # L1
        nlines = self.cache_sizes[0]/self.cache_line_sizes[0]
        nvars  = avg_reuse_dist/self.cache_line_sizes[0]
        vvars  = math.sqrt( 2.0*(stdev_reuse_dist/self.cache_line_sizes[0])**2 )
        L1_hitrate = 0.5 * ( 1.0 + math.erf( (nlines-nvars)/vvars ) )

        L1_int_hits   = int(L1_hitrate*nonreg_index_loads)
        L1_float_hits = int(L1_hitrate*float_loads)

        L1_int_misses   = nonreg_index_loads - L1_int_hits
        L1_float_misses = float_loads - L1_float_hits

        print "l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses,\
                        L1_float_misses

        # L2
        nlines = self.cache_sizes[1]/self.cache_line_sizes[1]
        nvars  = avg_reuse_dist/self.cache_line_sizes[1]
        vvars  = math.sqrt( 2.0*(stdev_reuse_dist/self.cache_line_sizes[1])**2 )
        L2_hitrate = 0.5 * ( 1.0 + math.erf( (nlines-nvars)/vvars ) )

        L2_int_hits   = int(L2_hitrate*L1_int_misses)
        L2_float_hits = int(L2_hitrate*L1_float_misses)
        
        L2_int_misses   = L1_int_misses - L2_int_hits
        L2_float_misses = L1_float_misses - L2_float_hits

        print "l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses,\
                        L2_float_misses

        # L3
        nlines = self.cache_sizes[2]/self.cache_line_sizes[2]
        nvars  = avg_reuse_dist/self.cache_line_sizes[2]
        vvars  = math.sqrt( 2.0*(stdev_reuse_dist/self.cache_line_sizes[2])**2 )
        L3_hitrate = 0.5 * ( 1.0 + math.erf( (nlines-nvars)/vvars ) )

        L3_int_hits   = int(L3_hitrate*L2_int_misses)
        L3_float_hits = int(L3_hitrate*L2_float_misses)

        L3_int_misses = L2_int_misses - L3_int_hits
        L3_float_misses = L2_float_misses - L3_float_hits

        print "l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses,\
                        L3_float_misses

        # Update the cycles count for cache hits
        # Registers
        cycles += num_reg_accesses*self.register_cycles
        print "registers, cycles= ", cycles

        # L1, data moves together in cache lines
        cycles += self.cache_cycles[0] * \
                  (L1_float_hits*self.fsize + L1_int_hits*self.isize) / \
                  self.cache_line_sizes[0]
        print "l1 ", cycles

        # L2
        cycles += self.cache_cycles[1] * \
                  (L2_float_hits*self.fsize + L2_int_hits*self.isize) / \
                  self.cache_line_sizes[1]
        print "l2 ", cycles

        # L3
        cycles += self.cache_cycles[2] * \
                  (L3_float_hits*self.fsize + L3_int_hits*self.isize) / \
                  self.cache_line_sizes[2]
        print "l3 ", cycles

        # L3 misses accumulate time from main memory
        cycles += self.ram_cycles * \
                  (L3_float_misses + L3_int_misses) / num_vars_per_page
        print "l3 misses", cycles

        # L2 misses accumulate time from L3
        #cycles += self.cache_cycles[2] * \
        #          (L2_float_misses*self.fsize + L2_int_misses*self.isize) / \
        #          self.cache_line_sizes[2]
        #print "l2 misses", cycles

        # L1 misses accumulate time from L2
        #cycles += self.cache_cycles[1] * \
        #          (L1_float_misses*self.fsize + L1_int_misses*self.isize) / \
        #          self.cache_line_sizes[2]
        #print "l1 misses", cycles

        print "memaccess cycles= ", cycles

        stats['L1_float_hits']   += L1_float_hits
        stats['L2_float_hits']   += L2_float_hits
        stats['L1_int_hits']     += L1_int_hits
        stats['L2_int_hits']     += L2_int_hits
        stats['L1_int_misses']   += L1_int_misses
        stats['L2_int_misses']   += L2_int_misses
        stats['L1_float_misses'] += L1_float_misses
        stats['L2_float_misses'] += L2_float_misses
        stats['RAM accesses']    += (2*L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page
        stats['L1 cycles']       += self.cache_cycles[0] * \
                                    (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles']       += self.cache_cycles[1] * \
                                    (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles']      += self.ram_cycles * \
                                    (2*L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page

      # TASK: Direct input of cache level hitrates
      elif item[0] == 'HITRATES':

        L1_hitrate       = item[1]
        L2_hitrate       = item[2]
        L3_hitrate       = item[3]
        num_index_vars   = item[4]
        num_float_vars   = item[5]
        avg_dist         = item[6]
        index_loads      = item[7]
        float_loads      = item[8]
        init_flag        = item[9]
        
        # Insert formula that turns these variables into actual L1, L2, ram
        #   accesses. This is a V0.1 model for realistic caching behavior. The
        #   arguments to MEM_ACCESS are architecture independent. We use index
        #   vars and integers vars interchageably, ie all int variables are
        #   treated as array indices, all floats are array elements.

        # Assume index variables are in registers as much as possible. Compute
        #   number of index loads that are not handled by registers
        avg_index_loads = index_loads / num_index_vars
        num_reg_accesses = int(self.num_registers * avg_index_loads)
        nonreg_index_loads = max(0, index_loads - num_reg_accesses)

        # Compute number of variables on a typical page in memory
        num_vars_per_page = self.ram_page_size / avg_dist

	# Check memory layout to begin
        if init_flag:
          # New function call (true); floats need loaded from main memory and
          #   fewer float_loads from cache.
          initial_ram_pages = num_float_vars / num_vars_per_page
          float_loads -= num_float_vars
        else:
          # float data already in cache
          initial_ram_pages = 0.0

        # Main memory: initial_ram_pages moving through the cache
        cycles += self.ram_cycles*initial_ram_pages

        print "inital_ram_pages, cycles=", cycles, initial_ram_pages

        # Compute probability that reload is required, assume normal
        #   distribution with reuse dist larger than cache page size

        # L1
        L1_int_hits   = int(L1_hitrate*nonreg_index_loads)
        L1_float_hits = int(L1_hitrate*float_loads)

        L1_int_misses   = nonreg_index_loads - L1_int_hits
        L1_float_misses = float_loads - L1_float_hits

        print "l1_hit", L1_hitrate, L1_int_hits, L1_float_hits, L1_int_misses,\
                        L1_float_misses

        # L2
        L2_int_hits   = int(L2_hitrate*L1_int_misses)
        L2_float_hits = int(L2_hitrate*L1_float_misses)
        
        L2_int_misses   = L1_int_misses - L2_int_hits
        L2_float_misses = L1_float_misses - L2_float_hits

        print "l2_hit", L2_hitrate, L2_int_hits, L2_float_hits, L2_int_misses,\
                        L2_float_misses

        # L3
        L3_int_hits   = int(L3_hitrate*L2_int_misses)
        L3_float_hits = int(L3_hitrate*L2_float_misses)

        L3_int_misses = L2_int_misses - L3_int_hits
        L3_float_misses = L2_float_misses - L3_float_hits

        print "l3_hit", L3_hitrate, L3_int_hits, L3_float_hits, L3_int_misses,\
                        L3_float_misses

        # Update the cycles count for cache hits
        # Registers
        cycles += num_reg_accesses*self.register_cycles
        print "registers, cycles= ", cycles

        # L1, data moves together in cache lines
        cycles += self.cache_cycles[0] * \
                  (L1_float_hits*self.fsize + L1_int_hits*self.isize) / \
                  self.cache_line_sizes[0]
        print "l1 ", cycles

        # L2
        cycles += self.cache_cycles[1] * \
                  (L2_float_hits*self.fsize + L2_int_hits*self.isize) / \
                  self.cache_line_sizes[1]
        print "l2 ", cycles

        # L3
        cycles += self.cache_cycles[2] * \
                  (L3_float_hits*self.fsize + L3_int_hits*self.isize) / \
                  self.cache_line_sizes[2]
        print "l3 ", cycles

        # L3 misses accumulate time from main memory
        cycles += self.ram_cycles * \
                  (L3_float_misses + L3_int_misses) / num_vars_per_page
        print "l3 misses", cycles

        print "memaccess cycles= ", cycles

        stats['L1_float_hits']   += L1_float_hits
        stats['L2_float_hits']   += L2_float_hits
        stats['L1_int_hits']     += L1_int_hits
        stats['L2_int_hits']     += L2_int_hits
        stats['L1_int_misses']   += L1_int_misses
        stats['L2_int_misses']   += L2_int_misses
        stats['L1_float_misses'] += L1_float_misses
        stats['L2_float_misses'] += L2_float_misses
        stats['RAM accesses']    += (2*L2_float_misses + L2_int_misses) / \
                                    num_vars_per_page
        stats['L1 cycles']       += self.cache_cycles[0] * \
                                    (2*L1_float_hits + L1_int_hits)
        stats['L2 cycles']       += self.cache_cycles[1] * \
                                    (2*L2_float_hits + L2_int_hits)
        stats['RAM cycles']      += self.ram_cycles * \
                                    num_vars_per_page

      # TASK: Direct input of L1 accesses
      elif item[0] == 'L1':

        num_accesses = item[1]
        cycles += num_accesses*self.cache_cycles[0]
        print "l1 ", cycles

      # TASK: Direct input of L2 accesses
      elif item[0] == 'L2':

        num_accesses = item[1]
        cycles += num_accesses*self.cache_cycles[1]           
        print "l2 ", cycles

      # TASK: Direct input of higher cache and memory accesses
      elif item[0] in ['L3', 'L4', 'L5', 'RAM', 'mem']:

        num_accesses = item[1]
        cycles += num_accesses*self.ram_cycles
        print "l3 ", cycles

      # TASK: CPU ops
      elif item[0] == 'CPU':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_CPU_ops
        stats['CPU cycles'] += num_ops*self.cycles_per_CPU_ops
        print "cpu ", cycles

      # TASK: Integer operations
      elif item[0] == 'iALU':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_iALU
        stats['iALU cycles'] +=   num_ops*self.cycles_per_iALU
        print "ialu ", cycles

      # TASK: Floating point operations
      elif item[0] == 'fALU':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_fALU
        stats['fALU cycles'] +=   num_ops*self.cycles_per_fALU
        print "falu ", cycles, num_ops

      # TASK: Floating point divisions
      elif item[0] == 'fDIV':

        num_ops = item[1]
        cycles += num_ops*self.cycles_per_division
        stats['fDIV cycles'] +=   num_ops*self.cycles_per_division
        print "fDIV ", cycles, num_ops

      # TASK: Integer vector operations
      elif item[0] == 'INTVEC':

        num_ops   = item[1]
        vec_width = item[2]

        # vec_ops is the actual number of operations necessary, even if
        #   required width was too high. The // operator is floor integer
        #   division.

        vec_ops = (1+vec_width//self.vector_width)*num_ops
        cycles += vec_ops*self.cycles_per_int_vec
        stats['INTVEC ops'] +=  vec_ops
        stats['INTVEC cycles'] += vec_ops*self.cycles_per_int_vec
        print "intvect ", cycles, num_ops, vec_ops

      # TASK: Vector operations
      elif item[0] == 'VECTOR':

        num_ops   = item[1]
        vec_width = item[2]

        # vec_ops is the actual number of operations necessary, even if
        #   required width was too high. The // operator is floor integer
        #   division.

        vec_ops = (1+vec_width//self.vector_width)*num_ops
        cycles += vec_ops*self.cycles_per_vector_ops
        stats['VECTOR ops'] +=  vec_ops
        stats['VECTOR cycles'] += vec_ops*self.cycles_per_vector_ops
        print "vector ", cycles, num_ops, vec_ops

      # TASK: communication across nodes, update time directly instead of cycles
      elif item[0] == 'internode':

        msg_size = item[1]
        tmp = msg_size / self.node.interconnect_bandwidth + \
               self.node.interconnect_latency
        time += tmp
        stats['internode comm time'] += tmp
        print "inter ", cycles

      # TASK: communication within a node treated as memory access
      elif item[0] == 'intranode':

        num_accesses = float(item[1])/self.ram_page_size
        cycles += num_accesses*self.ram_cycles
        stats['intranode comm time'] += num_accesses*self.ram_cycles
        print "intra ", cycles

      # TASK: memory management allocation
      elif item[0] == 'alloc':

        mem_size = item[1]
        if mem_size < 0:
          mem_size = - mem_size
        mem_alloc_success = self.node.mem_alloc(mem_size)
        if mem_alloc_success:
           # Count this as a single memory access for timing purposes
           cycles += self.ram_cycles
        else:
          # File system access or just an exception, add to time not cycles
          time += self.node.filesystem_access_time

        print "alloc ", cycles

      # TASK: memory management deallocation; changes memory footprint, not time
      elif item[0] == 'unalloc':

        mem_size = item[1]
        if mem_size > 0:
          mem_size = - mem_size
        mem_alloc_success = self.node.mem_alloc(mem_size)
        print "unalloc ", cycles

      # Invalid task
      else:

        print 'Warning: task list item', item,' cannot be parsed, ignoring it'

    # Divide cumulative cycles by clockspeed to get time; apply thread eff
    time += cycles / self.clockspeed * self.thread_efficiency()
    stats['Thread Efficiency'] = self.thread_efficiency()

    if statsFlag:
      return time, stats
    else:
      return time


  def thread_efficiency(self):
    """
    Gives the efficiency back as a function of the number of active threads.
    Function chosen as inverse of active threads. This is a cheap way of
    mimicing time slicing.
    """

    efficiency = 0.0
    if self.activethreads <=self.hwthreads:
      efficiency = 1.0
    else:
      efficiency = float(self.hwthreads)/float(self.activethreads)

    return efficiency
