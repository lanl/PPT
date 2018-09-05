# Copyright (c) 2018, Los Alamos National Security, LLC
# All rights reserved.
# Copyright 2017. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
# Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.



'''
*********** Performance Prediction Toolkit PPT *********

File: phit.py
Description main library of hardware core definitions.
'''
import sys

import math

'''
def __init__(self, node):
    super(GenericProcessor, self).__init__(node)
    # Units
    self.ns = 1.0 * 10 ** (-9)  # nanoseconds
    self.kb = 1024.0  # Kilobytes
    self.mb = 1024.0 ** 2  # Megabytes
    self.isize = 4.0  # bytes per word
    self.fsize = 8.0  # bytes per word

    # Threads and vectors
    self.maxthreads = 16  # upper bound on number active threads
    self.clockspeed = 2.40 * 10 ** 9  # Hertz
    self.hwthreads = 1  # number of hardware threads
    self.vector_width = 16  # width of vector unit in bytes, 128 bit?

    # Registers
    self.num_registers = 16.0  # number of registers [bytes]
    self.register_cycles = 1  # cycles per register access

    # Cache details
    self.cache_levels = 3
    self.cache_sizes = [64.0 * self.kb, 512.0 * self.kb, 1.5 * self.mb]  # bytes
    self.cache_line_sizes = [64.0, 64.0, 64.0]  # bytes
    self.cache_latency = [1.0 * self.ns, 5.0 * self.ns, 25.0 * self.ns]  # seconds
    self.cache_cycles = [self.cache_latency[0] * self.clockspeed,
                        self.cache_latency[1] * self.clockspeed,
                        self.cache_latency[2] * self.clockspeed]

    # Main memory
    self.ram_page_size = 4096.0  # bytes
    self.ram_latency = 60.0 * self.ns  # seconds
    self.ram_cycles = self.ram_latency * self.clockspeed

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

'''

ns = 1.0 * 10 ** (-9)  # nanoseconds
kb = 1024.0  # Kilobytes
mb = 1024.0 ** 2  # Megabytes
isize = 4.0  # bytes per word
fsize = 8.0  # bytes per word


associativity = [8.0, 8.0, 20.0]
cache_sizes = [256.0 * kb, 1024.0 * kb, 6.0 * mb]
cache_line_sizes = [64.0, 64.0, 64.0]

sd_psd = open(sys.argv[1],'r').read().split('\n')
sd_psd = filter(None,sd_psd)
list_sd_psd = [item.split(',') for item in sd_psd]

stack_dist = zip(*list_sd_psd)[0]
stack_dist = map(float,stack_dist)


probability_sd = zip(*list_sd_psd)[1]
probability_sd = map(float,probability_sd)


def ncr( n, m):
      """
      n choose m
      """
      if(m>n): return 0
      r = 1
      for j in xrange(1,m+1):
          r *= (n-m+j)/float(j)
      return r

#   def phit_D(self, D=9.0, A, cs, ls): #We can keep 9.0 as the default Stack distance if Associativity=8
def phit_D( D, A, cs, ls):
  """
  Calculate the probability of hit (given stack distance, D) for a give cache level
  Output: Gives the probability of a hit given D -- P(h/D)
  """
  #D = 4   stack distance (need to take either from tasklist or use grammatical function)
  # A (Associativity)
  phit_D = 0.0 #To compute probability of a hit given D
  B = (1.0 * cs)/ls  # B = Block size (cache_size/line_size)

  if (D <= A):
      if (D == -1):   D = cache_sizes[2]
      elif (D == 0):  phit_D = 1.0
      else:    phit_D = math.pow((1 - (1/B)), D)
  # Don't be too creative to change the follow condition to just 'else:'
  # I am changing the value of D in the previous condition.
  if(D > A):
    for a in xrange(int(A)):
      term_1 = ncr(D,a)
      #term_1 = math.gamma(D + 1) / (1.0 * math.gamma(D - a + 1) * math.gamma(a + 1))
      term_2 = math.pow((A/B), a)
      term_3 = math.pow((1 - (A/B)), (D - a))
      phit_D += (term_1 * term_2 * term_3)

  return phit_D

def phit_sd( stack_dist, assoc, c_size, l_size):
    """
    Calculate probability of hits for all the stack distances
    """
    phit_sd = [phit_D(d, assoc, c_size, l_size) for d in stack_dist]
    return phit_sd

def phit( Pd, Phd):
  """
  Calculate probability of hit (given P(D), P(h/D) for a given cache level
  Output: Gives the probability of a hit -- P(h) = P(D)*P(h/D)
  """
  phit = 0.0 #Sum (probability of stack distance * probability of hit given D)
  Ph = map(lambda pd,phd:pd*phd,Pd,Phd)
  phit = sum(Ph)
  return phit

def effective_cycles(phit_L1, phit_L2, phit_L3, cycles, ram_penality):
  """
  Calculate effective clock cycles for the given arguments
  """
  eff_clocks = 0.0
  #X#print "Latencies/ReciprocalThroughput(1/BW):", cycles
  eff_clocks=(cycles[0]*phit_L1+(1.0-phit_L1)* (cycles[1]*phit_L2 + (1.0-phit_L2)* \
            (cycles[2]*phit_L3+(1.0-phit_L3)* ram_penality)))
  return eff_clocks





L3_phits_d = phit_sd(stack_dist, associativity[2], cache_sizes[2], cache_line_sizes[2])
#L3_phit = p(d) * p(h/d) i.e; psd * phits_d
L3_phit = phit(probability_sd, L3_phits_d)
print ("l3_hit ", L3_phit)
