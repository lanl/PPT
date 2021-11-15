##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##############################################################################

from .warps import Warp

class Scheduler(object):
    '''
    class that represents a warp scheduler inside an SM
    '''
    def __init__(self, num_warp_schedulers, policy):
       self.num_warp_schedulers = num_warp_schedulers
       self.policy = policy

    def step(self, warp_list, cycles):
        '''
        advance computation for the active warps by one cycle,
        choose which step function to execute depending on the scheduling policy
        '''
        if self.policy == "LRR":
            return self.step_LRR(warp_list, cycles)
        elif self.policy == "GTO":
            return self.step_GTO(warp_list, cycles)
        elif self.policy == "TL":
            return self.step_TL(warp_list, cycles)
        

    def step_LRR(self, warp_list, cycles):
        '''
        loop over every available warp and issue warp if ready, 
        if warp is not ready, skip and issue next ready warp
        '''
        warps_executed = 0
        ints_executed = 0

        for warp in warp_list:
            # executed max warps in current clock cycles
            # selected warps per multiprocessor per cycle is bounded to the range of [0 .. # warp schedulers per SM]
            if warps_executed >= self.num_warp_schedulers:
                break
            # see if we can execute warp
            if warp.is_active():
                current_inst_executed = warp.step(cycles)
                # warp executed current instruction successfully  
                if current_inst_executed:
                    ints_executed += current_inst_executed
                    warps_executed += 1
            else:
                print("Error")
                break
                
        return ints_executed