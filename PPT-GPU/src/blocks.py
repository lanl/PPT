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

class Block(object):
    '''
    class that represents a block (CTA) being executed on an SM
    '''
    def __init__(self, gpu, id, num_warps, num_warp_schedulers, tasklist, kernel_id, isa, avg_mem_lat, avg_atom_lat):
        self.num_warps = num_warps 
        self.num_active_warps = 0
        self.sync_warps = 0
        self.id = id
        self.active = False
        self.waiting_to_execute = True
        self.actual_end = 0
        self.warp_list = []
        
        if isa == "SASS":
            # Here: len(self.kernel_tasklist) <= self.num_warps
            for warp in list(tasklist):
                self.warp_list.append(Warp(self, gpu, tasklist[warp], kernel_id, avg_mem_lat, avg_atom_lat))
        else:    
            for i in range(self.num_warps):
                self.warp_list.append(Warp(self, gpu, tasklist, kernel_id, avg_mem_lat, avg_atom_lat))
        
        self.num_warp_schedulers = num_warp_schedulers


    def is_waiting_to_execute(self):
        return self.waiting_to_execute


    def is_active(self):
        '''
        return whether the block has an active warp or not
        '''
        res = False
        if not self.active:
            return False

        # check whether all warps have reached a sync point
        # if self.sync_warps == len(self.warp_list):
        #     # allow warps to resume computations
        #     self.sync_warps = 0
        #     for warp in self.warp_list:
        #         warp.syncing = False

        actual_end = 0
        for warp in self.warp_list:
            if warp.is_active():
                return True
            max_warp_completions = max(warp.completions)
            actual_end = max(actual_end, max_warp_completions)
        self.active = False
        self.actual_end = actual_end


    def count_active_warps(self):
        achieved_active_warps = 0
        for warp in self.warp_list:
            if warp.is_active():
                achieved_active_warps += 1
        return achieved_active_warps
