# Copyright (c) 2017, Los Alamos National Security, LLC
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


# summihttp://www.intel.com/content/www/us/en/high-performance-computing-fabrics/omni-path-edge-switches-100-series.htmlt_config.py :- configuration for a fat-tree-based Summit supercomputer

# http://richardcaseyhpc.com/summit/
# Summit interconnect employs a 2:1 oversubscribed Fat-Tree topology.
# This is one of the most common and well-tested interconnect topologies in the HPC community.
# On Summit it consists of 48-port Level-2 OPA switch Core nodes, 48-port Level-1 OPA switch Leaf nodes,
# and 376 compute nodes.  2:1 oversubscription represents a cost-performance tradeoff,
# with a reduction in the number of (relatively expensive) network switches for a small performance hit.

# http://www.intel.com/content/www/us/en/high-performance-computing-fabrics/omni-path-edge-switches-100-series.html
# Summit uses Intel OmniPath switched-based interconnect.
# Omnipath Architecture Interconnect (OPA) bandwidth: 100Gb/s
# 100-110ns switch latency

summit_intercon = {
    "num_ports_per_switch": 64, # m-port
    "num_levels": 2, # n-tree

    "switch_link_up_delay": 0.105e-6, # 105ns
    "switch_link_down_delay": 0.105e-6,
    "host_link_delay": 0.105e-6,

    "switch_link_up_bdw": 12.5e9, # 100Gb/s
    "switch_link_down_bdw": 12.5e9,
    "host_link_bdw": 12.5e9,

    "switch_link_dups": 1,
    "route_method": "multiple_lid_nca", # multiple LID nearest common ancestor
}
