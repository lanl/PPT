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
