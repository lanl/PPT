# stampede_config.py :- configuration for a fat-tree-based Stampede topology

# https://portal.tacc.utexas.edu/user-guides/stampede
# The 56GB/s FDR InfiniBand interconnect consists of Mellanox switches, 
# fiber cables and HCAs (Host Channel Adapters). Eight 648-port SX6536 core switches
# and over 320 36-port SX6025 endpoint switches (2 in each compute-node rack) 
# form a 2-level Clos fat tree topology, illustrated in Figure 1.4. 
# Core and endpoint switches have 4.0 and 73 Tb/s capacities, respectively. 
# There is a 5/4 oversubscription at the endpoint (leaf) switches (20 node input 
# ports: 16 core-switch output ports). Any MPI message is only 5 hops or less from source to destination.

# http://www.mellanox.com/page/performance_infiniband
# Latency for Mellanox FDR IB is: 0.7us

stampede_intercon = {
    "num_ports_per_switch": 4, # m-port
    "num_levels": 3, # n-tree
    "switch_link_up_delay": 0.7e-6,
    "switch_link_down_delay": 0.7e-6,
    "host_link_delay": 0.7e-6,
    "switch_link_up_bdw": 44.8e10,  #56GB/s
    "switch_link_down_bdw": 44.8e10,
    "host_link_bdw": 56e9, #56Gb/s
    "switch_link_dups": 1,
    "route_method": "multiple_lid_nca", # multiple LID nearest common ancestor
}

