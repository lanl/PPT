# omnipath_config.py :- configuration for an omnipath-based Grizzly supercomputer

omnipath_intercon = {
    # Number of ports per leaf switch in Grizzly is 48,
    # using number of ports as 64, as for m-port n-tree structure,
    # m should be a power of 2.
    # For omni-path structure, first 48 ports will be used.
    "num_ports_per_switch": 64,  
    "num_levels": 2,
    
    # link delays:
    # Ref: "Transforming the economics of HPC fabrics with intel omni-path architecture"
    "switch_link_up_delay": 105e-9,
    "switch_link_down_delay": 105e-9,
    "host_link_delay": 105e-9,
    
    # bandwidths:
    # Ref: "Transforming the economics of HPC fabrics with intel omni-path architecture"
    "switch_link_up_bdw": 100e10,
    "switch_link_down_bdw": 100e10,
    "host_link_bdw": 100e10, 

    "switch_link_dups": 1,
    "route_method": "multiple_lid_nca", # multiple LID nearest common ancestor
}

