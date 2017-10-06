# dragonfly_config.py :- configuration for a simple dragonfly and Aries topo confgis

dragonfly_intercon = {
    # This is a simple dragonfly topology taken as example from the paper
    # "Technology-Driven, Highly-Scalable Dragonfly Topology" by
    # William J. Dally
    "num_groups": 9,
    "num_switches_per_group": 50, #50
    "num_hosts_per_switch": 2, #2
    "num_ports_per_host": 7,
    "num_inter_links_per_switch": 2,
    "num_intra_links_per_switch": 49, #49 
    
    #"inter_group_bdw" : 4.75e9,
    #"intra_group_bdw" : 5.25e9,
    #"switch_host_bdw" : 1.6e9,
    
    "inter_group_bdw" : 7e7,
    "intra_group_bdw" : 7e7,
    "switch_host_bdw" : 7e7,

    "inter_group_delay" : 1.92e-6, 
    "intra_group_delay" : 1.545e-6,
    "switch_host_delay" : 1.498e-6, 

    "inter_link_dups": 1, 
    "intra_group_topology": "all_to_all",
    "inter_group_topology": "consecutive",
    "intra_link_dups": 1,
    "route_method": "minimal",  # minimal or non_minimal
}

# also contains a simple Aries topology configurations
aries_intercon = {
    # This is a simple Aries topology.
    "num_groups": 8,
    "num_switches_per_group": 96, #96 or 16
    "num_hosts_per_switch": 4,
    "num_ports_per_host": 7,
    "num_inter_links_per_switch": 10, #2 or 1 (8, when links are bundled together)
    
    "inter_group_bdw": 4.75e9,
    "inter_chassis_bdw": 5.25e9, 
    "intra_chassis_bdw": 5.25e9,
    "switch_host_bdw": 16e9,

    "inter_group_delay": 2.2e-6, 
    "inter_chassis_delay": 1.5e-6, 
    "intra_chassis_delay": 1.5e-6,
    "switch_host_delay": 0.65e-6,

    "num_chassis_per_group": 6, #6 or 2
    "num_blades_per_chassis": 16,   #16 or 8
    "intra_group_topology": "cascade",
    "inter_group_topology": 'consecutive_aries',
    "num_inter_links_grouped": 4, # for Aries, 4 inter-links are grouped into 1 bigger link
    "num_intra_links_grouped": 3, # for Aries, 3 intra-links (among chassis) are grouped into 1 bigger link
    "route_method": "non_minimal",  # minimal or non_minimal

    "inter_link_dups": 1, 
    "inter_chassis_dups": 1, 
    "intra_chassis_dups": 1, 
}

