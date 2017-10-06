# trinity_config.py :- configuration for a dragonfly-based Trinity supercomputer

# Aries has 96 switches per group, 4 nodes per switch; 40 network ports per host
# and 8 processor tiles:  "Cray XC Series Network" whitepaper

# The Aries router connects 8 NIC ports to 40 network ports, operating at rates
# of 4.7 to 5.25GB/s per direction per port: "Cray XC series Network" whitepaper

# Peak bandwidth of the 16X PCI-Express Gen3 interface connecting each Cray XC series
# node to its Aries is 16GB/s, 8 giga-transfers per second, each of 16 bits: 
# "Cray XC series Network" whitepaper

# Inter group link bandwidth 4.7 GB/s per direction and intra group link 
# bandwidth 5.25 GB/s per direction: "Analyzing Network Health and Congestion 
# in Dragonfly-based Systems"

# Delay references:
# "Characterization of the Cray Aries Network" by Brian Austin (slide 7)
# Blade latency: 1.3us. host-switch latency: 1.3/2 = 0.65us
# Rank-1 latency: 1.5us. blade-blade latency: 1.5us
# Rank-2 latency: 1.5us. chassis-chassis latency: 1.5us
# Rank-3 latency: 2.2us. inter-group latency: 2.2us

darter_intercon = {
    "num_groups": 2,
    "num_switches_per_group": 96,
    "num_hosts_per_switch": 4,
    "num_ports_per_host": 48,
    "num_inter_links_per_switch": 10,

    "inter_group_bdw": 3.8e10,      #4.75e9 (bytes)
    "inter_chassis_bdw": 4.2e10,    #5.25e9
    "intra_chassis_bdw": 4.2e10,    #5.25e9
    "switch_host_bdw": 12.8e10,     #16e9

    "inter_group_delay": 0.1e-6,    #2.2e-6
    "inter_chassis_delay": 0.1e-6,  #1.5e-6 
    "intra_chassis_delay": 0.1e-6,  #1.5e-6
    "switch_host_delay": 0.4e-6,   #0.65e-6

    "inter_link_dups": 1,
    "inter_chassis_dups": 1,
    "intra_chassis_dups": 1, # i.e., among blades
    
    "num_chassis_per_group": 6,
    "num_blades_per_chassis": 16,
    "intra_group_topology": "cascade",  # cascade (aries) or all-to-all (dragonfly)
    "inter_group_topology": 'consecutive_aries', # consecutive_aries (aries) or consecutive (dragonfly)
    "num_inter_links_grouped": 4, # for aries, 4 inter-links are grouped into 1 bigger link
    "num_intra_links_grouped": 3, # for aries, 3 intra-links (among chassis) are grouped into 1 bigger link
    "route_method": "minimal",  # minimal or non_minimal
}

