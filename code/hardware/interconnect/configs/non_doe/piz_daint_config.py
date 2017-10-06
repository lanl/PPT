# piz_daint_config.py :- configuration for a dragonfly-based Piz Daint supercomputer

# Aries has 96 switches per group, 4 nodes per switch; 40 network ports per host
# and 8 processor tiles:  "Cray XC Series Network" whitepaper

# Piz Daint has 14 groups (out of possible 241 groups): 
# "Piz Daint :" Application driven co-design of a supercomputer based 
# on Crays adaptive system design" by Sadaf Alam and Thomas Schulthess

# The Aries router connects 8 NIC ports to 40 network ports, operating at rates
# of 4.7 to 5.25GB/s per direction per port: "Cray XC series Network" whitepaper

# Peak bandwidth of the 16X PCI-Express Gen3 interface connecting each Cray XC series
# node to its Aries is 16GB/s, 8 giga-transfers per second, each of 16 bits: 
# "Cray XC series Network" whitepaper

# Inter group link bandwidth 4.7 GB/s per direction and intra group link 
# bandwidth 5.25 GB/s per direction: "Analyzing Network Health and Congestion 
# in Dragonfly-based Systems"

############### 
# Delay references: 
# According to Cray XC whitepaper, measured end-to-end latencies for user-space
# communication are 0.8us for an 8-byte put.
# For a quiet network, each router-to-router hop adds approximately 100ns of latency.
# Quiet network latency between any pair of nodes is less than 2us on the target 
# systems.
# 2us/2 = 1us
#"inter_group_delay" : 100e-9,   
#"intra_group_delay" : 100e-9, # Assumption: inter- and intra-group delay are same
#"switch_host_delay" : 1e-6,

# From Cray XC30 system:overview
# Maximum Inter-node latency (farthest-node pair): 1.920
# Maximum Intra-node latency (farthest-node pair): 0.545
# Minimum Inter-node latency (nearest-node pair): 1.498

# USED THE FOLLOWING:
# Following presentation with all the latency details (slide 7): 
# "Characterization of the Cray Aries Network"
# by Brian Austin
# Blade latency: 1.3us. host-switch latency: 1.3/2 = 0.75us
# Rank-1 latency: 1.5us. blade-blade latency: 1.5us
# Rank-2 latency: 1.5us. chassis-chassis latency: 1.5us
# Rank-3 latency: 2.2us. inter-group latency: 2.2us

piz_daint_intercon = {
    "num_groups": 14,
    "num_switches_per_group": 96,
    "num_hosts_per_switch": 4,
    "num_ports_per_host": 48,
    "num_inter_links_per_switch": 10,

    "inter_group_bdw": 4.75e9,
    "inter_chassis_bdw": 5.25e9, 
    "intra_chassis_bdw": 5.25e9,
    "switch_host_bdw": 16e9,

    "inter_group_delay": 2.2e-6, 
    "inter_chassis_delay": 1.5e-6, 
    "intra_chassis_delay": 1.5e-6,
    "switch_host_delay": 0.65e-6,

    "inter_link_dups": 1,
    "inter_chassis_dups": 1,
    "intra_chassis_dups": 1, 
    
    "num_chassis_per_group": 6,
    "num_blades_per_chassis": 16,
    "intra_group_topology": "cascade",
    "inter_group_topology": 'consecutive_aries',
    "num_inter_links_grouped": 4, # for Aries, 4 inter-links are grouped into 1 bigger link
    "num_intra_links_grouped": 3, # for Aries, 3 intra-links (among chassis) are grouped into 1 bigger link
    "route_method": "minimal",  # minimal or non-minimal
}

