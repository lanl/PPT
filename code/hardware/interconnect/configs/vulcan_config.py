# vulcan_config.py :- configuration for Vulcan

vulcan_intercon = {
    # Vulcan is a 24-rack system (either in six rows of four racks or two rows of twelve racks)
    # With 24,576 nodes, Vulcan's torus has dimension: 8 x 12 x 16 x 8 x 2 or 8 x 8 x 12 x 16 x 2
    # Ref:https://computing.llnl.gov/tutorials/bgq/images/bgqTorusDimensions.pdf 
    
    "dima" : 6,
    "dimb" : 2,
    "dimc" : 6,
    "dimd" : 6,
    "dime" : 2,

    # 2GB/s raw bandwidth on all 10 links (each direction)
    # Ref: Blue Gene/Q Overview and Update (November 2011) 
    "bdwa" : 2e9,
    "bdwb" : 2e9,
    "bdwc" : 2e9,
    "bdwd" : 2e9,
    "bdwe" : 2e9,
    
    # MPI ping-pong, zero byte latency, 
    # nearest neighbors: 80ns
    # across full machine: 3 us
    # 80/2ns = 40ns = 0.04us, (3-0.04)/16 = 0.185

    "host_link_delay" : 0.04e-6,
    "switch_link_delay" : 0.185e-6,
}
