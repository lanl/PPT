# sequoia_config.py :- configuration for Sequoia

sequoia_intercon = {
    # The dimensions of Sequoia's torus are 16 x 12 x 16 x 16 x 2. 
    # Ref: https://computing.llnl.gov/tutorials/bgq/index.htm://computing.llnl.gov/tutorials/bgq/index.html 
    #"dima" : 16,
    #"dimb" : 12,
    #"dimc" : 16,
    #"dimd" : 16,
    #"dime" : 2,
    
    "dima" : 4,
    "dimb" : 4,
    "dimc" : 4,
    "dimd" : 4,
    "dime" : 2,

    # 2GB/s raw bandwidth on all 10 links (each direction)
    # Ref: Blue Gene/Q Overview and Update (November 2011) 
    #"bdwa" : 1.6e10,
    #"bdwb" : 1.6e10,
    #"bdwc" : 1.6e10,
    #"bdwd" : 1.6e10,
    #"bdwe" : 1.6e10,
    
    "bdwa" : 1.6e10,
    "bdwb" : 1.6e10,
    "bdwc" : 1.6e10,
    "bdwd" : 1.6e10,
    "bdwe" : 1.6e10,
    
    # MPI ping-pong, zero byte latency, 
    # nearest neighbors: 80ns
    # across full machine: 3 us
    # 80/2ns = 40ns = 0.04us, (3-0.04)/16 = 0.185

    #"host_link_delay" : 0.04e-6,
    #"switch_link_delay" : 0.185e-6,
    
    # From "The IBM Blue Gene/Q interconnection network and message unit" 
    # The on-chip per hop latency for point-to-point packets on BG/Q is approximately 40 ns
    "host_link_delay" : 4e-8,   # 4e-8
    "switch_link_delay" : 4e-8, # 4e-8

}
