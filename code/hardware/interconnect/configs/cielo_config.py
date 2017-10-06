# cielo_config.py :- configuration for Cielo

cielo_intercon = {
    # The dimensions of Cielo's torus are 16 x 12 x 24. (Ref: "Using
    # the Cray Gemini Performance Counters", by Kevin Pedretti,
    # Courtenay Vaughan, Richard Barrett, Karen Devine, K. Scott
    # Hemmer).
    "dimx" : 16,
    "dimy" : 12,
    "dimz" : 24,

    # Each of the 10 Gemini torus connections is comprised of 12 lanes
    # in each direction operating at 3.125 to 6.25 GHz, yielding link
    # bandwidths of 4.68 to 9.375 GBytes/sec per direction, although
    # MPI can see only 2.9 to 5.8 GB/sec per direction.
    "bdwx" : 3.744e10,
    "bdwy" : 3.744e10,
    "bdwz" : 3.744e10,
    
    # Maximum injection bandwidth per node is 20 GB/s.
    "bdwh" : 1.6e11,

    # We keep the same set of values as those in Hopper
    "host_link_delay" : 0.635e-6,
    "switch_link_delay" : 0.10875e-6,
}
