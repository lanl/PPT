# titan_config.py :- configuration for Titan

titan_intercon = {
    # The dimensions of Titan's torus are 25 x 16 x 24. (Ref: "Titan: 
    # Early experience with the Cray XK6 at Oak Ridge National Laboratory", 
    # by Arthur S. Bland,  Jack C. Wells ,  Otis E. Messer,  Oscar R. Hernandez, James H. Rogers
    
    "dimx" : 25,
    "dimy" : 16,
    "dimz" : 24,

    # MPI unidirectional bandwidth, as given in paper: 
    # "Titan: Early experience with Cray XK6 at Oak Ridge National Laboratory"
    "bdwx" : 5.87e9,
    "bdwy" : 3.47e9,
    "bdwz" : 6.09e10,
    
    # Maximum injection bandwidth per node is 20 GB/s.
    # Ref: "Scalable Algorithm for Radiative Heat Transfer Using Reverse Monte Carlo Ray Tracing"
    "bdwh" : 1.6e11,

    # MPI ping-pong, zero byte latency, 
    # nearest neighbors: 1.5 us (x-dim), 1.7 us (y-dim), 1.55 us (z-dim)
    # across full machine: 3.96 us
    # 1.5/2 = 0.75, (3.96-1.5)/25 = 0.0984

    "host_link_delay" : 0.75e-6,
    "switch_link_delay" : 0.0984e-6,
}
