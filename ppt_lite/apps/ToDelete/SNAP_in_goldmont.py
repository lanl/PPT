"""
Input example file for SNAP C mini-app, can be used as template for other applications
"""
# To run: python ppt.py in-example.py

task_graph = "SNAP_taskgraph"
reuse_distance_functions = "SNAP_rd"
basic_block_count_functions = "SNAP_bbc"
basic_block_ids = "SNAP_bbId"

hardware_platforms = [ \
#    "KNL",
    "goldmont"
]

input_params = [# Order matters, that is why it is a list and not a dictionary
    [10],      # Nx = Number of cells in x direction
    [100],    # Ny = Number of cells in y direction
    [10],    # Nz = Number of cells in z direction
    [2],        # Ic = ichunk
    [2],        # Nm = Number of moments
    [2],        # Nang = Number of angles
    [2],        # Ng = Number of energy groups
    [2],        # Li = Number of inner iterations
    [2]         # Lo = Number of outer iterations
]
