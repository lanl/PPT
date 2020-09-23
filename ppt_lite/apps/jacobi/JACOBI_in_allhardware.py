"""
Input example file for JACOBI C mini-app, can be used as template for other applications
"""
# To run: python ppt.py in-example

task_graph = "JACOBI_taskgraph"
reuse_distance_functions = "JACOBI_rd"
basic_block_count_functions = "JACOBI_bbc"
basic_block_ids = "JACOBI_bbId"

hardware_platforms = [ \
#    "KNL",
    "AMDbobcat",
    "AMDbulldozer",
    "AMDexcavator",
    "AMDjaguar",
    "AMDpiledriver",
    "AMDryzen",
    "AMDsteamroller",
    "broadwell",
    "goldmontPlus",
    "goldmont",
    "haswell",
    "ivybridge",
    "knl",
    "merom",
    "nehalem",
    "sandybridge",
    "silvermont",
    "skylake",
    "skylake-X",
    "wolfdale"
]

input_params = [# Order matters, that is why it is a list and not a dictionary
    [10, 100, 1000]      # Nx = Number of entries in a row or column of a matrix (matrix is square) 
]
