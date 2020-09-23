"""
Input example file for MMULT C mini-app, can be used as template for other applications
"""
# To run: python ppt.py in-example

task_graph = "MMULT_taskgraph"
reuse_distance_functions = "MMULT_rd"
basic_block_count_functions = "MMULT_bbc"
basic_block_ids = "MMULT_bbId"

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
    #"goldmontPlus",
    #"goldmont",
    #"haswell",
    #"ivybridge",
    #"knl",
    #"merom",
    #"nehalem",
    #"sandybridge",
    #"silvermont",
    #"skylake",
    #"skylake-X",
    #"wolfdale"
]

input_params = [# Order matters, that is why it is a list and not a dictionary
    [1000, 10000, 65000]      # Nx = Number of entries in a row or column of a matrix (matrix is square) 
]


# Sensitivity Analysis flags

print_block_info = False
sensitivity_analysis = False