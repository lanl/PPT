"""
Input example file for SNAP C mini-app, can be used as template for other applications
"""
# To run: python ppt.py in-example

task_graph = "SNAP_taskgraph"
reuse_distance_functions = "SNAP_rd"
basic_block_count_functions = "SNAP_bbc"
basic_block_ids = "SNAP_bbId"

hardware_platforms = [ \
    #"AMDbobcat",
    #"AMDbulldozer",
    #"AMDexcavator",
    #"AMDjaguar",
    #"AMDpiledriver",
    #"AMDryzen",
    #"AMDsteamroller",
    #"broadwell",
    #"goldmontPlus",
    #"goldmont",
    #"haswell",
    #"ivybridge",
    #"knl",
    #"merom",
    #"nehalem",
    #"sandybridge",
    #"silvermont",
    "skylake",
    #"skylake-X",
    #"wolfdale"
]

input_params = [# Order matters, that is why it is a list and not a dictionary
    [1000],      # Nx = Number of cells in x direction
    [1000, 10000],    # Ny = Number of cells in y direction
    [10, 100],    # Nz = Number of cells in z direction
    [2],        # Ic = ichunk
    [2],        # Nm = Number of moments
    [12, 24],        # Nang = Number of angles
    [20, 40],        # Ng = Number of energy groups
    [10],        # Li = Number of inner iterations
    [5]         # Lo = Number of outer iterations
]


# Sensitivity Analysis flags

print_block_info = False

sensitivity_analysis = False
sensitivity_parameters = [ \
	# comment out parameters not needed in sensitivity analysis
	# General parameters
	'clockspeed',
	# Memory Parameters
	'cache_sizes', 
	#'cache_line_sizes', 
	'cache_cycles',
	'cache_bandwidth_cycles',
	'associativity',
	#'ram_latency',
	#'bw_ram',
	# Pipeline Parameters
	'pipelinecounts',
	'pipelinelatencies',
	'pipelinethroughputs'
]
ppl_ops = [
	# Pipeline Operations types 
	# (note: load and store are controlled through memory parameters)
	'iadd', 
	'fadd', 
	'idiv', 
	'fdiv', 
	'imul', 
	'fmul', 
	#'br', 
	#'alu', 
	#'unknown'
]