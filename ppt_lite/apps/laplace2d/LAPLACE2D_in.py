"""
Input example file for LAPLACE2D C mini-app, can be used as template for other applications
"""
# To run: python ppt.py in-example

task_graph = "LAPLACE2D_taskgraph"
reuse_distance_functions = "LAPLACE2D_rd"
basic_block_count_functions = "LAPLACE2D_bbc"
basic_block_ids = "LAPLACE2D_bbId"


hardware_platforms = [ \
    #"KNL",
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
    [4],      # Nx = Number of cells in x direction
    #[4,6,8,10, 2000]    # Ny = Number of cells in y direction
    [ 2000]    # Ny = Number of cells in y direction
]


# Sensitivity Analysis flags

print_block_info = False

sensitivity_analysis = True
sensitivity_parameters = [ \
	# comment out parameters not needed in sensitivity analysis
	# General parameters
	'clockspeed',
	# Memory Parameters
	'cache_sizes', 
	'cache_line_sizes', 
	'cache_cycles',
	'cache_bandwidth_cycles',
	'associativity',
	'ram_latency',
	'bw_ram',
	# Pipeline Parameters
	#'pipelinecounts',
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
