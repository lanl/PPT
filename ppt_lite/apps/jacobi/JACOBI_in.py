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
    #"silvermont", 
    "skylake"
]

input_params = [# Order matters, that is why it is a list and not a dictionary
    [ 100000000]      # Nx = Number of entries in a row or column of a matrix (matrix is square) 
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
