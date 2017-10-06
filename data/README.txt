If you have any questions, please let me know. 

EJ Park
ejpark@lanl.gov

--------------------------------------------------
Data: 

1. data/hw_specs
A descrition of each machine containing OS, CPU info, and memory info. They are coming from:
> uname -a
> cat /proc/cpuinfo
> getconf -a


2. data/app_runs
 
The first column in Each file is composed of the program name and the input configuration.
For example, 2mm_N1024 means the program is 2mm and the input configuration used is 1024. 

* polybench_byfl_<date in year-month-date format>.csv
  They are dynamic architecture-independent input-dependent features coming from ByFl.

  - name: 	 	program name with the input configuration used 
  - integer_ops: 	number of integer operations 
  - flops: 	 	number of float operations
  - FAdd: 	 	number of float ADD operations.
  - FMul: 	 	number of float MUL operations. 
  - memory_ops:  	number of memory operations (loads + stores)
  - loads:	 	number of load operations
  - stores:	 	number of store operations
  - branch_ops:	 	number of branch operations (unconditional branches + conditional branches)
  - uncond_branch_ops: 	number of unconditional branch operations
  - cond_branch_ops:   	number of conditional branch operations
  - comparison: 	number of comparisons 
  - cpu_ops:		number of getelementptr instructions (LLVM instruction calculating addresses in array)
  - total_ops:		number of total operations
  - vector_ops:		number of vector operations


* polybench_irf_<date in year-month-date format>.csv
  They are static architecture-input-independent features that are collected during compiling a program.

  - name: 	 	program name with the input configuration used 
  - num_loops:		number of loops
  - num_vec_loops:	number of vectorizable loops 
  - num_par_loops:	number of parallelizable loops 
  - num_par_dep_loops:	number of loops that cannot be parallelized because of dependences.
  - num_vec_dep_loops:  number of loops that cannot be vectorized because of dependences.
  - raw_dep:		number of read-after-write dependences
  - war_dep:		number of write-after-read dependences
  - waw_dep:		number of write-after-write dependences
  - num_bbs:		number of basic blocks in control flow graphs
  - num_edges:		number of edges in control flow graphs


* polybench_<machine name>_<date in year-month-date format>.csv
  They are architecture-input-dependent features that are collected during actual runs. 
  For <machine name>, I used partition name (especially for Darwin cluster where we have a collection of different nodes) or cluster name. 

  - name: 	 	program name with the input configuration used 
  - estimated_data_size (eds in bytes): estimated input data size based on number of matrices, matrix size, and data type. 
  - eds/L1_cache:	eds divided by L1 cache size
  - eds/L2_cache:	eds divided by L2 cache size
  - eds/L3_cache:	eds divided by L3 cache size
  - PAPI_L1_DCM:	Level 1 data cache misses
  - PAPI_L2_DCM:	Level 2 data cache misses
  - PAPI_L1_TCM:	Level 1 cache misses
  - PAPI_L2_TCM:	Level 2 cache misses
  - PAPI_TLB_DM:	Data translation lookaside buffer misses
  - PAPI_BR_TKN:	Conditional branch instructions taken
  - PAPI_BR_MSP:	Conditional branch instructions mispredicted
  - PAPI_L2_DCA:	Level 2 data cache accesses
  - PAPI_L2_TCA:	Level 2 total cache accesses
  - Runtime (secs):	Execution time measured in seconds, also averaged over 5 runs.

 
