##############################################################################
# This configuration file models NVIDIA Turing TITAN RTX GPU

# * GPU Microarchitecture adopted from:
# - https://www.techpowerup.com/gpu-specs/titan-rtx.c3311
# - https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
# - https://arxiv.org/pdf/1903.07486.pdf
# - https://developer.nvidia.com/cuda-gpus

##############################################################################

uarch = {

    "gpu_name"                         :  "TITAN RTX",
    "gpu_arch"                         :  "Turing", #This name has to match one of the files in ISA module
    
    # compute capabilty defines the physical limits of GPUs 
    # options available:
    #   - Kepler: 35, 37
    #   - Maxwell: 50, 52, 53
    #   - Pascal: 60, 61
    #   - Volta: 70 
    #   - Turing: 75
    "compute_capabilty"                 :  75,
    
    # base GPU clock speed in HZ                
    "clockspeed"                        :  1350 * 10**6, 

    # streaming multiprocessors (SMs)
    "num_SMs"                           :  72,
    # represents [INT] units; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # responsible for int instructions
    "num_INT_units_per_SM"              :  64,
    # represents [FP32] units 
    # responsible for Single-Precision floating point instructions 
    "num_SP_units_per_SM"               :  64,
    # represents [FP64] units in volta & Turing
    # responsible for Double-Precision floating point instructions
    "num_DP_units_per_SM"               :  2,
    # special function unites per SM
    # responsible for transcendental instructions  
    "num_SF_units_per_SM"               :  16,
    # tensor core units per SM               
    "num_TC_units_per_SM"               :  4,
    # load & store units per SM
    "num_LDS_units_per_SM"              :  32,
    # branch units per SM; ** THIS UNIT IS IN VOLTA & TURING ONLY ** 
    # to handle and execute branch instructions             
    "num_BRA_units_per_SM"              :  4,
    # texture units per SM               
    "num_TEX_units_per_SM"              :  4,
    # warp scheduler units per SM
    "num_warp_schedulers_per_SM"        :  4,
    # instructions issued per warp
    "num_inst_dispatch_units_per_SM"    :  1,

    # L1 cache configs can be skipped if this option is True
    "l1_cache_bypassed"                 :  False,
    
    # In Turing, the L1 cache data storage is unified with SMEM data storage
    # for a total of 96KB size for both
    # SMEM size can be: 64KB, 32KB
    # defualt config is 32KB for L1 cache size and 64KB for SMEM
    # ** Sizes are in Byte **
    "l1_cache_size"                     :  32 *1024,
    "l1_cache_line_size"                :  32,                
    "l1_cache_associativity"            :  64,
    "l2_cache_size"                     :  6 * 1024*1024,
    "l2_cache_line_size"                :  64,                 
    "l2_cache_associativity"            :  32,  
    "shared_mem_size"                   :  64 * 1024,    

    # L2 total size is 6 MB, each subpartition is 128 KB. This gives ~ 48 memory parition
    "num_l2_partitions"	                :  48,
    # Turing has HBM which has 24 channels each (128 bits) 16 bytes width
    "num_dram_channels"	                :  24,
    # DRAM theoritical BW, measured through microbenchmarking
    "dram_th_bandwidth"                 :  936 * 10**9, #B/s
    # base GPU DRAM clock speed in HZ                
    "dram_clockspeed"                   :  3500 * 10**6,
    # NOC theoritical BW, measured through microbenchmarking
    "noc_th_bandwidth"                  :  1140 * 10**9, #B/s

    # warp scheduling: to select which warp to execute from the active warp pool 
    # options available:
    #   - LRR: Loosely Round Robin
    #   - GTO: Greedy Then Oldest -- currently not available, TO BE IMPLEMEMNTED --
    #   - TL: Two Level           -- currently not available, TO BE IMPLEMEMNTED --
    "warp_scheduling"                   :  "LRR",
    
}