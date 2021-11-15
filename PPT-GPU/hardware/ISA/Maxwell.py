##############################################################################
# SASS instructions adpoted from:
# - https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html

# Instructions Latencies adopted from: 
# Y. Arafa et al., "Low Overhead Instruction Latency Characterization for NVIDIA GPGPUs," HPEC'19
# https://github.com/NMSU-PEARL/GPUs-ISA-Latencies

##############################################################################

units_latency = {

    #ALU Units Latencies
    "iALU"              :   6,
    "fALU"              :   6,
    "hALU"              :   6,
    "dALU"              :   45,
    "SFU"               :   4,
    "dSFU"              :   8,
    #Memory Units Latencies
    "dram_mem_access"   :   379,
    "l1_cache_access"   :   110,
    "l2_cache_access"   :   250,
    "local_mem_access"  :   379,
    "const_mem_access"  :   20,
    "shared_mem_access" :   35,
    "tex_mem_access"    :   357,
    "tex_cache_access"  :   86,
    "atomic_operation"  :   245,
    #ThreadBlock
    "TB_launch_ovhd"    :   700

}


sass_isa = {

    # Integer Instructions
    "BFE"               : "iALU",
    "BFI"               : "iALU",
    "FLO"               : "iALU",
    "IADD"              : "iALU",
    "IADD3"             : "iALU",
    "ICMP"              : "iALU",
    "IMAD"              : "iALU",
    "IMADSP"            : "iALU",
    "IMNMX"             : "iALU",
    "IMUL"              : "iALU",
    "ISCADD"            : "iALU",
    "ISET"              : "iALU",
    "ISETP"             : "iALU",
    "LEA"               : "iALU",
    "LOP"               : "iALU",
    "LOP3"              : "iALU",
    "POPC"              : "iALU",
    "SHF"               : "iALU",
    "SHL"               : "iALU",
    "SHR"               : "iALU",
    "XMAD"              : "iALU",
    # Single-Precision Floating Instructions
    "FADD"              : "fALU",
    "FCHK"              : "fALU",
    "FCMP"              : "fALU",
    "FFMA"              : "fALU",
    "FMNMX"             : "fALU",
    "FMUL"              : "fALU",
    "FSET"              : "fALU",
    "FSETP"             : "fALU",
    "FSWZADD"           : "fALU",
    # Half-Precision Floating Instructions
    "HADD2"             : "hALU",
    "HFMA2"             : "hALU",
    "HMUL2"             : "hALU",
    "HSET2"             : "hALU",
    "HSETP2"            : "hALU",
    # Double-Precision Floating Instructions
    "DADD"              : "dALU",
    "DFMA"              : "dALU",
    "DMNMX"             : "dALU",
    "DMUL"              : "dALU",
    "DSET"              : "dALU",
    "DSETP"             : "dALU",
    # SFU Special Instructions
    "MUFU"              : "SFU",
    # Conversion Instructions
    "F2F"               : "iALU",
    "F2I"               : "iALU",
    "I2F"               : "iALU",
    "I2I"               : "iALU",
    # Movement Instructions
    "MOV"               : "iALU",
    "PRMT"              : "iALU",
    "SEL"               : "iALU",
    "SHFL"              : "iALU",
    # Predicate Instructions
    "CSET"              : "iALU",
    "CSETP"             : "iALU",
    "PSET"              : "iALU",
    "PSETP"             : "iALU",
    "P2R"               : "iALU",
    "R2P"               : "iALU",
    # Control Instructions
    "BRA"               : "BRA",
    "BRX"               : "BRA",
    "JMP"               : "BRA",
    "JMX"               : "BRA",
    "SSY"               : "BRA",
    "SYNC"              : "BRA",
    "CAL"               : "BRA",
    "JCAL"              : "BRA",
    "PRET"              : "BRA",
    "RET"               : "BRA",
    "BRK"               : "BRA",
    "PBK"               : "BRA",
    "CONT"              : "BRA",
    "PCNT"              : "BRA",
    "EXIT"              : "BRA",
    "PEXIT"             : "BRA",
    "BPT"               : "BRA",
    # Miscellaneous Instructions
    "NOP"               : "iALU",
    "CS2R"              : "iALU",
    "S2R"               : "iALU",
    "B2R"               : "iALU",
    "BAR"               : "iALU",
    "R2B"               : "iALU",
    "VOTE"              : "iALU"
}


ptx_isa = { # ---> (ptx v.72)

    # Integer Instructions
    "add"               : "iALU",
    "sub"               : "iALU",
    "mul"               : ["iALU", 12],
    "mad"               : ["iALU", 12],
    "mul24lo"           : ["iALU", 12],
    "mul24hi"           : ["iALU", 32],
    "mad24lo"           : ["iALU", 12],
    "mad24hi"           : ["iALU", 32],
    "sad"               : "iALU",
    "mad"               : "iALU",
    "div"               : ["iALU", 130],
    "rem"               : ["iALU", 130],
    "abs"               : "iALU",
    "neg"               : "iALU",
    "min"               : "iALU",
    "max"               : "iALU",
    "popc"              : "iALU",
    "clz"               : ["iALU", 6],
    "bfind"             : "iALU",
    "fns"               : "iALU",
    "brev"              : ["iALU", 6],
    "bfe"               : ["iALU", 12],
    "bfi"               : ["iALU", 12],
    "dp4a"              : "iALU",
    "dp2a"              : "iALU",
    "ret"               : "iALU",
    "exit"              : "iALU",
    "bar"               : "iALU",
    # Logic and Shift Instructions
    "and"               : "iALU",
    "or"                : "iALU",
    "not"               : "iALU",
    "xor"               : "iALU",
    "cnot"              : ["iALU", 12],
    "lop3"              : "iALU",
    "shf"               : "iALU",
    "shl"               : "iALU",
    "shr"               : "iALU",
    # Extended-Precision Integer Instructions
    "addc"              : "iALU",
    "sub.cc"            : "iALU",
    "subc"              : "iALU",
    "mad.cc"            : "iALU",
    "madc"              : "iALU",
    # Single-Precision Floating Instructions
    "ftestp"            : "fALU",
    "fcopysign"         : "fALU",
    "fadd"              : "fALU",
    "fsub"              : "fALU",
    "fmul"              : "fALU",
    "ffma"              : "fALU",
    "fmad"              : "fALU",
    "fdiv"              : ["fALU", 450],
    "fabs"              : "fALU",
    "fneg"              : "fALU",
    "fmin"              : "fALU",
    "fmax"              : "fALU",
    "frcp"              : ["SFU", 347],
    "Fastfrcp"          : ["SFU", 90],
    "fsqrt"             : ["SFU", 360],
    "Fastfsqrt"         : ["SFU", 47],
    "frsqrt"            : ["SFU", 360],
    "Fastfrsqrt"        : ["SFU", 47],
    "fsin"              : ["SFU", 6],
    "fcos"              : ["SFU", 6],
    "fex2"              : ["SFU", 16],
    "flg2"              : ["SFU", 16],
    "ftanh"             : ["SFU", 6],
    # Half-Precision Floating Instructions
    "hfadd"             : "hALU",
    "hfsub"             : "hALU", 
    "hfmul"             : "hALU",
    "hfma"              : "hALU",  
    "hneg"              : "hALU",
    "habs"              : "hALU",  
    "hmin"              : "hALU", 
    "hmax"              : "hALU", 
    "htanh"             : "hALU", 
    "hex2"              : "hALU",
    # Double-Precision Floating Instructions
    "dadd"              : "dALU",
    "dsub"              : "dALU",
    "dmul"              : "dALU",
    "dmad"              : "dALU",
    "dfma"              : "dALU",
    "dabs"              : "dALU",
    "dneg"              : "dALU",
    "dmin"              : "dALU",
    "dmax"              : "dALU",
    "dmax"              : "dALU",
    "ddiv"              : ["dALU", 1550],
    "Fastddiv"          : ["dALU", 709],
    "drcp"              : ["dALU", 342],
    "dsqrt"             : ["dALU", 454],
    "drsqrt"            : ["dALU", 454],
    # Conversion & Movement Instructions
    "mov"               : "iALU",
    "shfl"              : "iALU",
    "prmt"              : "iALU",
    "cvta"              : "iALU",
    "cvt"               : "iALU",
    # Comparision & Selection Instructions
    "set"               : "iALU",
    "setp"              : "iALU",
    "selp"              : "iALU",
    # Control Instructions
    "bra"               : "BRA",
    "call"              : "BRA",
    
}