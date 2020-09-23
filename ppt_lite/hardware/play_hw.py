########################################################### 
#play_hw.py
''' 
A hw resource for debugging porposes
''' 
# Units 
ns = 1.0 * 10 ** (-9)  # nanoseconds 
kb = 1024.0  # Kilobytes 
mb = 1024.0 ** 2  # Megabytes 

# Threads and vectors 
clockspeed = ( 
       #4.30 * 10 ** 9 
       1* 10**9
)  #cycles per second

# Cache details (retrieved using 'lscpu') 
cache_levels = 3 
cache_sizes = [1000 * kb, 1000 * kb, 1000 * kb]  # bytes 
cache_line_sizes = [64, 64, 64 ]  # bytes 
associativity = [8, 8, 6] 

cache_cycles = [10, 1, 1]  # cycles from microarhitechture.pdf from agner fog 
 
cache_bandwidth_cycles = [ 1,1,1
#       4 / 8.0, 
#       ((2.2 + 2.3) / 2.0 + 6.1) / 2.0 / 8.0, 
#       ((4.7 + 5.0) / 2.0 + 8.4) / 2.0 / 8.0, 
]



bw_ram = ( 
        2.0937 * mb 
)  # mega bytes/sec (calculated from Nandu's measured values) 
bw_ram_miss_penality = 1/(bw_ram) * clockspeed / 8.0 # cycles/byte 

 
# Main memory 
ram_page_size = 4096.0  # bytes 
ram_latency = ( 
        10000.8 * ns 
)  # Calculated from Nandu's measured values (from Intel MLC tool) 
ram_cycles = ram_latency * clockspeed 


# Pipelines 
pipelinetypes = [ 
        "iadd", 
        "fadd", 
        "idiv", 
        "fdiv", 
        "imul", 
        "fmul", 
        "load", 
        "store", 
        "alu",
        "br", 
        "unknown", 
] 
pipelinecounts = { 
        "iadd": 2, 
        "fadd": 2, 
        "idiv": 2, 
        "fdiv": 2, 
        "imul": 2, 
        "fmul": 2, 
        "load": 2, 
        "store": 2, 
        "alu": 2, 
        "br": 1, 
        "unknown": 1, 
} 
pipelinelatencies = {  # in seconds 
        "iadd": 1.000 / clockspeed, 
        "fadd": 1.000 / clockspeed, 
        "idiv": 1.00 / clockspeed, 
        "fdiv": 1.000 / clockspeed, 
        "imul": 1.000 / clockspeed, 
        "fmul": 1.000 / clockspeed, 
        "load": 1.000 / clockspeed, 
        "store": 1.000 / clockspeed, 
        "alu": 1.000 / clockspeed, 
        "br":13.000 / clockspeed, 
        "unknown": 1.000 / clockspeed, 
} 
pipelinethroughputs = {  # in seconds 
        "iadd": 1.00 / clockspeed, 
        "fadd": 1.000 / clockspeed, 
        "idiv": 1.000 / clockspeed, 
        "fdiv": 1.000 / clockspeed, 
        "imul": 1.000 / clockspeed, 
        "fmul": 1.000 / clockspeed, 
        "load": 1.000 / clockspeed, 
        "store": 1.000 / clockspeed, 
        "alu": 1.00 / clockspeed, 
        "br": 1.000 / clockspeed, 
        "unknown": 1.000 / clockspeed, 
} 
 
