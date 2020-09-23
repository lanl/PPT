########################################################### 
#merom.py 
''' 
  A Simian resource that represents merom core. It has
  3 cache levels with L3 being a SmartCache. 
''' 
# Units 
ns = 1.0 * 10 ** (-9)  # nanoseconds 
kb = 1024.0  # Kilobytes 
mb = 1024.0 ** 2  # Megabytes 
isize = 4.0  # bytes per word 
fsize = 8.0  # bytes per word 

# Threads and vectors 
maxthreads = 36  # upper bound on number active threads 
clockspeed = ( 
        2.80 * 10 ** 9  # next highest clock 2.60 Highest Turbo Clocks  2.80 2.60 
)  # Hertz (Highest normal frequency, next highest then Turbo Clocks) 
hwthreads = 36  # number of hardware threads 
vector_width = 32  # width of vector unit in bytes, 256 bit? (I don't know this value, how can get this) 

# Registers (we do not need them for reuse distance based model) 
num_registers = 16.0  # number of registers [bytes] 
register_cycles = 1.0  # cycles per register access 

# Cache details (retrieved using 'lscpu') 
cache_levels = 2 
cache_sizes = [32 * kb, 32 * kb]  # bytes 
cache_line_sizes = [64, 64]  # bytes 
associativity = [8, 24] 
num_cache_lines = [ 
        cache_sizes[0] / cache_line_sizes[0], 
        cache_sizes[1] / cache_line_sizes[1], 
]  # #_of_cache_lines = cache_size / cache_line_size 


# Add following line to all hardware units
block_size =  8.0 # in bytes (Data bus width -- 64-bit)


# Operations 
# Each operation takes 1 cycle but can execute more than 1 instruction per 
#   cycle - microarchitecture; e.g., a lot of multiplication, alawys need 
#   to use ALU0, so not 3 muls/cycle. But together AGU/iALU/fALU can do 
#   3 ops/cycle. Need ILP_Efficiency? For now, put a throughput with some 
#   assumptions about not perfect efficiency. 
''' 
cycles_per_CPU_ops = 1.0 
cycles_per_iALU = 0.1 
cycles_per_int_vec = 0.075 
cycles_per_fALU = 0.1 
cycles_per_vector_ops = 0.075 * 2 
cycles_per_division = 1.0 
''' 
cycles_per_CPU_ops = 1 
cycles_per_iALU = 1 
cycles_per_int_vec = 1 
cycles_per_fALU = 1 
cycles_per_vector_ops = 1 
cycles_per_division = 1 

# Following parameters are specific to Reuse based hardware model 

# cache_cycles = [0.0, 0.0, 0.0]  # cycles from www.7-cpu.com where they can be found 
cache_cycles = [0.0, 15.0]  # cycles from microarhitechture.pdf from agner fog 
# normalizing these latencies as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor) 
# cache_cycles = [4/8.0, 10/8.0, 65/8.0]  # cycles (Intel forums for i7-Xeon cpu) 
# cache_cycles = [5, 12, 40]  # cycles (From 7-cpu) 
cache_latency = [ 
       cache_cycles[0] / clockspeed, 
       cache_cycles[1] / clockspeed 
]  # seconds 

# cache_bandwidth_cycles = [0.5/64, (2.3+6.1)/2/64, (5.0+8.4)/2/64] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes) 
cache_bandwidth_cycles = [ 
        4 / 8.0, 
        ((2.2 + 2.3) / 2.0 + 6.1) / 2.0 / 8.0, 
        ((4.7 + 5.0) / 2.0 + 8.4) / 2.0 / 8.0, 
]  # From 7-cpu: cycles (read + write) per cache line (cycles/bytes) 
# However, normalizing these value as per the data bus width (which is 64-bits for 64-bit processor, 32-bits for 32-bit processor) 
# cache_bandwidth_cycles = [0.5, 6.1/64.0, 8.4/64.0] #From 7-cpu: cycles (read + write) per cache line (cycles/bytes) 
# bandwidth = [128/(64.0+32.0), 128/64.0, 128/32] #It is 1/bandwidth (cycle/bytes) values read from the spec sheet 
# ram_bandwidth = 128/16  #which is also 1/bandwidth (cycle/bytes) wrong values 
bw_ram = ( 
        20222.0937 * mb 
)  # mega bytes/sec (calculated from Nandu's measured values) 
bw_ram_miss_penality = 1/(bw_ram) * clockspeed / 8.0 # cycles/byte 
# bw_ram_miss_penality = 1/bw_ram * clockspeed # cycles/byte
 
# Main memory 
ram_page_size = 4096.0  # bytes 
ram_latency = ( 
        10.8 * ns 
)  # Calculated from Nandu's measured values (from Intel MLC tool) 
# ram_latency = 36.0 / clockspeed + 57 * ns #from 7-cpu.com -- (36 cycles + 57 ns) 
# ram_latency = 60 * ns #from Intel forums 
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
        "fadd": 3.000 / clockspeed, 
        "idiv": 55.500 / clockspeed, 
        "fdiv": 22.000 / clockspeed, 
        "imul": 5.000 / clockspeed, 
        "fmul": 5.000 / clockspeed, 
        "load": 2.000 / clockspeed, 
        "store": 3.000 / clockspeed, 
        "alu": 2.000 / clockspeed, 
        "br": 0.100 / clockspeed, 
        "unknown": 1.000 / clockspeed, 
} 
pipelinethroughputs = {  # in seconds 
        "iadd": 0.330 / clockspeed, 
        "fadd": 1.000 / clockspeed, 
        "idiv": 34.000 / clockspeed, 
        "fdiv": 21.000 / clockspeed, 
        "imul": 2.000 / clockspeed, 
        "fmul": 2.000 / clockspeed, 
        "load": 1.000 / clockspeed, 
        "store": 1.000 / clockspeed, 
        "alu": 2.000 / clockspeed, 
        "br": 1.500 / clockspeed, 
        "unknown": 1.000 / clockspeed, 
} 
 
