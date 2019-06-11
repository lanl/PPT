"""
*********** Performance Prediction Toolkit PPT *********

File: PTXParser.py
Description: This script converts the application's PTX into its corresponding tasklist
Author: Yehia Arafa 
"""

from sys import path
import sys 
path.append('configs/')
import string
from arch_latencies_config import *


# For dependencies 
out_inst_ctr = 0
dependencies = {}

#For error checking
num_of_loops = 0
provided_loops = 0


def Error(type):
    if type == 1: #arguments
        print "\n====================================================="
        print "\tError: Incorrect number of arguments"
        print "=====================================================\n"
        usage()
        exit(1)
    elif type == 2: #ptx file
        print "\n=========================================================="
        print "\tError: PTX file provided doesn't exist "
        print "==========================================================\n"
        usage()
        exit(1)
    elif type == 3: #kernel name
        print "\n=========================================================="
        print "\tError: Kernel function provided doesn't exist "
        print "==========================================================\n"
        usage()
        exit(1)
    elif type == 4: #arch
        print "\n================================================================================================"
        print "\tError: Target GPU Generation doesn't exist in [configs/arch_latencies_config] "
        print "================================================================================================\n"
        usage()
        exit(1)
    elif type == 5: #loop
        print "\n====================================================================================================================="
        print "\tError: There are",num_of_loops,"loops in the code but number of iterations for only",provided_loops,"is provided in the arguments"
        print "=====================================================================================================================\n"
        usage()
        exit(1)
   

def usage():
    print("[USAGE] python PTXParser.py <PTX File> <Target Kernel Name> <Loop Iteration Counts> <Target GPU Generation>")
    print("\t[ARG] PTX File: path to the ptx file")
    print("\t[ARG] Target Kernel Name: name of the kernel function")
    print("\t[ARG] Loop Iteration Counts: comma separated iteration counts for the loops in the kernel (in order in which they are executed)")
    print("\t[ARG] Target GPU Generation: GPU generation (i.e. Kepler, Maxwel) from configs/arch_latencies_config\n")


def check_args():
    NB_ARGS = 5 
    if len(sys.argv) != NB_ARGS:
        Error(1)

def parse_args():
    ptx_file = sys.argv[1]
    kernel_name = sys.argv[2]
    loop_iterations_counts = sys.argv[3].split(",")
    loop_iterations_counts = filter(None, loop_iterations_counts)
    gpu_genertion = sys.argv[4]
    return ptx_file, kernel_name, loop_iterations_counts, gpu_genertion

def clean_up(arg):
    return string.strip(arg, ",;[]%")

def get_kernel_insts(file, kernel_name):
    '''
    Parse the given kernel instructions
    '''
    lines = []
    pattern = ".entry "+kernel_name+"("
    try:
        with open(file, "r") as f:
            status = "out"
            for line in f:
                if status == "out":
                    if pattern in line:
                        status = "found"
                elif status == "found":
                    if "{" in line:
                        status = "in"
                elif status == "in":
                    if "}\n" in line:
                        break
                    else:
                        lines.append(line)
            if not lines:
                Error(3)
    except EnvironmentError:
        Error(2)
    return lines


def get_basic_blocks(insts):
    '''
    Divide the kernel's instructions into its corresponding basic blocks
    '''
    blocks = []
    current_block = {'name':"entry", 'insts':[], 'is_loop':False} 
    for inst in insts:
        if ":" in inst: #to check if we have another block
            blocks.append(current_block)
            name = inst.split(":")[0]
            current_block = {'name':name, 'insts':[], 'is_loop':False}
        elif inst != "\n": #just eliminating the empty line 
            current_block['insts'].append(inst)
    blocks.append(current_block)
    return blocks


def extract_loops(blocks, loop_counts):
    '''
    extract the basic blocks that represent loops
    '''
    global num_of_loops
    structure = []
    index = dict()
    for block in blocks:
        index[block['name']] = block
        lastInst = block['insts'][-1]
        if "bra" in lastInst: #check for branches (loops)
            dest = lastInst.split()[-1].split(";")[0]
            if dest in index: 
                index[dest]['is_loop'] = True
                index[dest]['last_block'] = block['name']
                num_of_loops+=1
    while len(blocks) > 0:
        b = None
        if blocks[0]['is_loop']:
            b, blocks, loop_counts = process_loops(blocks, loop_counts)
        else:
            b = blocks.pop(0)
        structure.append(b)
    return structure


def process_loops(blocks, loop_counts):
    '''
    updates the loop counts for each basic block
    '''
    loop = blocks.pop(0)
    loop['body'] = []
    if not loop_counts:
        Error(5)
    loop['loop_counts'] = loop_counts.pop(0)
    if not loop['last_block'] == loop['name']:
        while len(blocks) > 0:
            b = None
            if blocks[0]['is_loop']:
                b, blocks, loop_counts = process_loops(blocks, loop_counts)
            else:
                b = blocks.pop(0)
            loop['body'].append(b)
            if b['name'] == loop['last_block']:
                break   
    return loop, blocks, loop_counts


def get_tasklist(structure, alu_latencies):
    task_list = []
    for blocks in structure:
        task_list += get_block_tasklist(blocks, alu_latencies)
    return task_list


def get_block_tasklist(block, alu_latencies):
    block_tl = []
    if block['is_loop']:
        count = int(block['loop_counts'])
        for i in xrange(count):
            block_tl += get_subtasklist(block['insts'], alu_latencies)
            block_tl += get_tasklist(block['body'], alu_latencies)
    else:
        block_tl += get_subtasklist(block['insts'], alu_latencies)
    return block_tl


def get_subtasklist(insts, alu_latencies):
    global out_inst_ctr
    sub_tl = []
    task = None 
    for line in insts:
        parts = string.split(line)
        if len(parts)>=2:
            inst = parts[0]
            args = parts[1:]
            for i in xrange(len(args)):
                args[i] = clean_up(args[i])
            task = process_inst(inst, args, alu_latencies)
            if task:
                sub_tl.append(task)
                out_inst_ctr+=1
    return sub_tl


def process_inst(inst, args, alu_latencies):
    global dependencies
    parts = string.split(inst, ".")
    opcode = parts[0]
    types = None
    task = None
    if len(parts)>1:
        types = parts[1:]
    if opcode == "ld" or\
         opcode == "ldu" or\
         opcode == "st" or\
         opcode == "prefetch" or\
         opcode == "prefetchu" or\
         opcode == "isspacep":
        task = generate_mem_access(opcode, types, args, isGlobal=False)
    elif opcode == "suld" or\
         opcode == "sust" or\
         opcode == "sured" or\
         opcode == "suq":
        task = generate_mem_access(opcode, types, args, isGlobal=True)
    elif opcode == "tex" or\
         opcode == "tld4" or\
         opcode == "txq":
        task = generate_tex_access(opcode, types, args)
    elif opcode == "bar" or\
         opcode == "membar" or\
         opcode == "atom" or\
         opcode == "red" or\
         opcode == "vote":
        task = generate_sync(opcode, types, args)
    elif opcode == "mov" or opcode == "shfl":
        task = generate_data_mvmnt(opcode, types, args, alu_latencies)
        if args[1] in dependencies:
            dependencies[args[0]] = dependencies[args[1]] 
    elif opcode == "cvta" or\
         opcode == "cvt" or\
         opcode == "selp" or\
         opcode == "setp":
        task = generate_data_mvmnt(opcode, types, args, alu_latencies)
    elif opcode == "rcp" or\
         opcode == "sqrt" or\
         opcode == "rsqrt" or\
         opcode == "sin" or\
         opcode == "cos" or\
         opcode == "lg2" or\
         opcode == "ex2" or\
         opcode == "copysign":
        task = generate_alu_access(opcode, types, args, alu_latencies, isSFU=True)    
    elif opcode and\
         opcode[0]!="@" and\
         opcode[0]!="." and\
         opcode!="//":
        task = generate_alu_access(opcode, types, args, alu_latencies, isSFU=False)
    return task
   

def generate_alu_access(opcode, types, args, alu_latencies, isSFU):
    global out_inst_ctr
    global dependencies
    if isSFU == True:
        task = ["SFU"]
        if opcode == "sqrt" and types[0] == "approx":
            opcode = "fasqrt"
        if opcode in alu_latencies.keys():
            latency = alu_latencies[opcode]
        else:
            return
        task.append(latency)
    else:
        type = types[len(types)-1]
        offset = types[len(types)-2]
        task = [""]
        if type[-2:] == "64": 
            task[0] = "dALU"
            if opcode == "mul" and offset == "hi":
                opcode_new= "dmulhi" 
            else:
                opcode_new = "d"+opcode
            if opcode_new in alu_latencies.keys():
                latency = alu_latencies[opcode_new]
            elif opcode in alu_latencies.keys():
                latency = alu_latencies[opcode]
            else:
                return
            task.append(latency)
        else:
            task[0] = ""
            if type[0] == "f":
                task[0]+="fALU"
                if type[-2:] == "16":
                    opcode_new = "hf" + opcode
                    if opcode_new in alu_latencies.keys():
                        latency = alu_latencies[opcode_new]
                    else:
                        return
                else:
                    opcode_new = "f" + opcode
                    if opcode_new in alu_latencies.keys():
                        latency = alu_latencies[opcode_new]
                    else:
                        return
                task.append(latency)
            else:
                task[0]+="iALU"
                if opcode=="div" or opcode =="rem":
                    opcode = str(type[0])+opcode
                elif opcode == "mul" and offset == "hi":
                    opcode = "mulhi" 
                if opcode in alu_latencies.keys():
                    latency = alu_latencies[opcode]
                else:
                    return
                task.append(latency)
    for i in range(1,len(args)): # Check for dependencies
        if args[i] in dependencies:
            task.append(dependencies[args[i]])
    dependencies[args[0]] = out_inst_ctr # Add dependency
    return task       
    

def generate_mem_access(opcode, types, args, isGlobal):
    global out_inst_ctr
    global dependencies
    type = types[0]
    task = None
    if isGlobal:
        task = ["GLOB_MEM_ACCESS"]
        task.append(opcode)
    else:
        if type == 'volatile':
            type = types[1]
        if type == 'global':
            task = ["GLOB_MEM_ACCESS"]
        elif type == 'param':
            task = ["PARAM_MEM_ACCESS"]
        if type == 'local':
            task = ["LOCAL_MEM_ACCESS"]
        elif type == 'shared':
            task = ["SHARED_MEM_ACCESS"]
        elif type == 'const':
            task = ["CONST_MEM_ACCESS"]
        if opcode == 'ld' or opcode == 'ldu':
            task.append('LOAD')
        elif opcode == "st":
            task.append("STORE")
        else:
            task.append(opcode)
    for i in range(1,len(args)): # Check for dependency
        if args[i] in dependencies:
            task.append(dependencies[args[i]]) # Add dependency
    dependencies[args[0]] = out_inst_ctr 
    return task


def generate_tex_access(opcode, types, args):
    global out_inst_ctr
    global dependencies
    task = ["TEX_MEM_ACCESS"]
    task.append(opcode)
    for i in range(1,len(args)): # Check for dependencies
      if args[i] in dependencies:
        task.append(dependencies[args[i]])
    dependencies[args[0]] = out_inst_ctr # Add dependency
    return task


def generate_data_mvmnt(opcode, types, args, alu_latencies):
    global dependencies
    task = [""]
    task[0]+="iALU"
    task.append(alu_latencies[opcode])
    return task  


def generate_sync(opcode, types, args):
    return ["THREAD_SYNC"]

if __name__ == "__main__":
    check_args()
    ptx_file, kernel_name, loop_iterations_counts, gpu_genertion = parse_args()
    provided_loops = len(loop_iterations_counts)
    kernel_instructions = get_kernel_insts(ptx_file, kernel_name)
    try:
       alu_latencies = get_alu_latencies(getattr(sys.modules[__name__], gpu_genertion))
    except AttributeError:
        Error(4)
    blocks = get_basic_blocks(kernel_instructions)
    structure = extract_loops(blocks, loop_iterations_counts)
    tasklist = get_tasklist(structure, alu_latencies)
    print tasklist