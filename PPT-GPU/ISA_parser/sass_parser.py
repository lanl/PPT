##############################################################################
# &copy 2017. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.

# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.

# Recall that this copyright notice must be accompanied by the appropriate open source license terms and conditions. Additionally, it is prudent to include a statement of which license is being used with the copyright notice. For example, the text below could also be included in the copyright notice file: This is open source software; you can redistribute it and/or modify it under the terms of the Performance Prediction Toolkit (PPT) License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the Performance Prediction Toolkit (PPT) License can be found in the License file in the main development branch of the repository.
##############################################################################

# Author: Yehia Arafa
# Last Update Date: April, 2021
# Copyright: Open source, must acknowledge original author

##############################################################################


def parse(units_latency, sass_instructions, sass_path, num_warps):
    
    sass_trace = open(sass_path,'r').read().strip().split('\n')
    task_list = {}
    depenedency_map = {}
    warp_inst_count = {}
    count_gmem_reqs = 0

    for inst in sass_trace:
        inst_list = []

        splitted_inst = inst.split(" ")
        warp_id = int(splitted_inst[0])
        current_inst = splitted_inst[1]
        opcodeAndOption = current_inst.split(".")
        opcode = opcodeAndOption[0]
        opcodeAndOption.pop(0)
        
        isOpcodeST = False
        if warp_id in warp_inst_count:
            warp_inst_count[warp_id] += 1
        else:
            warp_inst_count[warp_id] = 0

        #(1) type of inst
        if "LDG" in opcode:
            inst_list.append("GLOB_MEM_ACCESS")
            inst_list.append("LD")
            count_gmem_reqs += 1
        elif "STG" in opcode:
            isOpcodeST = True
            inst_list.append("GLOB_MEM_ACCESS")
            inst_list.append("ST")
            count_gmem_reqs += 1
        elif "LDL" in opcode:
            inst_list.append("LOCAL_MEM_ACCESS")
            inst_list.append("LD")
        elif "STL" in opcode:
            isOpcodeST = True
            inst_list.append("LOCAL_MEM_ACCESS")
            inst_list.append("ST")
        elif "LDS" in opcode:
            inst_list.append("SHARED_MEM_ACCESS")
            inst_list.append("LD")
        elif "STS" in opcode:
            isOpcodeST = True
            inst_list.append("SHARED_MEM_ACCESS")
            inst_list.append("ST")
        elif "LDC" in opcode:
            inst_list.append("CONST_MEM_ACCESS")
            inst_list.append("LD")
        elif "STC" in opcode:
            isOpcodeST = True
            inst_list.append("CONST_MEM_ACCESS")
            inst_list.append("ST")
        elif "ATOM" in opcode or "RED" in opcode:
            inst_list.append("ATOMIC_OP")
            inst_list.append("") #for now just put an empty holder; need to be changed to the type of atomic operation later
        elif "BAR" in opcode:
            inst_list.append("BarrierSYNC")
        elif "MEMBAR" in opcode:
            inst_list.append("MEMBAR")
        else:
            try:
                if "MUFU" in opcode:
                    unit = "SFU"
                    if "64" in opcodeAndOption:
                        unit_64 = "dSFU"
                        latency = units_latency[unit_64]
                    else:
                        latency = units_latency[unit]
                else:
                    unit = sass_instructions[opcode]
                    latency = units_latency[unit]
            except:
                print("\n[Error]\n"+"\""+current_inst+"\""+" is not available in SASS instructions table")
                exit(0)
            
            # add the instruction HW unit to the warp tasklist
            inst_list.append(unit)
            # add the instruction latency to the warp tasklist
            inst_list.append(latency)
        
        
        #(2) add current instruction dependencies
        destination = None
        if warp_id not in depenedency_map:
            depenedency_map[warp_id] = {}

        for i in range(2, len(splitted_inst)):
            if i == 2 and not isOpcodeST:
                destination = splitted_inst[i]
            else:
                source = splitted_inst[i]
                if source in depenedency_map[warp_id]:
                    inst_list.append(depenedency_map[warp_id][source])


        if destination is not None:
            depenedency_map[warp_id][destination] = warp_inst_count[warp_id]
                


        #(3) commit the instruction list to the tasklist
        if warp_id not in task_list:      
            task_list[warp_id] = []
        task_list[warp_id].append(inst_list)

    return task_list, count_gmem_reqs