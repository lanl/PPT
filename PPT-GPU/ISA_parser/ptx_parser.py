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


def parse(units_latency, ptx_instructions, ptx_path, num_warps):
    
    ptx_trace = open(ptx_path,'r').read().strip().split('\n')
    # task_list = [[]] * num_warps
    task_list = []
    count_gmem_reqs = 0

    for inst in ptx_trace:
        inst_list = []
        
        splitted_inst = inst.split(" ")
        current_inst = splitted_inst[0]
        opcodeAndOption = current_inst.split(".")
        opcode = opcodeAndOption[0]
        opcodeAndOption.pop(0)

        #(1) add the inst HW unit and latency to the current instruction list 
        if "ld" in opcode:
            if "global" in opcodeAndOption:
                inst_list.append("GLOB_MEM_ACCESS")
                inst_list.append("LD")
                count_gmem_reqs += 1
            elif "local" in opcodeAndOption:
                inst_list.append("LOCAL_MEM_ACCESS")
                inst_list.append("LD")
            elif "shared" in opcodeAndOption:
                inst_list.append("SHARED_MEM_ACCESS")
                inst_list.append("LD")
            elif "const" in opcodeAndOption:
                inst_list.append("CONST_MEM_ACCESS")
                inst_list.append("LD")
            elif "param" in opcodeAndOption:
                inst_list.append("CONST_MEM_ACCESS")
                inst_list.append("LD")
            else:
                print("\n[Warning]\n"+"\"memory unit in: "+current_inst+" is not defined, adding GLOB_MEM_ACCESS")
                inst_list.append("GLOB_MEM_ACCESS")
                inst_list.append("LD")
        elif "st" in opcode:
            if "global" in opcodeAndOption:
                inst_list.append("GLOB_MEM_ACCESS")
                inst_list.append("ST")
                count_gmem_reqs += 1
            elif "local" in opcodeAndOption:
                inst_list.append("LOCAL_MEM_ACCESS")
                inst_list.append("ST")
            elif "shared" in opcodeAndOption:
                inst_list.append("SHARED_MEM_ACCESS")
                inst_list.append("ST")
            elif "const" in opcodeAndOption:
                inst_list.append("CONST_MEM_ACCESS")
                inst_list.append("ST")
            elif "param" in opcodeAndOption:
                inst_list.append("CONST_MEM_ACCESS")
                inst_list.append("ST")
            else:
                print("\n[Warning]\n"+"\"memory unit  in: "+current_inst+" is not defined, adding GLOB_MEM_ACCESS")
                inst_list.append("GLOB_MEM_ACCESS")
                inst_list.append("ST")
        elif "atom" in opcode or "red" in opcode:
            inst_list.append("ATOMIC_OP")
            inst_list.append("")
        elif "barrier" in opcode:
             inst_list.append("BarrierSYNC")
        else:
            try:
                if opcode == "mov" or opcode == "shfl" or opcode == "prmt" or opcode == "cvta" or opcode == "cvt"\
                    or opcode == "set" or opcode == "setp" or opcode == "selp" or opcode == "bra" or opcode == "call"\
                    or opcode == "ret" or opcode == "exit" or opcode == "bar":
                    pass
                else:
                    if "f" in opcodeAndOption[-1]:
                        if "64" in opcodeAndOption[-1]:
                            if opcode == "div" and opcodeAndOption[-2] == "approx":
                                opcode = "Fastddiv"
                            else:
                                opcode = "d" + opcode
                        elif "16" in opcodeAndOption[-1]:
                            opcode = "h" + opcode
                        else:
                            opcode = "f" + opcode
                            if opcode == "rcp" or opcode == "sqrt" or opcode == "rsqrt" and opcodeAndOption[-2] == "approx":
                                opcode = "Fast" + opcode    
                    else:
                        if "sub.cc" in current_inst:
                            opcode = "sub.cc"
                        elif "mad.cc" in current_inst:
                             opcode = "mad.cc"

                unit = ptx_instructions[opcode]
                if type(unit) != list:
                    currentUnit = unit
                    latency = units_latency[unit]
                else:
                    currentUnit = unit[0]
                    latency = unit[1]
                
            except:
                print("\n[Error]\n"+"\""+current_inst+"\""+" is not available in PTX instructions table")
                exit(0)

            # add the instruction HW unit to the tasklist
            inst_list.append(currentUnit)
            
            # add the instruction latency to the tasklist
            inst_list.append(latency)
        

        #(2) add instruction dependencies to the current instruction list  
        splitted_inst.pop(0)
        for dependency in splitted_inst:
            if dependency:
                inst_list.append(int(dependency))
        

        #(3) commit the current instruction list to the tasklist
        task_list.append(inst_list)

    return task_list, count_gmem_reqs