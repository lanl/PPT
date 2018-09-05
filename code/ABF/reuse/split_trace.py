
import sys
import numpy as np
import random
import gc
from operator import itemgetter
from collections import OrderedDict

sample_size = 5 #random number of sample

def usuage():
    print "[USUAGE]: "+__file__+" BasicBlocks_table mem_trace"

def process_split_orig_trace(bb_table,trace_file):
    '''
    Preprocess original trace to include BasicBlock names and their END
    '''
    orig_mem_trace = open(trace_file,'r').read().strip()
    print type(orig_mem_trace)
    #print bb_table
    basic_blocks = []
    for b_t in bb_table:
        b_t_info = b_t.split(': ')
        basic_blocks.append(b_t_info[2])
        
        if b_t_info[1] in orig_mem_trace:
            orig_mem_trace = orig_mem_trace.replace(b_t_info[1],b_t_info[2])            

    prep_orig_mem_trace = orig_mem_trace
    prep_orig_mem_trace = prep_orig_mem_trace.replace('LOAD: ','')
    prep_orig_mem_trace = prep_orig_mem_trace.replace('STORE: ','')

            #orig_mem_trace = orig_mem_trace[:bb_index]+"BasicBlock: END\n"+orig_mem_trace[bb_index:]
    prep_orig_mem_trace = prep_orig_mem_trace.split('\n')
    bb_dict = {}
    for i in range(0, len(prep_orig_mem_trace)):
        for bb in basic_blocks:
            if bb in prep_orig_mem_trace[i]:
                bb_dict[prep_orig_mem_trace[i]]
    
    '''
    o_t_file = open('processed_trace.dat','w')
    for addr in prep_orig_mem_trace:
        print >>o_t_file,addr
    '''
    #return orig_mem_trace

#def main(bb_file,trace_file,i, orig_trace):
def main(bb_file,trace_file):
    bb_table = open(bb_file,'r').read().strip().split('\n') # list of bb table [ex: BasicBlock: 539a6dfb142c2644: (main, entry)]
        
    process_split_orig_trace(bb_table, trace_file)
    

if __name__ == "__main__":
    #if(len(sys.argv) <= 5):   usuage()
    if(len(sys.argv) != 3):   usuage()
    else:    
        #main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4:])
        main(sys.argv[1],sys.argv[2])
