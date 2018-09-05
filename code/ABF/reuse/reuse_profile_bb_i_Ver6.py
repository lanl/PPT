"""
Author          :       Gopinath Chennupati
Last Modified   :       8 Dec 2016
File            :       reuse_profile_bb_i_Ver6.py
Purpose         :       Preprocess the raw trace resulted out of hacked Byfl
                        Compute the reuse distribution of a given BasicBlock
                        Compute p(sd/BBi)
Description     :       Sample (10 or 20 or 25) the window sizes in order to speedup the process
Output          :       Write the sd,p(sd/BBi) into a file
"""

import sys
import cPickle as cp 
import numpy as np
import random
import gc
from operator import itemgetter
from collections import OrderedDict

sample_size = 5 #random number of sample

def usuage():
    print "[USUAGE]: "+__file__+" BasicBlocks_table mem_trace"

def preprocess_orig_trace(bb_table,trace_file):
    '''
    Preprocess original trace to include BasicBlock names and their END
    '''
    print "Step1: Preprocessing original trace to include BasicBlock names and their END"
    orig_mem_trace = open(trace_file,'r').read().strip()    
    #print bb_table
    for b_t in bb_table:
        b_t_info = b_t.split(': ')
        #print b_t_info[0]+":", b_t_info[2]
        #print b_t_info
        if b_t_info[1] in orig_mem_trace:
            #orig_mem_trace = orig_mem_trace.replace('BB:', b_t_info[0]+'--')
            orig_mem_trace = orig_mem_trace.replace(b_t_info[1],b_t_info[2]) 
            #bb_index = orig_mem_trace.index(b_t_info[0]+"-- "+b_t_info[2])
            #orig_mem_trace = orig_mem_trace.replace(b_t_info[0]+"-- "+b_t_info[2],"BasicBlock--END\n"+b_t_info[0]+"--"+b_t_info[2])    
    prep_orig_mem_trace = orig_mem_trace
    prep_orig_mem_trace = prep_orig_mem_trace.replace('LOAD: ','')
    prep_orig_mem_trace = prep_orig_mem_trace.replace('STORE: ','')    
            #orig_mem_trace = orig_mem_trace[:bb_index]+"BasicBlock: END\n"+orig_mem_trace[bb_index:]
    #print prep_orig_mem_trace
    prep_orig_mem_trace = prep_orig_mem_trace.split('\n')
    
    #orig_mem_trace = orig_mem_trace.split('\n')

    #o_t_file = open(trace_file[-4]+'_processed.dat','w')
    o_t_file = open('processed_trace.dat','w')
    for addr in prep_orig_mem_trace:
        print >>o_t_file,addr
    print "Done. Processed trace written to file"
    return prep_orig_mem_trace
    #return orig_mem_trace

def read_prep_trace_from_file(prep_trace_file):
    return open(prep_trace_file,'r').read().strip().split('\n')

def read_trace_dict_from_file(dict_file):
    with open(dict_file, 'r') as f:
        trace_dict = eval(f.read())
    #print trace_dict
    return trace_dict

def get_trace_dict(prep_mem_trace):
    print "Step 2: Generating trace dictionary, this may take some time"
    trace_dict = {}
    last_key = ''
    print len(prep_mem_trace)
    for adx, addr in enumerate(prep_mem_trace):
        #trace[addr] = {}
        #print adx , " " , addr
        if addr not in trace_dict:
            i = 0
            trace_dict[addr] = OrderedDict()
            trace_dict[addr][str(i)] = adx
        else:
            i = int(trace_dict[addr].keys()[-1])
            i = i + 1
            trace_dict[addr][str(i)] = adx
    #print "************** Finished initializing the trace_dict *************"
    ls = []
    ls.append(trace_dict)    
    f = open('trace_dict.dat','w')
    f.write('\n'.join(map(lambda x: str(x), ls)))
    f.close()
    '''
    for t, v in trace.iteritems() :
        print t, v
    '''
    print "Done. Trace dictionary contains all each BB_START, BB_END & memory access's occurences in trace"
    return trace_dict

def get_bbi_windows(bbi_name, trace_dict):
    '''
    Return all the windows of i^th BB
    '''
    bbi_windows = []
    
    bb_start_key = 'BB START: '+bbi_name
    bb_end_key = 'BB DONE: '+bbi_name
    #print len(trace_dict[bb_start_key]), len(trace_dict[bb_end_key])
    if bb_start_key in trace_dict:
        dict1, dict2 = trace_dict[bb_start_key], trace_dict[bb_end_key]
        for d1, d2 in zip(dict1, dict2):
            if dict1[d1]+1 != dict2[d2]:
                bbi_windows.append([dict1[d1], dict2[d2]])
    else: bbi_windows.append([0,0])
    #print bbi_windows
    return bbi_windows

def sample_bbi_wins(bbi_win_sizes, sample_size):
    '''
    Randomly sample 'sample_size' number of bbi_win_sizes
    '''
    sampled_wins = []
    len_bbi_win_sizes = len(bbi_win_sizes)
    if len_bbi_win_sizes > sample_size: 
        indices = list(range(len_bbi_win_sizes))
        sample_indices = random.sample(indices,sample_size)        
        '''
        if not len_bbi_win_sizes-1 in sample_indices: 
            sample_indices[-1] = len_bbi_win_sizes-1
        '''
        sampled_wins = [bbi_win_sizes[s_idx] for s_idx in sample_indices]
            #sampled_wins.append(bbi_win_sizes[s_idx])
    else: sampled_wins = bbi_win_sizes
    return sampled_wins

def get_bbi_reuse_prof_fast(bb_i_size, orig_trace):
    sd = [0,0]
    if bb_i_size[1] > bb_i_size[0]+1:
        bbi_trace = orig_trace[bb_i_size[0]+1:bb_i_size[1]]
        sd = []     
        # Calculate the SDs for each addr in BBi
        for addr, idx in zip(bbi_trace,range(bb_i_size[0]+1,bb_i_size[1])):
            window_trace = orig_trace[:idx]
            dict_sd = {}
            addr_found = False
            for w_adx in range(0,len(window_trace)):
                w_addr = window_trace[-w_adx -1]
                if addr == w_addr:
                    addr_found = True
                    break
                if w_addr[:2] == '0x':
                    dict_sd[w_addr] = True
            if addr_found: sd.append(len(dict_sd))
            else: sd.append(-1)
    sd = np.array(sd)
    #print sd
    return sd

def get_bb_i_trace(bb_i_start,bb_i_end,orig_trace):
    '''
    Return the memory trace of a given BasicBlock
    '''
    return orig_trace[bb_i_start:bb_i_end]

def get_prob_sd_bb_i(sd_vals):
    '''
    Return the unique stack distances (sd) and their probabilities (p(sd/BBi)))
    for a given BasicBlock
    '''
    len_sd = len(sd_vals)
    uniq_sd,counts = np.unique(sd_vals,return_counts=True)
    #Compute probabilities
    p_uniq_sd_bbi = map(lambda x: x/float(len_sd),counts)
    print 'Sum of p(sd/bbi) : ', np.sum(p_uniq_sd_bbi)
    return zip(uniq_sd,p_uniq_sd_bbi)

def get_final_prob_bb_i(reuse_profile_bbi,bbi_prob):
    '''
    Multiply conditional probabilities of a given Basicblock with 'weighted probability'
    '''
    final_reuse_prof = []
    for i in range(0,len(reuse_profile_bbi)):
        reuse_profile_bbi[i][1] = float(bbi_prob) * float(reuse_profile_bbi[i][1])

    return reuse_profile_bbi

def get_all_bbi_reuse_profile(final_reuse_profile):
    '''
    Merge the reuse profile probabilities of duplicate SDs
    '''
    print "Start of final_reuse_profile"
    res = {}
    for item in final_reuse_profile:
    	if item[0] not in res:
    	    res[item[0]] = 0.0
    	res[item[0]] += item[1]
    print "********** Done adding the duplicate probabilities ***********"
    return res

#def main(bb_file,trace_file,i, orig_trace):
def main(bb_file,trace_file):
    
    bb_table = open(bb_file,'r').read().strip().split('\n') # list of bb table [ex: BasicBlock: 539a6dfb142c2644: (main, entry)]
    '''
    for ent in bb_table:
        print ent, "\n"
    '''
    print len(bb_table)," Basic Blocks\n"
    
    #orig_trace = preprocess_orig_trace(bb_table,trace_file) # list of original preprocessed trace
    
    orig_trace = read_prep_trace_from_file(trace_file)
            
    trace_dict = get_trace_dict(orig_trace)    
    #trace_dict = read_prep_trace_from_file('processed_trace.dat')
    
    final_reuse_profile = []
    sum_pbbs = 0
    #for i in range(3,len(bb_table)-6):
    for i in range(0,len(bb_table)):
        #bb_i_size = get_bb_i_window(bb_table[3],bb_table[4],orig_trace)
        print "\nBasicblock Number : ", i
        bb_table_row = bb_table[i]
        b_t_info = bb_table_row.split(': ')        
        sum_pbbs += float(b_t_info[-1])
        print "Basicblock Name: ", b_t_info[2], " Probability: ", b_t_info[-1]
        bbi_win_sizes = get_bbi_windows(b_t_info[2],trace_dict)
        #bbi_win_sizes = get_bb_i_window(bb_table_row,orig_trace) #Returns a list of lists
        print "Number of bb", i, " windows = ", len(bbi_win_sizes)
        #print "All bbi_win_sizes : ", bbi_win_sizes
        bbi_win_sizes = sample_bbi_wins(bbi_win_sizes,sample_size)
        print "Sampled bb", i, " windows : ", bbi_win_sizes
        zipped_sd_psd = []
        res_sds = np.array([])
        #f_i_res_sd_psd = []
        for bb_i_size in bbi_win_sizes:
            res_sds = np.concatenate([res_sds, get_bbi_reuse_prof_fast(bb_i_size,orig_trace)])
        #print res_sds
        zipped_sd_psd_i = get_prob_sd_bb_i(res_sds)
        #print zipped_sd_psd_i
        res_sd_psd_i = [list(sd_psd) for sd_psd in zipped_sd_psd_i]
        #print "Conditional reuse profile of ", b_t_info[0], " is ", res_sd_psd_i
        
        #print "Probability of BasicBlock ",b_t_info[2]," is ", b_t_info[-1]
        
        f_res_sd_psd_i = get_final_prob_bb_i(res_sd_psd_i, b_t_info[-1])
        #print "Weighted probability of BasicBlock", b_t_info[1]," is ",f_res_sd_psd_i
        for f_sd_psd_i in f_res_sd_psd_i:
            final_reuse_profile.append(f_sd_psd_i)
        #final_reuse_profile.append(f_res_sd_psd_i)
        
        #np.savetxt(b_t_info[0]+b_t_info[2]+'_sd_psd.dat',zipped_sd_psd_i,fmt='%i,%s', delimiter=',', newline='\n')

        #print sd_vals
        print "Stack Distance Stats of BasicBlock ", b_t_info[2]
        print "mean %f min %f max %f std %f" %(np.mean(res_sds),np.min(res_sds),np.max(res_sds),np.std(res_sds))
        
    #Get the final reuse profile for the program by merging the probabilities for all the BBi
    all_bbi_sd_profile = get_all_bbi_reuse_profile(final_reuse_profile)
    print "\nSize of the final reuse profile : ", len(all_bbi_sd_profile)
    
    sum_f = sum(value for key,value in all_bbi_sd_profile.iteritems())
    print "Final sum of the reuse profile probabilities", sum_f 
    
    with open(trace_file+"_reuse_profile.dat","w") as f:
        for key,value in all_bbi_sd_profile.iteritems():
            f.write(str(key)+","+str(value)+"\n")
    
    print "Finished preparing the final reuse_profile of the program"
    
    print "Sum of p(BBi) : ", sum_pbbs
    

if __name__ == "__main__":
    #if(len(sys.argv) <= 5):   usuage()
    if(len(sys.argv) != 3):   usuage()
    else:    
        #main(sys.argv[1],sys.argv[2],sys.argv[3], sys.argv[4:])
        main(sys.argv[1],sys.argv[2])
