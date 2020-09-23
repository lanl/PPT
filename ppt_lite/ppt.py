#!/usr/bin/env python3
"""
 ppt.py: The main ppt-lite file that takes as input:
 (i)    input file
 which contains the following elements
 (ii)   task_graph (filename)
 (iii)  reuse_distance_functions (filename of reuse distance functions)
 (iv)   basic_block_count_functions (filename of basic block count functions)
 (v)    hardware (a list of hardware platforms that this should simulate on, imported)
 (vi)   input_params (a list of values for each of the input parameters)

 ppt.py imports the files corresponding to these dictionary entries and then runs a
 fully factorial experiment of all combinations of input_params and hardware.

 Output to screen as a list of (input param values, hardware platform, predicted runtime)
"""

# To run: python ppt.py input_params.py
# Output: To screen and to file ?

from __future__ import print_function
import sys
import itertools as it
import math
import importlib
from simian import Simian, Entity

###############################################################################
# Add paths where to look for auxillary (eg: hardware, app etc) models
###############################################################################
sys.path.append('./')
sys.path.append('../../hardware/')

###############################################################################
# Helper functions for computing the reuse distance functions
###############################################################################
def ncr(n, m):
    """
    n choose m
    """
    if(m>n): return 0
    r = 1
    for j in range(1,m+1): r *= (n-m+j)/float(j)
    return r

def phit_D(D, A, cs, ls):
    """
    Calculate the probability of hit (given stack distance, D) for a give cache level
    Output: Gives the probability of a hit given D -- P(h/D)
    """
    phit_D = 0.0 #To compute probability of a hit given D
    B = (1.0 * cs)/ls  # B = Block size (cache_size/line_size)

    if (D <= A):
        if (D == -1):   D = hw.cache_sizes[2]
        elif (D == 0):  phit_D = 1.0
        else:    phit_D = math.pow((1 - (1/B)), D)
    # Don't be too creative to change the follow condition to just 'else:'
    # I am changing the value of D in the previous condition.
    if(D > A):
      for a in range(int(A)):
        term_1 = ncr(D,a)
        #term_1 = math.gamma(D + 1) / (1.0 * math.gamma(D - a + 1) * math.gamma(a + 1))
        term_2 = math.pow((A/B), a)
        term_3 = math.pow((1 - (A/B)), (D - a))
        phit_D += (term_1 * term_2 * term_3)

    return phit_D

def phit_sd(stack_dist, assoc, c_size, l_size):
  """
  Calculate probability of hits for all the stack distances
  """
  phit_sd = [phit_D(d, assoc, c_size, l_size) for d in stack_dist]
  return phit_sd

def phit(Pd, Phd):
    """
    Calculate probability of hit (given P(D), P(h/D) for a given cache level
    Output: Gives the probability of a hit -- P(h) = P(D)*P(h/D)
    """
    phit = 0.0 #Sum (probability of stack distance * probability of hit given D)
    Ph = map(lambda pd,phd:pd*phd,Pd,Phd)
    phit = sum(Ph)
    #print("in phit, Pd, Phd, phit:", Pd, Phd, phit)
    return phit

def effective_cycles(phits, cycles, penalty):
    """
    Calculate effective clock cycles for the given arguments
    """
    eff_clocks = penalty # TODO: double check accuracy, could be index error here
    #print("in effective_cycles with phits, cycles, penalty: ", phits, cycles, penalty)
    for l in range(len(phits)):
        i = len(phits) - l -1
        eff_clocks = eff_clocks * (1.0-phits[i]) + cycles[i]*phits[i]
        #print("eff_clocks, i, l: ", eff_clocks, i, l)
    return eff_clocks



################################################################################
# DropIn entity with processses for pipelines, actual PDES funcationlity is here
################################################################################

class dropCore(Entity):
    def __init__(self, baseInfo, taskgraph, *args):
        super(dropCore, self).__init__(baseInfo)
        # Create and start a process for each pipeline
        self.q, self.pipe_active = {}, {}
        #print("DropCore Entity created: ", baseInfo)
        for pipeline_type in hw.pipelinecounts:
            self.q[pipeline_type], self.pipe_active[pipeline_type] = {}, {}
            for i in range(math.ceil(hw.pipelinecounts[pipeline_type])):
                self.q[pipeline_type][i] = [] # q for this pipeline
                self.pipe_active[pipeline_type][i] = False
                self.createProcess(pipeline_type+str(i), pipeline_process)
                #print("Pipeline ", pipeline_type, i, " created with throughput: ", hw.pipelinethroughputs[pipeline_type])
                self.startProcess(pipeline_type+str(i), pipeline_type, i, \
                    hw.pipelinelatencies[pipeline_type], \
                    hw.pipelinethroughputs[pipeline_type])

        # Create master process, e.g, the drop in process
        self.done_Q = []
        self.createProcess("dropin_process", dropin_process)
        self.startProcess("dropin_process", taskgraph)

###########################################################
def dropin_process(this, taskgraph):
    """
    A process that executes the taskgraph in the dropin engine
    """
    # Taskgraph is a list of instruction vertices each being a dictionary
    # with entries  'children', 'instr' with dictionary entries lists

    # Initialization
    #print this
    core = this.entity
    g = taskgraph
    #print g, len(g)
    # Iterate over each item (an item is a task-graphlet/BB-graph) in the list
    # For each item gt the key values, key is the vertex and values are
    '''
    for item in g:
        for vert, value in item.iteritems():
            print vert, value['children']
    '''

    v_ready = [] # List of vertices ready to be executed
    num_outstanding = len(g) # number of nodes still to be processed
    #print(core.engine.now, ": dropinprocess before while, num_outstanding: ", num_outstanding)
    core.active = True
    # Count number of parents for each vertex
    for v in g:
        g[v]['num_parents_outstanding'] = 0
        #v['num_parents_outstanding'] = 0
    for v, val in g.items():
    #for v in g:
        #print 'Children at each vertex : ', val['children']
        for w in val['children']:
        #for w in v['children']:
            g[w]['num_parents_outstanding'] += 1
            #print g[w]['num_parents_outstanding']
    for v in g:
        g[v]['done'] = False
        g[v]['in_pipeline'] = False
        if g[v]['num_parents_outstanding'] == 0:
            v_ready.append(v)
    # Main loop
    while num_outstanding > 0: # Not all instruction vertex completed yet
        # 1: Check for new messages
        #print(v_ready, core.done_Q)
        while core.done_Q: # A pipeline has completed an instruction
            v_done = core.done_Q.pop()
            num_outstanding -= 1
            # 'v_done : ', v_done, num_outstanding, core.done_Q
            g[v_done]['time_done'] = core.engine.now
            for w in g[v_done]['children']:
                g[w]['num_parents_outstanding'] -= 1
                if g[w]['num_parents_outstanding'] == 0:
                    v_ready.append(w)
        # 2: Get vertices without outstanding parents sent to pipelines
        for v in v_ready: #
            instr_type = g[v]['inst']
            #print instr_type
            if not instr_type in hw.pipelinetypes: # "unknown instruction"
                instr_type = 'unknown'
            n_pipes = math.ceil(hw.pipelinecounts[instr_type])
            dispatched = False
            for i in range(n_pipes): # send to first sleeping queue
                if not core.pipe_active[instr_type][i] and not dispatched:
                    dispatched = True
                    #print(core.engine.now, ": drop assigned ", v, " to sleeping pipeline ", instr_type, i )
                    core.q[instr_type][i].insert(0, v)
                    core.wakeProcess(instr_type+str(i))
            if not dispatched: # all queues are active, insert in shortest one
                min_q_length, min_i = 1000000, 0 # what is MAXINT in Python?
                for i in range(n_pipes):
                    if len(core.q[instr_type][i]) < min_q_length:
                        min_q_length = len(core.q[instr_type][i])
                        min_i = i
                #print(instr_type, min_i)
                core.q[instr_type][min_i].insert(0, v)
                #print(core.engine.now, ": drop assigned ", v, " to active pipeline ", instr_type, i )
        #print 'V_ready : ',v_ready
        v_ready = []
        core.active = False
        g["totalBBTime"] = core.engine.now
        this.hibernate()
        core.active = True
        #print(core.engine.now, ": dropinprocess end of while, num_outstanding: ", num_outstanding)
    print('End of loop Main',g)
    # We are done, let's report the time this took
    # We do this by appending an element to the task graph list at the end.

###########################################################
def pipeline_process(this, type, id, latency, throughput_time):
    """
    The pipeline process mimicking a single, pipelined, hadware resource, such as an ADD pipeline
    """
    core = this.entity
    in_q = core.q[type][id] # incoming instructions are in this queue
    depth = int(latency/float(throughput_time))
    if depth < 1: depth = 1 # necesssary for rest of code to work
    #print(core.engine.now, " Pipeline Process ", type, id, " at beginning with latency, throughput_time, depth ", latency, throughput_time, depth)

    #print core.engine.now, " DropIn Pipeline Process ", id, " with depth ", depth, " started"
    pipe = {} # the actual pipeline that processes instructions
              # instructions enter at pipe[depth] and work their way to
              # pipe[0] position, then get put back into dropIn proc queue
    for i in range(depth):
        pipe[i] = None
    i = 0 # counter
    last = 0
    core.pipe_active[type][id] = False
    #print(core.engine.now, " Pipeline Process ", type, id, " before initial hibernate ", pipe)
    this.hibernate() # wait to be woken up with an instruction to process
    #print(core.engine.now, " Pipeline Process ", type, id, " after initial hibernate ", pipe)
    core.pipe_active[type][id] = True
    while core.pipe_active[type][id]:
        #print("Pipeline Process ", type, id, "before sleep with throughput_time ",core.engine.now,  throughput_time, pipe)
        this.sleep(throughput_time)
        #print("Pipeline Process ", type, id, " after sleep with throughput_time: ",core.engine.now,  throughput_time, pipe)
        if pipe[0] != None: # an instruction is done
            core.done_Q.append(pipe[0])
            if not core.active: # main process is hibernating, so let's wake it up
                core.wakeProcess('dropin_process')
            #print(core.engine.now, " DropIn Pipeline Process ", type, id, " completed instr ", pipe[0])
        for i in range(depth-1):  # advance the pipeline
            pipe[i] = pipe[i+1]
        if in_q: # another instruction in the q
            pipe[depth-1] = in_q.pop()
            last = 0
        else:
            pipe[depth-1] = None
            last += 1
            if last == depth: # the entire pipeline is empty
                core.pipe_active[type][id] = False
                this.hibernate() # sleep until woken up again
                core.pipe_active[type][id] = True


###############
ss = 0   # global counter for sensitivty_parameters list
ssl = 0  # global counter for element within list (or dict)
plusminusbase = 0 # 0 = base, 1= plus, 2=minus

delta = 0.01
plus_factor = 1 + delta
minus_factor = 1/plus_factor * (1- delta)
base_factor = 1/(1-delta)

def changeParameter(hw): # change next parameter 
    done, call_self = False, False
    global ss
    global ssl
    global plusminusbase
    base_param_value = 0
    params = in_file.sensitivity_parameters
    #print("changeParameter called with s, sl:", ss, ssl)
    p = getattr(hw, params[ss])
    if isinstance(p, list):
        #print(p, " is a list")
        plusminusbase = (plusminusbase +1) % 3
        para_name = params[ss]+'['+str(ssl)+']'
        if plusminusbase == 1: 
            p[ssl] = p[ssl] * plus_factor
            #print("Check if getattr copy worked:", getattr(hw, params[ss]), " = ", p)
        elif plusminusbase == 2: 
            p[ssl] = p[ssl] * minus_factor
            #print("Check if getattr copy worked:", getattr(hw, params[ss]), " = ", p)
        elif plusminusbase == 0:
            p[ssl] = p[ssl] * base_factor # reset to orig value
            #print("Check if getattr copy worked:", getattr(hw, params[ss]), " = ", p)
            call_self = True
            if len(p)>ssl+1: #continue through list
                ssl += 1
            else: # done with list, move to next sensitivity_parameter
                ssl = 0
                ss += 1
                if ss>=len(params): 
                    done = True
        base_param_value = p[ssl]
    elif isinstance(p, dict):
        #print(p, " is a dict with params[ss],ss, plusminusbase:", params[ss],ss, plusminusbase)
        dl = sorted(list(p.keys()))
        para_name = params[ss]+'['+dl[ssl]+']'
        if dl[ssl] in in_file.ppl_ops: # the pipeline key is listed in the parameters
            plusminusbase = (plusminusbase +1) % 3
            if plusminusbase == 1: 
                p[dl[ssl]] = p[dl[ssl]] * plus_factor
                #print("Check if getattr copy worked: ", getattr(hw, params[ss]), " = ", p)
            elif plusminusbase == 2: 
                p[dl[ssl]] = p[dl[ssl]]* minus_factor
                #print("Check if getattr copy worked: ", getattr(hw, params[ss]), " = ", p)
            elif plusminusbase == 0:
                p[dl[ssl]] = p[dl[ssl]] * base_factor # reset to orig value
                #print("Check if getattr copy worked:", getattr(hw, params[ss]), " = ", p)
                call_self = True
                if len(p)>ssl+1: #continue through list
                    ssl += 1
                else: # done with list, move to next sensitivity_parameter
                    ssl = 0
                    ss += 1
                    if ss>=len(params): 
                        done = True
            base_param_value = p[dl[ssl]]
        else: # pipeline op not listed, just skip, but adjust s and sl
            plusminusbase = 0
            if len(p)>ssl+1: #continue through list
                ssl += 1
            else: # done with list, move to next sensitivity_parameter
                ssl = 0
                ss += 1
                if ss>=len(params): 
                    done = True
            call_self = True
    else: # its a single value
        #print(p, " is a single value and plusminusbase = ", plusminusbase)
        plusminusbase = (plusminusbase +1) % 3
        if plusminusbase == 1: 
            p = p * plus_factor
            if params[ss] == 'clockspeed':
                # hack to reset those values
                for item in hw.pipelinelatencies:
                    hw.pipelinelatencies[item] /= plus_factor
                    hw.pipelinethroughputs[item] /= plus_factor
        elif plusminusbase == 2: 
            p = p * minus_factor
            if params[ss] == 'clockspeed':
                # hack to reset those values
                for item in hw.pipelinelatencies:
                    hw.pipelinelatencies[item] /= minus_factor
                    hw.pipelinethroughputs[item] /= minus_factor
        if plusminusbase == 0:
            p = p * base_factor # reset to orig value
            if params[ss] == 'clockspeed':
                # hack to reset those values
                for item in hw.pipelinelatencies:
                    hw.pipelinelatencies[item] /= base_factor
                    hw.pipelinethroughputs[item] /= base_factor
            #print("Just reset to basefactor ", base_factor, p)
        setattr(hw,params[ss],p)
        para_name = params[ss]
        #print("Check if getattr copy worked:", getattr(hw, params[ss]), " = ", p)
        if plusminusbase == 0:
            call_self = True
            ss += 1
            if ss>=len(params): 
                done = True
        base_param_value = p
    plus = False
    if plusminusbase == 1: 
        plus = True
        base_param_value = base_param_value/plus_factor
    elif plusminusbase == 2:
        base_param_value = base_param_value*base_factor
    #print("... returning ", done, para_name, plus, call_self)
    if done:
        ss, ssl = 0, 0 
    return done, para_name, plus, call_self, base_param_value

####################
######  MAIN  ######
####################
######
# 0. Initializations
######
# read in/import input_params, taskgraph, reuse_distance_functions, bbc functions, hardware parameters

in_file = importlib.import_module(str(sys.argv[1])) #application input file
tg = importlib.import_module(in_file.task_graph)
rd =  importlib.import_module(in_file.reuse_distance_functions)
bbcf =  importlib.import_module(in_file.basic_block_count_functions)
bbid =  importlib.import_module(in_file.basic_block_ids) #NOTE: Nandu: Mapping to take from BB string-label to numerical-id

params = in_file.input_params    # List of list of values for input parameters, order matters
hw_list = in_file.hardware_platforms     # List of hardware platforms to be simulated
task_graph = tg.task_graph
bb_id_map = bbid.bb_id

######
#  1.Loop over all factorial design cases
######
runs = list(it.product(*params)) # Creates all combinations of input parameters
predicted_times, sensitivities = {}, {}

print("\n")
#print("runs data structure:", runs)
j=0
sens_done, init_sens_run = False, True
while j < len(hw_list):
    hw_name = hw_list[j]
    hw = importlib.import_module(hw_list[j])
    print("Simulating for hardware platform ", hw_name)
    #predicted_times[hw_name] = {}
    i = 0
    while i < len(runs):
        print("Running input parameters (%i/%i): %s" %(i,len(runs), runs[i]))
        #print(hw.clockspeed)
        ######
        # 2. Calculate reuse distances
        ######
        reuse_distances = rd.get_RDcountsArray(runs[i])
        ######
        # 3. Calculate memory access time
        ######
        sd = reuse_distances.keys() #stack distance TODO: fix rights
        psd_raw = reuse_distances.values()  #probability p(sd) TODO: fix
        # normalize reuse distances
        total = sum(psd_raw)
        psd = []
        for t in psd_raw: psd.append(t/total)
        phits = {}
        for l in range(hw.cache_levels):
            phits_d = phit_sd(sd, hw.associativity[l], hw.cache_sizes[l], hw.cache_line_sizes[l])
            #print("phits_sd after call (sd, phits_sd:", sd, phits_d)
            phits[l] = phit(psd, phits_d)
            #print("phit after call", phits)
       
        #Reset a few values here just to make sure we have the right values in the sensitivity case
        hw.ram_cycles = hw.ram_latency * hw.clockspeed 
        hw.bw_ram_miss_penality = 1/(hw.bw_ram) * hw.clockspeed / 8.0 # cycles/byte 
         # Measure the effective latency ( L_eff) in cycles
        L_eff = effective_cycles(phits, hw.cache_cycles, hw.ram_cycles)
        # Measure effective bandwidth (B_eff) in cycles -- We need to fix the input for this later
        B_eff = effective_cycles(phits, hw.cache_bandwidth_cycles,hw.bw_ram_miss_penality)
        #print("phits: ", phits, " L_eff, B_eff, clockspeed: ", L_eff, B_eff, hw.clockspeed)

        #try: hw.block_size
        #except NameError: hw.block_size = 8.0
        hw.pipelinelatencies['load'] = L_eff  /hw.clockspeed
        hw.pipelinethroughputs['load'] =  B_eff / hw.clockspeed
        #print("L_eff, B_eff for load ops and Phits:", L_eff, B_eff, phits)

        ######
        # 3. Compute run times of each block in task graph
        ######
        #simName, startTime, endTime, minDelay, useMPI = "ppt", 0.0, 100000000000.0, 0.1, False
        simName = "ppt"
        bb_times, counts = {}, 0
        for bbName, bb_graph in task_graph.items():
            if bbName in bb_id_map:
                #print("INFO:", bbName, "in task_graph also in bb_id_map") #NOTE: Nandu: These are the good BBs
                bb_id = bb_id_map[bbName]
                #print("before dropCore with bbName, bb_graph: ", bbName, bb_id, len(task_graph.items()))
                simEngine = Simian(simName, silent=True)
                simEngine.addEntity('dropCore', dropCore, 0, bb_graph)
                simEngine.run()
                simEngine.exit()
                # By convention, dropEngine added an element to the taskgraph with the time
                # it took to complete it. Every instruction vertex in the taskgraph also has an
                # attribute 'time_done' that lists when that instruction was completed.
                bb_times[bb_id] = bb_graph["totalBBTime"]
                #NOTE: Nandu: Otherwise taskgraph traversal in pipeline simulator (for future input settings) gets confused by this extra node
                del bb_graph["totalBBTime"]
                counts +=1
            else:
                pass 
                #print("WARNING:", bbName, "in task_graph but not in bb_id_map") #NOTE: Nandu: Need to investigate these warnings further

        ######
        # 4. Calculate sum of basic block functions
        ######

        bb_counts = bbcf.get_BBcountsArray(runs[i])
        #print("Calculating sum of basic block functions for run", hw_name, runs[i], " successful bbs:", counts )
        predicted_times[(hw_name, runs[i])] = 0.0
        if in_file.print_block_info:
                    print("Block id \tBlock Count \tBlock Time [s]\tTotal Time for block [s]")
        for bb in sorted(bb_times.keys()):
            if bb in bb_counts: 
                predicted_times[(hw_name, runs[i])] += bb_counts[bb] * bb_times[bb]
                if in_file.print_block_info:
                    prod = bb_counts[bb] * bb_times[bb]
                    print("%7s      %8.7e       %8.7e         %8.7e"%(bb, bb_counts[bb], bb_times[bb],prod))
            else:
                pass 
                #print("WARNING:", bb, "in bb_times but not in bb_counts") #NOTE: Nandu: Need to investigate these warnings further
        # Dealing with Sensitivity runs, filling information into sensitivities list
        
        #print ("Predicted time calculated: ",  predicted_times[(hw_name, runs[i])], hw.clockspeed)
        
        ######
        # 5. Sensitivity analysis runs
        ######       
        
        
        if in_file.sensitivity_analysis:
            #print("in sensitivity analysis")
            if init_sens_run == True:
                init_sens_run = False
                sensitivities[(hw_name, runs[i])] = []
                print("in initial sense branch with (hw_name, runs[i]", hw_name, runs[i])
                baserun_time = predicted_times[(hw_name, runs[i])]
                call_again = True # indicates whether changeParameter actually changed a parameter
                while call_again and not sens_done:
                    sens_done, para_name, plus, call_again, base_hw_param_value  = changeParameter(hw)
            else: # not first sensitivity run
                if not sens_done:
                    
                    sens_time = predicted_times[(hw_name, runs[i])]
                    if plus: #this one gets done first
                        #print("+in sense branch with (sens_time, baserun_time, plus_factor):", sens_time, baserun_time, plus_factor)
                        #plus_ratio = ((sens_time - baserun_time)/baserun_time) / plus_factor
                        plus_ratio = (sens_time-baserun_time)/baserun_time / delta
                        print ("+sens_time, baserun_time, para_name: ",  sens_time, baserun_time, para_name)
                        predicted_times[(hw_name, runs[i])] = baserun_time 
                    else: # this is the minus run
                        #print("-in sense branch with (sens_time, baserun_time, plus_factor):", sens_time, baserun_time, plus_factor)
                        #minus_ratio = (( baserun_time - sens_time)/baserun_time) / 1/plus_factor
                        minus_ratio = (sens_time -baserun_time)/baserun_time  / delta
                        sensitivities[(hw_name, runs[i])].append((para_name,plus_ratio,minus_ratio))
                        print ("+sens_time, baserun_time, para_name: ",  sens_time, baserun_time, para_name)
                        predicted_times[(hw_name, runs[i])] = baserun_time 
                    call_again = True # indicates whether changeParameter actually changed a parameter
                    while call_again and not sens_done:
                        sens_done, para_name, plus, call_again, base_hw_param_value  = changeParameter(hw)
            if sens_done:
                init_sens_run = True
        if sens_done or not in_file.sensitivity_analysis: 
            #print("about to increase i ", i)
            # reset orig value
            #if sens_done: predicted_times[(hw_name, runs[i])] = baserun_time 
            sens_done = False
            i += 1
    j += 1

######
# 6. Prepare output
######
print("\n")
print("Runtime Prediction Results (%d input parameter combinations on %d hw platforms):"%(len(runs), len(hw_list)))
print("\n")
#print(sensitivities)

print("========================================================================================")
print("%-20s  %-15s %-30s %-15s "%("App", "HW", "Predicted Time [seconds]", "Input"))
print("========================================================================================")
for item in predicted_times:
    (hw, run) = item
    t = predicted_times[item] # comes out as seconds directly
    print("%-20s  %-15s %10.10e %33s "%(str(sys.argv[1]), hw, t, run))
    if in_file.sensitivity_analysis:
        print("-----------------------------------------------------------------------------------")
        print("Runtime sensitivities to (decrease/increase) in parameter values with delta=", delta, ":")
        print("(Interpretation: negative values = GOOD, 0 = no effect, +-1 = direct effect)")
        print("-----------------------------------------------------------------------------------")
        for item2 in sensitivities[item]:
            (parameter, sensplus, sensminus) = item2
            print("%-30s  %-3.4f     %-3.4f "%(parameter, sensminus, sensplus))
        print("========================================================================================")
print("========================================================================================")

###############################################################################
# END of main
###############################################################################

