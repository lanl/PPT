##############################
# definitions of major functions of the kernel IMC(operations) for all the functions in the original IMC app.
################################

from operator import add
#list3 = map(add, list1, list2)

#these are the main timestep functions(ops and functions are nested):
def time_step_kernel(num_photons, num_timesteps, num_cell):
  #total_ops_list = map(add, calculate_photon_energy(num_photons, num_timesteps, num_cell),get_photon_list_E(num_photons, num_timesteps, num_cell))
  #total_ops_list = map(add,total_ops_list, total_ops_list2)
  return transport_photons(num_photons, num_timesteps, num_cell)

#this is the main computational intensive function (calls nested functions in this file). Work in progress
#becouse it depends on event probabilites estimations. Currently the simplest estimate is implemented.
def transport_photons(num_photons, timestep_i, num_cell):
  
  ######-----------PHOTON operations variation--------#########
  #x,y=photons, loads per invocation of transport_photons
  #y = 124.59x + 2337.5
  float_loads_p = (124.59*num_photons) + 2337.5

  #x,y=photons, flp per invocation of transport_photons
  #no direct divisions in transport_photons
  #y = 11.434x + 193.49
  float_alu_ops_p = (11.434*num_photons) + 193.49

  #x,y=photons, intops per invocation of transport_photons
  #y = 273.67x + 5187.5
  int_alu_ops_p =(273.67*num_photons) + 5187.5

  ######-----------Cell operations variation--------#########
  #really bad float estimate (focuses on large cell materials)
  float_loads_c=(25084*num_cell) + 1E+08

  #x,y=cells, flp per invocation of transport_photons
  #no direct divisions in transport_photons
  #float_alu_ops_c= 1971.9*num_cell + 1E+07
  float_alu_ops_c= (1971.9*num_cell) + 1E+07

  int_alu_ops_c= (48142*num_cell) + 3E+08
  
  # Put together task list to send to hardware 
  num_index_vars = 3         # number of index variables
  #may set to 0
  num_float_vars = 12
  index_loads = 3
  float_loads = float_loads_c+float_loads_p
  
  int_alu_ops = int_alu_ops_c+int_alu_ops_p
  
  float_alu_ops = float_alu_ops_c+float_alu_ops_p
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  listt = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  #nested function to get rng(construction)
  listt = map(add,listt, get_rng())

  #Experimental event counts
  event_dict = get_event_counts(num_photons)
  avg_num_photon_deaths=event_dict['avg_num_photon_deaths']
  #avg number of times each photon traverses the for loop:
  avg_pht_travel=event_dict['avg_pht_travel']
  avg_scatters=event_dict['avg_scatters']
  avg_bound=event_dict['avg_bound']
  avg_b_vacuum=event_dict['avg_b_vacuum']
  avg_b_reflect=event_dict['avg_b_reflect']
  avg_sne=event_dict['avg_sne']
  avg_census=event_dict['avg_census']

  total_loops = int(num_photons*avg_pht_travel)

  '''#TIME INTENSIVE FOR LOOP DEPENDS ON #PHOTONS
  
  for p in range(0,total_loops):
    listt = map(add,listt, get_distance_to_scatter(num_photons, total_loops, num_cell, p))
    listt = map(add,listt, get_distance_to_bound(num_photons, timestep_i, num_cell, p))
    listt = map(add,listt, get_distance_remaining(num_photons, timestep_i, num_cell, p))
    list2 = map(lambda x: x * 2, min_fun(num_photons, timestep_i, num_cell))
    listt = map(add,listt, list2)
    listt = map(add,listt, tally_line_abs_E())
    listt = map(add,listt, move())
    listt = map(add,listt, below_cutoff())
  '''
  ###this is the above for loop compressed version:
  listnew = map(add,get_distance_to_bound(num_photons, timestep_i, num_cell), get_distance_remaining(num_photons, timestep_i, num_cell))
  list2 = map(lambda x: x * 2, min_fun(num_photons, timestep_i, num_cell))
  list3 = map(add, move(), tally_line_abs_E())
  list4 = map(add, below_cutoff(), list3)
  listnew = map(add, list4, map(add,list2, listnew))
  listt = map(add, listt, map(lambda x: x * total_loops, listnew))
  listt = map(add, listt, get_distance_to_scatter(num_photons, total_loops, num_cell))

  #count operations for the 4 types of events:
  ####SCATTER EVENT####
  lists = map(add, tally_scatter_event(), get_uniform_angle())
  lists = map(add,lists, set_angle())
  listt = map (add, listt, map(lambda x: x * avg_scatters, lists))
  ####BOUNDARY CROSS EVENT####
  #vacuum input condition
  listb_v = map(add, tally_exit_event(), tally_exit_E())
  listb_v = map(add, listb_v, get_E())
  listb_v = map(lambda x: x * avg_b_vacuum, listb_v)
  #reflect input condition
  listb_r = map(lambda x: x * avg_b_reflect, reflect())
  #else set next element condition
  listt = map(add, listt, map(lambda x: x * avg_sne, set_element()))

  listt = map(add, listt, map(add,listb_v, listb_r))

  ####REACH CENSUS EVENT####
  listt = map(add, listt, map(lambda x: x * avg_census, set_census_flag()) )

  ####PHOTON DEATH EVENT####
  listt = map(add, listt, map(lambda x: x * avg_num_photon_deaths, map(add, tally_point_abs_E(), get_E())) )

  return listt

# Get experimental event counts
def get_event_counts(n_photons):
  event_dict = {}
  if n_photons==80000:
    #p_80000 event count
    event_dict['avg_num_photon_deaths']=79994
    #avg number of times each photon traverses the for loop:
    event_dict['avg_pht_travel']=17.08
    event_dict['avg_scatters']=1.29E+06
    event_dict['avg_bound']=84921
    event_dict['avg_b_vacuum']=0
    event_dict['avg_b_reflect']=84921
    event_dict['avg_sne']=0
    event_dict['avg_census']=0
    
  if n_photons==180000:
    #p_180000 event counts
    event_dict['avg_num_photon_deaths']=179986
    #avg number of times each photon traverses the for loop:
    event_dict['avg_pht_travel']=17.16
    event_dict['avg_scatters']=2.90E+06
    event_dict['avg_bound']=191926
    event_dict['avg_b_vacuum']=0
    event_dict['avg_b_reflect']=191926
    event_dict['avg_sne']=0
    event_dict['avg_census']=0

  if n_photons==200000:
    #p_180000 event counts
    event_dict['avg_num_photon_deaths']=199985
    #avg number of times each photon traverses the for loop:
    event_dict['avg_pht_travel']=17.15
    event_dict['avg_scatters']=3.22E+06
    event_dict['avg_bound']=211807
    event_dict['avg_b_vacuum']=0
    event_dict['avg_b_reflect']=211807
    event_dict['avg_sne']=0
    event_dict['avg_census']=0

  if n_photons==300000:
    #p_300000 event counts
    event_dict['avg_num_photon_deaths']=299997
    #avg number of times each photon traverses the for loop:
    event_dict['avg_pht_travel']=17.16
    event_dict['avg_scatters']=4.83E+06
    event_dict['avg_bound']=319365
    event_dict['avg_b_vacuum']=0
    event_dict['avg_b_reflect']=319365
    event_dict['avg_sne']=0
    event_dict['avg_census']=0

  return event_dict


def reflect():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 3
  int_alu_ops = 4
  float_alu_ops = 1
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def tally_point_abs_E():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 4
  int_alu_ops = 7
  float_alu_ops = 1
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def set_census_flag():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 5
  int_alu_ops = 7
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def set_element():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 3
  int_alu_ops = 4
  float_alu_ops = 1
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def tally_exit_E():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 3
  int_alu_ops = 4
  float_alu_ops = 1
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def tally_exit_event():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 2
  int_alu_ops = 4
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def set_angle():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 2
  int_alu_ops = 4
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def get_uniform_angle():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 1
  int_alu_ops = 2
  float_alu_ops = 2
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1


def tally_scatter_event():
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 3
  int_alu_ops = 7
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def below_cutoff():
  num_index_vars = 0         # number of index variables
  num_float_vars = 2
  index_loads = 0
  float_loads = 6
  int_alu_ops = 8
  float_alu_ops = 1
  float_div_ops = 1
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  list1 = map(add,list1, get_E())
  list1 = map(add,list1, set_E())
  return list1

def move():
  num_index_vars = 0         # number of index variables
  num_float_vars = 4
  index_loads = 0
  float_loads = 8
  int_alu_ops = 8
  float_alu_ops = 3
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0
  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def tally_line_abs_E():
  num_index_vars = 3         # number of index variables
  num_float_vars = 7
  index_loads = 3
  float_loads = 14
  int_alu_ops = 22
  float_alu_ops = 7
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  list1 = map(add,list1, get_E())
  list1 = map(add,list1, set_E())
  return list1

def get_E():
  num_index_vars = 2         # number of index variables
  num_float_vars = 0
  index_loads = 2
  float_loads = 0
  int_alu_ops = 3
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def set_E():
  num_index_vars = 3         # number of index variables
  num_float_vars = 0
  index_loads = 3
  float_loads = 0
  int_alu_ops = 4
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

#op count for function min(...)
def min_fun(num_photons, timestep_i, num_cell):
  num_index_vars = 0         # number of index variables
  num_float_vars = 2
  index_loads = 0
  float_loads = 6
  int_alu_ops = 3
  float_alu_ops = 1
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

def get_distance_to_bound(num_photons, timestep_i, num_cell):
  num_index_vars = 0         # number of index variables
  #may set to 0
  num_float_vars = 5
  index_loads = 0
  float_loads = 10
  
  int_alu_ops = 13
  
  float_alu_ops = 5
  float_div_ops = 1
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

#needs to be filled out at a later date
def get_distance_remaining(num_photons, timestep_i, num_cell):
  num_index_vars = 0         # number of index variables
  num_float_vars = 0
  index_loads = 0
  float_loads = 2
  int_alu_ops = 3
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1


def get_distance_to_scatter(num_photons, total_loops, num_cell):
  num_index_vars = 2*total_loops        # number of index variables
  #may set to 0
  num_float_vars = 6*total_loops
  index_loads = 0
  float_loads = 6*total_loops
  
  int_alu_ops = 14*total_loops
  
  float_alu_ops = 3*total_loops
  #I included 1 log op in the division * weight of 6 divisions
  float_div_ops = (1+6)*total_loops
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  list1 = map(add,list1, generate_random_number(total_loops))
  return list1

#this function includes the extra cost of calling nested function regenerate_rng for some number of invocations
def generate_random_number(num_invoc):
  num_index_vars = 6*num_invoc
  num_float_vars = 0*num_invoc
  index_loads = 6*num_invoc
  float_loads = 0*num_invoc
  int_alu_ops = 13*num_invoc
  float_alu_ops = 0*num_invoc
  float_div_ops = 0*num_invoc
  float_vector_ops = 0*num_invoc
  int_vector_ops = 0*num_invoc

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  #nested function (will only call it once to save time)
  list1 = map(add,list1, regenerate_random_numbers(num_invoc))
  return list1

def regenerate_random_numbers(num_invoc):
  #get number of regenerate invocations this is based on array size of generated rngs:
  num_calls = num_invoc/10000

  num_index_vars = 10000
  num_float_vars = 30000
  index_loads = 2850004/2
  float_loads = 2850004/2
  int_alu_ops = 9480091
  float_alu_ops = 20000
  float_div_ops = 10000
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  #nested 3fishfun
  list2 = threefry2x64()
  #multiply list2 by 10000 (this is the default rng array size)
  list2 = map(lambda x: x * 10000, list2)
  list1 = map(add,list1, list2)

  #get the op count for all the regenerate_random_number calls for num_invoc
  list1 = map(lambda x: x * num_calls, list1)
  return list1

def threefry2x64():
  #get the ops count for each invocation of this function
  num_index_vars = 0
  num_float_vars = 1
  index_loads = 0
  float_loads = 281
  int_alu_ops = 935
  float_alu_ops = 1
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

#becouse it depends on event probabilites estimations. Currently the simplest estimate is implemented.
def get_rng():
  # Put together task list to send to hardware
  #does not depend on cells 
  num_index_vars = 0         # number of index variables
  #may set to 0
  num_float_vars = 0
  index_loads = 0
  float_loads = 5
  
  int_alu_ops = 23
  
  float_alu_ops = 0
  float_div_ops = 0
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]
  return list1

"""
def build_census_list(num_photons, num_timesteps, num_cell):
  # Put together task list to send to hardware 
  num_index_vars = 10        # number of index variables
  # Per cell basis
  num_float_vars = 11
  # Per cell basis
  index_loads = 55
  # Per cell basis
  float_loads = 26
  
  int_alu_ops = 89
  
  float_alu_ops = 7
  float_div_ops = 2
  float_vector_ops = 22
  int_vector_ops = 0

  listt = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_div_ops, float_vector_ops, int_vector_ops]}

  return listt

def pow():
  # Put together task list to send to hardware 
  num_index_vars = 0         # number of index variables
  # Per cell basis
  num_float_vars = 2
  # Per cell basis
  index_loads = 0
  # Per cell basis
  float_loads = 2
  
  int_alu_ops = 3
  float_alu_ops = 1
  float_vector_ops = 0
  int_vector_ops = 0

  list1 = [num_index_vars, num_float_vars, index_loads, float_loads, int_alu_ops, float_alu_ops, float_vector_ops, int_vector_ops]

  return list1
"""
