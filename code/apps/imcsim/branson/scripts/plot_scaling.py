import matplotlib.pyplot as plt
import matplotlib
import numpy as np


###############################################################################
def method_to_num(method_str):
  if (method_str=="CELL_PASS"): return 0
  if (method_str=="PARTICLE_PASS"): return 1
###############################################################################
def get_scaling_from_file(filename):
  data = np.loadtxt(filename, converters={3:method_to_num})
  map_particles = data[:,2]
  map_algorithms = data[:,3]
  unique_particles = set(map_particles)
  unique_schemes = set(map_algorithms)
 
  results = []
  results_map = {}
  count = 0
  for i,n_p in enumerate(unique_particles):
    for j, dd_type in enumerate(unique_schemes):
      results.append([])
      results_map["{0}_{1}".format(int(n_p),int(dd_type))] =count
      count=count+1

  for run in data:
    # use the key to get the index into the results vector
    key = "{0}_{1}".format(int(run[2]), int(run[3]))
    r_index = results_map[key]
    results[r_index].append(run)

  return results
###############################################################################

scale_1_filename = "results/scale_1_results.txt"
scale_2_filename = "results/scale_2_results.txt"

plot_scale_1 = True
plot_scale_2 = True

#different symbol types: 'o', '+', 'x', '*'
#use 'markersize' option to set size of symbols
#set font size
matplotlib.rcParams.update({'font.size': 20})


###############################################################################
if  plot_scale_2:
  # weak scaling data and plot
  fig = plt.figure(1)
  ax = fig.add_subplot(111)
  
  results = get_scaling_from_file(scale_2_filename)

  d_label = {"1024001_0":"Cell Pass--1.0e6", "1024001_1":"Particle Pass--1.0e6", \
             "10240001_0":"Cell Pass--1.0e7", "10240001_1":"Particle Pass--1.0e7",\
             "102400001_0":"Cell Pass--1.0e8", "102400001_1":"Particle Pass--1.0e8"}
  d_format = {"1024001_0":"mo-", "1024001_1":"mo--", \
              "10240001_0":"bo-", "10240001_1":"bo--",\
              "102400001_0":"go-", "102400001_1":"go--"}

  max_cores = 0
  max_scale = 0.0
  for result in results:  
    time = []
    p = []
    for run in result:
      p.append(run[0])
      time.append(run[4])
    t_1 = time[0]
    strong_scale = np.divide(t_1, time) 
    max_scale = np.max([max_scale, np.max(strong_scale)])
    max_cores = np.max([max_cores, np.max(p)])
    key = "{0}_{1}".format(int(result[0][2]), int(result[0][3]))
    ax.plot(p, strong_scale, d_format[key], label=d_label[key] )

  ideal_x = np.linspace(0,max_cores,100)
  ideal_y = np.linspace(0,max_cores,100)
  ax.plot(ideal_x, ideal_y, 'k-', label="Ideal Strong")

  #ax.set_xscale('log')
  #ax.set_yscale('log')
  plt.xlabel("cores")
  plt.ylabel("strong scaling")
  
  plt.axis([0.9, 1.001*max_cores, 0.0, max_scale])
  ax.legend(loc = 'best')
  #save figure
  fig.set_size_inches(12.0,10.0)
  plt.savefig('dd_scaling_weak.pdf',dpi=100, bbox_inches='tight')
###############################################################################
