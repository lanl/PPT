import os
import sys
import subprocess
import re
import lxml.etree as ET
import numpy as np

################################################################################
def make_input_file(base_file, dd_type, buffer_size, n_photons, n_cells):
  r_tree = ET.parse(base_file)
  r_root = r_tree.getroot()
  # set photons
  r_common = r_root.find("common")
  e_photons = r_common.find("photons")
  e_photons.text = "{0}".format(n_photons)
  # set buffer size
  r_common.find("particle_message_size").text = "{0}".format(buffer_size)
  # set dd algorithm 
  r_common.find("dd_transport_type").text = "{0}".format(dd_type)
  # set cells per dimension
  e_spatial = r_root.find("spatial")
  e_spatial.find("n_x_cells").text = "{0}".format(n_cells)
  e_spatial.find("n_y_cells").text = "{0}".format(n_cells)
  e_spatial.find("n_z_cells").text = "{0}".format(n_cells)  
  # set spacing in each dimension
  side_length = 1.0
  #mesh_spacing = side_length/n_cells 
  mesh_spacing = 1.0
  e_spatial.find("dx").text = "{0}".format(mesh_spacing)
  e_spatial.find("dy").text = "{0}".format(mesh_spacing)
  e_spatial.find("dz").text = "{0}".format(mesh_spacing)
  
  new_filename = "temp_input_n_{0}_cell_{1}.xml".format(n_photons, n_cells)
  r_tree.write(new_filename, pretty_print=True )
  return new_filename
################################################################################

if (len(sys.argv) != 2):
  print("usage: {0} <basic_input_file_name>".format(sys.argv[0]))
  sys.exit();

base_filename = sys.argv[1]

path_to_exe = "/net/scratch1/along/branson/build"
exe_name = "BRANSON"

time_r = re.compile('runtime: (.*?) $')

buffer_size = 2000
c_per_dim = [40]
base_photons = 1024000
np_list = [base_photons+1, 10*base_photons+1, 100*base_photons+1]
proc_list = [128, 256, 512]
dd_methods = ["CELL_PASS",  "PARTICLE_PASS" ]
samples = 4

#temp_input_file = make_input_file(base_filename, "CELL_PASS", 2000, 1000000, 40)
# make output file for results
results_filename = "scaling_results.txt"
f_results = open(results_filename,'w')

for dd_method in dd_methods:
  for n_particles in np_list:
    for n_cells in c_per_dim:
      for p in proc_list:
        times = []
        temp_input_file = make_input_file(base_filename, dd_method,\
          buffer_size, n_particles, n_cells)
        for s in range(samples):
          time = 0.0
          output_file = "temp_output.txt"
          subprocess.call(["mpirun -np {0} {1}/{2} {3} >> {4}".format( \
            p, path_to_exe, exe_name, temp_input_file, output_file) ], shell=True)
          f_out = open(output_file,'r')
          for line in f_out:
            if (time_r.search(line)):
              time = float(time_r.findall(line)[0])
          times.append(time)
          os.remove("{0}".format(output_file))
          print("{0} {1} {2} {3} {4}".format( \
            p, n_cells, n_particles, dd_method, time))
        os.remove("{0}".format(temp_input_file))
        # calculate average runtime and standard deviation
        runtime = np.average(times)
        stdev_runtime = np.std(times)
        f_results.write("{0} {1} {2} {3} {4} {5}\n".format(\
          p, n_cells, n_particles, dd_method, runtime, stdev_runtime))
f_results.close()
