import os
import sys
import subprocess
import numpy as np
from multiprocessing import Pool

def s(i):
  pht_per_rank = 1600000
  #np_list = [16, 32, 64, 128, 256, 512, 1280, 2000]
  #rank_list = [(2,4,2), (4,4,2),(4,4,4), (4,4,8), (4,8,8), (8,8,8), (8,8,20), (10,10,20)]
  np_list = [16, 32, 64, 128, 256, 512, 1280]
  rank_list = [(2,4,2), (4,4,2),(4,4,4), (4,4,8), (4,8,8), (8,8,8), (8,8,20)]
  time_list = []
  c_time_list = []
  results_filename = "test_results1.txt"
  f_results = open(results_filename,'w')
  #for index, np in enumerate(np_list):
  time = 0.0
  (x,y,z) = rank_list[i]
  output_file = "rank{0}_weak1.txt".format((x*y*z))
  p_s = "python2.7 IMCSim-mpi.py {0} {1} {2} {3} > {4}".format(\
    (pht_per_rank*(x*y*z)), x, y, z, output_file)
  subprocess.call([p_s], shell=True)
  ppr = (pht_per_rank*(x*y*z))/(x*y*z)
  f_out = open(output_file,'r')
  for line in f_out:
    if 'total_time=' in line:
      time = float(line.strip().split('=')[1].strip().split(',')[0].strip())
      c_time = line.split('=')[2]
      time_list.append(time)
      c_time_list.append(c_time)
      print p_s + "\n:c_t:{0},t:{1}---ppr:{2}".format(c_time, time, ppr)
      break
  f_results.write("weak_scaling\n"+p_s + "\nTime:{0}\nCompute_time:{1}".format(time, c_time))
  '''
  pht=1024000000
  time_list = []
  #for index, np in enumerate(np_list):
  time = 0.0
  (x,y,z) = rank_list[i]
  output_file = "rank{0}_strong1.txt".format(np_list[i])
  process_string = "python2.7 IMCSim-mpi.py {0} {1} {2} {3} > {4}".format(\
    pht, x, y, z, output_file)
  subprocess.call([process_string], shell=True)
  f_out = open(output_file,'r')
  for line in f_out:
    if 'total_time=' in line:
      time = float(line.strip().split('=')[1].strip().split(',')[0].strip())
      time_list.append(time)
      print process_string + "\n:{0}".format(time)
      break
  f_results.write("\nSTRONG_SCALING:\n"+process_string + "\n:{0}".format(time))
  '''
  f_results.close()

if __name__ == '__main__':
  pool = Pool(processes=7)
  print pool.map(s, range(7))

