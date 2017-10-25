import os
import sys
import subprocess
import numpy as np
result_list = []
# weak scaling
#np_list =  [ 1, 2, 4, 8, 16, 32, 64] #[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
rank_list = [(1,1,1), (4,4,4), (4,8,8), ]
batchsize_list = [10000]#
buffersize_list = [10000]#
bandwidths = [40e9]  #[40e9] interconnect
latencies = [.7e-6] #[1e-10] interconnect
mem_bandwidths = [5e11]#[5e11]
mem_latencies = [1e-7]# [1e-7]
n_photons = [128000001]#[1.28e6, 1.28e7, 1.28e8] # per rank
compute_times = [3.3e-7] #[9.7e-7]
mpi_call_times = [1e-7]
platform = ['crossbar', 'bypass', 'stampede', 'mustang', 'moonlight', 'edison', 'hopper', 'titan', 'sequoia', 'cielo', 'darter' ]
# argv[1] can be one of the following:
# cielo, hopper, titan (gemini)
# sequoia, mira, vulcan (bluegeneq)
# darter, edison (aries)
# stampede, moonlight, mustang (fattree)
# crossbar, bypass
time_list = []

for ct in compute_times:
	for (x, y, z) in rank_list:
		for pf in platform:
			for ph in n_photons:
				for buf in buffersize_list:
					for bat in batchsize_list: 
						result_list.append(subprocess.call(["/Users/eidenben/PERFORMANCEPREDICTION/Code/pypy2-v5.6.0-osx64/bin/pypy IMCSim-mpiwarpV4_hw.py {0} {1} {2} {3} {4} {5} {6} {7} ".format(\
							pf, ph, x, y, z, bat, buf, ct) ], shell=True))
						print "done with one "
print result_list
exit()

