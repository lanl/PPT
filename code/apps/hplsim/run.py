#!/usr/bin/env python

import os
import string
import time
import numpy
#import matplotlib.pyplot as plt
from datetime import datetime

# INPUT:
#PQ_in = [[1,1],
#         [2,1],
#         [2,2],
#         [4,2],
#         [4,4],
#         [6,4],
#         [8,4],
#         [8,6],
#         [8,8],
#         [10,8],
#         [12,8]]
PQ_in = [[1,1],[2,1],[2,2]]
NS_in = [1000,2000,4000]
NB_in = [10,50,100]
DEPTH_in = [0,1]

PQ_in = [[1,1],
         [2,2],
         [4,4]]
NS_in = [1000,2000,4000,8000]
NB_in = [10,100]
DEPTH_in = [0]

if True:
    PQ_in = [[4,4],
             [8,4],
             [8,6],
             [8,8],
             [10,8],
             [12,8]]
    NS_in = [6400]
    NB_in = [32]
    DEPTH_in = [0]

# LABEL DETAILS:
binaryfile = "hplsim-mpi.py"
outdir = "out"
run_id = "production" # this becomes part of the output file
skip_run = False
plot_on = True
first_at_figure = True

# Setup:
GFLOPs_vs_NP = []
GFLOPs_vs_NP = []

# Output Summary: summary.out
if not os.path.isdir(str(outdir)): os.system(str("mkdir "+outdir))
g = open(str(outdir+"/summary.out"), 'w')
g.write("Filename: "+str(outdir+"/summary.out")+"\n")
#g.write("depth\tNP\tNS\tNB\tP\tQ\tGFLOPs\n")
g.write("depth\tNP\tP\tQ\tNS\tNB\tGFLOPs\t\tt_PDFACT\tt_PDUPDATE\n")

# LOOP:
for DEPTH in DEPTH_in:
    for PQ in PQ_in:
        P = PQ[0]
        Q = PQ[1]
        NP = P * Q
        for NS in NS_in:
            for NB in NB_in:
                filename = outdir+"/run_d"+str(int(DEPTH))+"_p"+str(int(P))+"_q"+str(int(Q))+"_ns"+str(int(NS))+"_nb"+str(int(NB))+".raw"
                cmd_params1 = " -d "+str(DEPTH)+" -p "+str(int(P))+" -q "+str(int(Q))+" -s "+str(int(NS))+" -b "+str(int(NB))+" >> "
                cmd_params2 = " 2>&1"
                cmd = str("pypy "+binaryfile+cmd_params1+filename+cmd_params2)
                print cmd
                if not skip_run:
                    os.system(cmd)
                performance = 0.0
                with open(str(filename),"r") as f:
                    for stringline in f:
                        line = string.split(stringline)
                        if line<>[]:
                            if line[0] == "PERFORMANCE(GFlops):":
                                performance = float(line[1])
                            if line[0] == "TIME_pdfact:":
                                t_pdfact = float(line[1])
                            if line[0] == "TIME_pdupdate:":
                                t_pdupdate = float(line[1])
                f.close()
                #g.write(str(DEPTH)+"\t"+str(NP)+"\t"+str(NS)+"\t"+str(NB)+"\t"+str(P)+"\t"+str(Q)+"\t"+str(performance)+"\n")
                g.write(str(DEPTH)+"\t"+str(NP)+"\t"+str(P)+"\t"+str(Q)+"\t"+str(NS)+"\t"+str(NB)+"\t"+str(performance)+"\t"+str(t_pdfact)+"\t"+str(t_pdupdate)+"\n")
g.close()
if plot_on:
    pass
    #plt.show()
