#!/usr/bin/env python

import os
import string

src_path="./"
binaryfile = "blas_test.exe"
binary_path = src_path+binaryfile
outdir_tag = "_timing"

#              M       N       K       ALPHA   BETA
PARAMS_in = [ [8192,   4096,   128,    1.0,    1.0 ] ]

n_trials = 1
skip_run = False
do_bfbin2csv = False #True

for PARAMS in PARAMS_in:
          
            [M,N,K,ALPHA,BETA] = PARAMS

            outdirname = "out_M"+str(int(M))+"_N"+str(int(N))+"_K"+str(int(K))+"_A"+str(int(float(ALPHA)))+"_B"+str(int(float(BETA)))+outdir_tag
            os.system("mkdir "+outdirname)
            filename = outdirname+"/result.out"

            for trial in range(0, n_trials):

                cmd_params1 = " "+str(N)+" "+str(M)+" "+str(K)+" "+str(ALPHA)+" "+str(BETA)+" >> "
                cmd_params2 = " 2>&1"
                cmd = str(binary_path+cmd_params1+filename+cmd_params2)

                print cmd
                if not skip_run:
                    os.system(cmd)
                    if do_bfbin2csv:
                        os.system("bfbin2csv "+binary_path+".byfl >> "+outdirname+"/result.csv")
                        os.system("mv "+binary_path+".byfl "+outdirname+"/.")
