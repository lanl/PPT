
------------------------------------------------------------
To generate scripts for each of the all polybench programs, 
simply run ./gen_script_time.sh or ./gen_script_papi.sh
Two groups of scripts, in total 60 scripts will be generated:
- runtime_1.sh to runtime_30.sh 
- runpapi_1.sh to runpapi_30.sh 

------------------------------------------------------------
To submit a job, you can use submit_job.sh. The first parameter should be either time or papi, 
and you can give the range of scripts with the 2nd and 3rd parameter. 
If you want to run runtime.sh runpapi.sh for all programs, you can submit like followings:

./submit_job.sh time 1 30 
./submit_job.sh papi 1 30 

If you want to submit only one single job - the N-th program in Polybench (the order is given in programlist.txt), 
./submit_job.sh time N N 
./submit_job.sh papi N N 


------------------------------------------------------------
To check the progress of running, you can use
./check_output.sh 


Any questions, or issues, you can send me email ejpark@lanl.gov


