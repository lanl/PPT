#!/bin/bash
# EJ Park
# ejpark@lanl.gov
# Last modified: Oct 28, 2015
# To submit jobs in a certain range 
# The first parameter should specify either time or papi
#########################################################


if [ $# -ne "3" ];
then
    echo "Usage $0 [time or papi] start end"
    exit;
fi;

metric=$1
partition_name="SAND_16_2:MEM64.0:Accelerator-1:TeslaK40c:10GbE::";

for((i=$2;i<=$3;i++));
do
    srun -p $partition_name -N 1 --time=72:00:00 ./run${metric}_${i}.sh &
done
