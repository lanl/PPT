#!/bin/bash
# EJ Park
# ejpark@lanl.gov
# Last modified: Oct 28, 2015
# To check the progress of running
##################################

i="1"
while read name;
do
    time_file="output/time/output_${name}.txt"
    papi_file="output/papi/output_papi_${name}.txt"
    config_file="input_config/${name}.txt";

    correct_onum=`wc -l $config_file | cut -d' ' -f1`;

    if [ -f $time_file ];
    then
	value=`tail -1 ${time_file} | awk 'BEGIN{FS=",\t"}{print $2}'`;
	if [ -z "$value" ]; then sed -i '$ d' ${time_file}; fi;
	current_t_onum=`wc -l $time_file | cut -d' ' -f1`;
    else current_t_onum="0"; fi;
    
    if [ -f $papi_file ];
    then
	value=`tail -1 ${papi_file} | awk 'BEGIN{FS=",\t"}{print $2}'`;
	if [ -z "$value" ]; then sed -i '$ d' ${papi_file}; fi;
    	current_p_onum=`wc -l $papi_file | cut -d' ' -f1`;
    else current_p_onum="0"; fi;

    if [ $correct_onum -eq $current_t_onum ] && [ $correct_onum -eq $current_p_onum ];
    then echo "$name is Done.";
    else
	echo -n "[$i] $name is incomplete:	";
	echo "time($current_t_onum/$correct_onum), PAPI($current_p_onum/$correct_onum)";
    fi;

    i=$[i+1]; 
done < "programlist.txt"
