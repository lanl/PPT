#!/bin/bash
# EJ Park
# ejpark@lanl.gov
# Last modified: Nov 17, 2015
# To create byfl features in csv format
########################################

iloc=$1
ofile=$2

list_to_find='
    integer_ops
    flops$
    FAdd
    FMul
    memory_ops_(
    branch_ops
    ICmp 
    GetElementPtr
    TOTAL_OPS
    vector_operations
'

grep_print(){
    keyword=`echo $byflcntr | sed -e 's/_/ /g'`;
    test=`grep "$keyword" $file`;
 
    # Test if the keyword exists or not. Grep only if this test passes.
    if ! [ -z "$test" ];
    then
    	echo -n `echo $test | awk '{gsub(",","",$2); print $2}'` >> $ofile;
    	if ! [ $byflcntr = "vector_operations" ];
    	then echo -n ", " >> $ofile;
	else echo "" >> $ofile; fi;

	case $byflcntr in
	    "memory_ops_(" )
		# loads and stores
		echo -n `echo $test | cut -d'(' -f2 | awk '{gsub(",","",$1); print $1}'` >> $ofile; 
		echo -n ", " >> $ofile;
		echo -n `echo $test | awk '{gsub(",","",$8);  print $8}'` >> $ofile; 
		echo -n ", " >> $ofile;
		;;
	    "branch_ops" )
		# unconditional,direct and condition,indirect
	        echo -n `echo $test | cut -d'(' -f2 | awk '{gsub(",","",$1); print $1}'` >> $ofile; 
		echo -n ", " >> $ofile;
    	        echo -n `echo $test | awk '{gsub(",","",$10);  print $10}'` >> $ofile; 
		echo -n ", " >> $ofile;
		;;
	esac;
    else 
	if ! [ $byflcntr = "vector_operations" ]; then echo -n "0, ";
	else echo "0" >> $ofile; fi; 
    fi;
}

first="1";
for file in $iloc/*.txt;
do
    # Print header for the first line
    if [ $first -eq "1" ];
    then
	echo -n "name, " > $ofile
	echo -n "integer_ops, flops, FAdd, FMul, " >> $ofile;
    	echo -n "memory_ops, loads, stores, " >> $ofile
	echo -n "branch_ops, uncond_branch_ops, cond_branch_ops, " >> $ofile;
	echo -n "comparison, " >> $ofile;
    	echo -n "cpu_ops, " >> $ofile;
    	echo -n "total_ops, " >> $ofile;
	echo "vec_ops" >> $ofile;
	first="0";
    fi;

    # If file is not empty
    if [ -s "$file" ];
    then
   	name=`basename $file .txt`;
	echo -n $name", " >> $ofile;

	for byflcntr in $list_to_find;
	do
	    grep_print;
	done
    fi;
done
