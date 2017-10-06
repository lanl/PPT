#!/bin/bash
# EJ Park
# ejpark@lanl.gov
# Last modified: Oct 28, 2015
# To collect PAPI info for each program 
##########################################


benchmark=$1	# program name without .c, e.g., For 2mm.c, we use 2mm

current_loc=`pwd`;
common_loc="${current_loc}/utilities";
programs_loc="${current_loc}/benchmarks_papi"
output_loc="${current_loc}/output/papi"
input_config_loc="${current_loc}/input_config";
input_loc="${programs_loc}/${benchmark}";

common_file="${current_loc}/utilities/polybench.c";
include_headers="-I. -I${common_loc}"
optimizations="-O0" # no optimization
link_libs="-lm"

# Set up PAPI counters to collect
papi_loc=`which papi_avail | xargs dirname | xargs dirname`;
include_headers=${include_headers}" -I${papi_loc}/include"
link_libs=${link_libs}" -lpapi -L${papi_loc}/lib";
what_to_collect="-DPOLYBENCH_PAPI"  # To collect PAPIcounters, we add -DPOLYBENCH_PAPI

if ! [ -f ${common_loc}/papi_counters.list ];
then
    while read line;
    do
    	counter=`echo $line | awk '{gsub("\"","",$0); gsub(",$","",$0); print $0}'`;
    	papi_avail | grep "$counter" | awk '{if($3 == "Yes") print "\""$1"\",";}' >> ${common_loc}/papi_counters.list
    done < "counters.txt"
fi;

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${papi_loc}/lib

output_file="${output_loc}/output_papi_${benchmark}.txt";
config_file="${input_config_loc}/${benchmark}.txt"

# Create the output directory if it does not exist
if [ ! -d ${output_loc} ]; then mkdir -p ${output_loc}; fi;

# Set up start_point and end_point
end_point=`wc -l ${config_file} | cut -d' ' -f1`;
if [ -f ${output_file} ]; 
then 
    papivalue=`tail -1 ${output_file} | awk 'BEGIN{FS=",\t"}{print $2}'`;
    if [ -z "$papivalue" ];
    then
	sed -i '$ d' ${output_file}
    fi;
    start_point=`wc -l ${output_file} | cut -d' ' -f1`;

    if [ $start_point -ge $end_point ]; then exit;
    else
	start_point=$[${start_point}+1];
    fi;
else start_point="1"; 
fi;

input_factor="1" 
case $benchmark in
    	"adi" | "dynprog" | "fdtd-2d" |  \
    	"jacobi-1d-imper" | "jacobi-2d-imper" | "seidel-2d" )
	    input_factor="2"
	    ;;
    	"reg_detect" )
	    input_factor="3"
	    ;;
esac;

input_setup(){
    case $benchmark in
        "2mm" )			input_define="-DNI=${isize} -DNJ=${isize} -DNK=${isize} -DNL=${isize}";;
    	"3mm" )			input_define="-DNI=${isize} -DNJ=${isize} -DNK=${isize} -DNL=${isize} -DNM=${isize}";;
    	"adi" )			input_define="-DTSTEPS=${isize} -DN=${isize2}";;
	"jacobi-1d-imper" )	input_define="-DTSTEPS=${isize} -DN=${isize2}";;
	"jacobi-2d-imper" )	input_define="-DTSTEPS=${isize} -DN=${isize2}";;
	"seidel-2d" )		input_define="-DTSTEPS=${isize} -DN=${isize2}";;
	"atax" | "bicg" )	input_define="-DNX=${isize} -DNY=${isize}";;
	"cholesky" | "durbin" | "gemver" | "gesummv" | "lu" | "ludcmp" | "mvt" | "trisolv" )
				input_define="-DN=${isize}";;
	"floyd-warshall" )	input_define="-DN=${isize}";;
	"covariance" | "correlation" )
				input_define="-DN=${isize} -DM=${isize}";;
	"doitgen" )		input_define="-DNQ=${isize} -DNR=${isize} -DNP=${isize}";;
	"dynprog" )		input_define="-DTSTEPS=${isize} -DLENGTH=${isize2}";;
	"fdtd-2d" )		input_define="-DTMAX=${isize} -DNX=${isize2} -DNY=${isize2}";;
	"fdtd-apml" )		input_define="-DCZ=${isize} -DCYM=${isize} -DCXM=${isize}";;
	"gemm" )		input_define="-DNI=${isize} -DNJ=${isize} -DNK=${isize}";;
	"gramschmidt" | "symm" | "syrk" | "syr2k" )
				input_define="-DNI=${isize} -DNJ=${isize}";;
	"trmm" )		input_define="-DNI=${isize}";;
	"reg_detect" )		input_define="-DNITER=${isize} -DLENGTH=${isize2} -DMAXGRID=${isize3}";;
    esac
}

input_run(){
    input_setup;
    input_str=`echo " "$input_define | awk '{gsub(" -D","_",$0); print $0}'`;
    echo -n ${benchmark}""${input_str}",	" 2>&1 | tee -a ${output_file};

    # Compile
    gcc ${optimizations} -o ${benchmark} ${common_file} ${benchmark}.c ${include_headers} \
	${what_to_collect} ${input_define} ${link_libs}

    ./${benchmark} | awk '{gsub(" ",",",$0); gsub(",$","",$0); print $0}'  2>&1 | tee -a ${output_file};
}

####### Main
cd ${input_loc}
for((j=${start_point};j<=${end_point};j++));
do
    isize=`sed -n ${j}p $config_file | cut -d'_' -f1`;
    isize2=`sed -n ${j}p $config_file | cut -d'_' -f2`;
    isize3=`sed -n ${j}p $config_file | cut -d'_' -f3`;
    input_run;
done

cd ${current_loc}
