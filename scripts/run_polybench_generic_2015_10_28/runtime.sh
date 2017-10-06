#!/bin/bash
# EJ Park
# ejpark@lanl.gov
# Last modified: Oct 28, 2015
# To collect timing info for each program 
##########################################

benchmark=$1	# program name without .c, e.g., For 2mm.c, we use 2mm
variant="5";

current_loc=`pwd`;
common_loc="${current_loc}/utilities";
programs_loc="${current_loc}/benchmarks_time"
output_loc="${current_loc}/output/time"
input_config_loc="${current_loc}/input_config";
input_loc="${programs_loc}/${benchmark}";

common_file="${current_loc}/utilities/polybench.c";
include_headers="-I. -I${common_loc}"
optimizations="-O0" # no optimization
what_to_collect="-DPOLYBENCH_TIME"  # To collect PAPI counters, we add -DPOLYBENCH_PAPI
link_libs="-lm"

output_file="${output_loc}/output_${benchmark}.txt";
config_file="${input_config_loc}/${benchmark}.txt"

# Create the output directory if it does not exist
if [ ! -d ${output_loc} ]; then mkdir -p ${output_loc}; fi;

# Set up start_point and end_point
end_point=`wc -l ${config_file} | cut -d' ' -f1`;
if [ -f ${output_file} ]; 
then 
    timevalue=`tail -1 ${output_file} | awk 'BEGIN{FS=",\t"}{print $2}'`;
    if [ -z "$timevalue" ];
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


    # First run with p
    # Compile
    gcc ${optimizations} -o ${benchmark} ${common_file} ${benchmark}.c ${include_headers} \
	${what_to_collect} ${input_define} ${link_libs}

    rawoutput="timing.out";
    if [ -f ${rawoutput} ]; then rm ${rawoutput}; fi;
		        	        
    for((i=1;i<=${variant};i++));
    do
	./${benchmark} >> ${rawoutput};
    done

    # Calculate mean, highest, lowest value among ${variant} numbers of run
    amean=`awk 'BEGIN{sum=0}{sum+=$1}END{printf("%.6f\n",sum/NR)}' ${rawoutput}`;
    echo $amean 2>&1 | tee -a ${output_file};
	
    rm ${rawoutput} ${benchmark};
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
