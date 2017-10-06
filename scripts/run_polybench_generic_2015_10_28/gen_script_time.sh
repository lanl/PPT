#!/bin/bash
# EJ Park
# ejpark@lanl.gov
# Last modified: Oct 28, 2015
# To create a runtime script for each program
##############################################

num="1"
while read progname;
do
    ofile="runtime_"${num}".sh"
    
    echo "#!/bin/bash" > $ofile
    echo "" >> $ofile

    echo "# SBATCH -N 1" >> $ofile
    echo "# SBATCH --time=72:00:00" >> $ofile
    echo "" >> $ofile

    echo `pwd`"/runtime.sh "${progname} >> $ofile;

    chmod +x $ofile;
    num=$[$num+1];
done < "programlist.txt"
