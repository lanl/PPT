#!/bin/bash
date
echo " Running All apps with allhardware... "
cd ./blackscholes
python3 ../../ppt.py BLACKSCHOLES_in_allhardware >> OutputOfAllHardwareTests.txt 
cd ../jacobi
python3 ../../ppt.py JACOBI_in_allhardware >> OutputOfAllHardwareTests.txt 
cd ../laplace2d
python3 ../../ppt.py LAPLACE2D_in_allhardware >> OutputOfAllHardwareTests.txt 
cd ../matrixmult
python3 ../../ppt.py MMULT_in_allhardware >> OutputOfAllHardwareTests.txt 
cd ../snap
python3 ../../ppt.py SNAP_in_allhardware >> OutputOfAllHardwareTests.txt 
cd ..

date
echo " Finished running all apps with allhardware! "
echo "     Check each subdirectory for an OoutputOfAllHardwareTests.txt for outputs and warnings of each test. "
