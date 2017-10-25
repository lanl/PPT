Author: Nandakishore Santhi
Date: Nov 2016

Average Bracketed Form (ABF) using the following as inputs:

1. LLVM-IR static trace with debug information:
cd ../IRStaticTrace; make
path/to/clang -g -gcolumn-info -c -S -emit-llvm -o blackscholes.llvm blackscholes.c
./bcread blackscholes.llvm > blackscholes.trace

2. llvm-cov branch probability coverage using LLVM/Clang 3.5.2:
path/to/clang blackscholes.c --coverage -o blackscholes #Generates blackscholes.gcno
./blackscholes 1 ../inputs/in_16.txt out.txt #Generates blackscholes.gcda
path/to/llvm-cov -b blackscholes.c #Generates blackscholes.c.gcov

3. To use parseTrace.lua on the generated static trace and using branch probability estimates (either from 2 above or other knowledge of code):
In this directory do:

luajit parseTrace.lua ./SimpleLoop/simple_loop.trace Out/
This produces:
$> ll Out/
rw-r--r--  1 nsanthi  staff  3211 Jan 17 10:25 module_1.lua
-rw-r--r--  1 nsanthi  staff   213 Jan 17 10:25 probabilities_module_1.lua

Now copy:
cp Out/prob_module_1.lua prob_simple_loop.lua
ln -sf prob_simple_loop.lua probabilities.lua

Then adjust the branch probability values in probabilities.lua.
Now run:

luajit Out/module_1.lua

which should output to stdout the final basic-block apriori probabilities, and runtime instruction counts.
