Author: Nandakishore Santhi
Date: Nov 2016

Ported to LLVM 10.0.0 by Atanu Barai
Date: August 2021

Average Bracketed Form (ABF) using the following as inputs:

1. LLVM-IR static trace with debug information:
cd ../IRStaticTrace; make
cd ../simpleExample
clang -g -gcolumn-info -c -S -emit-llvm -o simple.llvm simple.c
../IRStaticTrace/bcread simple.llvm > simple.trace

2. llvm-cov branch probability coverage using LLVM/Clang 10.0.0 (optional):
clang simple.c --coverage -o simple #Generates simple.gcno
./simple #Generates simple.gcda
llvm-cov gcov -b simple.c #Generates simple.c.gcov

3. To use parseTrace.lua on the generated static trace and using branch probability estimates (either from 2 above or other knowledge of code):
In this ABF directory do:

luajit parseTrace.lua ./simpleExample/simple.trace Out/
This produces:
$> ls Out/
module_1.lua probabilities_module_1.lua

Now:
cp Out/probabilities_module_1.lua prob_simple_loop.lua
ln -sf prob_simple_loop.lua probabilities.lua

Then manually adjust the branch probability values in probabilities.lua. In this file you will find
the line numbers in the code to look for the probabilities. The probabilities can also be found using step 2.
Now run:

luajit Out/module_1.lua > simpleExample/simple.analysis

which should output to stdout the final basic-block apriori probabilities, counts and instruction counts.
