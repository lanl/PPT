#!/bin/bash

rm -rf build *.bc *.o

mkdir -p build
cd build
# cmake ..
# Sometimes its needed to point to the correct compiler
cmake -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ ..
make && printf '>>>>> Pass Created! <<<<<\n\n' || exit
cd ..
# clang -emit-llvm -c simple.cpp
# clang -emit-llvm -c adder.cpp
# llvm-link adder.bc simple.bc -o source-compiled.bc
# clang -fno-discard-value-names -Xclang -load -Xclang ../Basic-Block-Counter/build/counter-pass/libBBCounterPass.so -c source-compiled.bc -o source-inst.o
printf ">>>>> Compiling runtime library bb-counter-rt.c* <<<<<\n"
clang -c bb-counter-rt.c* && printf ">>>>> Success <<<<<\n\n" || exit
if [ "$#" -lt 1 ]; then
    echo "You must pass target program file name: ./build.sh filename"
else
    echo ">>>>> Instrumenting input source files <<<<<"
    for i in "$@"    # same as your "for i"
    do
        echo "$i"
        clang -emit-llvm -c $i
    done
    echo ">>>>> Linking all the bitcode files <<<<<"
    llvm-link *.bc -o source-compiled.bc
    clang -fno-discard-value-names -Xclang -load -Xclang ../Basic-Block-Counter/build/counter-pass/libBBCounterPass.so -c source-compiled.bc -o source-inst.o
    clang source-inst.o bb-counter-rt.o -o bb_count_instrumented.out
fi
echo ">>>>>Done!<<<<<"
