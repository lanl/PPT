#!/bin/bash

for ni in {2..20}
do
    for nj in {2..20}
    do
        for nk in {2..20}
        do
            for nl in {2..20}
            do
                clang -emit-llvm -I ../../../utilities "-DNI="$ni "-DNJ="$nj "-DNK="$nk "-DNL="$nl -c *.c
                llvm-link *.bc ../../../utilities/polybench.bc -o source-compiled.bc
                clang -fno-discard-value-names -Xclang -load -Xclang ./build/counter-pass/libBBCounterPass.so -c source-compiled.bc -o source-inst.o
                clang source-inst.o ../../../../Basic-Block-Counter/bb-counter-rt.o -o bb_count_instrumented.out
                echo -n "$ni $nj $nk $nl " >> basic-block-counts.txt
                ./bb_count_instrumented.out
                rm source-compiled.bc
            done
        done
    done
done
