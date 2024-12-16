A LLVM pass to count number of times each basic block is executed.
It's for LLVM 10.

Build:

    $ cd Basic-Block-Counter
    $ ./build.sh simple.cpp adder.cpp
Run:
    $ ./bb_count_instrumented.out
