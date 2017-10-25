#Where is LLVM and Clang installed?
LLVM_DIR = /Users/nsanthi/Work/HWSW/LLVM-Tracer/x86-64

bcread: bcRead.cpp
	clang++ -O3 -o $@ `${LLVM_DIR}/bin/llvm-config --cxxflags --libs --ldflags --system-libs` $^

clean:
	rm -f bcread
