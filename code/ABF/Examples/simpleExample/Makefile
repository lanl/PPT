CLANG ?= $(HOME)/Work/HWSW/LLVM-Tracer/x86-64/bin/clang
#DBG ?= -g -gcolumn-info
DBG =

all: simple.O0.llvm simple.O3.llvm simple

simple.O0.llvm: simple.c
	$(CLANG) -O0 $(DBG) -c -S -emit-llvm -o $@ $^

simple.O3.llvm: simple.c
	$(CLANG) -O3 $(DBG) -c -S -emit-llvm -o $@ $^

simple: simple.c
	$(CLANG) -O3 -o $@ $^

clean:
	rm -f simple.O0.llvm simple.O3.llvm simple
