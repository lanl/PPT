#DEBUG = 1
#OMP = 1
#MPI = 1

BASE=gcc


CFLAGS += -Wall 
ifdef DEBUG
CFLAGS+= -g -O0
else
CFLAGS+= -O2
endif
CFLAGS += $(shell pkg-config --cflags glib-2.0)
LIBS    = $(shell pkg-config --libs glib-2.0 --libs gthread-2.0)
OBJS+= main.o splay.o parda.o parda_print.o narray.o process_args.o seperate.o
HEADERS= splay.h parda.h narray.h process_args.h seperate.h

ifdef OMP
OBJS+= parda_omp.o
HEADERS+= parda_omp.h
CFLAGS+=-Denable_omp
ifeq ($(BASE),icc)
CFLAGS+=-openmp
else
CFLAGS+=-fopenmp
endif
endif

ifeq ($(BASE),icc)
CFLAGS+=-limf
endif

SOURCES=$(subst .o,.c, $(OBJS) )
EXE=parda.x
.PHONY: all clean gnuplots run
all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $+ $(LIBS)
$(OBJS):$(HEADERS) makefile
%.d: %.c
	set -e; rm -f $@; \
	$(CC) -M $(CPPFLAGS) $< > $@.$$$$; \
        sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
        rm -f $@.$$$$
include $(sources:.c=.d)
clean:
	rm -f $(EXE) *.o 
run:
