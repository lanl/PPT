# source files.
SRC =   BitOp.c
 
# C++ compiler flags
CCFLAGS = -fPIC -O0 -Wall -g -ggdb -G
 
# compiler
CC = gcc

OUT := libOp.so
OBJ := BitOp.o 
default: $(OUT)
 
.c.o:
	$(CC) $(INCLUDES) $(CCFLAGS) -c $< -o $@

libOp.so: BitOp.o
	$(CC) -shared -g -ggdb $< -o $@

clean: ; rm -f $(OBJ) $(OUT) Makefile.bak
