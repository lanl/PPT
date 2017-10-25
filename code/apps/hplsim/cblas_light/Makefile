CC = clang
#CC = bf-clang -bf-strides -bf-vectors -bf-types -bf-by-func -bf-inst-mix
CFLAGS = -O0 -Wall
#CFLAGS = -O3 -Wall
DEPS = blas0.h blas2.h blas3.h
OBJ = blas_test.o blas3.o blas2.o blas0.o

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) -c -o $@ $<

blas_test.exe: $(OBJ)
	$(CC) $(CFLAGS) -lm -o $@ $^

clean:
	rm -f *.o *~ core *.exe