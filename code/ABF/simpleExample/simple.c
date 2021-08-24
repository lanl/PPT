#include <stdio.h>

#define COUNT 1000

long sum(long A[], long B[], long C[], long count) {
    long i, j, sum=0;
    for (i=0; i<count; i++) {
        sum += A[i]*A[i+1];
        C[i] = B[i] + A[i];
        for (j=0; j<count; j++) sum += i*j;
    }
    return sum;
}

int main() {
    long i;
    static long A[COUNT+1];
    static long B[COUNT+1];
    static long C[COUNT+1];
    for (i=0; i<COUNT+1; i++) A[i] = 20;
    printf("%ld\n", sum(A, B, C, COUNT));
}
