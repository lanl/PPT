#include <stdio.h>

#include "adder.h"

int multiplier(int limit)
{
    int num = 1;
    for(int i = 1; i <= limit; i++)
    {
        num *= i;
    }
    return num;
}


int main(int argc, const char** argv)
{
    unsigned sum = 0;
    printf("Adder result: %d\n", adder(200));
    printf("Multiplier result: %d\n", multiplier(10));
    for(int i = 0; i < 5; i++)
    {
        sum += i*i - i;
    }
    printf("Sum: %d\n", sum);
    return 0;
}
