#include "adder.h"

int adder(int limit)
{
    int num = 0;
    for(int i = 1; i <= limit; i++)
    {
        num += i;
    }
    for(int i = 1; i <= limit; i++)
    {
        num += i;
    }
    return num;
}