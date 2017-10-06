#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
/* Number of bits dedicated to each dimension
   ie. max height of the octree
   21 allows keys to fit in 64 bits (3*21+1 placeholder bit)
   31 means 94 bit keys ie. 2 long ints
*/
#if LONG_KEYS
#define NB_BITS 42
#else
#define NB_BITS 31
#endif
double X,Y,Z;

void set_sizes(double x, double y, double z)
{
    X = x;
    Y = y;
    Z = z;
}
#if LONG_KEYS
void print_bitset(uint64_t bitset[2])
{
    uint64_t mask=(1UL<<62);
    int i;
    for(i=62; i>=0;--i)
    {
        int tmp=0;
        if(bitset[0]&mask)
        {
            tmp=1;
        }    
        printf("%d", tmp);
        mask>>=1;
    }
    mask=(1UL<<63);
    for(i=63; i>=0;--i)
    {       
        int tmp=0;
        if(bitset[1]&mask)
        {
            tmp=1;
        }    
        printf("%d", tmp);
        mask>>=1;
    }
}
void hash(
    uint64_t bitset[2],
    double x, 
    double y, 
    double z
    )
{
  uint64_t bins[3] = {(uint64_t) ((x/X)*((double)(1UL<<NB_BITS))),
                      (uint64_t) ((y/Y)*((double)(1UL<<NB_BITS))),
                      (uint64_t) ((z/Z)*((double)(1UL<<NB_BITS)))};                  
  uint64_t mask = 1;
  int i;
  for(i=0; i<21;++i)
  {
    bitset[1] |= (bins[0]&mask)<<(2*i);
    bitset[1] |= (bins[1]&mask)<<(2*i+1);
    bitset[1] |= (bins[2]&mask)<<(2*i+2);
    mask<<=1;
  }
  bitset[1] |= (bins[0]&mask)<<42;
  bitset[0] |= (bins[1]&mask)>>21;
  bitset[0] |= (bins[2]&mask)>>20;
  for(i=0; i<10;++i)
  {
    mask <<= 1;
    bitset[0] |= (bins[0]&mask)>>(20-2*i);
    bitset[0] |= (bins[1]&mask)>>(20-(2*i+1));
    bitset[0] |= (bins[2]&mask)>>(20-(2*i+2));
  }
  for(i=0; i<10;++i)
  {
    mask <<= 1;
    bitset[0] |= (bins[0]&mask)<<(2*i);
    bitset[0] |= (bins[1]&mask)<<(2*i+1);
    bitset[0] |= (bins[2]&mask)<<(2*i+2);
  }
  bitset[0] |= (1UL<<62 );
}
#else
void print_bitset(uint64_t bitset[2])
{
    uint64_t mask=(1UL<<29);
    int i;
    for(i=29; i>=0;--i)
    {
        int tmp=0;
        if(bitset[0]&mask)
        {
            tmp=1;
        }    
        printf("%d", tmp);
        mask>>=1;
    }
    mask=(1UL<<63);
    for(i=63; i>=0;--i)
    {       
        int tmp=0;
        if(bitset[1]&mask)
        {
            tmp=1;
        }    
        printf("%d", tmp);
        mask>>=1;
    }
}

void hash(
    uint64_t bitset[2],
    double x, 
    double y, 
    double z
    )
{
  uint64_t bins[3] = {(uint64_t) ((x/X)*(1UL<<NB_BITS)),
                      (uint64_t) ((y/Y)*(1UL<<NB_BITS)),
                      (uint64_t) ((z/Z)*(1UL<<NB_BITS))};
  uint64_t mask = 1;
  int i;
  for(i=0; i<21;++i)
  {
    bitset[1] |= (bins[0]&mask)<<(2*i);
    bitset[1] |= (bins[1]&mask)<<(2*i+1);
    bitset[1] |= (bins[2]&mask)<<(2*i+2);
    mask<<=1;
  }
  bitset[1] |= (bins[0]&mask)<<42;
  bitset[0] |= (bins[1]&mask)>>21;
  bitset[0] |= (bins[2]&mask)>>20;
  for(i=0; i<9;++i)
  {
    mask <<= 1; 
    bitset[0] |= (bins[0]&mask)>>(20-2*i);
    bitset[0] |= (bins[1]&mask)>>(20-(2*i+1));
    bitset[0] |= (bins[2]&mask)>>(20-(2*i+2));
  }
  bitset[0] |= (1UL<<29);
}
#endif
bool equals(uint64_t key1[2], uint64_t key2[2])
{
    return key1[1]==key2[1] && key1[0]==key2[0];
}

bool lt(uint64_t key1[2], uint64_t key2[2])
{
    if(key1[0]==key2[0])
    {
        return key1[1]<key2[1];
    }
    return key1[0]<key2[0];
}

bool gt(uint64_t key1[2], uint64_t key2[2])
{
    if(key1[0]==key2[0])
        return key1[1]>key2[1];
    return key1[0]>key2[0];
}
void rshift(uint64_t res[2], uint64_t key[2], int shift)
{
    signed int diff = 64-shift;
    if (diff == 0) 
    {
        res[0]=key[0];
        res[1]=key[1];
    } 
    else if (diff > 0) 
    {
        res[1] = (key[1] >> shift) | (key[0] << diff);
        res[0] = key[0] >> shift;
    }
    else 
    {
        res[1] = key[0] >> -diff;
        res[0] = 0;
    }
}
void lshift(uint64_t res[2], uint64_t key[2], int shift)
{
    signed int diff = 64-shift;

    if (diff == 0) 
    {
        res[0]=key[0];
        res[1]=key[1];
    } 
    else if (diff > 0) 
    {
        res[0] = (key[0] << shift) | (key[1] >> diff);
        res[1] = key[1] << shift;
    } 
    else 
    {
        res[0] = key[1] << -diff;
        res[1] = 0;
    }
}
void get_daughter(uint64_t res[2], uint64_t key[2], uint64_t daughter)
{
    lshift(res, key,3);
    res[1] = res[1] | daughter;
}

bool is_one(uint64_t key[2])
{
    return key[1]==0x1 && key[0]==0;
}
void create_root(uint64_t res[2])
{
    res[0] = 0;
    res[1] = 1ULL;
}
int bor(int val1, int val2)
{
    return val1|val2;
}

int band(int val1, int val2)
{
    return val1&val2;
}
int get_short_hash(uint64_t val[2])
{
    return val[1]&0x7fff;
}
/*int main ( int arc, char **argv ) {
    double x,y,z;
    uint64_t bitset[2] = {0,0};
    x=2097151.999999999;
    y=2097151.999999999;
    z=2097151.999999999;
    set_sizes(2097152UL,2097152UL,2097152UL);
    hash(bitset,x,y,z);
    print_bitset(bitset);
    return 0;
}*/