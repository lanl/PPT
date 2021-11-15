#ifndef _NARRAY_H
#define _NARRAY_H

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#ifdef enable_mdebugging
#define mdebug(cmd) cmd
#else
#define mdebug(cmd)
#endif

typedef struct narray_s {
    void* data;
    unsigned len, capacity, element_size;
}narray_t;

narray_t* narray_heaparray_new(void* data, unsigned len, unsigned element_size);
narray_t* narray_new(unsigned element_size, unsigned capacity);
void narray_append_val(narray_t* na, const void* value);
void narray_free(narray_t* na);
void narray_print(narray_t* na, void (*show_element)(void*, int, FILE*), FILE* fp);

static inline unsigned narray_get_len(const narray_t* na) {
    return na->len/na->element_size;
}

#endif
