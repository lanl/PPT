#include "narray.h"

narray_t* narray_new(unsigned element_size, unsigned capacity) {
    narray_t* na = (narray_t*)malloc(sizeof(narray_t));
    na->element_size = element_size;
    na->len = 0;
    na->capacity = capacity * element_size;
    na->data = calloc(capacity, element_size);
    return na;
}

void narray_append_val(narray_t* na, const void* value) {
   if(na->len == na->capacity) {
       unsigned new_capacity = na->capacity + na->capacity + 10 * na->element_size;
       void* ndata = calloc(new_capacity, 1);
       memcpy(ndata, na->data, na->len);
       free(na->data);
       na->data = ndata;
       na->capacity = new_capacity;
   }
   memcpy(na->data + na->len, value, na->element_size);
	 na->len += na->element_size;
}

void narray_free(narray_t* na) {
		free(na->data);
		free(na);
}

void narray_print(narray_t* na, void (*show_element)(void*, int, FILE*), FILE* fp) {
  mdebug(fprintf(fp, "enter narray_print len=%u\n",na->len);)
  unsigned len = narray_get_len(na);
  int i;
	for (i = 0; i < len; i++) {
      show_element(na->data, i, fp);
			mdebug(printf("%s ", ((HKEY*)ga->data)[i]);)
	}
}

narray_t* narray_heaparray_new(void* data, const unsigned len, const unsigned element_size) {
    narray_t* na = (narray_t*)malloc(sizeof(narray_t));
    na->data = data;
    na->len = len;
    na->capacity = len;
    na->element_size = element_size;
    return na;
}
