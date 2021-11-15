#ifndef _PARDA_H
#define _PARDA_H

#include "narray.h"
#include "process_args.h"
#include "splay.h"

#include <assert.h>
#include <glib.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <unistd.h>

/*
  An implementation of Parda, a fast parallel algorithm
to compute accurate reuse distances by analysis of reference
traces.
  Qingpeng Niu
*/

#ifdef enable_mpi
#ifdef enable_omp
#define enable_hybrid
#endif
#endif

//#define enable_debugging
#ifdef enable_debugging
#define DEBUG(cmd) cmd
#else
#define DEBUG(cmd)
#endif

#ifdef enable_profiling
#define PROF(cmd) cmd
#else
#define PROF(cmd)
#endif

//#define DEFAULT_NBUCKETS 100000
#define DEFAULT_NBUCKETS 1000000
#define B_OVFL   nbuckets
#define B_INF    nbuckets+1
#define SLEN 20
/* Tunable parameters */
extern int nbuckets;
#ifdef ENABLE_PROFILING
extern char pfile[30];
extern FILE* pid_fp;
#endif
/*data structure for parda*/
typedef char HKEY[SLEN];

typedef struct end_keytime_s {
		narray_t* gkeys;
		narray_t* gtimes;
}end_keytime_t;

typedef struct processor_info_s {
	int pid,psize;
	long tstart,tlen,tend,sum;
}processor_info_t;

typedef struct program_data_s {
  GHashTable* gh;
	narray_t* ga;
	end_keytime_t ekt;
  Tree* root;
  unsigned int *histogram;
}program_data_t;


int lmem_used;
int shared_l2;

//void classical_tree_based_stackdist(char* inputFileName, long lines);
void classical_tree_based_stackdist(char* inputFileName, int kernel_id, int sm_id, long lines, long assoc, char* output_directory);
/*functions for glib*/
gboolean compare_strings(gconstpointer a, gconstpointer b);
void iterator(gpointer key, gpointer value, gpointer ekt);

/*functions for parda core*/
program_data_t parda_init(void);
void parda_input_with_filename(char* inFileName, int kernel_id, int sm_id, long assoc, program_data_t* pdt_umem, program_data_t* pdt_gmem_lds, program_data_t* pdt_gmem, program_data_t* pdt_lmem, long begin, long end, char* output_directory);
void parda_input_with_textfilepointer(FILE* fp, int kernel_id, int sm_id, long assoc, program_data_t* pdt_umem, program_data_t* pdt_gmem_lds, program_data_t* pdt_gmem, program_data_t* pdt_lmem, long begin, long end, char* output_directory);
void parda_input_with_binaryfilepointer(FILE* fp, program_data_t* pdt, long begin, long end);
void parda_free(program_data_t* pdt);
end_keytime_t parda_generate_end(const program_data_t* pdt);
processor_info_t parda_get_processor_info(int pid, int psize, long sum);
void parda_get_abfront(program_data_t* pdt_a, const narray_t* gb, const processor_info_t* pit_a);
int parda_get_abend(program_data_t* pdt_b, const end_keytime_t* ekt_a);
program_data_t parda_merge(program_data_t* pdt_a, program_data_t* pdt_b,
				const processor_info_t* pit_b);
/*functions for parda print*/
void parda_print_front(const program_data_t* pdt);
void parda_print_end(const end_keytime_t* ekt);
void parda_print_tree(const program_data_t* pdt);
void parda_print_hash(const program_data_t* pdt);
void parda_print(const program_data_t* pdt);
void print_iterator(gpointer key, gpointer value, gpointer ekt);
void parda_print_histogram(FILE* fp, const unsigned* histogram);

int parda_findopt(char *option, char **value, int *argc, char ***argv);
void parda_process(char* input, T tim, program_data_t* pdt);
/*functions for mpi communication*/
void show_hkey(void* data, int i, FILE* fp);
void show_T(void* data, int i, FILE* fp);
/*functions for clock*/
double rtclock(void);

/*parda inline functions*/
static inline T parda_low(int pid, int psize, T sum) {
	return (((long long)(pid))*(sum)/(psize));
}

static inline T parda_high(int pid, int psize, T sum) {
		return parda_low(pid + 1, psize, sum)-1;
}

static inline T parda_size(int pid, int psize, T sum) {
		return (parda_low(pid + 1, psize, sum)) - (parda_low(pid, psize, sum));
}

static inline T parda_owner(T index, int psize, T sum) {
		return (((long long)psize)*(index+1)-1)/sum;
}

static inline char* parda_generate_pfilename(char filename[], int pid, int psize) {
	char pfilename[30];
	sprintf(pfilename, "%d_%s_p%d.txt", psize, filename, pid);
	return strdup(pfilename);
}



static inline void process_one_access(char* inst, char* mem_id, char* warp_id, char* input, long assoc, program_data_t* pdt_umem, program_data_t* pdt_gmem_lds, program_data_t* pdt_gmem, program_data_t* pdt_lmem, const long tim) {

	int distance;
	int *lookup;

	if (lmem_used){
		
		if (strcmp(mem_id,"0") == 0){ /*global memory*/

			lookup = g_hash_table_lookup(pdt_gmem->gh, input);
			// Cold start: Not in the list yet
			if (lookup == NULL) {
				char *data = strdup(input);
				pdt_gmem->root=insert(tim,pdt_gmem->root);
				long *p_data;
				narray_append_val(pdt_gmem->ga,input);
				if ( !(p_data = (long*)malloc(sizeof(long))) )
				{
					printf("no memory for p_data\n");assert(0);exit(-1);
				}
				*p_data = tim;
				g_hash_table_insert(pdt_gmem->gh, data, p_data);  // Store pointer to list element
			}
			// Hit: We've seen this data before
			else {
				char *data = strdup(input);
				pdt_gmem->root = insert((*lookup), pdt_gmem->root);
				distance = node_size(pdt_gmem->root->right);
				pdt_gmem->root = delete(*lookup, pdt_gmem->root);
				pdt_gmem->root = insert(tim, pdt_gmem->root);
				int *p_data;
				if ( !(p_data = (int*)malloc(sizeof(int)))) {
					printf("no memory for p_data\n");
					assert(0); 
					exit(-1);
				}
				*p_data = tim;
				g_hash_table_replace(pdt_gmem->gh, data, p_data);
				// Is distance greater than the largest bucket
				if (distance > nbuckets)
					pdt_gmem->histogram[B_OVFL] += 1;
				else
					pdt_gmem->histogram[distance] += 1;
			}

		}else{ /*local memory*/

			lookup = g_hash_table_lookup(pdt_lmem->gh, input);
			// Cold start: Not in the list yet
			if (lookup == NULL) {
				char *data = strdup(input);
				pdt_lmem->root=insert(tim,pdt_lmem->root);
				long *p_data;
				narray_append_val(pdt_lmem->ga,input);
				if ( !(p_data = (long*)malloc(sizeof(long))) )
				{
					printf("no memory for p_data\n");assert(0);exit(-1);
				}
				*p_data = tim;
				g_hash_table_insert(pdt_lmem->gh, data, p_data);  // Store pointer to list element
			}
			// Hit: We've seen this data before
			else {
				char *data = strdup(input);
				pdt_lmem->root = insert((*lookup), pdt_lmem->root);
				distance = node_size(pdt_lmem->root->right);
				pdt_lmem->root = delete(*lookup, pdt_lmem->root);
				pdt_lmem->root = insert(tim, pdt_lmem->root);
				int *p_data;
				if ( !(p_data = (int*)malloc(sizeof(int)))) {
					printf("no memory for p_data\n");
					assert(0); 
					exit(-1);
				}
				*p_data = tim;
				g_hash_table_replace(pdt_lmem->gh, data, p_data);
				// Is distance greater than the largest bucket
				if (distance > nbuckets)
					pdt_lmem->histogram[B_OVFL] += 1;
				else
					pdt_lmem->histogram[distance] += 1;
			}
		}
	}

	/*gmem memory --> load inst*/
	if (strcmp(mem_id,"0") == 0 && strcmp(inst,"0") == 0){ 
		
		lookup = g_hash_table_lookup(pdt_gmem_lds->gh, input);
		// Cold start: Not in the list yet
		if (lookup == NULL) {
			char *data = strdup(input);
			pdt_gmem_lds->root=insert(tim,pdt_gmem_lds->root);
			long *p_data;
			narray_append_val(pdt_gmem_lds->ga,input);
			if ( !(p_data = (long*)malloc(sizeof(long))) )
			{
				printf("no memory for p_data\n");assert(0);exit(-1);
			}
			*p_data = tim;
			g_hash_table_insert(pdt_gmem_lds->gh, data, p_data);  // Store pointer to list element
		}
		// Hit: We've seen this data before
		else {
			char *data = strdup(input);
			pdt_gmem_lds->root = insert((*lookup), pdt_gmem_lds->root);
			distance = node_size(pdt_gmem_lds->root->right);
			pdt_gmem_lds->root = delete(*lookup, pdt_gmem_lds->root);
			pdt_gmem_lds->root = insert(tim, pdt_gmem_lds->root);
			int *p_data;
			if ( !(p_data = (int*)malloc(sizeof(int)))) {
				printf("no memory for p_data\n");
				assert(0); 
				exit(-1);
			}
			*p_data = tim;
			g_hash_table_replace(pdt_gmem_lds->gh, data, p_data);
			// Is distance greater than the largest bucket
			if (distance > nbuckets)
				pdt_gmem_lds->histogram[B_OVFL] += 1;
			else
				pdt_gmem_lds->histogram[distance] += 1;
		}
	}
	
	/*all accesses --> umem memory*/
	lookup = g_hash_table_lookup(pdt_umem->gh, input);
	// Cold start: Not in the list yet
	if (lookup == NULL) {
		char *data = strdup(input);
		pdt_umem->root=insert(tim,pdt_umem->root);
		long *p_data;
		narray_append_val(pdt_umem->ga,input);
		if ( !(p_data = (long*)malloc(sizeof(long))) )
		{
			printf("no memory for p_data\n");assert(0);exit(-1);
		}
		*p_data = tim;
		g_hash_table_insert(pdt_umem->gh, data, p_data);  // Store pointer to list element
	}
	// Hit: We've seen this data before
	else {
		char *data = strdup(input);
		pdt_umem->root = insert((*lookup), pdt_umem->root);
		distance = node_size(pdt_umem->root->right);
		pdt_umem->root = delete(*lookup, pdt_umem->root);
		pdt_umem->root = insert(tim, pdt_umem->root);
		int *p_data;
		if ( !(p_data = (int*)malloc(sizeof(int)))) {
			printf("no memory for p_data\n");
			assert(0); 
			exit(-1);
		}
		*p_data = tim;
		g_hash_table_replace(pdt_umem->gh, data, p_data);
		// Is distance greater than the largest bucket
		if (distance > nbuckets)
			pdt_umem->histogram[B_OVFL] += 1;
		else
			pdt_umem->histogram[distance] += 1;
	}
}



static inline void process_one_access_shared(char* input, program_data_t* pdt, const long tim) {
	int distance;
	int *lookup;
	lookup = g_hash_table_lookup(pdt->gh, input);
    //printf("gh=%p process_one\n",pdt->gh);
	// Cold start: Not in the list yet
	if (lookup == NULL) {
		char *data = strdup(input);
		pdt->root=insert(tim,pdt->root);
		long *p_data;
        narray_append_val(pdt->ga,input);
		if (!(p_data = (long*)malloc(sizeof(long)))){
			printf("no memory for p_data\n");
			assert(0);
			exit(-1);}
		*p_data = tim;
		g_hash_table_insert(pdt->gh, data, p_data);  // Store pointer to list element
	}
	// Hit: We've seen this data before
	else {
		char *data = strdup(input);
		pdt->root = insert((*lookup), pdt->root);
		distance = node_size(pdt->root->right);
		pdt->root = delete(*lookup, pdt->root);
		pdt->root = insert(tim, pdt->root);
		int *p_data;
		if (!(p_data = (int*)malloc(sizeof(int)))){
			printf("no memory for p_data\n");
			assert(0); 
			exit(-1);}
			*p_data = tim;
			g_hash_table_replace(pdt->gh, data, p_data);
			// Is distance greater than the largest bucket
			if (distance > nbuckets)
				pdt->histogram[B_OVFL] += 1;
			else
				pdt->histogram[distance] += 1;
	}
}
#endif
