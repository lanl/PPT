#include "parda.h"
#include "narray.h"

#include<omp.h>

/*
  An implementation of Parda, a fast parallel algorithm
to compute accurate reuse distances by analysis of reference
traces.
  Qingpeng Niu
*/

int nbuckets  = DEFAULT_NBUCKETS;

void iterator(gpointer key, gpointer value, gpointer ekt) {
	HKEY temp;
	strcpy(temp, key);
	narray_append_val(((end_keytime_t*)ekt)->gkeys, temp);
	narray_append_val(((end_keytime_t*)ekt)->gtimes, value);
}

end_keytime_t parda_generate_end(const program_data_t* pdt) {
	GHashTable *gh = pdt->gh;
	end_keytime_t ekt;
	ekt.gkeys = narray_new(sizeof(HKEY), 1000);
	ekt.gtimes = narray_new(sizeof(T), 1000);
	g_hash_table_foreach(gh, (GHFunc)iterator, &ekt);

	return ekt;
}

program_data_t parda_init() {
	program_data_t pdt;
	GHashTable *gh;
	narray_t* ga = narray_new(sizeof(HKEY), 1000);
	Tree* root;
	unsigned int *histogram;
	histogram = malloc(sizeof(unsigned int) * (nbuckets+2));
	gh = g_hash_table_new_full(g_str_hash, compare_strings, free, free);
	root = NULL;
	memset(histogram, 0, (nbuckets + 2) * sizeof(unsigned int));
	pdt.ga = ga;
	pdt.gh = gh;
	pdt.root = root;
	pdt.histogram = histogram;

	return pdt;
}

gboolean compare_strings(gconstpointer a, gconstpointer b) {
  if (strcmp(a,b) == 0)
    return TRUE;
  else
    return FALSE;
}

void parda_process(char* input, T tim, program_data_t* pdt) {
	GHashTable *gh = pdt->gh;
	Tree* root = pdt->root;
	narray_t* ga = pdt->ga;
	unsigned int *histogram = pdt->histogram;
	int distance;
	T *lookup;
	lookup = g_hash_table_lookup(gh, input);
	if (lookup == NULL) {
		char* data = strdup(input);
		root = insert(tim, root);
		T *p_data;

		narray_append_val(ga, input);
		if (!(p_data = (T*)malloc(sizeof(T)))) exit(1);
		*p_data = tim;
		g_hash_table_insert(gh, data, p_data);  // Store pointer to list element
	}

	// Hit: We've seen this data before
	else {
		root = insert((*lookup), root);
		distance = node_size(root->right);
		root = delete(*lookup, root);
		root = insert(tim, root);
		int *p_data;
		if (!(p_data = (int*)malloc(sizeof(int)))) exit(1);
		*p_data = tim;
		g_hash_table_replace(gh, strdup(input), p_data);

		// Is distance greater than the largest bucket
		if (distance > nbuckets)
			histogram[B_OVFL]++;
		else
			histogram[distance]++;
	}
	pdt->root = root;
}

processor_info_t parda_get_processor_info(int pid, int psize, long sum) {
	processor_info_t pit;
	pit.tstart = parda_low(pid, psize, sum);
	pit.tend = parda_high(pid, psize, sum);
	pit.tlen = parda_size(pid, psize, sum);
	pit.sum = sum;
	pit.pid = pid;
	pit.psize = psize;
	return pit;
}

void parda_get_abfront(program_data_t* pdt_a, const narray_t* gb, const processor_info_t* pit_a) {
	//printf("enter abfront and before for loopheihei\n");
	T tim = pit_a->tend + 1;
	GHashTable* gh = pdt_a->gh;
	narray_t* ga = pdt_a->ga;
	Tree* root = pdt_a->root;
	unsigned* histogram = pdt_a->histogram;
	int i;
	T *lookup;
	T distance;
	unsigned len = narray_get_len(gb);
	//printf("enter abfront and before for loop\n");
	for(i=0; i < len; i++, tim++) {
		HKEY entry;
		char* temp=((HKEY*)gb->data)[i];
		strcpy(entry, temp);
		lookup = g_hash_table_lookup(gh, entry);
		//printf("merge entry %s\n",entry);
		if(lookup==NULL) {
			narray_append_val (ga, entry);
			root = insert(tim, root);
		} else {
			root = insert((*lookup), root);
			distance = node_size(root->right);
			root = delete(*lookup, root);
			root = insert(tim, root);
			if (distance > nbuckets)
				histogram[B_OVFL]++;
			else
				histogram[distance]++;
		}
	}
	pdt_a->root = root;
	pdt_a->gh = gh;
}

int parda_get_abend(program_data_t* pdt_b,
	const	end_keytime_t* ekt_a ) {
	Tree* root = pdt_b->root;
	GHashTable* gh = pdt_b->gh;
	narray_t* gkeys = ekt_a->gkeys;
	narray_t* gtimes = ekt_a->gtimes;
	unsigned len = narray_get_len(gkeys);
	unsigned i;
	HKEY key;
	T tim;
	T* lookup;
	for (i = 0; i < len; i++) {
		char* temp = ((HKEY*)gkeys->data)[i];
		strcpy(key, temp);
		tim = ((T*)(gtimes->data))[i];
		lookup = g_hash_table_lookup(gh, key);
		if (lookup == NULL) {
			char* data = strdup(key);
			root = insert(tim,root);
			T *p_data;
			if ( !(p_data = (T*)malloc(sizeof(T))) ) return -1;
			*p_data = tim;
			g_hash_table_insert(gh, data, p_data);
		}
	}
	pdt_b->root = root;
	pdt_b->gh = gh;

	return 0;
}

program_data_t parda_merge(program_data_t* pdt_a, program_data_t* pdt_b,
	const processor_info_t* pit_b) {
	program_data_t pdt;
	parda_get_abfront(pdt_a, pdt_b->ga, pit_b);
	DEBUG(printf("after get abfront %d\n", pit_b->pid);)
	narray_free(pdt_b->ga);
	pdt_a->ekt = parda_generate_end(pdt_a);
	parda_get_abend(pdt_b, &pdt_a->ekt);
	narray_free(pdt_a->ekt.gkeys);
	narray_free(pdt_a->ekt.gtimes);
	pdt.ga = pdt_a->ga;
	pdt.root = pdt_b->root;
	pdt.gh = pdt_b->gh;
	pdt.histogram = pdt_a->histogram;
	int i;
	for (i = 0; i < nbuckets+2; i++)
		pdt.histogram[i] += pdt_b->histogram[i];
	free(pdt_b->histogram);

	return pdt;
}


void classical_tree_based_stackdist(char* inputFileName, int kernel_id, int sm_id, long lines, long assoc, char* output_directory){
	
	char umem_file[200], gmem_file_lds[200], gmem_file[200], lmem_file[200];
	program_data_t pdt_c_umem, pdt_c_gmem_lds, pdt_c_gmem, pdt_c_lmem;

	if(shared_l2 == 0){
		pdt_c_umem = parda_init();
		pdt_c_gmem_lds = parda_init();

		if(lmem_used){
			pdt_c_gmem = parda_init();
			pdt_c_lmem = parda_init();
		}
	}
	else{
		pdt_c_umem = parda_init();
	}
	
    parda_input_with_filename(inputFileName, kernel_id, sm_id, assoc, &pdt_c_umem, &pdt_c_gmem_lds, &pdt_c_gmem, &pdt_c_lmem, 0, lines - 1, output_directory);
	program_data_t* pdt_umem = &pdt_c_umem;
	pdt_umem->histogram[B_INF] += narray_get_len(pdt_umem->ga);
	if(shared_l2 == 0){
		sprintf(umem_file, "%s/K%d_UMEM_SM%d.rp", output_directory, kernel_id, sm_id);
		sprintf(gmem_file_lds, "%s/K%d_GMEM_SM%d_lds.rp", output_directory, kernel_id, sm_id);
		FILE* umem_fp = fopen(umem_file, "w");
		FILE* gmem_fp_lds = fopen(gmem_file_lds, "w");
		program_data_t* pdt_gmem_lds = &pdt_c_gmem_lds;
		pdt_gmem_lds->histogram[B_INF] += narray_get_len(pdt_gmem_lds->ga);
		parda_print_histogram(umem_fp, pdt_umem->histogram);
		parda_print_histogram(gmem_fp_lds, pdt_gmem_lds->histogram);
		parda_free(pdt_umem);
		parda_free(pdt_gmem_lds);
	}else{
		char shared_file[200];
		sprintf(shared_file, "%s/K%d_shared.rp", output_directory, kernel_id);
		FILE* shared_l2_fp = fopen(shared_file, "w");
		parda_print_histogram(shared_l2_fp, pdt_umem->histogram);
		parda_free(pdt_umem);
	}

	if(lmem_used){
		sprintf(gmem_file, "%s/K%d_GMEM_SM%d.rp", output_directory, kernel_id, sm_id);
		sprintf(lmem_file, "%s/K%d_LMEM_SM%d.rp", output_directory, kernel_id, sm_id);
		FILE* gmem_fp = fopen(gmem_file, "w");
		FILE* lmem_fp = fopen(lmem_file, "w");
		program_data_t* pdt_gmem = &pdt_c_gmem;
		pdt_gmem->histogram[B_INF] += narray_get_len(pdt_gmem->ga);
		program_data_t* pdt_lmem = &pdt_c_lmem;
		pdt_lmem->histogram[B_INF] += narray_get_len(pdt_lmem->ga);
		parda_print_histogram(gmem_fp, pdt_gmem->histogram);
		parda_print_histogram(lmem_fp, pdt_lmem->histogram);
		parda_free(pdt_gmem);
		parda_free(pdt_lmem);
	}


}

void parda_input_with_filename(char* inFileName, int kernel_id, int sm_id, long assoc, program_data_t* pdt_umem, program_data_t* pdt_gmem_lds, program_data_t* pdt_gmem, program_data_t* pdt_lmem, long begin, long end, char* output_directory) {
	
	DEBUG(printf("enter parda_input < %s from %ld to %ld\n", inFileName, begin, end);)
    FILE* fp;
	if(!is_binary) {
		fp = fopen(inFileName, "r");
		parda_input_with_textfilepointer(fp, kernel_id, sm_id, assoc, pdt_umem, pdt_gmem_lds, pdt_gmem, pdt_lmem, begin, end, output_directory);
	} else {
		fp = fopen(inFileName, "rb");
		parda_input_with_binaryfilepointer(fp, pdt_umem, begin, end);
	}
	fclose(fp);
}

void parda_input_with_textfilepointer(FILE* fp, int kernel_id, int sm_id, long assoc, program_data_t* pdt_umem, program_data_t* pdt_gmem_lds, program_data_t* pdt_gmem, program_data_t* pdt_lmem, long begin, long end, char* output_directory) {
	HKEY inst_id;
	HKEY mem_id;
	HKEY warp_id;
	HKEY address;
	long i;

	if(shared_l2 == 0){
		for(i = begin; i <= end; i++) {
			assert(fscanf(fp, "%s %s %s %s", inst_id, mem_id, warp_id, address) != EOF);
			DEBUG(printf("%s %d\n", address, i);)
			process_one_access(inst_id, mem_id, warp_id, address, assoc, pdt_umem, pdt_gmem_lds, pdt_gmem, pdt_lmem, i);
		}
	}else{
		for(i = begin; i <= end; i++) {
			assert(fscanf(fp, "%s", address) != EOF);
			DEBUG(printf("%s %d\n", address, i);)
			process_one_access_shared(address, pdt_umem, i);
		}
	}

}


void parda_input_with_binaryfilepointer(FILE* fp, program_data_t* pdt, long begin,long end) {
	HKEY inst_id;
	HKEY mem_id;
	HKEY warp_id;
	HKEY address;
	long t, i;
	long count;
	void** buffer = (void**)malloc(buffersize * sizeof(void*));
	for (t = begin; t <= end; t += count) {
		count = fread(buffer, sizeof(void*), buffersize, fp);
		for(i=0; i < count; i++) {
			sprintf(inst_id, mem_id, warp_id, address, "%p %p %p %p", buffer[i]);
			DEBUG(printf("%s %d\n",address,i+t);)
			//process_one_access(inst_id, mem_id, warp_id, address, pdt, pdt, pdt, pdt, i);
		}
	}
}


void parda_free(program_data_t* pdt) {
  narray_free(pdt->ga);
  //g_hash_table_foreach(pdt->gh, free_key_value, NULL);
  g_hash_table_destroy(pdt->gh);
  free(pdt->histogram);
  freetree(pdt->root);
}
