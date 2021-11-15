#include "parda_omp.h"

processor_info_t parda_get_thread_info(long lines, long begin, int pid, int psize) {
    processor_info_t pit_c;
    pit_c.pid = pid;
    pit_c.psize = psize;
    pit_c.sum = lines;
    pit_c.tstart = parda_low(pit_c.pid, pit_c.psize, lines);
    pit_c.tstart += begin;
    pit_c.tend = parda_high(pit_c.pid, pit_c.psize, lines);
    pit_c.tend += begin;
    pit_c.tlen = parda_size(pit_c.pid, pit_c.psize, lines);
    return pit_c;
}

program_data_t* parda_omp_init(int nthreads) {
    g_thread_init(NULL);
    program_data_t* pdt_a = (program_data_t*)malloc(nthreads * sizeof(program_data_t));
    int i;
    for(i=0;i<nthreads;i++)
        pdt_a[i]=parda_init();
    omp_set_num_threads(nthreads);
    return pdt_a;
}

void parda_omp_openfile(char inputFileName[],
    const int pid, const int nthreads, const int psize, FILE* fpa[]) {
    int i;
    for(i = 0; i < nthreads; i++) {
        char* pfilename = parda_generate_pfilename(inputFileName, pid * nthreads + i, psize * nthreads);
        fpa[i] = fopen(pfilename, "r");
    }
}

program_data_t parda_omp_input_with_filename(char inputFileName[],
    program_data_t* pdt_a, long begin, long end, int pid, int psize) {
    int nthreads = threads;
    long lines = end + 1 - begin;
    FILE* fpa[8];
    parda_omp_openfile(inputFileName, pid, nthreads, psize, fpa);
#pragma omp parallel default(none) firstprivate(begin, pid, psize, nthreads, lines, is_binary) shared(pdt_a, fpa)
    {
        int i = omp_get_thread_num();
        FILE* fp = fpa[i];
        processor_info_t pit = parda_get_thread_info(lines, begin, i, nthreads);
        program_data_t pdt_c = pdt_a[i];
				if (!is_binary) {
						parda_input_with_textfilepointer(fp,&pdt_c, pit.tstart, pit.tend);
				} else {
						parda_input_with_binaryfilepointer(fp,&pdt_c, pit.tstart, pit.tend);
				}
				pdt_a[i] = pdt_c;
        int tid = i;
        int var, len;
        int mlen = nthreads >> 1;
        for (var = tid, len = 1; len <= mlen; len = (len << 1)) {
            if(var&1) {
                program_data_t pdt_A = pdt_a[tid - len];
                program_data_t pdt_B = pdt_a[tid];
                pdt_a[tid] = parda_merge(&pdt_A, &pdt_B, &pit);
                var >>= 1;
            }
#pragma omp barrier
        }
    }
    program_data_t pdt_c = pdt_a[nthreads - 1];
    return pdt_c;
}

program_data_t parda_omp_input(char inputFileName[], program_data_t* pdt_a, long begin, long end,
    int pid, int psize)
{
    int nthreads = threads;
    //omp_set_num_threads(nthreads);
    long lines = end + 1 - begin;
    processor_info_t pit_a[8];
    int syn[8 << 6];
    memset(syn, 0, sizeof(syn));
    int i;
#pragma omp parallel default(none) private(i) firstprivate(begin, pid, psize, nthreads, lines) shared(pdt_a, pit_a, syn, inputFileName)
    {
        DEBUG(printf("enter parallel for\n");)
            __sync_synchronize();
#pragma omp for
        for(i=0 ;i < nthreads; i++)
        {
						printf("i=%d executed by thread=%d\n", i, omp_get_thread_num());
            pit_a[i] = parda_get_thread_info(lines, begin, i, nthreads);
            parda_input_with_filename(parda_generate_pfilename(inputFileName, pid * nthreads + i, psize * nthreads), &pdt_a[i], pit_a[i].tstart, pit_a[i].tend);
#ifdef enable_debugging
            printf("after input in for\n");
#endif
            int tid = i;
            int var, len;
            for(var = tid, len = 1; var % 2 == 1; var = (var >> 1), len = (len << 1)) {
                DEBUG(printf("before while in for %d and %d\n", tid - len, tid);)
                while(syn[(tid - len) << 8] == 0) {
#pragma omp flush(syn)
								}
                DEBUG(printf("after while in for and will merge %d and %d\n",tid - len, tid);)
                pdt_a[tid] = parda_merge(&pdt_a[tid - len], &pdt_a[tid], &pit_a[tid]);
                DEBUG(printf("after merged %d and %d\n",tid-len,tid);)
            }
            syn[tid << 8]++;
#pragma omp flush(syn)
        }
    }
    program_data_t pdt = pdt_a[nthreads - 1];
    return pdt;
}

void parda_omp_free(program_data_t* pdt_a, int psize)
{
    int i;
#pragma omp parallel private(i) shared(pdt_a, psize)
    {
#pragma omp for
        for(i = 0; i < psize - 1; i++) {
            g_hash_table_destroy(pdt_a[i].gh);
        }
    }
    free(pdt_a);
}

void parda_omp_stackdist(char* inputFileName, long lines, int threads)
{
#ifdef enable_timing
        double ts, te, t_init, t_input, t_print, t_free;
        ts=rtclock();
#endif
    program_data_t* pdt_a = parda_omp_init(threads);
    PTIME(te = rtclock();)
    PTIME(t_init = te - ts;)
    DEBUG(printf("after omp init\n");)
    program_data_t pdt_c = parda_omp_input_with_filename(inputFileName, pdt_a, 0, lines - 1, 0, 1);
    //program_data_t pdt_c=parda_omp_input(inputFileName,pdt_a,0,lines-1,0,1);
    DEBUG(printf("after omp input\n");)
    program_data_t* pdt = &pdt_c;
    pdt->histogram[B_INF] += narray_get_len(pdt->ga);
    PTIME(ts = rtclock();)
    PTIME(t_input = ts - te;)
    parda_print_histogram(pdt->histogram);
    PTIME(te = rtclock();)
    PTIME(t_print = te - ts;)
    parda_omp_free(pdt_a, threads);
    parda_free(pdt);
    PTIME(ts = rtclock();)
    PTIME(t_free = ts - te;)
#ifdef enable_timing
    printf("omp\n");
    printf("init time is %lf\n", t_init);
    printf("input time is %lf\n", t_input);
    printf("print time is %lf\n", t_print);
    printf("free time is %lf\n", t_free);
#endif
}
