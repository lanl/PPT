#ifndef _PARDA_OMP_H
#define _PARDA_OMP_H

#include "parda.h"
#include <omp.h>

processor_info_t parda_get_thread_info(long lines, long begin, int pid, int psize);
program_data_t* parda_omp_init(int psize);
program_data_t parda_omp_input(char inputFileName[], program_data_t* pdt_a, long begin, long end, int pid, int psize);
void parda_omp_free(program_data_t* pdt_a, int psize);
void parda_omp_stackdist(char* inputFileName, long lines, int threads);
#endif
