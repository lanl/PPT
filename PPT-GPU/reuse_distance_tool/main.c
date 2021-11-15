#include "parda.h"
#ifdef enable_omp
#include "parda_omp.h"
#endif
#include "process_args.h"
#include "seperate.h"


int main(int argc, char **argv){
    
	process_args(argc,argv);

	if(is_lmem == 0){
		lmem_used = 0;
	}
	else{
		lmem_used = 1;
	}
	if(is_l2 == 0){
		shared_l2 = 0;
	}
	else{
		shared_l2 = 1;
	}
	if (is_seperate == 1) {
		parda_seperate_file(inputFileName, threads, lines);
	}else if (is_omp == 0) {
		DEBUG(printf("This is seq stackdist\n");)
		classical_tree_based_stackdist(inputFileName, kernel_id, sm_id, lines, assoc, output_directory);
	}else if (is_omp == 1) {
		DEBUG(printf("This is omp stackdist\n");)
		#ifdef enable_omp
				parda_omp_stackdist(inputFileName, lines, threads);
		#else
			printf("openmp is not enabled, try to define enable_omp and add OMP variable in Makefile\n");
			abort();
	#endif
	}

return 0;
}
