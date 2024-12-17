#include <stdio.h>
#include <stdlib.h>

unsigned *bb_counts = NULL;
unsigned max_bb_id = 0;

void initCounter(int32_t n_bb) {
    max_bb_id = n_bb;
    bb_counts = (unsigned*)calloc(n_bb, sizeof(unsigned));
    printf("Instrumentation: Basic block counter initialization function called -------- n_bb = %u\n", n_bb);
}

void bbCounter(int32_t id) {
    // printf("bbCounter called with basic block id %u\n", id);
    if(id >= max_bb_id) {
       printf("Instrumentation: Reached Max BB_LIMIT\nAborting!\n");
       exit(1);
    }
    bb_counts[id]++;
    // printf("execution count = %u, max_bb = %u\n", bb_counts[id], max_bb_id);
}

void dumpCounts(int32_t a) {
    FILE *fptr;
    fptr = fopen("basic-block-counts.txt", "a");
    for(unsigned id = 0; id < max_bb_id; id++) {
        fprintf(fptr, "%u ", bb_counts[id]);
    }
    fprintf(fptr, "\n");
    fclose(fptr);
    printf("Instrumentation: Done dumping basic block counts.\nBasic Block counts written/appended in basic-block-counts.txt file.\n");
    free(bb_counts);
}
