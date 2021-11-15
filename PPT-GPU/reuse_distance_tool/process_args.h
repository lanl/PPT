#ifndef _PROCESS_ARGS_H
#define _PROCESS_ARGS_H

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

extern int is_omp;
extern int is_seperate;
extern int is_lmem;
extern int is_l2;
extern int is_binary;
extern char inputFileName[200];
extern int sm_id;
extern long assoc;
extern long lines;
extern int kernel_id;
extern int threads;
extern int buffersize;
extern char output_directory[200];

int process_args(int argc,char **argv);
#endif
