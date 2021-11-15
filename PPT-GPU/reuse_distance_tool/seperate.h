#ifndef _SEPERATE_H
#define _SEPERATE_H

#include "parda.h"

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

long get_file_lines(char filename[]);
long seperate_textfile(char filename[], int processor_number, long lines);
long seperate_binaryfile(char inputFileName[], int processor_number, long lines);
long parda_seperate_file(char inputFileName[], int processor_number, long lines);
#endif
