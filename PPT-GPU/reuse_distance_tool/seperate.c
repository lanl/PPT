#include "seperate.h"

long get_file_lines(char filename[]) {
  FILE* fp=fopen(filename,"r");
  char input[20];
  long sum=0;
  while (fscanf(fp,"%s",input) != EOF)
  {
    sum++;
    if(sum<0)
    {
      printf("Trace length is out of 32 bit integer type\n");
      return -1;
    }
  }
  fclose(fp);
  return sum;
}

long seperate_textfile(char filename[], int processor_number, long lines) {
  FILE* fp = fopen(filename,"r");
  char input[20];
  long sum = lines;
  int i;
  long tstart,tend;
  long tim;
  for (i = 0; i < processor_number; ++i) {
    FILE* fw = fopen(parda_generate_pfilename(filename, i, processor_number), "w");
    tstart = parda_low(i, processor_number, sum);
    tend = parda_high(i, processor_number, sum);
    for ( tim = tstart; tim <= tend; ++tim) {
      assert(fscanf(fp, "%s", input) != EOF);
      int len = strlen(input);
      if(len >= 20)  {
        printf("line %ld length is larger than SLEN, please make sure all line less than SLEN\n",tim+1);
        //return -2;
      }
      fprintf(fw, "%s\n", input);
    }
    fclose(fw);
  }
  fclose(fp);
  return sum;
}

long seperate_binaryfile(char filename[],int processor_number,long lines) {
  FILE* fp = fopen(filename,"rb");
  long sum = lines;
  int i;
  long tstart, tend;
  long t, count;
  void** buffer = (void**)malloc(buffersize * sizeof(void*));
  for (i = 0; i < processor_number; ++i) {
    FILE* fw = fopen(parda_generate_pfilename(filename, i, processor_number), "wb");
    tstart = parda_low(i, processor_number, sum);
    tend = parda_high(i, processor_number, sum);
    for(t = tstart; t <= tend; t += count) {
      count = min(tend + 1 - t, buffersize);
      count = fread(buffer, sizeof(void*), count, fp);
      fwrite(buffer, sizeof(void*), count, fw);
    }
    fclose(fw);
  }
  fclose(fp);
  return sum;
}

long parda_seperate_file(char inputFileName[], int processor_number, long lines) {
  if (lines == -1)
    lines = get_file_lines(inputFileName);
  int psize = processor_number;
  if(!is_binary) seperate_textfile(inputFileName, psize, lines);
  else seperate_binaryfile(inputFileName, psize, lines);
  char linesFile[50];
  sprintf(linesFile, "%s_lines_%ld.txt", inputFileName, lines);
  FILE* tfile = fopen(linesFile,"w");
  fprintf(tfile, "%ld", lines);
  fclose(tfile);
  return lines;
}
