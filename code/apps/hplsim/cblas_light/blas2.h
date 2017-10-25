void dgemv ( char trans, int m, int n, double alpha, double a[], int lda, 
  double x[], int incx, double beta, double y[], int incy );
void dger ( int m, int n, double alpha, double x[], int incx, double y[], 
  int incy, double a[], int lda );
void dtrmv ( char uplo, char trans, char diag, int n, double a[], int lda, 
  double x[], int incx );

void sgemv ( char trans, int m, int n, float alpha, float a[], int lda, 
  float x[], int incx, float beta, float y[], int incy );
void sger ( int m, int n, float alpha, float x[], int incx, float y[], 
  int incy, float a[], int lda );
void strmv ( char uplo, char trans, char diag, int n, float a[], int lda, 
  float x[], int incx );
