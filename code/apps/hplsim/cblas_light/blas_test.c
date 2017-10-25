# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <complex.h>
# include <time.h>

# include "blas0.h"
# include "blas2.h"
# include "blas3.h"

#define USEBYFL       0
#define USEBYFL_DGEMM 0
#define USEBYFL_DGEMV 0
#define USEBYFL_DSTRM 0

int main ( );
void test01 ( ); // DGEMM
void test02 ( ); // DGEMV
void test03 ( ); // DTRSM

int  N_in, M_in, K_in;
double ALPHA_in, BETA_in;

/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
  Purpose:
    MAIN is the main program for BLAS_TEST.

  Discussion:
    BLAS_TEST was derived from the BLAS3_PRB and BLAS2_PRB codes at
    (https://people.sc.fsu.edu/~jburkardt/c_src/blas0/blas0.html).

  Licensing:
    The original code is distributed under the GNU LGPL license.

  Modified:
    17 October 2016

  Author:
    RJ Zamora (Most of code by John Burkardt)
*/
{
#if USEBYFL
  bf_enable_counting (0);
#endif
  int i;
  timestamp ( );
  printf ( "\n" );
  printf ( "BLAS_TEST\n" );
  printf ( "  C version\n" );
  printf ( "  Test important HPL BLAS library routines...\n\n" );
  
  // args: N, M, K, ALPHA, BETA
  for (i=1; i<argc; i++) {
      if        (i==1) {
          N_in = atoi(argv[i]);
          printf("Setting N_in to %s\n", argv[i]);
      } else if (i==2) {
          M_in = atoi(argv[i]);
          printf("Setting M_in to %s\n", argv[i]);
      } else if (i==3) {
          K_in = atoi(argv[i]);
          printf("Setting K_in to %s\n", argv[i]);
      } else if (i==4) {
          ALPHA_in = strtod(argv[i], NULL);
          printf("Setting ALPHA_in to %s\n", argv[i]);
      } else if (i==5) {
          BETA_in = strtod(argv[i], NULL);
          printf("Setting BETA_in to %s\n", argv[i]);
      }
  }

  test01 ( );
  test02 ( );
  test03 ( );

/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "BLAS3_PRB\n" );
  printf ( "  Normal end of execution.\n" );
  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

void test01 ( )

/******************************************************************************/
/*
  Purpose:
    TEST01 tests DGEMM.
*/
{
  double *a;
  double alpha;
  double *b;
  double beta;
  double *c;
  int k;
  int lda;
  int ldb;
  int ldc;
  int m;
  int n;
  char transa;
  char transb;
  char transc;
  clock_t time, diff;

  printf ( "\n" );
  printf ( "TEST01 - DGEMM\n" );
  //printf ( "  DGEMM carries out matrix multiplications\n" );
  //printf ( "  for double precision real matrices.\n" );
  //printf ( "\n" );
  //printf ( "  1: C = alpha * A  * B  + beta * C;\n" );
  //printf ( "  2: C = alpha * A' * B  + beta * C;\n" );
  //printf ( "  3: C = alpha * A  * B' + beta * C;\n" );
  //printf ( "  4: C = alpha * A' * B' + beta * C;\n" );
  //printf ( "\n" );
  //printf ( "  We carry out all four calculations, but in each case,\n" );
  //printf ( "  we choose our input matrices so that we get the same result.\n" );

  // C = alpha * A * B + beta * C.

  transa = 'N';
  transb = 'N';
  transc = 'N';
  m = M_in; //4;
  n = N_in; //5;
  k = K_in; //3;
  alpha = ALPHA_in; //2.0;
  lda = m;
  a = r8mat_test ( transa, lda, m, k );
  ldb = k;
  b = r8mat_test ( transb, ldb, k, n );
  beta = BETA_in; //3.0;
  ldc = m;
  c = r8mat_test ( transc, ldc, m, n );

  time = clock();
  
#if USEBYFL_DGEMM
  bf_enable_counting (1);
#endif
  dgemm ( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );
#if USEBYFL
  bf_enable_counting (0);
#endif

  diff = clock() - time;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("DGEMM WCT: %d seconds %d milliseconds\n", msec/1000, msec%1000);

  //r8mat_print ( m, n, c, "  C = alpha * A * B + beta * C:" );

  free ( a );
  free ( b );
  free ( c );
  
  /*
  // C = alpha * A' * B + beta * C.

  transa = 'T';
  transb = 'N';
  transc = 'N';
  m = 4;
  n = 5;
  k = 3;
  alpha = 2.0;
  lda = k;
  a = r8mat_test ( transa, lda, m, k );
  ldb = k;
  b = r8mat_test ( transb, ldb, k, n );
  beta = 3.0;
  ldc = m;
  c = r8mat_test ( transc, ldc, m, n );

  dgemm ( transa, transb,  m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );

  r8mat_print ( m, n, c, "  C = alpha * A' * B + beta * C:" );

  free ( a );
  free ( b );
  free ( c );

  // C = alpha * A * B' + beta * C.

  transa = 'N';
  transb = 'T';
  transc = 'N';
  m = 4;
  n = 5;
  k = 3;
  alpha = 2.0;
  lda = m;
  a = r8mat_test ( transa, lda, m, k );
  ldb = n;
  b = r8mat_test ( transb, ldb, k, n );
  beta = 3.0;
  ldc = m;
  c = r8mat_test ( transc, ldc, m, n );

  dgemm ( transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );

  r8mat_print ( m, n, c, "  C = alpha * A * B' + beta * C:" );

  free ( a );
  free ( b );
  free ( c );

  // C = alpha * A' * B' + beta * C.

  transa = 'T';
  transb = 'T';
  transc = 'N';
  m = 4;
  n = 5;
  k = 3;
  alpha = 2.0;
  lda = k;
  a = r8mat_test ( transa, lda, m, k );
  ldb = n;
  b = r8mat_test ( transb, ldb, k, n );
  beta = 3.0;
  ldc = m;
  c = r8mat_test ( transc, ldc, m, n );

  dgemm ( transa, transb,  m, n, k, alpha, a, lda, b, ldb, beta, c, ldc );

  r8mat_print ( m, n, c, "  C = alpha * A' * B' + beta * C:" );

  free ( a );
  free ( b );
  free ( c );
  
  */

  return;
}
/******************************************************************************/

void test02 ( )

/******************************************************************************/
/*
  Purpose:
    TEST02 tests DGEMV.
*/
{
  double *a;
  double alpha;
  double beta;
  int i;
  int incx;
  int incy;
  //int j;
  int lda;
  int m;
  int n;
  char trans;
  double *x;
  double *y;
  clock_t time, diff;

  printf ( "\n" );
  printf ( "TEST02 - DGEMV\n" );
  //printf ( "  For a general matrix A,\n" );
  //printf ( "  DGEMV computes y := alpha * A * x + beta * y\n" );
  //printf ( "  or             y := alpha * A'' * x + beta * y.\n" );

  // y = alpha * A * x + beta * y

  trans = 'N';
  m = M_in; //5;
  n = N_in; //4;
  alpha = ALPHA_in; //2.0;
  lda = m;
  a = r8mat_test ( trans, lda, m, n );
  x = ( double * ) malloc ( n * sizeof ( double ) );
  for ( i = 0; i < n; i++ )
  {
    x[i] = ( double ) ( i + 1 );
  }
  incx = 1;
  beta = BETA_in; //3.0;
  y = ( double * ) malloc ( m * sizeof ( double ) );
  for ( i = 0; i < m; i++ )
  {
    y[i] = ( double ) ( 10 * ( i + 1 ) );
  }
  incy = 1;

  //r8mat_print ( m, n, a, "  Matrix A:" );
  //r8vec_print ( n, x, "  Vector X:" );
  //r8vec_print ( m, y, "  Vector Y:" );

  time = clock();
  
#if USEBYFL_DGEMV
  bf_enable_counting (1);
#endif
  dgemv ( trans, m, n, alpha, a, lda, x, incx, beta, y, incy );
#if USEBYFL
  bf_enable_counting (0);
#endif

  diff = clock() - time;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("DGEMV WCT: %d seconds %d milliseconds\n", msec/1000, msec%1000);

  //r8vec_print ( m, y, "  Result Y = alpha * A  * x + beta * y" );

  free ( a );
  free ( x );
  free ( y );
  
  /*
  // y = alpha * A' * x + beta * y

  trans = 'T';
  m = 5;
  n = 4;
  alpha = 2.0;
  lda = m;
  a = r8mat_test ( trans, lda, n, m );
  x = ( double * ) malloc ( m * sizeof ( double ) );
  for ( i = 0; i < m; i++ )
  {
    x[i] = ( double ) ( i + 1 );
  }
  incx = 1;
  beta = 3.0;
  y = ( double * ) malloc ( n * sizeof ( double ) );
  for ( i = 0; i < n; i++ )
  {
    y[i] = ( double ) ( 10 * ( i + 1 ) );
  }
  incy = 1;

  r8mat_print ( m, n, a, "  Matrix A:" );
  r8vec_print ( m, x, "  Vector X:" );
  r8vec_print ( n, y, "  Vector Y:" );

  dgemv ( trans, m, n, alpha, a, lda, x, incx, beta, y, incy );

  r8vec_print ( n, y, "  Result Y = alpha * A  * x + beta * y" );

  free ( a );
  free ( x );
  free ( y );
  */

  return;
}
/******************************************************************************/

void test03 ( )

/******************************************************************************/
/*
  Purpose:
    TEST03 tests DTRSM.
*/
{
  double *a;
  double alpha;
  double *b;
  char diag;
  int i;
  int j;
  int lda;
  int ldb;
  int m;
  int n;
  char side;
  char transa;
  char transb;
  char uplo;
  clock_t time, diff;

  printf ( "\n" );
  printf ( "TEST03 - DTRSM\n" );
  //printf ( "  DTRSM solves a linear system involving a triangular\n" );
  //printf ( "  matrix A and a rectangular matrix B.\n" );
  //printf ( "\n" );
  //printf ( "  1: Solve A  * X  = alpha * B;\n" );
  //printf ( "  2: Solve A' * X  = alpha * B;\n" );
  //printf ( "  3: Solve X  * A  = alpha * B;\n" );
  //printf ( "  4: Solve X  * A' = alpha * B;\n" );

  // Solve A * X = alpha * B.

  side = 'L';
  uplo = 'U';
  transa = 'N';
  diag = 'N';
  m = M_in; //4;
  n = N_in; //5;
  alpha = ALPHA_in; //2.0;
  lda = m;
  ldb = m;

  a = ( double * ) malloc ( lda * m * sizeof ( double ) );

  for ( j = 0; j < m; j++ )
  {
    for ( i = 0; i <= j; i++ )
    {
      a[i+j*lda] = ( double ) ( i + j + 2 );
    }
    for ( i = j + 1; i < m; i++ )
    {
      a[i+j*lda] = 0.0;
    }
  }

  transb = 'N';
  b = r8mat_test ( transb, ldb, m, n );

  time = clock();
  
#if USEBYFL_DSTRM
  bf_enable_counting (1);
#endif
  dtrsm ( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb );
#if USEBYFL
  bf_enable_counting (0);
#endif
  
  diff = clock() - time;
  int msec = diff * 1000 / CLOCKS_PER_SEC;
  printf("DTRSM WCT: %d seconds %d milliseconds\n", msec/1000, msec%1000);

  //r8mat_print ( m, n, b, "  X = inv ( A ) * alpha * B:" );

  free ( a );
  free ( b );
  
  /*
  // Solve A' * X = alpha * B.

  side = 'L';
  uplo = 'U';
  transa = 'T';
  diag = 'N';
  m = 4;
  n = 5;
  alpha = 2.0;
  lda = m;
  ldb = m;

  a = ( double * ) malloc ( lda * m * sizeof ( double ) );
  for ( j = 0; j < m; j++ )
  {
    for ( i = 0; i <= j; i++ )
    {
      a[i+j*lda] = ( double ) ( i + j + 2 );
    }
    for ( i = j + 1; i < m; i++ )
    {
      a[i+j*lda] = 0.0;
    }
  }

  transb = 'N';
  b = r8mat_test ( transb, ldb, m, n );

  dtrsm ( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb );

  r8mat_print ( m, n, b, "  X = inv ( A' ) * alpha * B:" );

  free ( a );
  free ( b );

  // Solve X * A = alpha * B.

  side = 'R';
  uplo = 'U';
  transa = 'N';
  diag = 'N';
  m = 4;
  n = 5;
  alpha = 2.0;
  lda = n;
  ldb = m;

  a = ( double * ) malloc ( lda * n * sizeof ( double ) );
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i <= j; i++ )
    {
      a[i+j*lda] = ( double ) ( i + j + 2 );
    }
    for ( i = j + 1; i < n; i++ )
    {
      a[i+j*lda] = 0.0;
    }
  }

  transb = 'N';
  b = r8mat_test ( transb, ldb, m, n );

  dtrsm ( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb );

  r8mat_print ( m, n, b, "  X = alpha * B * inv ( A ):" );

  free ( a );
  free ( b );

  // Solve X * A'' = alpha * B.

  side = 'R';
  uplo = 'U';
  transa = 'T';
  diag = 'N';
  m = 4;
  n = 5;
  alpha = 2.0;
  lda = n;
  ldb = m;

  a = ( double * ) malloc ( lda * n * sizeof ( double ) );
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i <= j; i++ )
    {
      a[i+j*lda] = ( double ) ( i + j + 2 );
    }
    for ( i = j + 1; i < n; i++ )
    {
      a[i+j*lda] = 0.0;
    }
  }

  transb = 'N';
  b = r8mat_test ( transb, ldb, m, n );

  dtrsm ( side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb );

  r8mat_print ( m, n, b, "  X = alpha * B * inv ( A' ):" );

  free ( a );
  free ( b );
  */

  return;
}

