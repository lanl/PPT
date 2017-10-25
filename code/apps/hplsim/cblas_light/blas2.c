# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <complex.h>

# include "blas0.h"
# include "blas2.h"

/******************************************************************************/

void dgemv ( char trans, int m, int n, double alpha, double a[], int lda, 
  double x[], int incx, double beta, double y[], int incy )

/******************************************************************************/
/*
  Purpose:

    DGEMV computes y := alpha * A * x + beta * y for general matrix A.

  Discussion:

    DGEMV performs one of the matrix-vector operations
      y := alpha*A *x + beta*y
    or
      y := alpha*A'*x + beta*y,
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    02 April 2014

  Author:

    C version by John Burkardt

  Parameters:

    Input, char TRANS, specifies the operation to be performed:
    'n' or 'N'   y := alpha*A *x + beta*y.
    't' or 'T'   y := alpha*A'*x + beta*y.
    'c' or 'C'   y := alpha*A'*x + beta*y.

    Input, int M, the number of rows of the matrix A.
    0 <= M.

    Input, int N, the number of columns of the matrix A.
    0 <= N.

    Input, double ALPHA, the scalar multiplier for A * x.

    Input, double A[LDA*N].  The M x N subarray contains
    the matrix A.

    Input, int LDA, the the first dimension of A as declared
    in the calling routine.  max ( 1, M ) <= LDA.

    Input, double X[*], an array containing the vector to be 
    multiplied by the matrix A.  
    If TRANS = 'N' or 'n', then X must contain N entries, stored in INCX 
    increments in a space of at least ( 1 + ( N - 1 ) * abs ( INCX ) ) 
    locations.
    Otherwise, X must contain M entries, store in INCX increments
    in a space of at least ( 1 + ( M - 1 ) * abs ( INCX ) ) locations.

    Input, int INCX, the increment for the elements of
    X.  INCX must not be zero.

    Input, double BETA, the scalar multiplier for Y.

    Input/output, double Y[*], an array containing the vector to
    be scaled and incremented by A*X.
    If TRANS = 'N' or 'n', then Y must contain M entries, stored in INCY
    increments in a space of at least ( 1 + ( M - 1 ) * abs ( INCY ) ) 
    locations.
    Otherwise, Y must contain N entries, store in INCY increments
    in a space of at least ( 1 + ( N - 1 ) * abs ( INCY ) ) locations.

    Input, int INCY, the increment for the elements of
    Y.  INCY must not be zero.
*/
{
  int i;
  int info;
  int ix;
  int iy;
  int j;
  int jx;
  int jy;
  int kx;
  int ky;
  int lenx;
  int leny;
  double temp;
/*
  Test the input parameters.
*/
  info = 0;
  if ( ! lsame ( trans, 'N' ) &&
       ! lsame ( trans, 'T' ) &&
       ! lsame ( trans, 'C' ) ) 
  {
    info = 1;
  }
  else if ( m < 0 )
  {
    info = 2;
  }
  else if ( n < 0 )
  {
    info = 3;
  }
  else if ( lda < i4_max ( 1, m ) )
  {
    info = 6;
  }
  else if ( incx == 0 )
  {
    info = 8;
  }
  else if ( incy == 0 )
  {
    info = 11;
  }

  if ( info != 0 )
  {
    xerbla ( "DGEMV", info );
    return;
  }
/*
  Quick return if possible.
*/
  if ( ( m == 0 ) ||
       ( n == 0 ) ||
       ( ( alpha == 0.0 ) && ( beta == 1.0 ) ) )
 {
   return;
  }
/*
  Set LENX and LENY, the lengths of the vectors x and y, and set
  up the start points in X and Y.
*/
  if ( lsame ( trans, 'N' ) )
  {
    lenx = n;
    leny = m;
  }
  else
  {
    lenx = m;
    leny = n;
  }

  if ( 0 < incx )
  {
    kx = 0;
  }
  else
  {
    kx = 0 - ( lenx - 1 ) * incx;
  }

  if ( 0 < incy )
  {
    ky = 0;
  }
  else
  {
    ky = 0 - ( leny - 1 ) * incy;
  }
/*
  Start the operations. In this version the elements of A are
  accessed sequentially with one pass through A.

  First form  y := beta*y.
*/
  if ( beta != 1.0 )
  {
    if ( incy == 1 )
    {
      if ( beta == 0.0 )
      {
        for ( i = 0; i < leny; i++ )
        {
          y[i] = 0.0;
        }
      }
      else
      {
        for ( i = 0; i < leny; i++ )
        {
          y[i] = beta * y[i];
        }
      }
    }
    else
    {
      iy = ky;
      if ( beta == 0.0 )
      {
        for ( i = 0; i < leny; i++ )
        {
          y[iy] = 0.0;
          iy = iy + incy;
        }
      }
      else
      {
        for ( i = 0; i < leny; i++ )
        {
          y[iy] = beta * y[iy];
          iy = iy + incy;
        }
      }
    }
  }

  if ( alpha == 0.0 )
  {
    return;
  }
/*
  Form y := alpha*A*x + y.
*/
  if ( lsame ( trans, 'N' ) )
  {
    jx = kx;
    if ( incy == 1 )
    {
      for ( j = 0; j < n; j++ )
      {
        if ( x[jx] != 0.0 )
        {
          temp = alpha * x[jx];
          for ( i = 0; i < m; i++ )
          {
            y[i] = y[i] + temp * a[i+j*lda];
          }
        }
        jx = jx + incx;
      }
    }
    else
    {
      for ( j = 0; j < n; j++ )
      {
        if ( x[jx] != 0.0 )
        {
          temp = alpha * x[jx];
          iy = ky;
          for ( i = 0; i < m; i++ )
          {
            y[iy] = y[iy] + temp * a[i+j*lda];
            iy = iy + incy;
          }
        }
        jx = jx + incx;
      }
    }
  }
/*
  Form y := alpha*A'*x + y.
*/
  else
  {
    jy = ky;
    if ( incx == 1 )
    {
      for ( j = 0; j < n; j++ )
      {
        temp = 0.0;
        for ( i = 0; i < m; i++ )
        {
          temp = temp + a[i+j*lda] * x[i];
        }
        y[jy] = y[jy] + alpha * temp;
        jy = jy + incy;
      }
    }
    else
    {
      for ( j = 0; j < n; j++ )
      {
        temp = 0.0;
        ix = kx;
        for ( i = 0; i < m; i++ )
        {
          temp = temp + a[i+j*lda] * x[ix];
          ix = ix + incx;
        }
        y[jy] = y[jy] + alpha * temp;
        jy = jy + incy;
      }
    }
  }

  return;
}
/******************************************************************************/

void dger ( int m, int n, double alpha, double x[], int incx, double y[], 
  int incy, double a[], int lda )

/******************************************************************************/
/*
  Purpose:

    DGER computes A := alpha*x*y' + A.

  Discussion:

    DGER performs the rank 1 operation

      A := alpha*x*y' + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    Original FORTRAN77 version by Jack Dongarra,  Jeremy Du Croz,  
    Sven Hammarling,  Richard Hanson.
    This C version by John Burkardt.

  Parameters:

    Input, int M, the number of rows of the matrix A.
    0 <= M.

    Input, int N, the number of columns of the matrix A.
    0 <= N.

    Input, double ALPHA, the scalar multiplier.

    Input, double X[1+(M-1)*abs(INCX)], the first vector.

    Input, int INCX, the increment for elements of X.
    INCX must not be zero.

    Input, double Y[1+(N-1)*abs(INCY)], the second vector.

    Input, int INCY, the increment for elements of Y.
    INCY must not be zero.

    Input/output, double A[LDA*N].  On entry, the leading M by N 
    part of the array contains the matrix of coefficients. On exit, A is
    overwritten by the updated matrix.

    Input, int LDA, the first dimension of A as declared
    in the calling program. max ( 1, M ) <= LDA.
*/
{
  int i;
  int info;
  int ix;
  int j;
  int jy;
  int kx;
  double temp;
/*
  Test the input parameters.
*/
  info = 0;
  if ( m < 0 )
  {
    info = 1;
  }
  else if ( n < 0 )
  {
    info = 2;
  }
  else if ( incx == 0 )
  {
    info = 5;
  }
  else if ( incy == 0 )
  {
    info = 7;
  }
  else if ( lda < i4_max ( 1, m ) )
  {
    info = 9;
  }

  if ( info != 0 )
  {
    xerbla ( "DGER", info );
    return;
  }
/*
  Quick return if possible.
*/
  if ( m == 0 || n == 0 || alpha == 0.0 )
  {
    return;
  }
/*
  Start the operations. In this version the elements of A are
  accessed sequentially with one pass through A.
*/
  if ( 0 < incy )
  {
    jy = 0;
  }
  else
  {
    jy = 0 - ( n - 1 ) * incy;
  }

  if ( incx == 1 )
  {
    for ( j = 0; j < n; j++ )
    {
      if ( y[jy] != 0.0 )
      {
        temp = alpha * y[jy];
        for ( i = 0; i < m; i++ )
        {
          a[i+j*lda] = a[i+j*lda] + x[i] * temp;
        }
      }
      jy = jy + incy;
    }
  }
  else
  {
    if ( 0 < incx )
    {
      kx = 0;
    }
    else
    {
      kx = 0 - ( m - 1 ) * incx;
    }
    for ( j = 0; j < n; j++ )
    {
      if ( y[jy] != 0.0 )
      {
        temp = alpha * y[jy];
        ix = kx;
        for ( i = 0; i < m; i++ )
        {
          a[i+j*lda] = a[i+j*lda] + x[ix] * temp;
          ix = ix + incx;
        }
      }
      jy = jy + incy;
    }
  }

  return;
}
/******************************************************************************/

void dtrmv ( char uplo, char trans, char diag, int n, double a[], int lda, 
  double x[], int incx )

/******************************************************************************/
/*
  Purpose:

    DTRMV computes x: = A*x or x = A'*x for a triangular matrix A.

  Discussion:

    DTRMV performs one of the matrix-vector operations

      x := A*x,   or   x := A'*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    04 April 2014

  Author:

    This C version by John Burkardt.

  Parameters:

    Input, char UPLO, specifies whether the matrix is an upper or
    lower triangular matrix as follows:
    'u' or 'U': A is an upper triangular matrix.
    'l' or 'L': A is a lower triangular matrix.

    Input, char TRANS, specifies the operation to be performed as
    follows:
    'n' or 'N': x := A*x.
    't' or 'T': x := A'*x.
    'c' or 'C': x := A'*x.

    Input, char DIAG, specifies whether or not A is unit
    triangular as follows:
    'u' or 'U': A is assumed to be unit triangular.
    'n' or 'N': A is not assumed to be unit triangular.

    Input, int N, the order of the matrix A.
    0 <= N.

    Input, double A[LDA*N].
    Before entry with  UPLO = 'u' or 'U', the leading n by n
    upper triangular part of the array A must contain the upper
    triangular matrix and the strictly lower triangular part of
    A is not referenced.
    Before entry with UPLO = 'l' or 'L', the leading n by n
    lower triangular part of the array A must contain the lower
    triangular matrix and the strictly upper triangular part of
    A is not referenced.
    Note that when  DIAG = 'u' or 'U', the diagonal elements of
    A are not referenced either, but are assumed to be unity.

    Input, int LDA, the first dimension of A as declared
    in the calling program. max ( 1, N ) <= LDA.

    Input/output, double X[1+(N-1)*abs( INCX)].
    Before entry, the incremented array X must contain the n
    element vector x. On exit, X is overwritten with the
    tranformed vector x.

    Input, int INCX, the increment for the elements of
    X.  INCX must not be zero.
*/
{
  int i;
  int info;
  int ix;
  int j;
  int jx;
  int kx;
  int nounit;
  double temp;
/*
  Test the input parameters.
*/
  info = 0;
  if  ( ! lsame ( uplo, 'U' ) && ! lsame ( uplo, 'L' )  )
  {
    info = 1;
  }
  else if ( ! lsame ( trans, 'N' ) && ! lsame ( trans, 'T' ) && 
            ! lsame ( trans, 'C' ) )
  {
    info = 2;
  }
  else if ( ! lsame ( diag, 'U' ) && ! lsame ( diag, 'N' ) )
  {
    info = 3;
  }
  else if ( n < 0 )
  {
    info = 4;
  }
  else if ( lda < i4_max ( 1, n ) )
  {
    info = 6;
  }
  else if ( incx == 0 )
  {
    info = 8;
  }

  if ( info != 0 )
  {
    xerbla ( "DTRMV", info );
    return;
  }
/*
  Quick return if possible.
*/
  if ( n == 0 ) 
  {
    return;
  }

  nounit = lsame ( diag, 'N' );
/*
  Set up the start point in X if the increment is not unity. This
  will be  ( N - 1 ) * INCX  too small for descending loops.
*/
  if ( incx <= 0 )
  {
    kx = 0 - ( n - 1 ) * incx;
  }
  else if ( incx != 1 )
  {
    kx = 0;
  }
/*
  Start the operations. In this version the elements of A are
  accessed sequentially with one pass through A.
*/
  if ( lsame ( trans, 'N' ) )
  {
/*
  Form x := A*x.
*/
    if ( lsame ( uplo, 'U' ) )
    {
      if ( incx == 1 )
      {
        for ( j = 0; j < n; j++ )
        {
          if ( x[j] != 0.0 )
          {
            temp = x[j];
            for ( i = 0; i < j; i++ )
            {
              x[i] = x[i] + temp * a[i+j*lda];
            }
            if ( nounit )
            {
              x[j] = x[j] * a[j+j*lda];
            }
          }
        }
      }
      else
      {
        jx = kx;
        for ( j = 0; j < n; j++ )
        {
          if ( x[jx] != 0.0 )
          {
            temp = x[jx];
            ix = kx;
            for ( i = 0; i < j; i++ )
            {
              x[ix] = x[ix] + temp * a[i+j*lda];
              ix = ix + incx;
            }
            if ( nounit )
            {
              x[jx] = x[jx] * a[j+j*lda];
            }
          }
          jx = jx + incx;
        }
      }
    }
    else
    {
      if ( incx == 1 )
      {
        for ( j = n - 1; 0 <= j; j-- )
        {
          if ( x[j] != 0.0 )
          {
            temp = x[j];
            for ( i = n - 1; j < i; i-- )
            {
              x[i] = x[i] + temp * a[i+j*lda];
            }
            if ( nounit )
            {
              x[j] = x[j] * a[j+j*lda];
            }
          }
        }
      }
      else
      {
        kx = kx + ( n - 1 ) * incx;
        jx = kx;
        for ( j = n - 1; 0 <= j; j-- )
        {
          if ( x[jx] != 0.0 )
          {
            temp = x[jx];
            ix = kx;
            for ( i = n - 1; j < i; i-- )
            {
              x[ix] = x[ix] + temp * a[i+j*lda];
              ix = ix - incx;
            }
            if ( nounit )
            {
              x[jx] = x[jx] * a[j+j*lda];
            }
          }
          jx = jx - incx;
        }
      }
    }
  }
/*
  Form x := A'*x.
*/
  else
  {
    if ( lsame ( uplo, 'U' ) )
    {
      if ( incx == 1 )
      {
        for ( j = n - 1; 0 <= j; j-- )
        {
          temp = x[j];
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j - 1; 0 <= i; i-- )
          {
            temp = temp + a[i+j*lda] * x[i];
          }
          x[j] = temp;
        }
      }
      else
      {
        jx = kx + ( n - 1 ) * incx;
        for ( j = n - 1; 0 <= j; j-- )
        {
          temp = x[jx];
          ix = jx;
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j - 1; 0 <= i; i-- )
          {
            ix = ix - incx;
            temp = temp + a[i+j*lda] * x[ix];
          }
          x[jx] = temp;
          jx = jx - incx;
        }
      }
    }
    else
    {
      if ( incx == 1 )
      {
        for ( j = 0; j < n; j++ )
        {
          temp = x[j];
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j + 1; i < n; i++ )
          {
            temp = temp + a[i+j*lda] * x[i];
          }
          x[j] = temp;
        }
      }
      else
      {
        jx = kx;
        for ( j = 0; j < n; j++ )
        {
          temp = x[jx];
          ix = jx;
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j + 1; i < n; i++ )
          {
            ix = ix + incx;
            temp = temp + a[i+j*lda] * x[ix];
          }
          x[jx] = temp;
          jx = jx + incx;
        }
      }
    }
  }

  return;
}
/******************************************************************************/

void sgemv ( char trans, int m, int n, float alpha, float a[], int lda, 
  float x[], int incx, float beta, float y[], int incy )

/******************************************************************************/
/*
  Purpose:

    SGEMV computes y := alpha * A * x + beta * y for general matrix A.

  Discussion:

    SGEMV performs one of the matrix-vector operations
      y := alpha*A *x + beta*y
    or
      y := alpha*A'*x + beta*y,
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    C version by John Burkardt

  Parameters:

    Input, char TRANS, specifies the operation to be performed:
    'n' or 'N'   y := alpha*A *x + beta*y.
    't' or 'T'   y := alpha*A'*x + beta*y.
    'c' or 'C'   y := alpha*A'*x + beta*y.

    Input, int M, the number of rows of the matrix A.
    0 <= M.

    Input, int N, the number of columns of the matrix A.
    0 <= N.

    Input, float ALPHA, the scalar multiplier for A * x.

    Input, float A[LDA*N].  The M x N subarray contains
    the matrix A.

    Input, int LDA, the the first dimension of A as declared
    in the calling routine.  max ( 1, M ) <= LDA.

    Input, float X[*], an array containing the vector to be 
    multiplied by the matrix A.  
    If TRANS = 'N' or 'n', then X must contain N entries, stored in INCX 
    increments in a space of at least ( 1 + ( N - 1 ) * abs ( INCX ) ) 
    locations.
    Otherwise, X must contain M entries, store in INCX increments
    in a space of at least ( 1 + ( M - 1 ) * abs ( INCX ) ) locations.

    Input, int INCX, the increment for the elements of
    X.  INCX must not be zero.

    Input, float BETA, the scalar multiplier for Y.

    Input/output, float Y[*], an array containing the vector to
    be scaled and incremented by A*X.
    If TRANS = 'N' or 'n', then Y must contain M entries, stored in INCY
    increments in a space of at least ( 1 + ( M - 1 ) * abs ( INCY ) ) 
    locations.
    Otherwise, Y must contain N entries, store in INCY increments
    in a space of at least ( 1 + ( N - 1 ) * abs ( INCY ) ) locations.

    Input, int INCY, the increment for the elements of
    Y.  INCY must not be zero.
*/
{
  int i;
  int info;
  int ix;
  int iy;
  int j;
  int jx;
  int jy;
  int kx;
  int ky;
  int lenx;
  int leny;
  float temp;
/*
  Test the input parameters.
*/
  info = 0;
  if ( ! lsame ( trans, 'N' ) &&
       ! lsame ( trans, 'T' ) &&
       ! lsame ( trans, 'C' ) ) 
  {
    info = 1;
  }
  else if ( m < 0 )
  {
    info = 2;
  }
  else if ( n < 0 )
  {
    info = 3;
  }
  else if ( lda < i4_max ( 1, m ) )
  {
    info = 6;
  }
  else if ( incx == 0 )
  {
    info = 8;
  }
  else if ( incy == 0 )
  {
    info = 11;
  }

  if ( info != 0 )
  {
    xerbla ( "SGEMV", info );
    return;
  }
/*
  Quick return if possible.
*/
  if ( ( m == 0 ) ||
       ( n == 0 ) ||
       ( ( alpha == 0.0 ) && ( beta == 1.0 ) ) )
 {
   return;
  }
/*
  Set LENX and LENY, the lengths of the vectors x and y, and set
  up the start points in X and Y.
*/
  if ( lsame ( trans, 'N' ) )
  {
    lenx = n;
    leny = m;
  }
  else
  {
    lenx = m;
    leny = n;
  }

  if ( 0 < incx )
  {
    kx = 0;
  }
  else
  {
    kx = 0 - ( lenx - 1 ) * incx;
  }

  if ( 0 < incy )
  {
    ky = 0;
  }
  else
  {
    ky = 0 - ( leny - 1 ) * incy;
  }
/*
  Start the operations. In this version the elements of A are
  accessed sequentially with one pass through A.

  First form  y := beta*y.
*/
  if ( beta != 1.0 )
  {
    if ( incy == 1 )
    {
      if ( beta == 0.0 )
      {
        for ( i = 0; i < leny; i++ )
        {
          y[i] = 0.0;
        }
      }
      else
      {
        for ( i = 0; i < leny; i++ )
        {
          y[i] = beta * y[i];
        }
      }
    }
    else
    {
      iy = ky;
      if ( beta == 0.0 )
      {
        for ( i = 0; i < leny; i++ )
        {
          y[iy] = 0.0;
          iy = iy + incy;
        }
      }
      else
      {
        for ( i = 0; i < leny; i++ )
        {
          y[iy] = beta * y[iy];
          iy = iy + incy;
        }
      }
    }
  }

  if ( alpha == 0.0 )
  {
    return;
  }
/*
  Form y := alpha*A*x + y.
*/
  if ( lsame ( trans, 'N' ) )
  {
    jx = kx;
    if ( incy == 1 )
    {
      for ( j = 0; j < n; j++ )
      {
        if ( x[jx] != 0.0 )
        {
          temp = alpha * x[jx];
          for ( i = 0; i < m; i++ )
          {
            y[i] = y[i] + temp * a[i+j*lda];
          }
        }
        jx = jx + incx;
      }
    }
    else
    {
      for ( j = 0; j < n; j++ )
      {
        if ( x[jx] != 0.0 )
        {
          temp = alpha * x[jx];
          iy = ky;
          for ( i = 0; i < m; i++ )
          {
            y[iy] = y[iy] + temp * a[i+j*lda];
            iy = iy + incy;
          }
        }
        jx = jx + incx;
      }
    }
  }
/*
  Form y := alpha*A'*x + y.
*/
  else
  {
    jy = ky;
    if ( incx == 1 )
    {
      for ( j = 0; j < n; j++ )
      {
        temp = 0.0;
        for ( i = 0; i < m; i++ )
        {
          temp = temp + a[i+j*lda] * x[i];
        }
        y[jy] = y[jy] + alpha * temp;
        jy = jy + incy;
      }
    }
    else
    {
      for ( j = 0; j < n; j++ )
      {
        temp = 0.0;
        ix = kx;
        for ( i = 0; i < m; i++ )
        {
          temp = temp + a[i+j*lda] * x[ix];
          ix = ix + incx;
        }
        y[jy] = y[jy] + alpha * temp;
        jy = jy + incy;
      }
    }
  }

  return;
}
/******************************************************************************/

void sger ( int m, int n, float alpha, float x[], int incx, float y[], 
  int incy, float a[], int lda )

/******************************************************************************/
/*
  Purpose:

    SGER computes A := alpha*x*y' + A.

  Discussion:

    SGER performs the rank 1 operation

      A := alpha*x*y' + A,

    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    Original FORTRAN77 version by Jack Dongarra,  Jeremy Du Croz,  
    Sven Hammarling,  Richard Hanson.
    This C version by John Burkardt.

  Parameters:

    Input, int M, the number of rows of the matrix A.
    0 <= M.

    Input, int N, the number of columns of the matrix A.
    0 <= N.

    Input, float ALPHA, the scalar multiplier.

    Input, float X[1+(M-1)*abs(INCX)], the first vector.

    Input, int INCX, the increment for elements of X.
    INCX must not be zero.

    Input, float Y[1+(N-1)*abs(INCY)], the second vector.

    Input, int INCY, the increment for elements of Y.
    INCY must not be zero.

    Input/output, float A[LDA*N].  On entry, the leading M by N 
    part of the array contains the matrix of coefficients. On exit, A is
    overwritten by the updated matrix.

    Input, int LDA, the first dimension of A as declared
    in the calling program. max ( 1, M ) <= LDA.
*/
{
  int i;
  int info;
  int ix;
  int j;
  int jy;
  int kx;
  float temp;
/*
  Test the input parameters.
*/
  info = 0;
  if ( m < 0 )
  {
    info = 1;
  }
  else if ( n < 0 )
  {
    info = 2;
  }
  else if ( incx == 0 )
  {
    info = 5;
  }
  else if ( incy == 0 )
  {
    info = 7;
  }
  else if ( lda < i4_max ( 1, m ) )
  {
    info = 9;
  }

  if ( info != 0 )
  {
    xerbla ( "SGER", info );
    return;
  }
/*
  Quick return if possible.
*/
  if ( m == 0 || n == 0 || alpha == 0.0 )
  {
    return;
  }
/*
  Start the operations. In this version the elements of A are
  accessed sequentially with one pass through A.
*/
  if ( 0 < incy )
  {
    jy = 0;
  }
  else
  {
    jy = 0 - ( n - 1 ) * incy;
  }

  if ( incx == 1 )
  {
    for ( j = 0; j < n; j++ )
    {
      if ( y[jy] != 0.0 )
      {
        temp = alpha * y[jy];
        for ( i = 0; i < m; i++ )
        {
          a[i+j*lda] = a[i+j*lda] + x[i] * temp;
        }
      }
      jy = jy + incy;
    }
  }
  else
  {
    if ( 0 < incx )
    {
      kx = 0;
    }
    else
    {
      kx = 0 - ( m - 1 ) * incx;
    }
    for ( j = 0; j < n; j++ )
    {
      if ( y[jy] != 0.0 )
      {
        temp = alpha * y[jy];
        ix = kx;
        for ( i = 0; i < m; i++ )
        {
          a[i+j*lda] = a[i+j*lda] + x[ix] * temp;
          ix = ix + incx;
        }
      }
      jy = jy + incy;
    }
  }
  return;
}
/******************************************************************************/

void strmv ( char uplo, char trans, char diag, int n, float a[], int lda, 
  float x[], int incx )

/******************************************************************************/
/*
  Purpose:

    STRMV computes x: = A*x or x = A'*x for a triangular matrix A.

  Discussion:

    STRMV performs one of the matrix-vector operations

      x := A*x,   or   x := A'*x,

    where x is an n element vector and  A is an n by n unit, or non-unit,
    upper or lower triangular matrix.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    04 April 2014

  Author:

    This C version by John Burkardt.

  Parameters:

    Input, char UPLO, specifies whether the matrix is an upper or
    lower triangular matrix as follows:
    'u' or 'U': A is an upper triangular matrix.
    'l' or 'L': A is a lower triangular matrix.

    Input, char TRANS, specifies the operation to be performed as
    follows:
    'n' or 'N': x := A*x.
    't' or 'T': x := A'*x.
    'c' or 'C': x := A'*x.

    Input, char DIAG, specifies whether or not A is unit
    triangular as follows:
    'u' or 'U': A is assumed to be unit triangular.
    'n' or 'N': A is not assumed to be unit triangular.

    Input, int N, the order of the matrix A.
    0 <= N.

    Input, float A[LDA*N].
    Before entry with  UPLO = 'u' or 'U', the leading n by n
    upper triangular part of the array A must contain the upper
    triangular matrix and the strictly lower triangular part of
    A is not referenced.
    Before entry with UPLO = 'l' or 'L', the leading n by n
    lower triangular part of the array A must contain the lower
    triangular matrix and the strictly upper triangular part of
    A is not referenced.
    Note that when  DIAG = 'u' or 'U', the diagonal elements of
    A are not referenced either, but are assumed to be unity.

    Input, int LDA, the first dimension of A as declared
    in the calling program. max ( 1, N ) <= LDA.

    Input/output, float X[1+(N-1)*abs( INCX)].
    Before entry, the incremented array X must contain the n
    element vector x. On exit, X is overwritten with the
    tranformed vector x.

    Input, int INCX, the increment for the elements of
    X.  INCX must not be zero.
*/
{
  int i;
  int info;
  int ix;
  int j;
  int jx;
  int kx;
  int nounit;
  float temp;
/*
  Test the input parameters.
*/
  info = 0;
  if  ( ! lsame ( uplo, 'U' ) && ! lsame ( uplo, 'L' )  )
  {
    info = 1;
  }
  else if ( ! lsame ( trans, 'N' ) && ! lsame ( trans, 'T' ) && 
            ! lsame ( trans, 'C' ) )
  {
    info = 2;
  }
  else if ( ! lsame ( diag, 'U' ) && ! lsame ( diag, 'N' ) )
  {
    info = 3;
  }
  else if ( n < 0 )
  {
    info = 4;
  }
  else if ( lda < i4_max ( 1, n ) )
  {
    info = 6;
  }
  else if ( incx == 0 )
  {
    info = 8;
  }

  if ( info != 0 )
  {
    xerbla ( "STRMV", info );
    return;
  }
/*
  Quick return if possible.
*/
  if ( n == 0 ) 
  {
    return;
  }

  nounit = lsame ( diag, 'N' );
/*
  Set up the start point in X if the increment is not unity. This
  will be  ( N - 1 ) * INCX  too small for descending loops.
*/
  if ( incx <= 0 )
  {
    kx = 0 - ( n - 1 ) * incx;
  }
  else if ( incx != 1 )
  {
    kx = 0;
  }
/*
  Start the operations. In this version the elements of A are
  accessed sequentially with one pass through A.
*/
  if ( lsame ( trans, 'N' ) )
  {
/*
  Form x := A*x.
*/
    if ( lsame ( uplo, 'U' ) )
    {
      if ( incx == 1 )
      {
        for ( j = 0; j < n; j++ )
        {
          if ( x[j] != 0.0 )
          {
            temp = x[j];
            for ( i = 0; i < j; i++ )
            {
              x[i] = x[i] + temp * a[i+j*lda];
            }
            if ( nounit )
            {
              x[j] = x[j] * a[j+j*lda];
            }
          }
        }
      }
      else
      {
        jx = kx;
        for ( j = 0; j < n; j++ )
        {
          if ( x[jx] != 0.0 )
          {
            temp = x[jx];
            ix = kx;
            for ( i = 0; i < j; i++ )
            {
              x[ix] = x[ix] + temp * a[i+j*lda];
              ix = ix + incx;
            }
            if ( nounit )
            {
              x[jx] = x[jx] * a[j+j*lda];
            }
          }
          jx = jx + incx;
        }
      }
    }
    else
    {
      if ( incx == 1 )
      {
        for ( j = n - 1; 0 <= j; j-- )
        {
          if ( x[j] != 0.0 )
          {
            temp = x[j];
            for ( i = n - 1; j < i; i-- )
            {
              x[i] = x[i] + temp * a[i+j*lda];
            }
            if ( nounit )
            {
              x[j] = x[j] * a[j+j*lda];
            }
          }
        }
      }
      else
      {
        kx = kx + ( n - 1 ) * incx;
        jx = kx;
        for ( j = n - 1; 0 <= j; j-- )
        {
          if ( x[jx] != 0.0 )
          {
            temp = x[jx];
            ix = kx;
            for ( i = n - 1; j < i; i-- )
            {
              x[ix] = x[ix] + temp * a[i+j*lda];
              ix = ix - incx;
            }
            if ( nounit )
            {
              x[jx] = x[jx] * a[j+j*lda];
            }
          }
          jx = jx - incx;
        }
      }
    }
  }
/*
  Form x := A'*x.
*/
  else
  {
    if ( lsame ( uplo, 'U' ) )
    {
      if ( incx == 1 )
      {
        for ( j = n - 1; 0 <= j; j-- )
        {
          temp = x[j];
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j - 1; 0 <= i; i-- )
          {
            temp = temp + a[i+j*lda] * x[i];
          }
          x[j] = temp;
        }
      }
      else
      {
        jx = kx + ( n - 1 ) * incx;
        for ( j = n - 1; 0 <= j; j-- )
        {
          temp = x[jx];
          ix = jx;
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j - 1; 0 <= i; i-- )
          {
            ix = ix - incx;
            temp = temp + a[i+j*lda] * x[ix];
          }
          x[jx] = temp;
          jx = jx - incx;
        }
      }
    }
    else
    {
      if ( incx == 1 )
      {
        for ( j = 0; j < n; j++ )
        {
          temp = x[j];
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j + 1; i < n; i++ )
          {
            temp = temp + a[i+j*lda] * x[i];
          }
          x[j] = temp;
        }
      }
      else
      {
        jx = kx;
        for ( j = 0; j < n; j++ )
        {
          temp = x[jx];
          ix = jx;
          if ( nounit )
          {
            temp = temp * a[j+j*lda];
          }
          for ( i = j + 1; i < n; i++ )
          {
            ix = ix + incx;
            temp = temp + a[i+j*lda] * x[ix];
          }
          x[jx] = temp;
          jx = jx + incx;
        }
      }
    }
  }

  return;
}
