# include <complex.h>
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <time.h>

# include "blas0.h"

/******************************************************************************/

float complex c4_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    C4_UNIFORM_01 returns a unit pseudorandom C4.

  Discussion:

    The angle should be uniformly distributed between 0 and 2 * PI,
    the square root of the radius uniformly distributed between 0 and 1.

    This results in a uniform distribution of values in the unit circle.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, float complex C4_UNIFORM_01, a pseudorandom complex value.
*/
{
  int i4_huge = 2147483647;
  int k;
  float r;
  float r4_pi = 3.1415926;
  float theta;
  float complex value;

  if ( *seed == 0 )
  {
    printf ( "\n" );
    printf ( "C4_UNIFORM_01 - Fatal error!\n" );
    printf ( "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = sqrt ( ( ( float ) ( *seed ) * 4.656612875E-10 ) );

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  theta = 2.0 * r4_pi * ( ( float ) ( *seed ) * 4.656612875E-10 );

  value = r * cos ( theta ) + r * sin ( theta ) * I;
  
  return value;
}
/******************************************************************************/

void c4mat_print ( int m, int n, float complex a[], char *title )

/******************************************************************************/
/*
  Purpose:

    C4MAT_PRINT prints a C4MAT.

  Discussion:

    A C4MAT is a matrix of single precision complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    09 March 2014

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input, float complex A[M*N], the matrix.

    Input, char *TITLE, a title.
*/
{
  c4mat_print_some ( m, n, a, 1, 1, m, n, title );

  return;
}
/******************************************************************************/

void c4mat_print_some ( int m, int n, float complex a[], int ilo, int jlo, 
  int ihi, int jhi, char *title )

/******************************************************************************/
/*
  Purpose:

    C4MAT_PRINT_SOME prints some of a C4MAT.

  Discussion:

    A C4MAT is a matrix of float complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    09 March 2014

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input, float complex A[M*N], the matrix.

    Input, int ILO, JLO, IHI, JHI, the first row and
    column, and the last row and column to be printed.

    Input, char *TITLE, a title.
*/
{
  float complex c;
  int i;
  int i2hi;
  int i2lo;
  int inc;
  int incx = 4;
  int j;
  int j2;
  int j2hi;
  int j2lo;

  printf ( "\n" );
  printf ( "%s\n", title );
/*
  Print the columns of the matrix, in strips of INCX.
*/
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + incx )
  {
    j2hi = j2lo + incx - 1;
    j2hi = i4_min ( j2hi, n );
    j2hi = i4_min ( j2hi, jhi );

    inc = j2hi + 1 - j2lo;

    printf ( "\n" );
    printf ( "  Col: " );
    for ( j = j2lo; j <= j2hi; j++ )
    {
      j2 = j + 1 - j2lo;
      printf ( "          %10d", j );
    }
    printf ( "\n" );
    printf ( "  Row\n" );
    printf ( "  ---\n" );
/*
  Determine the range of the rows in this strip.
*/
    i2lo = i4_max ( ilo, 1 );
    i2hi = i4_min ( ihi, m );

    for ( i = i2lo; i <= i2hi; i++ )
    {
/*
  Print out (up to) INCX entries in row I, that lie in the current strip.
*/
      for ( j2 = 1; j2 <= inc; j2++ )
      {
        j = j2lo - 1 + j2;
        c = a[i-1+(j-1)*m];
        printf ( "  %8g  %8g", creal ( c ), cimag ( c ) );
      }
      printf ( "\n" );
    }
  }
  return;
}
/******************************************************************************/

float complex *c4mat_test ( int n )

/******************************************************************************/
/*
  Purpose:

    C4MAT_TEST returns a test matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.

    Output, float complex C4MAT_TEST[N*N], the Fourier matrix.
*/
{
  float complex *a;
  float complex angle;
  int i;
  int j;
  float pi = 3.141592653589793;

  a = ( float complex * ) malloc ( n * n * sizeof ( float complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      angle = 2.0 * pi * ( float ) ( i * j ) * I / ( float ) ( n );

      a[i+j*n] = cexp ( angle ) / sqrt ( ( float ) ( n ) );
    }
  }
  return a;
}
/******************************************************************************/

float complex *c4mat_test_inverse ( int n )

/******************************************************************************/
/*
  Purpose:

    C4MAT_TEST_INVERSE returns the inverse of a test matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.

    Output, float complex C4MAT_TEST_INVERSE[N*N], the matrix.
*/
{
  float complex *a;
  int i;
  int j;
  float complex t;

  a = c4mat_test ( n );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < j; i++ )
    {
      t        = conj ( a[i+j*n] );
      a[i+j*n] = conj ( a[j+i*n] );
      a[j+i*n] = t;
    }
  }
  return a;
}
/******************************************************************************/

double complex c8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    C8_UNIFORM_01 returns a unit pseudorandom C8.

  Discussion:

    The angle should be uniformly distributed between 0 and 2 * PI,
    the square root of the radius uniformly distributed between 0 and 1.

    This results in a uniform distribution of values in the unit circle.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    06 October 2010

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, double complex C8_UNIFORM_01, a pseudorandom complex value.
*/
{
  int i4_huge = 2147483647;
  int k;
  double r;
  double r8_pi = 3.141592653589793;
  double theta;
  double complex value;

  if ( *seed == 0 )
  {
    printf ( "\n" );
    printf ( "C8_UNIFORM_01 - Fatal error!\n" );
    printf ( "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = sqrt ( ( ( double ) ( *seed ) * 4.656612875E-10 ) );

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  theta = 2.0 * r8_pi * ( ( double ) ( *seed ) * 4.656612875E-10 );

  value = r * cos ( theta ) + r * sin ( theta ) * I;
  
  return value;
}
/******************************************************************************/

void c8mat_print ( int m, int n, double complex a[], char *title )

/******************************************************************************/
/*
  Purpose:

    C8MAT_PRINT prints a C8MAT.

  Discussion:

    A C8MAT is a matrix of double precision complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input, double complex A[M*N], the matrix.

    Input, char *TITLE, a title.
*/
{
  c8mat_print_some ( m, n, a, 1, 1, m, n, title );

  return;
}
/******************************************************************************/

void c8mat_print_some ( int m, int n, double complex a[], int ilo, int jlo, 
  int ihi, int jhi, char *title )

/******************************************************************************/
/*
  Purpose:

    C8MAT_PRINT_SOME prints some of a C8MAT.

  Discussion:

    A C8MAT is a matrix of double precision complex values.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    08 July 2011

  Author:

    John Burkardt

  Parameters:

    Input, int M, N, the number of rows and columns in the matrix.

    Input, double complex A[M*N], the matrix.

    Input, int ILO, JLO, IHI, JHI, the first row and
    column, and the last row and column to be printed.

    Input, char *TITLE, a title.
*/
{
  double complex c;
  int i;
  int i2hi;
  int i2lo;
  int inc;
  int incx = 4;
  int j;
  int j2;
  int j2hi;
  int j2lo;

  printf ( "\n" );
  printf ( "%s\n", title );
/*
  Print the columns of the matrix, in strips of INCX.
*/
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + incx )
  {
    j2hi = j2lo + incx - 1;
    j2hi = i4_min ( j2hi, n );
    j2hi = i4_min ( j2hi, jhi );

    inc = j2hi + 1 - j2lo;

    printf ( "\n" );
    printf ( "  Col: " );
    for ( j = j2lo; j <= j2hi; j++ )
    {
      j2 = j + 1 - j2lo;
      printf ( "          %10d", j );
    }
    printf ( "\n" );
    printf ( "  Row\n" );
    printf ( "  ---\n" );
/*
  Determine the range of the rows in this strip.
*/
    i2lo = i4_max ( ilo, 1 );
    i2hi = i4_min ( ihi, m );

    for ( i = i2lo; i <= i2hi; i++ )
    {
/*
  Print out (up to) INCX entries in row I, that lie in the current strip.
*/
      for ( j2 = 1; j2 <= inc; j2++ )
      {
        j = j2lo - 1 + j2;
        c = a[i-1+(j-1)*m];
        printf ( "  %8g  %8g", creal ( c ), cimag ( c ) );
      }
      printf ( "\n" );
    }
  }
  return;
}
/******************************************************************************/

double complex *c8mat_test ( int n )

/******************************************************************************/
/*
  Purpose:

    C8MAT_TEST returns a test matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.

    Output, double complex C8MAT_TEST[N*N], the Fourier matrix.
*/
{
  double complex *a;
  double complex angle;
  int i;
  int j;
  double pi = 3.141592653589793;

  a = ( double complex * ) malloc ( n * n * sizeof ( double complex ) );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      angle = 2.0 * pi * ( double ) ( i * j ) * I / ( double ) ( n );

      a[i+j*n] = cexp ( angle ) / sqrt ( ( double ) ( n ) );
    }
  }
  return a;
}
/******************************************************************************/

double complex *c8mat_test_inverse ( int n )

/******************************************************************************/
/*
  Purpose:

    C8MAT_TEST_INVERSE returns the inverse of a test matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    04 April 2014

  Author:

    John Burkardt

  Parameters:

    Input, int N, the order of the matrix.

    Output, double complex C8MAT_TEST_INVERSE[N*N], the matrix.
*/
{
  double complex *a;
  int i;
  int j;
  double complex t;

  a = c8mat_test ( n );

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < j; i++ )
    {
      t        = conj ( a[i+j*n] );
      a[i+j*n] = conj ( a[j+i*n] );
      a[j+i*n] = t;
    }
  }
  return a;
}
/******************************************************************************/

float cabs1 ( float complex z )

/******************************************************************************/
/*
  Purpose:

    CABS1 returns the L1 norm of a number.

  Discussion:

    This routine uses single precision complex arithmetic.

    The L1 norm of a complex number is the sum of the absolute values
    of the real and imaginary components.

    CABS1 ( Z ) = abs ( real ( Z ) ) + abs ( imaginary ( Z ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for FORTRAN usage,
    ACM Transactions on Mathematical Software,
    Volume 5, Number 3, pages 308-323, 1979.

  Parameters:

    Input, float complex Z, the number whose norm is desired.

    Output, float CABS1, the L1 norm of Z.
*/
{
  float value;

  value = fabs ( creal ( z ) ) + fabs ( cimag ( z ) );

  return value;
}
/******************************************************************************/

float cabs2 ( float complex z )

/******************************************************************************/
/*
  Purpose:

    CABS2 returns the L2 norm of a number.

  Discussion:

    This routine uses single precision complex arithmetic.

    The L2 norm of a complex number is the square root of the sum 
    of the squares of the real and imaginary components.

    CABS2 ( Z ) = sqrt ( ( real ( Z ) )^2 + ( imaginary ( Z ) )^2 )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    John Burkardt

  Reference:

    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for FORTRAN usage,
    ACM Transactions on Mathematical Software,
    Volume 5, Number 3, pages 308-323, 1979.

  Parameters:

    Input, float complex Z, the number whose norm is desired.

    Output, float CABS2, the L2 norm of Z.
*/
{
  float value;

  value = sqrt ( pow ( creal ( z ), 2 ) 
               + pow ( cimag ( z ), 2 ) );

  return value;
}
/******************************************************************************/

float cmach ( int job )

/******************************************************************************/
/*
  Purpose:

    CMACH computes machine parameters for complex arithmetic.

  Discussion:

    Assume the computer has

      B = base of arithmetic;
      T = number of base B digits;
      L = smallest possible exponent;
      U = largest possible exponent;

    then

      EPS = B^(1-T)
      TINY = 100.0 * B^(-L+T)
      HUGE = 0.01 * B^(U-T)

    If complex division is done by

      1 / (X+i*Y) = (X-i*Y) / (X^2+Y^2)

    then

      TINY = sqrt ( TINY )
      HUGE = sqrt ( HUGE )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for FORTRAN usage,
    ACM Transactions on Mathematical Software,
    Volume 5, Number 3, pages 308-323, 1979.

  Parameters:

    Input, int JOB:
    1, EPS is desired;
    2, TINY is desired;
    3, HUGE is desired.

    Output, float CMACH, the requested value.
*/
{
  float eps;
  float huge;
  float s;
  float complex temp1;
  float complex temp2;
  float complex temp3;
  float tiny;
  float value;

  eps = 1.0;

  for ( ; ; )
  {
    eps = eps / 2.0;
    s = 1.0 + eps;
    if ( s <= 1.0 )
    {
      break;
    }
  }

  eps = 2.0 * eps;

  s = 1.0;

  for ( ; ; )
  {
    tiny = s;
    s = s / 16.0;

    if ( s * 1.0 == 0.0 )
    {
      break;
    }
  }

  tiny = ( tiny / eps ) * 100.0;
/*
  Had to insert this manually!
*/
  tiny = sqrt ( tiny );

  if ( 0 )
  {
    temp1 = 1.0; 
    temp2 = tiny;
    temp3 = temp1 / temp2;

    s = creal ( temp3 );

    if ( s != 1.0 / tiny )
    {
      tiny = sqrt ( tiny );
    }
  }

  huge = 1.0 / tiny;

  if ( job == 1 )
  {
    value = eps;
  }
  else if ( job == 2 )
  {
    value = tiny;
  }
  else if ( job == 3 )
  {
    value = huge;
  }
  else
  {
    value = 0.0;
  }

  return value;
}
/******************************************************************************/

float complex csign1 ( float complex z1, float complex z2 )

/******************************************************************************/
/*
  Purpose:

    CSIGN1 is a transfer-of-sign function.

  Discussion:

    This routine uses single precision complex arithmetic.

    The L1 norm is used.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, float complex Z1, Z2, the arguments.

    Output, float complex CSIGN1,  a complex value, with the magnitude of
    Z1, and the argument of Z2.
*/
{
  float complex value;

  if ( cabs1 ( z2 ) == 0.0 )
  {
    value = 0.0;
  }
  else
  {
    value = cabs1 ( z1 ) * ( z2 / cabs1 ( z2 ) );
  }

  return value;
}
/******************************************************************************/

float complex csign2 ( float complex z1, float complex z2 )

/******************************************************************************/
/*
  Purpose:

    CSIGN2 is a transfer-of-sign function.

  Discussion:

    This routine uses single precision complex arithmetic.

    The L2 norm is used.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, float complex Z1, Z2, the arguments.

    Output, float complex CSIGN2,  a complex value, with the magnitude of
    Z1, and the argument of Z2.
*/
{
  float complex value;

  if ( cabs2 ( z2 ) == 0.0 )
  {
    value = 0.0;
  }
  else
  {
    value = cabs2 ( z1 ) * ( z2 / cabs2 ( z2 ) );
  }

  return value;
}
/******************************************************************************/

double dmach ( int job )

/******************************************************************************/
/*
  Purpose:

    DMACH computes machine parameters of double precision real arithmetic.

  Discussion:

    This routine is for testing only.  It is not required by LINPACK.

    If there is trouble with the automatic computation of these quantities,
    they can be set by direct assignment statements.

    We assume the computer has

      B = base of arithmetic;
      T = number of base B digits;
      L = smallest possible exponent;
      U = largest possible exponent.

    then

      EPS = B^(1-T)
      TINY = 100.0 * B^(-L+T)
      HUGE = 0.01 * B^(U-T)

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 May 2005

  Author:

    Original FORTRAN77 version by Charles Lawson, Richard Hanson, 
    David Kincaid, Fred Krogh.
    C version by John Burkardt.

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for Fortran Usage,
    Algorithm 539, 
    ACM Transactions on Mathematical Software, 
    Volume 5, Number 3, September 1979, pages 308-323.

  Parameters:

    Input, int JOB:
    1: requests EPS;
    2: requests TINY;
    3: requests HUGE.

    Output, double DMACH, the requested value.
*/
{
  double eps;
  double huge;
  double s;
  double tiny;
  double value;

  eps = 1.0;
  for ( ; ; )
  {
    value = 1.0 + ( eps / 2.0 );
    if ( value <= 1.0 )
    {
      break;
    }
    eps = eps / 2.0;
  }

  s = 1.0;

  for ( ; ; )
  {
    tiny = s;
    s = s / 16.0;

    if ( s * 1.0 == 0.0 )
    {
      break;
    }

  }

  tiny = ( tiny / eps ) * 100.0;
  huge = 1.0 / tiny;

  if ( job == 1 )
  {
    value = eps;
  }
  else if ( job == 2 )
  {
    value = tiny;
  }
  else if ( job == 3 )
  {
    value = huge;
  }
  else
  {
    printf ( "\n" );
    printf ( "DMACH - Fatal error!\n" );
    printf ( "  Illegal input value of JOB = %d\n", job );
    exit ( 1 );
  }

  return value;
}
/******************************************************************************/

int i4_max ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MAX returns the maximum of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, are two integers to be compared.

    Output, int I4_MAX, the larger of I1 and I2.
*/
{
  int value;

  if ( i2 < i1 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

int i4_min ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MIN returns the smaller of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, two integers to be compared.

    Output, int I4_MIN, the smaller of I1 and I2.
*/
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

int lsame ( char ca, char cb )

/******************************************************************************/
/*
  Purpose:

    LSAME returns TRUE if CA is the same letter as CB regardless of case.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for Fortran Usage,
    Algorithm 539, 
    ACM Transactions on Mathematical Software, 
    Volume 5, Number 3, September 1979, pages 308-323.

  Parameters:

    Input, char CA, CB, the characters to compare.

    Output, int LSAME, is 1 if the characters are equal,
    disregarding case.
*/
{
  if ( ca == cb )
  {
    return 1;
  }

  if ( 'A' <= ca && ca <= 'Z' )
  {
    if ( ca - 'A' == cb - 'a' )
    {
      return 1;
    }    
  }
  else if ( 'a' <= ca && ca <= 'z' )
  {
    if ( ca - 'a' == cb - 'A' )
    {
      return 1;
    }
  }

  return 0;
}
/******************************************************************************/

float r4_abs ( float x )

/******************************************************************************/
/*
  Purpose:

    R4_ABS returns the absolute value of an R4.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, float X, the quantity whose absolute value is desired.

    Output, float R4_ABS, the absolute value of X.
*/
{
  float value;

  if ( 0.0 <= x )
  {
    value = x;
  } 
  else
  {
    value = -x;
  }
  return value;
}
/******************************************************************************/

float r4_max ( float x, float y )

/******************************************************************************/
/*
  Purpose:

    R4_MAX returns the maximum of two R4's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    29 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, float X, Y, the quantities to compare.

    Output, float R4_MAX, the maximum of X and Y.
*/
{
  float value;

  if ( y < x )
  {
    value = x;
  } 
  else
  {
    value = y;
  }
  return value;
}
/******************************************************************************/

float r4_sign ( float x )

/******************************************************************************/
/*
  Purpose:

    R4_SIGN returns the sign of an R4.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, float X, the number whose sign is desired.

    Output, float R4_SIGN, the sign of X.
*/
{
  float value;

  if ( x < 0.0 )
  {
    value = -1.0;
  } 
  else
  {
    value = 1.0;
  }
  return value;
}
/******************************************************************************/

float r4_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    R4_UNIFORM_01 returns a real pseudorandom R4.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2^31 - 1 )
      r4_uniform_01 = seed / ( 2^31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

    If the initial seed is 12345, then the first three computations are

      Input     Output      R4_UNIFORM_01
      SEED      SEED

         12345   207482415  0.096616
     207482415  1790989824  0.833995
    1790989824  2035175616  0.947702

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    16 November 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation
    edited by Jerry Banks,
    Wiley Interscience, page 95, 1998.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, pages 136-143, 1969.

  Parameters:

    Input/output, int *SEED, the "seed" value.  Normally, this
    value should not be 0.  On output, SEED has been updated.

    Output, float R4_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
  int k;
  float r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
/*
  Although SEED can be represented exactly as a 32 bit integer,
  it generally cannot be represented exactly as a 32 bit real number!
*/
  r = ( float ) ( *seed ) * 4.656612875E-10;

  return r;
}
/******************************************************************************/

float r4_uniform_ab ( float a, float b, int *seed )

/******************************************************************************/
/*
  Purpose:

    R4_UNIFORM_AB returns a scaled pseudorandom R4.

  Discussion:

    The pseudorandom number should be uniformly distributed
    between A and B.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    19 April 2011

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input, float A, B, the limits of the interval.

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, float R4_UNIFORM_AB, a number strictly between A and B.
*/
{
  int i4_huge = 2147483647;
  int k;
  float value;

  if ( *seed == 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "R4_UNIFORM_AB - Fatal error!\n" );
    fprintf ( stderr, "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  value = ( float ) ( *seed ) * 4.656612875E-10;

  value = a + ( b - a ) * value;

  return value;
}
/******************************************************************************/

void r4mat_print ( int m, int n, float a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R4MAT_PRINT prints an R4MAT.

  Discussion:

    An R4MAT is a doubly dimensioned array of R4 values, stored as a vector
    in column-major order.

    Entry A(I,J) is stored as A[I+J*M]

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 May 2008

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows in A.

    Input, int N, the number of columns in A.

    Input, float A[M*N], the M by N matrix.

    Input, char *TITLE, a title.
*/
{
  r4mat_print_some ( m, n, a, 1, 1, m, n, title );

  return;
}
/******************************************************************************/

void r4mat_print_some ( int m, int n, float a[], int ilo, int jlo, int ihi,
  int jhi, char *title )

/******************************************************************************/
/*
  Purpose:

    R4MAT_PRINT_SOME prints some of an R4MAT.

  Discussion:

    An R4MAT is a doubly dimensioned array of R4 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    26 June 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows of the matrix.
    M must be positive.

    Input, int N, the number of columns of the matrix.
    N must be positive.

    Input, float A[M*N], the matrix.

    Input, int ILO, JLO, IHI, JHI, designate the first row and
    column, and the last row and column to be printed.

    Input, char *TITLE, a title.
*/
{
# define INCX 5

  int i;
  int i2hi;
  int i2lo;
  int j;
  int j2hi;
  int j2lo;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );

  if ( m <= 0 || n <= 0 )
  {
    fprintf ( stdout, "\n" );
    fprintf ( stdout, "  (None)\n" );
    return;
  }
/*
  Print the columns of the matrix, in strips of 5.
*/
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX )
  {
    j2hi = j2lo + INCX - 1;
    if ( n < j2hi )
    {
      j2hi = n;
    }
    if ( jhi < j2hi )
    {
      j2hi = jhi;
    }

    fprintf ( stdout, "\n" );
/*
  For each column J in the current range...

  Write the header.
*/
    fprintf ( stdout, "  Col:  ");
    for ( j = j2lo; j <= j2hi; j++ )
    {
      fprintf ( stdout, "  %7d     ", j - 1 );
    }
    fprintf ( stdout, "\n" );
    fprintf ( stdout, "  Row\n" );
    fprintf ( stdout, "\n" );
/*
  Determine the range of the rows in this strip.
*/
    if ( 1 < ilo )
    {
      i2lo = ilo;
    }
    else
    {
      i2lo = 1;
    }
    if ( m < ihi )
    {
      i2hi = m;
    }
    else
    {
      i2hi = ihi;
    }

    for ( i = i2lo; i <= i2hi; i++ )
    {
/*
  Print out (up to) 5 entries in row I, that lie in the current strip.
*/
      fprintf ( stdout, "%5d:", i - 1 );
      for ( j = j2lo; j <= j2hi; j++ )
      {
        fprintf ( stdout, "  %14g", a[i-1+(j-1)*m] );
      }
      fprintf ( stdout, "\n" );
    }
  }

  return;
# undef INCX
}
/******************************************************************************/

float *r4mat_test ( char trans, int lda, int m, int n )

/******************************************************************************/
/*
  Purpose:

    R4MAT_TEST sets up a test matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.
    
  Modified:

    10 February 2014

  Author:

    John Burkardt.

  Parameters:

    Input, char TRANS, indicates whether matrix is to be transposed.
    'N', no transpose.
    'T', transpose the matrix.

    Input, int LDA, the leading dimension of the matrix.

    Input, int M, N, the number of rows and columns of the matrix.

    Output, float R4MAT_TEST[LDA*?], the matrix.
    if TRANS is 'N', then the matrix is stored in LDA*N entries,
    as an M x N matrix;
    if TRANS is 'T', then the matrix is stored in LDA*M entries,
    as an N x M matrix.
*/
{
  float *a;
  int i;
  int j;

  if ( trans == 'N' )
  {
    a = ( float * ) malloc ( lda * n * sizeof ( float ) );

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        a[i+j*lda] = ( float ) ( 10 * ( i + 1 ) + ( j + 1 ) );
      }
    }
  }
  else
  {
    a = ( float * ) malloc ( lda * m * sizeof ( float ) );

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        a[j+i*lda] = ( float ) ( 10 * ( i + 1 ) + ( j + 1 ) );
      }
    }
  }
  return a;
}
/******************************************************************************/

void r4mat_uniform_01 ( int m, int n, int *seed, float r[] )

/******************************************************************************/
/*
  Purpose:

    R4MAT_UNIFORM_01 fills a float array with unit pseudorandom values.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2**31 - 1 )
      unif = seed / ( 2**31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    23 June 2009

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, L E Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    P A Lewis, A S Goodman, J M Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, pages 136-143, 1969.

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input/output, int *SEED, the "seed" value.  Normally, this
    value should not be 0, otherwise the output value of SEED
    will still be 0, and D_UNIFORM will be 0.  On output, SEED has
    been updated.

    Output, float R4MAT_UNIFORM_01[M*N], a matrix of pseudorandom values.
*/
{
  int i;
  int j;
  int k;

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
      {
        *seed = *seed + 2147483647;
      }
/*
  Although SEED can be represented exactly as a 32 bit integer,
  it generally cannot be represented exactly as a 32 bit real number!
*/
      r[i+j*m] = ( float ) ( *seed ) * 4.656612875E-10;
    }
  }

  return;
}
/******************************************************************************/

void r4vec_print ( int n, float a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R4VEC_PRINT prints an R4VEC.

  Discussion:

    An R4VEC is a vector of R4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 April 2009

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input, float A[N], the vector to be printed.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );
  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %8d: %14f\n", i, a[i] );
  }

  return;
}
/******************************************************************************/

double r8_abs ( double x )

/******************************************************************************/
/*
  Purpose:

    R8_ABS returns the absolute value of a R8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    02 April 2005

  Author:

    John Burkardt

  Parameters:

    Input, double X, the quantity whose absolute value is desired.

    Output, double R8_ABS, the absolute value of X.
*/
{
  double value;

  if ( 0.0 <= x )
  {
    value = x;
  } 
  else
  {
    value = -x;
  }
  return value;
}
/******************************************************************************/

double r8_max ( double x, double y )

/******************************************************************************/
/*
  Purpose:

    R8_MAX returns the maximum of two R8's.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 August 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, Y, the quantities to compare.

    Output, double R8_MAX, the maximum of X and Y.
*/
{
  double value;

  if ( y < x )
  {
    value = x;
  } 
  else
  {
    value = y;
  }
  return value;
}
/******************************************************************************/

double r8_sign ( double x )

/******************************************************************************/
/*
  Purpose:

    R8_SIGN returns the sign of a R8.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 October 2004

  Author:

    John Burkardt

  Parameters:

    Input, double X, the number whose sign is desired.

    Output, double R8_SIGN, the sign of X.
*/
{
  double value;

  if ( x < 0.0 )
  {
    value = -1.0;
  } 
  else
  {
    value = 1.0;
  }
  return value;
}
/******************************************************************************/

double r8_uniform_01 ( int *seed )

/******************************************************************************/
/*
  Purpose:

    R8_UNIFORM_01 returns a pseudorandom R8 scaled to [0,1].

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2^31 - 1 )
      r8_uniform_01 = seed / ( 2^31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

    If the initial seed is 12345, then the first three computations are

      Input     Output      R8_UNIFORM_01
      SEED      SEED

         12345   207482415  0.096616
     207482415  1790989824  0.833995
    1790989824  2035175616  0.947702

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    11 August 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation
    edited by Jerry Banks,
    Wiley Interscience, page 95, 1998.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    P A Lewis, A S Goodman, J M Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, pages 136-143, 1969.

  Parameters:

    Input/output, int *SEED, the "seed" value.  Normally, this
    value should not be 0.  On output, SEED has been updated.

    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
  const int i4_huge = 2147483647;
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = ( ( double ) ( *seed ) ) * 4.656612875E-10;

  return r;
}
/******************************************************************************/

double r8_uniform_ab ( double a, double b, int *seed )

/******************************************************************************/
/*
  Purpose:

    R8_UNIFORM_AB returns a pseudorandom R8 scaled to [A,B].

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    21 November 2004

  Author:

    John Burkardt

  Parameters:

    Input, double A, B, the limits of the interval.

    Input/output, int *SEED, the "seed" value, which should NOT be 0.
    On output, SEED has been updated.

    Output, double R8_UNIFORM_AB, a number strictly between A and B.
*/
{
  const int i4_huge = 2147483647;
  int k;
  double r;
  double value;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + i4_huge;
  }

  r = ( ( double ) ( *seed ) ) * 4.656612875E-10;

  value = a + ( b - a ) * r;

  return value;
}
/******************************************************************************/

void r8mat_print ( int m, int n, double a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R8MAT_PRINT prints an R8MAT.

  Discussion:

    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
    in column-major order.

    Entry A(I,J) is stored as A[I+J*M]

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    28 May 2008

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows in A.

    Input, int N, the number of columns in A.

    Input, double A[M*N], the M by N matrix.

    Input, char *TITLE, a title.
*/
{
  r8mat_print_some ( m, n, a, 1, 1, m, n, title );

  return;
}
/******************************************************************************/

void r8mat_print_some ( int m, int n, double a[], int ilo, int jlo, int ihi,
  int jhi, char *title )

/******************************************************************************/
/*
  Purpose:

    R8MAT_PRINT_SOME prints some of an R8MAT.

  Discussion:

    An R8MAT is a doubly dimensioned array of R8 values, stored as a vector
    in column-major order.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    26 June 2013

  Author:

    John Burkardt

  Parameters:

    Input, int M, the number of rows of the matrix.
    M must be positive.

    Input, int N, the number of columns of the matrix.
    N must be positive.

    Input, double A[M*N], the matrix.

    Input, int ILO, JLO, IHI, JHI, designate the first row and
    column, and the last row and column to be printed.

    Input, char *TITLE, a title.
*/
{
# define INCX 5

  int i;
  int i2hi;
  int i2lo;
  int j;
  int j2hi;
  int j2lo;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );

  if ( m <= 0 || n <= 0 )
  {
    fprintf ( stdout, "\n" );
    fprintf ( stdout, "  (None)\n" );
    return;
  }
/*
  Print the columns of the matrix, in strips of 5.
*/
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX )
  {
    j2hi = j2lo + INCX - 1;
    if ( n < j2hi )
    {
      j2hi = n;
    }
    if ( jhi < j2hi )
    {
      j2hi = jhi;
    }

    fprintf ( stdout, "\n" );
/*
  For each column J in the current range...

  Write the header.
*/
    fprintf ( stdout, "  Col:  ");
    for ( j = j2lo; j <= j2hi; j++ )
    {
      fprintf ( stdout, "  %7d     ", j - 1 );
    }
    fprintf ( stdout, "\n" );
    fprintf ( stdout, "  Row\n" );
    fprintf ( stdout, "\n" );
/*
  Determine the range of the rows in this strip.
*/
    if ( 1 < ilo )
    {
      i2lo = ilo;
    }
    else
    {
      i2lo = 1;
    }
    if ( m < ihi )
    {
      i2hi = m;
    }
    else
    {
      i2hi = ihi;
    }

    for ( i = i2lo; i <= i2hi; i++ )
    {
/*
  Print out (up to) 5 entries in row I, that lie in the current strip.
*/
      fprintf ( stdout, "%5d:", i - 1 );
      for ( j = j2lo; j <= j2hi; j++ )
      {
        fprintf ( stdout, "  %14g", a[i-1+(j-1)*m] );
      }
      fprintf ( stdout, "\n" );
    }
  }

  return;
# undef INCX
}
/******************************************************************************/

double *r8mat_test ( char trans, int lda, int m, int n )

/******************************************************************************/
/*
  Purpose:

    R8MAT_TEST sets up a test matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.
    
  Modified:

    10 February 2014

  Author:

    John Burkardt.

  Parameters:

    Input, char TRANS, indicates whether matrix is to be transposed.
    'N', no transpose.
    'T', transpose the matrix.

    Input, int LDA, the leading dimension of the matrix.

    Input, int M, N, the number of rows and columns of the matrix.

    Output, double R8MAT_TEST[LDA*?], the matrix.
    if TRANS is 'N', then the matrix is stored in LDA*N entries,
    as an M x N matrix;
    if TRANS is 'T', then the matrix is stored in LDA*M entries,
    as an N x M matrix.
*/
{
  double *a;
  int i;
  int j;

  if ( trans == 'N' )
  {
    a = ( double * ) malloc ( lda * n * sizeof ( double ) );

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        a[i+j*lda] = ( double ) ( 10 * ( i + 1 ) + ( j + 1 ) );
      }
    }
  }
  else
  {
    a = ( double * ) malloc ( lda * m * sizeof ( double ) );

    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < m; i++ )
      {
        a[j+i*lda] = ( double ) ( 10 * ( i + 1 ) + ( j + 1 ) );
      }
    }
  }
  return a;
}
/******************************************************************************/

void r8mat_uniform_01 ( int m, int n, int *seed, double r[] )

/******************************************************************************/
/*
  Purpose:

    R8MAT_UNIFORM_01 returns a unit pseudorandom R8MAT.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2**31 - 1 )
      unif = seed / ( 2**31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    03 October 2005

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Second Edition,
    Springer, 1987,
    ISBN: 0387964673,
    LC: QA76.9.C65.B73.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, December 1986, pages 362-376.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation,
    edited by Jerry Banks,
    Wiley, 1998,
    ISBN: 0471134031,
    LC: T57.62.H37.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, Number 2, 1969, pages 136-143.

  Parameters:

    Input, int M, N, the number of rows and columns.

    Input/output, int *SEED, the "seed" value.  Normally, this
    value should not be 0.  On output, SEED has
    been updated.

    Output, double R[M*N], a matrix of pseudorandom values.
*/
{
  int i;
  int i4_huge = 2147483647;
  int j;
  int k;

  if ( *seed == 0 )
  {
    printf ( "\n" );
    printf ( "R8MAT_UNIFORM_01 - Fatal error!\n" );
    printf ( "  Input value of SEED = 0.\n" );
    exit ( 1 );
  }

  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      k = *seed / 127773;

      *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

      if ( *seed < 0 )
      {
        *seed = *seed + i4_huge;
      }
/*
  Although SEED can be represented exactly as a 32 bit integer,
  it generally cannot be represented exactly as a 32 bit real number!
*/
      r[i+j*m] = ( double ) ( *seed ) * 4.656612875E-10;
    }
  }

  return;
}
/******************************************************************************/

void r8vec_print ( int n, double a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R8VEC_PRINT prints an R8VEC.

  Discussion:

    An R8VEC is a vector of R8's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 April 2009

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input, double A[N], the vector to be printed.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );
  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %8d: %14g\n", i, a[i] );
  }

  return;
}
/******************************************************************************/

float smach ( int job )

/******************************************************************************/
/*
  Purpose:

    SMACH computes machine parameters of single precision real arithmetic.

  Discussion:

    This routine is for testing only.  It is not required by LINPACK.

    If there is trouble with the automatic computation of these quantities,
    they can be set by direct assignment statements.

    We assume the computer has

      B = base of arithmetic;
      T = number of base B digits;
      L = smallest possible exponent;
      U = largest possible exponent.

    then

      EPS = B^(1-T)
      TINY = 100.0 * B^(-L+T)
      HUGE = 0.01 * B^(U-T)

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    29 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Algorithm 539: 
    Basic Linear Algebra Subprograms for Fortran Usage,
    ACM Transactions on Mathematical Software, 
    Volume 5, Number 3, September 1979, pages 308-323.

  Parameters:

    Input, int JOB:
    1: requests EPS;
    2: requests TINY;
    3: requests HUGE.

    Output, float SMACH, the requested value.
*/
{
  float eps;
  float huge;
  float s;
  float tiny;
  float value;

  eps = 1.0;
  for ( ; ; )
  {
    value = 1.0 + ( eps / 2.0 );
    if ( value <= 1.0 )
    {
      break;
    }
    eps = eps / 2.0;
  }

  s = 1.0;

  for ( ; ; )
  {
    tiny = s;
    s = s / 16.0;

    if ( s * 1.0 == 0.0 )
    {
      break;
    }

  }

  tiny = ( tiny / eps ) * 100.0;
  huge = 1.0 / tiny;

  if ( job == 1 )
  {
    value = eps;
  }
  else if ( job == 2 )
  {
    value = tiny;
  }
  else if ( job == 3 )
  {
    value = huge;
  }
  else
  {
    printf ( "\n" );
    printf ( "SMACH - Fatal error!\n" );
    printf ( "  Illegal input value of JOB = %d\n", job );
    exit ( 1 );
  }

  return value;
}
/******************************************************************************/

void timestamp ( )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
/******************************************************************************/

void xerbla ( char *srname, int info )

/******************************************************************************/
/*
  Purpose:

    XERBLA is an error handler for the LAPACK routines.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    30 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Jim Bunch, Cleve Moler, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979,
    ISBN13: 978-0-898711-72-1,
    LC: QA214.L56.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for Fortran Usage,
    Algorithm 539, 
    ACM Transactions on Mathematical Software, 
    Volume 5, Number 3, September 1979, pages 308-323.

  Parameters:

    Input, char *SRNAME, the name of the routine
    which called XERBLA.

    Input, int INFO, the position of the invalid parameter in
    the parameter list of the calling routine.
*/
{
  printf ( "\n" );
  printf ( "XERBLA - Fatal error!\n" );
  printf ( "  On entry to routine %s\n", srname );
  printf ( "  input parameter number %d had an illegal value.\n", info );
  exit ( 1 );
}
/******************************************************************************/

double zabs1 ( double complex z )

/******************************************************************************/
/*
  Purpose:

    ZABS1 returns the L1 norm of a number.

  Discussion:

    This routine uses double precision complex arithmetic.

    The L1 norm of a complex number is the sum of the absolute values
    of the real and imaginary components.

    ZABS1 ( Z ) = abs ( real ( Z ) ) + abs ( imaginary ( Z ) )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for FORTRAN usage,
    ACM Transactions on Mathematical Software,
    Volume 5, Number 3, pages 308-323, 1979.

  Parameters:

    Input, double complex Z, the number whose norm is desired.

    Output, double ZABS1, the L1 norm of Z.
*/
{
  double value;

  value = fabs ( creal ( z ) ) + fabs ( cimagf ( z ) );

  return value;
}
/******************************************************************************/

double zabs2 ( double complex z )

/******************************************************************************/
/*
  Purpose:

    ZABS2 returns the L2 norm of a number.

  Discussion:

    This routine uses double precision complex arithmetic.

    The L2 norm of a complex number is the square root of the sum 
    of the squares of the real and imaginary components.

    ZABS2 ( Z ) = sqrt ( ( real ( Z ) )**2 + ( imaginary ( Z ) )**2 )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 March 2007

  Author:

    John Burkardt

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for FORTRAN usage,
    ACM Transactions on Mathematical Software,
    Volume 5, Number 3, pages 308-323, 1979.

  Parameters:

    Input, double complex Z, the number whose norm is desired.

    Output, float ZABS2, the L2 norm of Z.
*/
{
  double value;

  value = sqrt ( pow ( creal ( z ), 2 ) 
               + pow ( cimag ( z ), 2 ) );

  return value;
}
/******************************************************************************/

double zmach ( int job )

/******************************************************************************/
/*
  Purpose:

    ZMACH computes machine parameters for double complex arithmetic.

  Discussion:

    Assume the computer has

      B = base of arithmetic;
      T = number of base B digits;
      L = smallest possible exponent;
      U = largest possible exponent;

    then

      EPS = B^(1-T)
      TINY = 100.0 * B^(-L+T)
      HUGE = 0.01 * B^(U-T)

    If complex division is done by

      1 / (X+i*Y) = (X-i*Y) / (X^2+Y^2)

    then

      TINY = sqrt ( TINY )
      HUGE = sqrt ( HUGE )

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 March 2007

  Author:

    C version by John Burkardt

  Reference:

    Jack Dongarra, Cleve Moler, Jim Bunch, Pete Stewart,
    LINPACK User's Guide,
    SIAM, 1979.

    Charles Lawson, Richard Hanson, David Kincaid, Fred Krogh,
    Basic Linear Algebra Subprograms for FORTRAN usage,
    ACM Transactions on Mathematical Software,
    Volume 5, Number 3, pages 308-323, 1979.

  Parameters:

    Input, int JOB:
    1, EPS is desired;
    2, TINY is desired;
    3, HUGE is desired.

    Output, double ZMACH, the requested value.
*/
{
  double eps;
  double huge;
  double s;
  double complex temp1;
  double complex temp2;
  double complex temp3;
  double tiny;
  double value;

  eps = 1.0;

  for ( ; ; )
  {
    eps = eps / 2.0;
    s = 1.0 + eps;
    if ( s <= 1.0 )
    {
      break;
    }
  }

  eps = 2.0 * eps;

  s = 1.0;

  for ( ; ; )
  {
    tiny = s;
    s = s / 16.0;

    if ( s * 1.0 == 0.0 )
    {
      break;
    }
  }

  tiny = ( tiny / eps ) * 100.0;
/*
  Had to insert this manually!
*/
  tiny = sqrt ( tiny );

  if ( 0 )
  {
    temp1 = 1.0; 
    temp2 = tiny;
    temp3 = temp1 / temp2;

    s = creal ( temp3 );

    if ( s != 1.0 / tiny )
    {
      tiny = sqrt ( tiny );
    }
  }

  huge = 1.0 / tiny;

  if ( job == 1 )
  {
    value = eps;
  }
  else if ( job == 2 )
  {
    value = tiny;
  }
  else if ( job == 3 )
  {
    value = huge;
  }
  else
  {
    value = 0.0;
  }

  return value;
}
/******************************************************************************/

double complex zsign1 ( double complex z1, double complex z2 )

/******************************************************************************/
/*
  Purpose:

    ZSIGN1 is a transfer-of-sign function.

  Discussion:

    This routine uses double precision complex arithmetic.

    The L1 norm is used.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, double complex Z1, Z2, the arguments.

    Output, double complex ZSIGN1,  a complex value, with the magnitude of
    Z1, and the argument of Z2.
*/
{
  double complex value;

  if ( zabs1 ( z2 ) == 0.0 )
  {
    value = 0.0;
  }
  else
  {
    value = zabs1 ( z1 ) * ( z2 / zabs1 ( z2 ) );
  }

  return value;
}
/******************************************************************************/

double complex zsign2 ( double complex z1, double complex z2 )

/******************************************************************************/
/*
  Purpose:

    ZSIGN2 is a transfer-of-sign function.

  Discussion:

    This routine uses double precision complex arithmetic.

    The L2 norm is used.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 March 2007

  Author:

    John Burkardt

  Parameters:

    Input, double complex Z1, Z2, the arguments.

    Output, double complex ZSIGN2,  a complex value, with the magnitude of
    Z1, and the argument of Z2.
*/
{
  double complex value;

  if ( zabs2 ( z2 ) == 0.0 )
  {
    value = 0.0;
  }
  else
  {
    value = zabs2 ( z1 ) * ( z2 / zabs2 ( z2 ) );
  }

  return value;
}
