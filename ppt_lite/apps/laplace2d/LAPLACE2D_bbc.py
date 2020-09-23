import math

def get_BBcountsArray(params):
    ( Nx, Ny ) = params
    import numpy as np
    BBcount = {}
    for i in range(0,30):
        BBcount[i] = 0

    x = math.sqrt( Nx * Ny )

    BBcount[ 2 ] = ( 1.000000 )
    BBcount[ 3 ] = 0
    BBcount[ 4 ] = ( 1.000000 )
    BBcount[ 5 ] = ( 1.000000 + 1.000000*x )
    BBcount[ 6 ] = ( 1.000000*x )
    BBcount[ 7 ] = ( 1.000000*x )
    BBcount[ 8 ] = ( 1.000000 )
    BBcount[ 9 ] = ( 1001.000000 )
    BBcount[ 10 ] = ( 1001.000000 )
    BBcount[ 11 ] = ( 1001.000000 )
    BBcount[ 12 ] = ( 1000.000000 )
    BBcount[ 13 ] = ( -1000.000000 + 1000.000000*x )
    BBcount[ 14 ] = ( -2000.000000 + 1000.000000*x )
    BBcount[ 15 ] = ( 1000.000000*( -1.000000 + Ny )*( -2.000000 + Nx ) )
    BBcount[ 16 ] = ( 1000.000000*( -2.000000 + Nx )*( -2.000000 + Ny ) )
    BBcount[ 17 ] = ( 1000.000000*( -2.000000 + Nx )*( -2.000000 + Ny ) )
    BBcount[ 18 ] = ( -2000.000000 + 1000.000000*x )
    BBcount[ 19 ] = ( -2000.000000 + 1000.000000*x )
    BBcount[ 20 ] = ( 1000.000000 )
    BBcount[ 21 ] = ( -1000.000000 + 1000.000000*x )
    BBcount[ 22 ] = ( -2000.000000 + 1000.000000*x )
    BBcount[ 23 ] = ( 1000.000000*( -1.000000 + Ny )*( -2.000000 + Nx ) )
    BBcount[ 24 ] = ( 1000.000000*( -2.000000 + Nx )*( -2.000000 + Ny ) )
    BBcount[ 25 ] = ( 1000.000000*( -2.000000 + Nx )*( -2.000000 + Ny ) )
    BBcount[ 26 ] = ( -2000.000000 + 1000.000000*x )
    BBcount[ 27 ] = ( -2000.000000 + 1000.000000*x )
    BBcount[ 28 ] = ( 1000.000000 )
    BBcount[ 29 ] = ( 1.000000 )

    return BBcount

