def get_BBcountsArray(params):
    ( Nx ) = params
    import numpy as np
    BBcount = {}
    for i in range(0,25):
        BBcount[i] = 0

    Nx = int(''.join(map(str,Nx)))
    
    BBcount[ 1 ] = 1
    BBcount[ 2 ] = 0
    BBcount[ 3 ] = 1
    BBcount[ 4 ] = 1
    BBcount[ 5 ] = ( 1.000000 + 1.000000*Nx )
    BBcount[ 6 ] = ( 1.000000*Nx )
    BBcount[ 7 ] = ( 1.000000*Nx*( 1.000000 + Nx ) )
    BBcount[ 8 ] = ( 1.000000*pow(Nx,2) )
    BBcount[ 9 ] = ( 100.000000 )
    BBcount[ 10 ] = ( 1.000000*Nx )
    BBcount[ 11 ] = ( 1.000000*Nx )
    BBcount[ 12 ] = 1
    BBcount[ 13 ] = ( 1.000000 + 1.000000*Nx )
    BBcount[ 14 ] = ( 1.000000*Nx )
    BBcount[ 15 ] = ( 1.000000*Nx*( 1.000000 + Nx ) )
    BBcount[ 16 ] = ( 81.000000 )
    BBcount[ 17 ] = ( 1.000000*pow(Nx,2)*( 1.000000 + Nx ) )
    BBcount[ 18 ] = ( 1.000000*pow(Nx,3) )
    BBcount[ 19 ] = ( 1.000000*pow(Nx,3) )
    BBcount[ 20 ] = ( 1.000000*pow(Nx,2) )
    BBcount[ 21 ] = ( 1.000000*pow(Nx,2) )
    BBcount[ 22 ] = ( 1.000000*Nx )
    BBcount[ 23 ] = ( 1.000000*Nx )
    BBcount[ 24 ] = 1

    return BBcount

