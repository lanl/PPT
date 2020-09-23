def get_BBcountsArray(params):
    ( Nx ) = params
    import numpy as np
    BBcount = {}
    for i in range(0,54):
        BBcount[i] = 0


    BBcount[ 1 ] = ( 200.000000*Nx ) 
    BBcount[ 2 ] = 1 
    BBcount[ 3 ] = 1 
    BBcount[ 4 ] = 1 
    BBcount[ 5 ] = 1 
    BBcount[ 6 ] = 1 
    BBcount[ 7 ] = 1 
    BBcount[ 8 ] = 1 
    BBcount[ 9 ] = 1 
    BBcount[ 10 ] = 1 
    BBcount[ 11 ] = 0 
    BBcount[ 12 ] = 0 
    BBcount[ 13 ] = 0 
    BBcount[ 14 ] = 1 
    BBcount[ 15 ] = 1 
    BBcount[ 16 ] = 1 
    BBcount[ 17 ] = 0 
    BBcount[ 18 ] = 0 
    BBcount[ 19 ] = 0 
    BBcount[ 20 ] = 0 
    BBcount[ 21 ] = 0 
    BBcount[ 22 ] = 0 
    BBcount[ 23 ] = 0 
    BBcount[ 24 ] = 0 
    BBcount[ 25 ] = 0 
    BBcount[ 26 ] = 0 
    BBcount[ 27 ] = 0 
    BBcount[ 28 ] = 0 
    BBcount[ 29 ] = 0 
    BBcount[ 30 ] = 0 
    BBcount[ 31 ] = 0 
    BBcount[ 32 ] = 1 
    BBcount[ 33 ] = 0 
    BBcount[ 34 ] = 0 
    BBcount[ 35 ] = 1 
    BBcount[ 36 ] = 0 
    BBcount[ 37 ] = 0 
    BBcount[ 38 ] = 0 
    BBcount[ 39 ] = (0.250000*Ny*Nz) 
    BBcount[ 40 ] = (0.250000*Ny*Nz) 
    BBcount[ 41 ] = (0.250000*Nx*Ny*Nz) 
    BBcount[ 42 ] = 0 
    BBcount[ 43 ] = 0 
    BBcount[ 44 ] = 0 
    BBcount[ 45 ] = 0 
    BBcount[ 46 ] = 0 
    BBcount[ 47 ] = 1 
    BBcount[ 48 ] = 1 
    BBcount[ 49 ] = 0 
    BBcount[ 50 ] = 1 
    BBcount[ 51 ] = 0 
    BBcount[ 52 ] = 0 
    BBcount[ 53 ] = 0 
    BBcount[ 54 ] = 0 

    return BBcount

