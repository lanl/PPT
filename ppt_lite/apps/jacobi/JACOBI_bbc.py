def get_BBcountsArray(params):
    ( Nx ) = params
    import numpy as np
    BBcount = {}
    for i in range(0,100):
        BBcount[i] = 0

    Nx = int(''.join(map(str,Nx)))

    BBcount[ 1 ] = 1 
    BBcount[ 2 ] = ( 1.000000 + 1.000000*Nx ) 
    BBcount[ 3 ] = ( 1.000000*Nx ) 
    BBcount[ 4 ] = ( 1.000000*Nx*( 1.000000 + Nx ) ) 
    BBcount[ 5 ] = ( 1.000000*pow(Nx,2) ) 
    BBcount[ 6 ] = ( 0.944444*Nx ) 
    BBcount[ 7 ] = ( 1.000000 + 1.000000*Nx*( -1.000000 + Nx ) ) 
    BBcount[ 8 ] = ( 1.000000*Nx ) 
    BBcount[ 9 ] = ( 1.000000*pow((( -1.000000 + Nx )),2) ) 
    BBcount[ 10 ] = ( -1.000000 + 1.000000*Nx ) 
    BBcount[ 11 ] = ( 2.000000 + 1.000000*Nx*( -3.000000 + Nx ) ) 
    BBcount[ 12 ] = ( 225.000000 ) 
    BBcount[ 13 ] = ( 1.000000 + 1.000000*Nx*( -1.000000 + Nx ) ) 
    BBcount[ 14 ] = ( 1.000000*pow(Nx,2) ) 
    BBcount[ 15 ] = ( 1.000000*pow(Nx,2) ) 
    BBcount[ 16 ] = ( 1.000000*Nx ) 
    BBcount[ 17 ] = ( 1.000000*Nx ) 
    BBcount[ 18 ] = 1 
    BBcount[ 19 ] = ( 2000.000000 ) 
    BBcount[ 20 ] = ( 2000.000000 + 2000.000000*Nx ) 
    BBcount[ 21 ] = ( 2000.000000*Nx ) 
    BBcount[ 22 ] = ( 2000.000000*Nx*( 1.000000 + Nx ) ) 
    BBcount[ 23 ] = ( 2000.000000*pow(Nx,2) ) 
    BBcount[ 24 ] = ( 2000.000000*Nx*( -1.000000 + Nx ) ) 
    BBcount[ 25 ] =      BBcount[ 26 ] = ( 2000.000000*pow(Nx,2) ) 
    BBcount[ 27 ] = ( 2000.000000*Nx ) 
    BBcount[ 28 ] = ( 2000.000000*Nx ) 
    BBcount[ 29 ] = ( 2000.000000 ) 
    BBcount[ 30 ] = 1 
    BBcount[ 31 ] = ( 1.000000 + 1.000000*Nx ) 
    BBcount[ 32 ] = ( 1.000000*Nx ) 
    BBcount[ 33 ] = ( 1.000000*Nx*( 1.000000 + Nx ) ) 
    BBcount[ 34 ] = ( 1.000000*pow(Nx,2) ) 
    BBcount[ 35 ] = ( 1.000000*pow(Nx,2) ) 
    BBcount[ 36 ] = ( 18.000000 ) 
    BBcount[ 37 ] = ( 1.000000*Nx ) 
    BBcount[ 38 ] = 1 
    BBcount[ 39 ] = ( 2001.000000 ) 
    BBcount[ 40 ] = ( 2001.000000 + 2001.000000*Nx ) 
    BBcount[ 41 ] = ( 2001.000000*Nx )    
    BBcount[ 42 ] = 4002.0*pow(Nx,2)
    BBcount[ 43 ] = 4002.0*pow(Nx,2)
    BBcount[ 44 ] = ( 2001.000000*pow(Nx,2) ) 
    BBcount[ 45 ] = ( 2001.000000*Nx ) 
    BBcount[ 46 ] = ( 2001.000000*Nx ) 
    BBcount[ 47 ] = ( 2001.000000 ) 
    BBcount[ 48 ] = ( 2001.000000 + 2001.000000*Nx ) 
    BBcount[ 49 ] = ( 2001.000000*Nx ) 
    BBcount[ 50 ] = ( 2001.000000*Nx ) 
    BBcount[ 51 ] = ( 2001.000000 ) 
    BBcount[ 52 ] = 0 
    BBcount[ 53 ] = 0 
    BBcount[ 54 ] = 0 
    BBcount[ 55 ] = 0 
    BBcount[ 56 ] = 0 
    BBcount[ 57 ] = ( 2000.000000 ) 
    BBcount[ 58 ] = ( 2000.000000 + 2000.000000*Nx ) 
    BBcount[ 59 ] = ( 2000.000000*Nx ) 
    BBcount[ 60 ] = ( 2000.000000*Nx ) 
    BBcount[ 61 ] = ( 2000.000000 ) 
    BBcount[ 62 ] = 1 
    BBcount[ 63 ] = ( 1.000000 + 1.000000*Nx ) 
    BBcount[ 64 ] = ( 1.000000*Nx ) 
    BBcount[ 65 ] = ( 1.000000*Nx ) 
    BBcount[ 66 ] = 1 
    BBcount[ 67 ] = ( 2.000000 ) 
    BBcount[ 68 ] = 1 
    BBcount[ 69 ] = 0 
    BBcount[ 70 ] = 1 
    BBcount[ 71 ] = 1 
    BBcount[ 72 ] = ( 1.000000 + 1.000000*Nx ) 
    BBcount[ 73 ] = ( 1.000000*Nx ) 
    BBcount[ 74 ] = ( 8.000000 ) 
    BBcount[ 75 ] = 1 
    BBcount[ 76 ] = ( 1.000000 + 1.000000*Nx ) 
    BBcount[ 77 ] = ( 1.000000*Nx ) 
    BBcount[ 78 ] = ( 1.000000*Nx ) 
    BBcount[ 79 ] = 1 
    BBcount[ 80 ] = ( 1.000000 + 1.000000*Nx ) 
    BBcount[ 81 ] = ( 1.000000*Nx ) 
    BBcount[ 82 ] = ( 1.000000*Nx ) 
    BBcount[ 83 ] = 1 
    BBcount[ 84 ] = ( 2002.000000 ) 
    BBcount[ 85 ] = ( 2001.000000 ) 
    BBcount[ 86 ] = ( 2001.000000 ) 
    BBcount[ 87 ] = 1 
    BBcount[ 88 ] = ( 2001.000000 ) 
    BBcount[ 89 ] = ( 2000.000000 ) 
    BBcount[ 90 ] = ( 2000.000000 + 2000.000000*Nx ) 
    BBcount[ 91 ] = ( 2000.000000*Nx ) 
    BBcount[ 92 ] = ( 2000.000000*Nx ) 
    BBcount[ 93 ] = ( 2000.000000 ) 
    BBcount[ 94 ] = ( 2000.000000 + 2000.000000*Nx ) 
    BBcount[ 95 ] = ( 2000.000000*Nx ) 
    BBcount[ 96 ] = ( 2000.000000*Nx ) 
    BBcount[ 97 ] = ( 2000.000000 ) 
    BBcount[ 98 ] = ( 2000.000000 ) 
    BBcount[ 99 ] = 1

    return BBcount
