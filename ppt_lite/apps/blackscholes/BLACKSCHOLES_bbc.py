def get_BBcountsArray(params):
    ( Nx ) = params
    import numpy as np
    BBcount = {}
    for i in range(0,55):
        BBcount[i] = 0


    Nx = int(''.join(map(str,Nx)))

    BBcount[ 1 ] = ( 200.000000*Nx )
    BBcount[ 2 ] = ( 80.558737*( -11.898675 + Nx + ( -333.735837 )/( ( Nx + -22.185310*np.sqrt(Nx) ) ) ) )
    BBcount[ 3 ] = 119.437216*( 8.221206 + Nx + ( -12.522910 )/( ( np.sqrt(Nx) + -0.002002*Nx*np.sqrt(Nx) ) ) )
    BBcount[ 4 ] = ( 200.000000*Nx )
    BBcount[ 5 ] = ( 0.000019*pow(Nx,3) )
    BBcount[ 6 ] = ( 200.000000*Nx )
    BBcount[ 7 ] = ( 100.000000*Nx )
    BBcount[ 8 ] = ( 162.499395*( 1.711295 + pow(Nx,0.25) + 0.306508*Nx ) )
    BBcount[ 9 ] = ( -1403.937008 + 50.123031*Nx )
    BBcount[ 10 ] = ( 100.000000*Nx )
    BBcount[ 11 ] = 0
    BBcount[ 12 ] = ( 101.000000 )
    BBcount[ 13 ] = ( 100.000000 )
    BBcount[ 14 ] = ( 100.000000 + 100.000000*Nx )
    BBcount[ 15 ] = ( 100.000000*Nx )
    BBcount[ 16 ] = ( 100.000000*Nx )
    BBcount[ 17 ] = ( 100.000000 )
    BBcount[ 18 ] = ( 100.000000 )
    BBcount[ 19 ] = 1
    BBcount[ 20 ] = 1
    BBcount[ 21 ] = 0
    BBcount[ 22 ] = 1
    BBcount[ 23 ] = 0
    BBcount[ 24 ] = 1
    BBcount[ 25 ] = 0
    BBcount[ 26 ] = 1
    BBcount[ 27 ] = 0
    BBcount[ 28 ] = 1
    BBcount[ 29 ] = 0
    BBcount[ 30 ] = 1
    BBcount[ 31 ] = ( 1.000000 + 1.000000*Nx )
    BBcount[ 32 ] = ( 1.000000*Nx )
    BBcount[ 33 ] = 0
    BBcount[ 34 ] = ( 8.000000 )
    BBcount[ 35 ] = ( 1.000000*Nx )
    BBcount[ 36 ] = 1
    BBcount[ 37 ] = 0
    BBcount[ 38 ] = 1
    BBcount[ 39 ] = ( 1.000000 + 1.000000*Nx )
    BBcount[ 40 ] = ( 1.000000*Nx )
    BBcount[ 41 ] = ( 1.000000*Nx )
    BBcount[ 42 ] = 1
    BBcount[ 43 ] = 0
    BBcount[ 44 ] = 1
    BBcount[ 45 ] = 0
    BBcount[ 46 ] = 1
    BBcount[ 47 ] = ( 1.000000 + 1.000000*Nx )
    BBcount[ 48 ] = ( 1.000000*Nx )
    BBcount[ 49 ] = 0
    BBcount[ 50 ] = ( 1.000000*Nx )
    BBcount[ 51 ] = ( 1.000000*Nx )
    BBcount[ 52 ] = 1
    BBcount[ 53 ] = 0
    BBcount[ 54 ] = 1


    return BBcount

