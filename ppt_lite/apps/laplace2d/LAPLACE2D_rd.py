import math

Rdist = {}


# def get_RDcountsArray( Nx ):
def get_RDcountsArray(params): # note: input is a single tuple
    ( Nx, Ny ) =  params
    for i in range(0,12):
         Rdist[i] = 0

    Rdist[ 1 ] = -1
    Rdist[ 2 ] = 0
    Rdist[ 3 ] = 1
    Rdist[ 4 ] = 3
    Rdist[ 5 ] = 4
    Rdist[ 6 ] = 5

    x = math.sqrt( Nx * Ny ) 

    Rdist[ 7 ] = (0.019)*pow(x,2) + (0.659)*x + (2.799)
    Rdist[ 8 ] = (1.781)*pow(x,2) + (-43.97)*x + (292.822)
    Rdist[ 9 ] = (5.457)*pow(x,2) + (-118.961)*x + (678.115)
    Rdist[ 10 ] = (4.442)*pow(x,2) + (-77.757)*x + (375.713)
    Rdist[ 11 ] = (4.525)*pow(x,2) + (-75.903)*x + (400.782)
    

    return Rdist

