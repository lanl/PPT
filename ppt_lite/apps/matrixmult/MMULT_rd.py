import math

Rdist = {}


# def get_RDcountsArray( Nx ):
def get_RDcountsArray(params): # note: input is a single tuple
    ( Nx ) =  params
    for i in range(0,12):
         Rdist[i] = 0


    Rdist[ 1 ] = -1
    Rdist[ 2 ] = 0
    Rdist[ 3 ] = 3
    Rdist[ 4 ] = 4
    Rdist[ 5 ] = 5
    Rdist[ 6 ] = 7

    Nx = int(''.join(map(str,Nx)))

    Rdist[ 7 ] = (0.076)*pow(Nx,2) + (-0.932)*Nx + (14.533)
    Rdist[ 8 ] = (8.525)*pow(Nx,2) + (-205.481)*Nx + (1235.287)
    Rdist[ 9 ] = (4.091)*pow(Nx,2) + (7.91)*Nx + (-543.574)
    Rdist[ 10 ] = (2.563)*pow(Nx,2) + (96.618)*Nx + (-1224.437)
    Rdist[ 11 ] = (2.229)*pow(Nx,2) + (142.334)*Nx + (-1571.221)

    return Rdist

