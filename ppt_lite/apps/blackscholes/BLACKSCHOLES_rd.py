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

    Rdist[ 7 ] = (447.779)*pow(Nx,2) + (-10347.514)*Nx + (58082.638)
    Rdist[ 8 ] = (691.454)*pow(Nx,2) + (-15385.259)*Nx + (83163.778)
    Rdist[ 9 ] = (776.267)*pow(Nx,2) + (-17242.206)*Nx + (93270.545)
    Rdist[ 10 ] = (864.47)*pow(Nx,2) + (-19212.246)*Nx + (104143.091)
    Rdist[ 11 ] = (952.496)*pow(Nx,2) + (-21170.164)*Nx + (114844.026)

    return Rdist

