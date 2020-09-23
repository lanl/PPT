import math

Rdist = {}


# def get_RDcountsArray( Nx ):
def get_RDcountsArray(params): # note: input is a single tuple
    ( Nx ) =  params
    for i in range(0,12):
         Rdist[i] = 0

    Nx = int(''.join(map(str,Nx)))

    Rdist[ 1 ] = -1
    Rdist[ 2 ] = 0
    Rdist[ 3 ] = 1
    Rdist[ 4 ] = 2
    Rdist[ 5 ] = 3
    Rdist[ 6 ] = 4

    Rdist[ 7 ] = (0.008)*pow(Nx,2) + (8.528)*Nx + (-60.399)
    Rdist[ 8 ] = (0.355)*pow(Nx,2) + (72.492)*Nx + (-693.712)
    Rdist[ 9 ] = (-2.85)*pow(Nx,2) + (290.958)*Nx + (-2439.752)
    Rdist[ 10 ] = (-3.763)*pow(Nx,2) + (366.054)*Nx + (-2936.209)
    Rdist[ 11 ] = (158.106)*pow(Nx,2) + (-5019.266)*Nx + (56879.716)

    return Rdist

