import math

Rdist = {}


# def get_RDcountsArray( Nx, Ny, Nz, Ichunk, Nmom, Nang, Ng, Li, Lo ):
def get_RDcountsArray(params): # note: input is a single tuple
    ( Nx, Ny, Nz, Ic, Nm, Na, Ng, Li, Lo ) =  params
    for i in range(0,59):
    	 Rdist[i] = 0


    Rdist[ 16 ] = (172.996639+(math.sqrt(6.251699*Nx*(Ny+97.234124*math.sqrt(Nz))))/(-91.844910))
    Rdist[ 17 ] = ((853.970196)/(Nm)+-0.002749*pow((Na+-0.751428*Nm),10))
    Rdist[ 18 ] = 2288704.418615*pow(math.sqrt((1.000000)/(Nz*(Nm+(153.288535)/(Nz)))),3)
    Rdist[ 19 ] = 3576091.229607*pow(math.sqrt((1.000000)/(Nz*(Nm+(157.044720)/(Nz)))),3)
    Rdist[ 20 ] = 4732504.550014*pow(math.sqrt((1.000000)/(Nz*(Nm+(147.632902)/(Nz)))),3)
    Rdist[ 21 ] = 5668756.182780*pow(math.sqrt((1.000000)/(Nz*(Nm+(131.789609)/(Nz)))),3)
    Rdist[ 22 ] = 10376552.588128*pow((1.000500)/(Ny*Nz*math.sqrt((Na+(2.000000)/(Nm)))),2)
    Rdist[ 23 ] = (5263203.166098*pow(1.000500/(Ny*Nz),2))
    Rdist[ 24 ] = (7808.122417+(-0.004280*pow(Nx,9)*pow((Li+-0.328661*Ng),12))/(Nz))
    Rdist[ 25 ] = (10117.055796+0.000058*pow(Na*Ng,7))
    Rdist[ 26 ] = (11258.369285+0.000062*pow(Na*Ng,7))
    Rdist[ 27 ] = (12296.503507+0.000066*pow(Na*Ng,7))
    Rdist[ 28 ] = (13278.382156+0.000069*pow(Na*Ng,7))
    Rdist[ 29 ] = (14198.234670+0.000073*pow(Na*Ng,7))
    Rdist[ 30 ] = (15095.697797+0.000076*pow(Na*Ng,7))
    Rdist[ 31 ] = (16728.104739+-0.006953*Na*pow((Na+-0.337674*Nx),12))
    Rdist[ 32 ] = (17859.785326+-0.012468*math.sqrt(Na)*pow((Na+-0.330916*Nx),12))
    Rdist[ 33 ] = (19420.832735+(-0.103526*pow((Na+-0.325352*Nx),12))/(Nx))
    Rdist[ 34 ] = (21699.109923+(3728.959824)/(Nz)+-0.028252*pow((Na+-0.320917*Nx),12))
    Rdist[ 35 ] = (26413.582509+-0.011439*Na*pow((Na+-0.340036*Nx),12))
    Rdist[ 36 ] = (1570.845976+(Ny)/(Nz*Li*(Ng+(-1.992151)/(Nm))))
    Rdist[ 37 ] = (710.300306+0.001688*pow((Ny+-1.195647*Nz),7))
    Rdist[ 38 ] = 187.311444*math.sqrt((Ng+(25.719888)/((Nz+7.619798*math.sqrt(Ny)))))
    Rdist[ 39 ] = ((956.666836)/(Ny)+7.425445*math.sqrt(0.064996*Nm*math.sqrt(Nx)*pow(Nz,2)))
    Rdist[ 40 ] = (87.658938+(179.880245)/(math.sqrt((Na+41.893330*math.sqrt(Nz)*math.sqrt(Ng)))))
    Rdist[ 41 ] = (124.879121)
    Rdist[ 42 ] = (47.230858+(math.sqrt(8.413347*Ny*pow(Nm,2)))/(-5.413347))
    Rdist[ 43 ] = (28.399341+(100.284688)/(math.sqrt((Nz+42.512138*pow(Nm,2)))))
    Rdist[ 44 ] = (20.234636+(math.sqrt((Ny+(202.332488)/(Nm)+22.688829*Nz)))/(Li))
    Rdist[ 45 ] = (36.512251+(math.sqrt((Li+5.544686*Ny)))/(-6.573249))
    Rdist[ 46 ] = (44.579963+(-1.431942*Nm*Na)/(pow(Ng,4))+-2.176085*Ng)
    Rdist[ 47 ] = (29.988015+pow(math.sqrt(0.000051*Nm*Ng*pow(Ny,4)),2))
    Rdist[ 48 ] = (29.357004+pow(math.sqrt(0.000101*Nz*Li*pow(Ny,4)),2))
    Rdist[ 49 ] = (25.913184+(math.sqrt((0.792664*Ny*Na)/(Nz)))/(-25.645297))
    Rdist[ 50 ] = (1.274907*(7.341445+math.sqrt(0.447745*Nz*pow(Ny,2))))
    Rdist[ 51 ] = 0.952905*(8.465174+math.sqrt(0.142261*Nz*Na*math.sqrt(3.000000*Ny*Nz)))
    Rdist[ 52 ] = (9.670250+math.sqrt(0.073234*(Ny+(2.000000)/(Nx)+3.000000*Nx)))
    Rdist[ 53 ] = (9.928254+0.025085*math.sqrt(Nx))
    Rdist[ 54 ] = (10.987135+-0.137787*pow((-2.511266+Nm),4))
    Rdist[ 55 ] = (10.780661+math.sqrt(0.025560*(Ny+2.000000*Li*math.sqrt(Nm))))

    return Rdist