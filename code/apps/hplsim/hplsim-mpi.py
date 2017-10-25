"""
 HPLSim, an application simulator for the LAPACK HPL Benchmark

"""
version = "2016.12.05.1"

# Set up path variables; PPT applications expect it in this fashion.
import sys
from sys import path
path.append('../..')
from ppt import *

# INTERNAL imports for this specific application
from hplobjects import *
from hplpdupdate import *
from hplauxil import *
from hpltimecompute import *

# EXGTERNAL imports for this specific application
import math
from optparse import OptionParser

#######################
# HPL.dat INPUT
#######################
NS        = 10000    #     # System size: [(NS)x(NS)]*[(NS)x(1)]=[(NS)x(1)]
NB        = 10       #     # Column thickness of panels
P         = 1        #     # The number of process ROWS on the PxQ process grid
Q         = 1        #     # The number of process COLUMNS on the PxQ process grid
DEPTH     = 0         # 0   # Lookahead depth
## START FIXED ## (Starting with some recomended values at netlib.org)
RFACT     = 2         # 2   # Recursive panel factorization algorithm (0=left, 1=Crout, 2=Right)
BCAST     = 1         # 1   # Broadcast algorithm (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
PMAP      = 0         #     # (0) ROW-major or (1) COLUMN-major process mapping
#PFACT     = 2         # 1   # Panel factorization algorithm (0=left, 1=Crout, 2=Right)
#NBMIN     = 1         # 4   # Recursive stopping criteria for RFACT
#NDIV      = 2         # 2   # Number of panels in recursion for RFACT
#FSWAP     = 2         # 2   # Swapping algorithm (0=bin-exch,1=long,2=mix)
#TSWAP     = 60        # 60  # Swapping threshold
#L1NOTRA   = 0         # 0   # L1 in (0=transposed,1=no-transposed) form
#UNOTRAN   = 0         # 0   # U  in (0=transposed,1=no-transposed) form
#EQUIL     = 1         # 1   # Equilibration (0=no,1=yes)
#ALIGN     = 8         # 8   # memory alignment in double (> 0)
### END FIXED ###

# Approximated latency and bandwidth:

LAT   = 25*1.5e-6 # Moonlight (-O0)
LAT   = 5*1.5e-6 # Moonlight (-O0)

#LAT   = 1.5e-6  # MLIntelNode
BDWTH = 1.0e10 #1.0 * 10**10 # MLIntelNode   # If course_grain_pdfact - Bandwidth to Use

# PARSER:
parser = OptionParser()
parser.add_option("-s", "--ns",    type="int", dest="ns",    default=NS,    help="set NS (system size)")
parser.add_option("-b", "--nb",    type="int", dest="nb",    default=NB,    help="set NB (block size)")
parser.add_option("-p", "--p",     type="int", dest="p",     default=P,     help="set P-dimension in process grid")
parser.add_option("-q", "--q",     type="int", dest="q",     default=Q,     help="set Q-dimension in process grid")
parser.add_option("-d", "--depth", type="int", dest="depth", default=DEPTH, help="set lookahead depth")
(options, args) = parser.parse_args()
NS    = options.ns
NB    = options.nb
P     = options.p
Q     = options.q
DEPTH = options.depth

###############################################################################
# LU PANEL Factorization (APPROXIMATION) -- HPL_pdfact 
###############################################################################
def HPL_pdfact(panel):
    
    # Check if there is no work to do:
    grid = panel.grid
    jb = panel.jb
    if ((grid.mycol <> panel.pcol) or (jb <= 0)):
        return

    # SIM Work:
    if (panel.mp>0 and jb>0):
        ts_args = [panel.mp, jb, panel.grid.nprow]
        pdfact_i_time = compute_pdfact_time(panel.grid.core, ts_args)
        mpi_ext_sleep(pdfact_i_time, panel.grid.all_comm)

    panel.nq -= jb

    return

###############################################################################
# LU Factorization Wrapper (DEPTH>0) -- HPL_pdgesvK2
###############################################################################
def HPL_pdgesvK2(grid):

    N = NS
    nb = NB
    if (N <= 0): return
    
    MSGID_BEGIN_FACT  =  2001
    MSGID_END_FACT    =  3000
    tag = MSGID_BEGIN_FACT
    mycol = grid.mycol
    npcol = grid.npcol
    depth = DEPTH
    icurcol = 0
    jj = 0
    
    start_time = mpi_wtime(grid.all_comm)
    gflops = 0.0
    time_pdfact = 0.0
    time_pdupdate = 0.0
    
    # Create and initialize the first depth panels
    nq = HPL_numrocI(N+1, 0, nb, nb, mycol, 0, npcol)
    nn = N
    jstart = 0
    panel = []
    for k in range(0,depth):
        jb = min(nn, nb)
        panel.append(HPL_pdpanel_init(grid, nn, nn+1, jb, jstart,
                     jstart, nb, RFACT, tag, BCAST))
        nn -= jb
        jstart += jb
        if (mycol == icurcol):
            jj += jb
            nq -= jb
            icurcol = MModAdd1(icurcol, npcol)
            tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT)

    # Create last depth+1 panel
    panel.append(HPL_pdpanel_init(grid, nn, nn+1, min(nn, nb), jstart,
                 jstart, nb, RFACT, tag, BCAST))
    tag = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT)
    
    # Initialize the lookahead - Factor jstart columns: panel[0..depth-1]
    j = 0
    for k in range(0,depth):
        jb = jstart - j
        jb = min(jb, nb)
        j += jb
        # Factor and broadcast k-th panel
        timei = mpi_wtime(grid.all_comm)
        HPL_pdfact(panel[k])
        timef = mpi_wtime(grid.all_comm)
        time_pdfact += (timef-timei)
        while True:
            test = HPL_bcast(panel[k])
            if test == 'HPL_SUCCESS': break
        # Partial update of the depth-k-1 panels in front of me
        if (k < depth - 1):
            nn = HPL_numrocI( jstart-j, j, nb, nb, mycol, 0, npcol)
            timei = mpi_wtime(grid.all_comm)
            HPL_pdupdateTT(None, panel[k], nn)
            timef = mpi_wtime(grid.all_comm)
            time_pdupdate += (timef-timei)

    # Loop over the columns of A, nb columns at a time...
    for j in range(jstart, N, nb):
        n = N - j
        jb = min(n, nb)
        # if this is process 0,0 and not the first panel */
        if ((grid.myrow == 0) and (grid.mycol == 0) and (j > 0)):
            time = mpi_wtime(grid.all_comm) - start_time
            if (time <= 0.0): time = 1e-6
            gflops = 2.0*(N*float(N*N) - n*float(n*n))/3.0/time/1e9
            print"Column= ",j,"\tFraction= ",j*100.0/N,"\tGflops= ",gflops,"\tDone"
        
        # Initialize current panel - Finish latest update, Factor and broadcast
        # current panel
        panel[depth] = HPL_pdpanel_init(grid, n, n+1, jb, j, j, nb, RFACT, tag, BCAST)
        if (mycol == icurcol):
            nn = HPL_numrocI( jb, j, nb, nb, mycol, 0, npcol)
            for k in range(0,depth):
                timei = mpi_wtime(grid.all_comm)
                HPL_pdupdateTT(None, panel[k], nn)
                timef = mpi_wtime(grid.all_comm)
                time_pdupdate += (timef-timei)
            timei = mpi_wtime(grid.all_comm)
            HPL_pdfact(panel[depth])
            timef = mpi_wtime(grid.all_comm)
            time_pdfact += (timef-timei)
        else:
            nn = 0
            
        # Finish the latest update and broadcast the current panel
        #print "nq, nn = ",nq, nn
        timei = mpi_wtime(grid.all_comm)
        test = HPL_pdupdateTT(panel[depth], panel[0], nq-nn)
        timef = mpi_wtime(grid.all_comm)
        time_pdupdate += (timef-timei)

        # Circular  of the panel pointers...
        # Go to next process row and column - update the message ids for broadcast
        p = panel[0]
        for k in range(0,depth):
            panel[k] = panel[k+1]
        panel[depth] = p

        if (mycol == icurcol):
            jj += jb
            nq -= jb
        icurcol = MModAdd1(icurcol, npcol)
        tag     = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT)

    # Clean-up: Finish updates - release panels and panel list
    nn = HPL_numrocI( 1, N, nb, nb, mycol, 0, npcol)
    for k in range(0,depth):
        timei = mpi_wtime(grid.all_comm)
        HPL_pdupdateTT(None, panel[k], nn)
        timef = mpi_wtime(grid.all_comm)
        time_pdupdate += (timef-timei)

    panel = None
            
    return [gflops, time_pdfact, time_pdupdate]

###############################################################################
# LU Factorization Wrapper (DEPTH==0) -- HPL_pdgesv0
###############################################################################
def HPL_pdgesv0(grid):

    N = NS
    if (N <= 0): return
    nb = NB
    
    start_time = mpi_wtime(grid.all_comm)
    gflops = 0.0
    time_pdupdate = 0.0
    time_pdfact = 0.0
    
    MSGID_BEGIN_FACT  =  2001
    MSGID_END_FACT    =  3000
    tag = MSGID_BEGIN_FACT
    
    # Loop over the columns of A, nb columns at a time...
    for j in range(0, N, nb):
    
        n = N - j
        jb = min(n, nb)
        
        # if this is process 0,0 and not the first panel */
        if ((grid.myrow == 0) and (grid.mycol == 0) and (j > 0)):
            time = mpi_wtime(grid.all_comm) - start_time
            if (time <= 0.0): time = 1e-6
            gflops = 2.0*(N*float(N*N) - n*float(n*n))/3.0/time/1e9
            print"Column= ",j,"\tFraction= ",j*100.0/N,"\tGflops= ",gflops,"\tDone"
        
        # Initialize the current panel:
        panel = HPL_pdpanel_init(grid, n, n+1, jb, j, j, nb, RFACT, tag, BCAST)

        # Factor current panel
        timei = mpi_wtime(grid.all_comm)
        HPL_pdfact(panel)
        timef = mpi_wtime(grid.all_comm)
        time_pdfact += (timef-timei)
        
        # Broadcast current panel
        while True:
            test = HPL_bcast(panel)
            if test == 'HPL_SUCCESS': break
        
        # Trailing Sub-matrix Update
        timei = mpi_wtime(grid.all_comm)
        HPL_pdupdateTT(None, panel, -1)
        timef = mpi_wtime(grid.all_comm)
        time_pdupdate += (timef-timei)
        
        # Update message tag for next factorization
        tag = MNxtMgid(tag, MSGID_BEGIN_FACT, MSGID_END_FACT)
        panel = None
        
    if grid.mycol == 0 and grid.myrow == 0:
      print "\n time_pdupdate = ",time_pdupdate
      print " time_pdfact   = ",time_pdfact,"\n"
      
    return [gflops, time_pdfact, time_pdupdate]

###############################################################################
# hpl_main
###############################################################################
def hpl_main(mpi_comm_world):

    """
    Driver for the HPL Benchmark.
    Input: mpi_comm_world -- communicator from the MPI model
    Output: nothing returned, but total simulation time printed
    """
    
    size = mpi_comm_size(mpi_comm_world)                   # total rank count
    rank = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

    # Compute time for core/process
    host = mpi_ext_host(mpi_comm_world)
    core_id = rank % cores_per_host
    core = host.cores[core_id]
    
    # How many rows needed for row-enchalant form:
    if NS % NB <> 0:
        print " !!!!!!!!!!!!!!!!!!!!!!!!!! "
        print " ERROR !! NS % NB MUST BE 0 "
        print " !!!!!!!!!!!!!!!!!!!!!!!!!! "

    # Setup the PxQ process grid:
    grid = hpl_grid_init(mpi_comm_world,rank,size,Q,P,PMAP,core)
    NPROW = grid.nprow
    NPCOL = grid.npcol
    MYROW = grid.myrow
    MYCOL = grid.mycol
    
    gflops = 0.0
    time_pdfact = 0.0
    time_pdupdate = 0.0
    
    # perform lu-factorization on linear system
    if ((DEPTH == 0) or (grid.npcol == 1)):
        [gflops,dtime_pdfact,dtime_pdupdate] = HPL_pdgesv0(grid)
    else:
        [gflops,dtime_pdfact,dtime_pdupdate] = HPL_pdgesvK2(grid)
    time_pdfact += dtime_pdfact
    time_pdupdate += dtime_pdupdate

    # Final Sync
    mpi_barrier(mpi_comm_world)

    # Print the results to screen
    if rank == 0:
        print "\nResult for HPL Benchmark:"
        print "==========================================="
        print "WCTime for entire run (sec): ", mpi_wtime(mpi_comm_world)
        print "PERFORMANCE(GFlops): ", gflops
        print "TIME_pdfact: ",time_pdfact
        print "TIME_pdupdate: ",time_pdupdate
        print "TIME_pdfact/TIME_pdupdate: ",(time_pdfact/time_pdupdate)*100.0,"%"

    # Finalize mpi and the simulation
    mpi_finalize(mpi_comm_world)

###############################################################################
# "MAIN"
###############################################################################
cores_per_host = 16
total_ranks = P * Q
nhosts = total_ranks/cores_per_host+1
modeldict = { "model_name"    : "hplsim",
              "sim_time"      : 100000000,
              "use_mpi"       : True,
              "mpi_path"      : "/opt/local/lib/mpich-mp/libmpich.dylib",
              "intercon_type" : "Bypass", # use interconnect bypass
              "bypass"        : {"nhosts" : nhosts,"bdw" : BDWTH,"link_delay" : LAT},
              "host_type"     : "MLIntelNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.infiniband_mpiopt,
              "debug_options" : set(["none"]),
              #"min_delay" : 0.001,
            }
print "\nHPLSim run with Simian PDES Engine\nVersion = ", version
cluster = Cluster(modeldict)
total_hosts = cluster.intercon.num_hosts()
total_cores = total_hosts*cores_per_host
if total_ranks >= total_cores:
  print "ERROR: the cluster doesn't have enough cores to run this job "
  sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(total_ranks)]

cluster.start_mpi(hostmap, hpl_main)
cluster.run()
###############################################################################
###############################################################################
