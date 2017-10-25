"""
 HPLSim, hplobjects.py
"""

# Set up path variables; PPT applications expect it in this fashion.
#from ppt import *

# INTERNAL imports for this specific application
from hplauxil import *

# EXGTERNAL imports for this specific application
import math

###############################################################################
# GRID Class
###############################################################################
class GRID(object):
    # See hpl_grid.h - HPL_S_grid struct
    def __init__(self):
        self.all_comm = None # grid communicator
        self.row_comm = None # row communicator
        self.col_comm = None # column communicator
        self.myrow = None # my row number in the grid
        self.mycol = None # the total # of columns in the grid
        self.nprow = None # the total # of rows in the grid
        self.npcol = None # the total # of columns in the grid
        self.nprocs = None # the total # of procs in the grid
        self.core = None

###############################################################################
# hpl_grid_init
###############################################################################
def hpl_grid_init(COMM,rank,nprocs,NPCOL,NPROW,ORDER,core):

    #######################
    # Constant Parameters
    #######################
    HPL_ROW_MAJOR = 0
    HPL_COL_MAJOR = 1
    
    # Row- or column-major ordering of the processes
    if ORDER == HPL_ROW_MAJOR:
        order = HPL_ROW_MAJOR
        myrow = rank / NPCOL
        mycol = rank - myrow * NPCOL
    else:
        order = HPL_COLUMN_MAJOR
        mycol = rank / NPROW
        myrow = rank - mycol * NPROW

    grid = GRID()
    grid.myrow = myrow
    grid.mycol  = mycol
    grid.nprow = NPROW
    grid.npcol = NPCOL
    grid.nprocs = nprocs

    # All communicator, leave if I am not part of this grid. Creation of the
    # row- and column communicators.
    color = 'mpi_undefined'
    if rank < nprocs:
        color = 0
    grid.all_comm = COMM #mpi_comm_split(COMM, color, rank)
    grid.row_comm = mpi_comm_split(grid.all_comm, myrow, mycol)
    grid.col_comm = mpi_comm_split(grid.all_comm, mycol, myrow)
    grid.core = core

    return grid

###############################################################################
# PANEL Class
###############################################################################
class PANEL(object):
    # See hpl_panel.h - HPL_S_panel struct
    def __init__(self):
        self.grid = None             # ptr to the process grid */
        self.request = [None]        # [1] # requests for panel bcast */
        self.nb = None               # distribution blocking factor */
        self.jb = None               # panel width */
        self.mp = None               # local # of rows of trailing part of A */
        self.nq = None               # local # of cols of trailing part of A */
        self.prow = None             # proc. row owning 1st row of trail. A */
        self.pcol = None             # proc. col owning 1st col of trail. A */
        self.msgid = None            # message id for panel bcast */
        self.rfact = None
        self.bcast = None
        self.nwait = 0

###############################################################################
# HPL_pdpanel_init -- Initialize PANEL object
###############################################################################
#def HPL_pdpanel_init(grid, M, N, JB, IA, JA, nb, PFACT, NBMIN, NDIV, RFACT, TAG, BCAST):
def HPL_pdpanel_init(grid, M, N, JB, IA, JA, nb, RFACT, TAG, BCAST):

    myrow = grid.myrow
    mycol = grid.mycol
    nprow = grid.nprow
    npcol = grid.npcol
    
    [ii, jj, icurrow, icurcol] = HPL_infog2l( IA, JA, nb, nb, nb, nb, 0, 0, myrow, mycol, nprow, npcol)
    mp = HPL_numrocI(M, IA, nb, nb, myrow, 0, nprow)
    nq = HPL_numrocI(N, JA, nb, nb, mycol, 0, npcol)
    
    panel = PANEL()
    panel.grid = grid   # ptr to the process grid
    panel.rfact = RFACT
    panel.bcast = BCAST

    # Local lengths, indexes process coordinates
    panel.nb      = nb      # distribution blocking factor */
    panel.jb      = JB      #                        /* panel width */
    panel.mp      = mp      # local # of rows of trailing part of A */
    panel.nq      = nq      # local # of cols of trailing part of A */
    panel.prow    = icurrow # proc row owning 1st row of trailing A */
    panel.pcol    = icurcol # proc col owning 1st col of trailing A */
    panel.msgid   = TAG     # message id to be used for panel bcast */

    return panel
