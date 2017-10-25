"""
 HPLSim, hplauxil.py
"""

# Set up path variables; PPT applications expect it in this fashion.
from ppt import *

# INTERNAL imports for this specific application

# EXGTERNAL imports for this specific application
import math

###############################################################################
# HPL_numrocI
###############################################################################
def HPL_numrocI(N, I, INB, NB, PROC, SRCPROC, NPROCS ):
    if ((SRCPROC == -1) or (NPROCS == 1)):
        return N
    srcproc = SRCPROC
    inb = INB - I
    if (inb <= 0):
        nblocks = (-inb) / NB + 1
        srcproc += nblocks
        srcproc -= (srcproc / NPROCS) * NPROCS
        inb     += nblocks * NB
    if (PROC == srcproc):
        if (N <= inb): return N
        nblocks = ( N - inb ) / NB + 1
        if (nblocks < NPROCS): return inb
        ilocblk = nblocks / NPROCS
        if (nblocks - ilocblk * NPROCS):
            return inb + ilocblk * NB
        else:
            return N + ( ilocblk - nblocks ) * NB
    else:
        if (N <= inb): return 0
        nblocks = ( N - inb ) / NB + 1
        mydist = PROC - srcproc
        if(mydist < 0): mydist += NPROCS
        if (nblocks < NPROCS):
            if (mydist < nblocks):
                return NB
            elif (mydist > nblocks):
                return 0
            else:
                return N - inb + NB * (1 - nblocks)
        ilocblk = nblocks / NPROCS
        mydist -= nblocks - ilocblk * NPROCS
        if (mydist < 0):
            return ( ilocblk + 1 ) * NB
        elif (mydist > 0):
            return ilocblk * NB
        else:
            return N - inb + NB * (ilocblk - nblocks + 1)
    return

###############################################################################
# HPL_infog2l --
###############################################################################
def HPL_infog2l(I, J, IMB, MB, INB, NB, RSRC, CSRC, MYROW, MYCOL, NPROW, NPCOL):
    # HPL_infog2l computes the starting local index II, JJ corresponding to
    # the submatrix starting globally at the entry pointed by  I,  J.  This
    # routine returns the coordinates in the grid of the process owning the
    # matrix entry of global indexes I, J, namely PROW and PCOL.
    imb   = IMB
    PROW = RSRC
    if ((PROW == -1) or (NPROW == 1)):
        II = I
    elif (I < imb):
        if (MYROW == PROW):
            II = I
        else:
            II = 0
    else:
        mb   = MB
        rsrc = PROW
        if (MYROW == rsrc):
            nblocks = ( I - imb ) / mb + 1
            PROW  += nblocks
            PROW  -= ( PROW / NPROW ) * NPROW

            if( nblocks < NPROW ):
                II = imb
            else:
                ilocblk = nblocks / NPROW
                if (ilocblk * NPROW >= nblocks):
                    if (MYROW == PROW):
                        II = I   + ( ilocblk - nblocks ) * mb
                    else:
                        II = imb + ( ilocblk - 1       ) * mb
                else:
                    II =  imb + ilocblk * mb
        else:
            I -= imb
            nblocks = I / mb + 1
            PROW  += nblocks
            PROW  -= ( PROW / NPROW ) * NPROW

            mydist = MYROW - rsrc
            if (mydist < 0): mydist += NPROW

            if (nblocks < NPROW):
                mydist -= nblocks
                if (mydist < 0):
                    II = mb
                elif (MYROW == PROW):
                    II = I + ( 1 - nblocks ) * mb
                else:
                    II = 0
            else:
                ilocblk = nblocks / NPROW
                mydist -= nblocks - ilocblk * NPROW
                if (mydist < 0):
                    II = ( ilocblk + 1 ) * mb
                elif (MYROW == PROW):
                    II = (ilocblk - nblocks + 1) * mb + I
                else:
                    II = ilocblk * mb
    inb   = INB
    PCOL = CSRC
    if ((PCOL == -1) or (NPCOL == 1)):
        JJ = J
    elif (J < inb):
        if (MYCOL == PCOL):
            JJ = J
        else:
            JJ = 0
    else:
      nb   = NB
      csrc = PCOL
      if (MYCOL == csrc):
          nblocks = (J - inb) / nb + 1
          PCOL  += nblocks
          PCOL  -= (PCOL / NPCOL) * NPCOL
          if (nblocks < NPCOL):
              JJ = inb
          else:
            ilocblk = nblocks / NPCOL
            if (ilocblk * NPCOL >= nblocks):
                if (MYCOL == PCOL):
                    JJ = J   + ( ilocblk - nblocks ) * nb
                else:
                    JJ = inb + ( ilocblk - 1       ) * nb
            else:
                JJ = inb + ilocblk * nb
      else:
          J -= inb
          nblocks = J / nb + 1
          PCOL  += nblocks
          PCOL  -= ( PCOL / NPCOL ) * NPCOL

          mydist = MYCOL - csrc
          if (mydist < 0): mydist += NPCOL

          if(nblocks < NPCOL):
              mydist -= nblocks
              if (mydist < 0):
                  JJ = nb
              elif (MYCOL == PCOL):
                  JJ = J + (1 - nblocks)*nb
              else:
                  JJ = 0
          else:
              ilocblk = nblocks / NPCOL
              mydist -= nblocks - ilocblk * NPCOL
              if (mydist < 0):
                  JJ = ( ilocblk + 1 ) * nb
              elif (MYCOL == PCOL):
                  JJ = ( ilocblk - nblocks + 1 ) * nb + J
              else:
                  JJ = ilocblk * nb
    return [II, JJ, PROW, PCOL]

###############################################################################
# MModSub
###############################################################################
def MModSub(I1, I2, d):
    if (I1) < (I2):
        return (d) + (I1) - (I2)
    else:
        return (I1) - (I2)

###############################################################################
# MModSub1
###############################################################################
def MModSub1(I, d):
    if ((I)<>0):
        return (I)-1
    else:
        return (d)-1

###############################################################################
# MModAdd
###############################################################################
def MModAdd(I1, I2, d):
    if (I1) + (I2) < (d):
        return (I1) + (I2)
    else:
        return (I1) + (I2) - (d)

###############################################################################
# MModAdd1
###############################################################################
def MModAdd1(I, d):
    if ((I) <> (d) - 1):
        return (I) + 1
    else:
        return 0

###############################################################################
# MNxtMgid
###############################################################################
def MNxtMgid(id_, beg_, end_):
    if ((id_)+1 > (end_)):
        return (beg_)
    else:
        return (id_)+1

###############################################################################
# HPL_sdrv - Wrapper for MPI_Sendrecv
###############################################################################
def HPL_sdrv(SCOUNT, RCOUNT, PARTNER, COMM):
    if (RCOUNT > 0):
        if (SCOUNT > 0):
            # Post asynchronous receive
            request = mpi_irecv(COMM, PARTNER)
            # Blocking send
            test = mpi_send(PARTNER, None, SCOUNT*8, COMM)
            # Wait for the receive to complete
            status = mpi_wait(request)
        else:
            # Blocking receive
            RBUF = mpi_recv(COMM, PARTNER)
    elif (SCOUNT > 0):
        # Blocking send
        test = mpi_send(PARTNER, None, SCOUNT*8, COMM)
    return

###############################################################################
# HPL_bcast - Wrapper for specific HPL_bcast algorithm
###############################################################################
def HPL_bcast(panel):
    if (panel == None):
        return 'HPL_SUCCESS'
    if (panel.grid.npcol <= 1):
        return 'HPL_SUCCESS'
    if panel.bcast == 1:
        iflag = HPL_bcast_1rinM(panel)
    else:
        print " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
        print " ERROR !! BCAST MUST BE 1 FOR NOW... "
        print " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
    return iflag

###############################################################################
# HPL_bcast_1rinM -
###############################################################################
def HPL_bcast_1rinM(panel):

    ## HACK to (GREATLY) speed up comm:
    #return 'HPL_SUCCESS'

    if (panel == None):
        print " EARLY return, rank = ",mpi_comm_rank(panel.grid.all_comm)
        return 'HPL_SUCCESS'
    size = panel.grid.npcol
    if (size <= 1):
        print " EARLY return, rank = ",mpi_comm_rank(panel.grid.all_comm)
        return 'HPL_SUCCESS'
    rank = panel.grid.mycol
    comm  = panel.grid.row_comm
    root = panel.pcol
    msgid = panel.msgid
    next = MModAdd1(rank, size)
    if (rank == root):
        ierr = 'MPI_FAILURE'
        msgid = str(panel.msgid)+"_"+str(mpi_comm_rank(comm))
        msize = max (1, panel.mp*panel.jb*8)
        if mpi_send(next, None, msize, comm, msgid): ierr = 'MPI_SUCCESS'
        if ((ierr == 'MPI_SUCCESS') and (size > 2)):
            ierr = 'MPI_FAILURE'
            msgid = str(panel.msgid)+"_"+str(mpi_comm_rank(comm))
            if mpi_send(MModAdd1(next, size), None, msize, comm, msgid ): ierr = 'MPI_SUCCESS'
    else:
        prev = MModSub1(rank, size)
        if ((size > 2) and (MModSub1(prev, size) == root)):
            partner = root
        else:
            partner = prev
        if panel.request[0] == None:
            panel.request[0] = mpi_irecv(comm, partner)
        msgid = str(panel.msgid)+"_"+str(partner)

        #mpi_wait(panel.request[0])

        if mpi_test(panel.request[0]):
            panel.nwait = 0
            ierr = 'MPI_SUCCESS'
            panel.request[0] = None
            msgid = str(panel.msgid)+"_"+str(partner)
            if ((prev <> root) and (next <> root)):
                ierr = 'MPI_FAILURE'
                msgid = str(panel.msgid)+"_"+str(mpi_comm_rank(comm))
                msize = max (1, panel.mp*panel.jb*8)
                if mpi_send(next, None, msize, comm, msgid ): ierr = 'MPI_SUCCESS'
        else:
            panel.nwait += 1
            timesleep = min( panel.nwait * panel.nwait * (1e-5) , 1.0)
            #timesleep = 1e-3 #4
            mpi_ext_sleep(timesleep, comm)
            return 'HPL_KEEP_TESTING'
    if (ierr == 'MPI_SUCCESS'):
        iflag = 'HPL_SUCCESS'
    else:
        iflag = 'HPL_FAILURE'
        print " ERROR ERROR ERROR ---        iflag = HPL_FAILURE           "
    return iflag
