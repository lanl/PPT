"""
 HPLSim, hplpdupdate.py
"""

# Set up path variables; PPT applications expect it in this fashion.
from ppt import *

# INTERNAL imports for this specific application
from hpltimecompute import *
from hplauxil import *

# EXGTERNAL imports for this specific application
import math

#global nupdates
#nupdates = 0

###############################################################################
# HPL_pdupdateTT broadcast -- forwards panel and simultaneously
# applies the row interchanges and updates part of the trailing submatrix.
###############################################################################
def HPL_pdupdateTT(pbcst, panel, nn):
    # Real Code Arguments:
    # =========
    #  PBCST   (local input/output)          HPL_T_panel *
    #          On entry,  PBCST  points to the data structure containing the
    #          panel (to be broadcast) information.
    #
    #  IFLAG   (local output)                int *
    #          On exit,  IFLAG  indicates  whether or not  the broadcast has
    #          been completed when PBCST is not NULL on entry. In that case,
    #          IFLAG is left unchanged.
    #
    #  PANEL   (local input/output)          HPL_T_panel *
    #          On entry,  PANEL  points to the data structure containing the
    #          panel (to be updated) information.
    #
    #  NN      (local input)                 const int
    #          On entry, NN specifies  the  local  number  of columns of the
    #          trailing  submatrix  to be updated  starting  at the  current
    #          position. NN must be at least zero.
    #  ---------------------------------------------------------------------
    
    global nupdates
    
    nb = panel.nb
    jb = panel.jb
    n = panel.nq
    NN = nn
    IFLAG = None
    
    if( NN >= 0 ): n = min(NN, n)
    
    # There is nothing to update, enforce the panel broadcast.
    if (( n <= 0 ) or ( jb <= 0 )):
        if (pbcst <> None):
            while (True):
                #(void) HPL_bcast( PBCST, IFLAG )
                IFLAG = HPL_bcast(pbcst)
                if (IFLAG == 'HPL_SUCCESS'): break
        #print "RETURNING"
        return IFLAG

    #(void) HPL_bcast( PBCST, &test );
    test = HPL_bcast(pbcst)

    # 1 x Q case
    if (panel.grid.nprow == 1):
        mp   = panel.mp - jb
        #iroff = panel.ii
        nq0   = 0
        # So far we have not updated anything -  test availability of the panel
        # to be forwarded - If detected forward it and finish the update in one
        # step.
        while (test == 'HPL_KEEP_TESTING'):
            nn = n - nq0
            nn = min(nb, nn)
            #HPL_dtrsm( HplColumnMajor, HplLeft, HplUpper, HplTrans,
            #        HplUnit, jb, nn, HPL_rone, L1ptr, jb, Aptr, lda );
            ts_args = [jb, nn, 'L']
            dtrsm_i_time = compute_dtrsm_time(panel.grid.core, ts_args)
            mpi_ext_sleep(dtrsm_i_time, panel.grid.all_comm)
            #HPL_dgemm( HplColumnMajor, HplNoTrans, HplNoTrans, mp, nn,
            #        jb, -HPL_rone, L2ptr, ldl2, Aptr, lda, HPL_rone,
            #        Mptr( Aptr, jb, 0, lda ), lda );
            ts_args = [mp, nn, jb]
            dgemm_i_time = compute_dgemm_time(panel.grid.core, ts_args)
            mpi_ext_sleep(dgemm_i_time, panel.grid.all_comm)
            nq0 += nn
            
            test = HPL_bcast(pbcst)
        
        # The panel has been forwarded at that point, finish the update
        nn = n - nq0
        
        #print "nn, n, mp, jb = ",nn, n, mp, jb
        
        if (nn > 0):
            #HPL_dtrsm( HplColumnMajor, HplLeft, HplUpper, HplTrans,
            #        HplUnit, jb, nn, HPL_rone, L1ptr, jb, Aptr, lda );
            ts_args = [jb, nn, 'L']
            dtrsm_i_time = compute_dtrsm_time(panel.grid.core, ts_args)
            mpi_ext_sleep(dtrsm_i_time, panel.grid.all_comm)
            #HPL_dgemm( HplColumnMajor, HplNoTrans, HplNoTrans, mp, nn,
            #        jb, -HPL_rone, L2ptr, ldl2, Aptr, lda, HPL_rone,
            #        Mptr( Aptr, jb, 0, lda ), lda );
            ts_args = [mp, nn, jb]
            dgemm_i_time = compute_dgemm_time(panel.grid.core, ts_args)
            mpi_ext_sleep(dgemm_i_time, panel.grid.all_comm)

            #nupdates += 1
            #print "(",panel.grid.myrow,",",panel.grid.mycol,") - nn, n, mp, jb = ",nn, n, mp, jb

    else: # nprow > 1 ...
    
        # Compute redundantly row block of U and update trailing submatrix
        nq0 = 0
        curr = int(panel.grid.myrow == panel.prow)
        if (curr <> 0):
            mp = panel.mp - jb
        else:
            mp = panel.mp - 0

        # Broadcast has not occured yet, spliting the computational part
        while (test == 'HPL_KEEP_TESTING'):
            nn = n - nq0
            nn = min(nb, nn)
            #HPL_dtrsm( HplColumnMajor, HplRight, HplUpper, HplNoTrans,
            #        HplUnit, nn, jb, HPL_rone, L1ptr, jb, Uptr, LDU );
            ts_args = [nn, jb, 'R']
            dtrsm_i_time = compute_dtrsm_time(panel.grid.core, ts_args)
            mpi_ext_sleep(dtrsm_i_time, panel.grid.all_comm)
            if (curr <> 0):
                #HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                #       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                #       Mptr( Aptr, jb, 0, lda ), lda );
                ts_args = [mp, nn, jb]
                dgemm_i_time = compute_dgemm_time(panel.grid.core, ts_args)
                mpi_ext_sleep(dgemm_i_time, panel.grid.all_comm)
            else:
                #HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                #       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                #       Aptr, lda );
                ts_args = [mp, nn, jb]
                dgemm_i_time = compute_dgemm_time(panel.grid.core, ts_args)
                mpi_ext_sleep(dgemm_i_time, panel.grid.all_comm)
            nq0 += nn
            test = HPL_bcast(pbcst)
            
        # The panel has been forwarded at that point, finish the update
        nn = n - nq0
        
        #print "nn, n, mp, jb = ",nn, n, mp, jb
        
        if (nn > 0):
            #HPL_dtrsm( HplColumnMajor, HplRight, HplUpper, HplNoTrans,
            #        HplUnit, nn, jb, HPL_rone, L1ptr, jb, Uptr, LDU );
            ts_args = [nn, jb, 'R']
            dtrsm_i_time = compute_dtrsm_time(panel.grid.core, ts_args)
            mpi_ext_sleep(dtrsm_i_time, panel.grid.all_comm)
            if (curr <> 0):
                #HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                #       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                #       Mptr( Aptr, jb, 0, lda ), lda );
                ts_args = [mp, nn, jb]
                dgemm_i_time = compute_dgemm_time(panel.grid.core, ts_args)
                mpi_ext_sleep(dgemm_i_time, panel.grid.all_comm)
            else:
                #HPL_dgemm( HplColumnMajor, HplNoTrans, HplTrans, mp, nn,
                #       jb, -HPL_rone, L2ptr, ldl2, Uptr, LDU, HPL_rone,
                #       Aptr, lda );
                ts_args = [mp, nn, jb]
                dgemm_i_time = compute_dgemm_time(panel.grid.core, ts_args)
                mpi_ext_sleep(dgemm_i_time, panel.grid.all_comm)
                
            #nupdates += 1
            #print "(",panel.grid.myrow,",",panel.grid.mycol,") - nn, n, mp, jb = ",nn, n, mp, jb
                
    panel.nq -= n
    #panel.jj += n
    # return the outcome of the probe  (should always be  HPL_SUCCESS,  the
    # panel broadcast is enforced in that routine).
    if (pbcst <> None): IFLAG = test
                
    return IFLAG
