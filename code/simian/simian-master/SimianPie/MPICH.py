#Copyright (c) 2015, Los Alamos National Security, LLC
#All rights reserved.
#
#Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
#Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
#	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
#	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
#THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Author: Nandakishore Santhi
#Date: 15 April, 2015
#Copyright: Open source, must acknowledge original author
#Purpose: PDES Engine in CPython and PyPy
#  Wrapper for MPICH-v3.1.3
import ctypes as C

#Dynamically link the MPICH library into the global namespace
def loadMPI(mpichLib):
    class MPI_Status(C.Structure):
        _fields_ = [
            ("count_lo", C.c_int),
            ("count_hi_and_cancelled", C.c_int),
            ("MPI_SOURC", C.c_int),
            ("MPI_TA", C.c_int),
            ("MPI_ERRO", C.c_int)
            ]


    mpi = C.CDLL(mpichLib) #Looks in current directory
    if not mpi:
        raise SimianError("Could not load mpich dynamic library")

    mpi.MPI_Status = MPI_Status

    mpi.MPI_SUCCESS = 0
    mpi.MPI_ANY_TAG = -1
    mpi.MPI_ANY_SOURCE = -2

    mpi.MPI_Comm = C.c_int
    mpi.MPI_COMM_WORLD = mpi.MPI_Comm(0x44000000)

    mpi.MPI_Datatype = C.c_int
    mpi.MPI_BYTE = mpi.MPI_Datatype(0x4c00010d)
    mpi.MPI_DOUBLE = mpi.MPI_Datatype(0x4c00080b)

    mpi.MPI_Op = C.c_int
    mpi.MPI_MIN = mpi.MPI_Op(0x58000002)
    mpi.MPI_SUM = mpi.MPI_Op(0x58000003)

    mpi.MPI_Request = C.c_int
    mpi.MPI_Init.restype = C.c_int
    mpi.MPI_Finalize.restype = C.c_int
    mpi.MPI_Comm_size.restype = C.c_int
    mpi.MPI_Comm_rank.restype = C.c_int
    mpi.MPI_Iprobe.restype = C.c_int
    mpi.MPI_Probe.restype = C.c_int
    mpi.MPI_Send.restype = C.c_int
    mpi.MPI_Isend.restype = C.c_int
    mpi.MPI_Recv.restype = C.c_int
    mpi.MPI_Get_count.restype = C.c_int
    mpi.MPI_Get_elements.restype = C.c_int
    mpi.MPI_Allreduce.restype = C.c_int
    mpi.MPI_Barrier.restype = C.c_int

    mpi.MPI_Init.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_char_p)]
    mpi.MPI_Finalize.argtypes = []
    mpi.MPI_Comm_size.argtypes = [mpi.MPI_Comm, C.POINTER(C.c_int)]
    mpi.MPI_Comm_rank.argtypes = [mpi.MPI_Comm, C.POINTER(C.c_int)]
    mpi.MPI_Iprobe.argtypes = [C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(C.c_int), C.POINTER(mpi.MPI_Status)]
    mpi.MPI_Probe.argtypes = [C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(mpi.MPI_Status)]
    mpi.MPI_Send.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_int, C.c_int, mpi.MPI_Comm]
    mpi.MPI_Isend.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(mpi.MPI_Request)]
    mpi.MPI_Recv.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(mpi.MPI_Status)]
    mpi.MPI_Get_count.argtypes = [C.POINTER(mpi.MPI_Status), mpi.MPI_Datatype, C.POINTER(C.c_int)]
    mpi.MPI_Get_elements.argtypes = [C.POINTER(mpi.MPI_Status), mpi.MPI_Datatype, C.POINTER(C.c_int)]
    mpi.MPI_Allreduce.argtypes = [C.c_void_p, C.c_void_p, C.c_int, mpi.MPI_Datatype, mpi.MPI_Op, mpi.MPI_Comm]
    mpi.MPI_Barrier.argtypes = [mpi.MPI_Comm]

    return mpi
