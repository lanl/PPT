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
#Purpose: PDES Engine in CPython/PyPy
#  Common MPI wrapper for MPICH3 and Open-MPI
#  NOTE: Currently, MPICH-v3.1.3 and Open-MPI-v1.6.5 work
#      There are some severe bugs in Open-MPI-v1.8.3
import ctypes as C

from MPICH import loadMPI
mpi = None

import umsgPack as msg #umsgpack-python-pure can be substituted with msgpack-pure or msgpack-python
Pack = msg.packb
Unpack = msg.unpackb

from utils import SimianError

class MPI(object):
    def __init__(self, libName):
        global mpi
        mpi = loadMPI(libName)

        if mpi.MPI_Init(None, None) != mpi.MPI_SUCCESS:
            raise SimianError("Could not initialize MPI")

        self.CBUF_LEN = 32*1024 #32kB

        self.comm = mpi.MPI_COMM_WORLD
        self.BYTE = mpi.MPI_BYTE
        self.DOUBLE = mpi.MPI_DOUBLE
        self.MIN = mpi.MPI_MIN
        self.SUM = mpi.MPI_SUM

        self.request = mpi.MPI_Request()
        self.status = mpi.MPI_Status()
        self.itemp = C.c_int()
        self.dtemp0 = C.c_double()
        self.dtemp1 = C.c_double()
        self.ctemp = C.create_string_buffer(self.CBUF_LEN) #Preallocate

    def finalize(self):
        if mpi.MPI_Finalize() == mpi.MPI_SUCCESS:
            return False
        raise SimianError("Could not finalize MPI")

    def rank(self):
        if mpi.MPI_Comm_rank(self.comm, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp.value
        raise SimianError("Could not get rank in MPI")

    def size(self):
        size = (C.c_int * 1)()
        if mpi.MPI_Comm_size(self.comm, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp.value
        raise SimianError("Could not get size in MPI")

    def iprobe(self, src=None, tag=None): #Non-blocking asynch
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        if mpi.MPI_Iprobe(src, tag, self.comm, C.byref(self.itemp), C.byref(self.status)) == mpi.MPI_SUCCESS:
            return (self.itemp.value != 0)
        raise SimianError("Could not Iprobe in MPI")

    def probe(self, src=None, tag=None): #Blocking synch
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        return (mpi.MPI_Probe(src, tag, self.comm, C.byref(self.status)) == mpi.MPI_SUCCESS)

    def send(self, x, dst, tag=None): #Blocking
        m = Pack(x)
        tag = tag or len(m) #Set to message length if None
        if mpi.MPI_Send(m, len(m), self.BYTE, dst, tag, self.comm) != mpi.MPI_SUCCESS:
            raise SimianError("Could not Send in MPI")

    def isend(self, x, dst, tag=None): #Non-Blocking
        m = Pack(x)
        tag = tag or len(m) #Set to message length if None
        if mpi.MPI_Isend(m, len(m), self.BYTE, dst, tag, self.comm, C.byref(self.request)) != mpi.MPI_SUCCESS:
            raise SimianError("Could not Isend in MPI")

    def recv(self, maxSize, src=None, tag=None): #Blocking
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        m = self.ctemp
        if maxSize > self.CBUF_LEN: #Temporary buffer is too small
            m = C.create_string_buffer(maxSize)
        if mpi.MPI_Recv(m, maxSize, self.BYTE, src, tag, self.comm, C.byref(self.status)) == mpi.MPI_SUCCESS:
            #return Unpack(m.raw)
            return Unpack(m[:maxSize])
        raise SimianError("Could not Recv in MPI")

    def getCount(self): #Non-blocking
        if mpi.MPI_Get_count(C.byref(self.status), self.BYTE, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp.value
        raise SimianError("Could not Get_count in MPI")

    def getElements(self): #Non-blocking
        if mpi.MPI_Get_elements(C.byref(self.status), self.BYTE, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp
        raise SimianError("Could not Get_elements in MPI")

    def recvAnySize(self, src=None, tag=None):
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        return self.recv(self.getCount(), src, tag)

    def allreduce(self, partial, op):
        self.dtemp0 = C.c_double(partial)
        if (mpi.MPI_Allreduce(C.byref(self.dtemp0), C.byref(self.dtemp1),
                    1, self.DOUBLE, #Single double operand
                    op, self.comm) != mpi.MPI_SUCCESS):
            raise SimianError("Could not Allreduce in MPI")
        return self.dtemp1.value

    def barrier(self):
        if (mpi.MPI_Barrier(self.comm) != mpi.MPI_SUCCESS):
            SimianError("Could not Barrier in MPI")
