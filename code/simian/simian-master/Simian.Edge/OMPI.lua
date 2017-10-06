--[[
Copyright (c) 2015, Los Alamos National Security, LLC
All rights reserved.

Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
]]

--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Wrapper for Open-MPI-v1.8.3
--  Currently, Open-MPI-v1.6.5 only works
--]]
local ffi = require("ffi")

ffi.cdef[[
typedef struct MPI_Status MPI_Status;
struct MPI_Status {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
  int _cancelled;
  size_t _ucount;
};

static const int MPI_SUCCESS = 0;
static const int MPI_ANY_TAG = -1;
static const int MPI_ANY_SOURCE = -1;

typedef struct ompi_communicator_t *MPI_Comm;
extern struct ompi_predefined_communicator_t MPI_COMM_WORLD __asm__("ompi_mpi_comm_world");

typedef struct ompi_datatype_t *MPI_Datatype;
extern struct ompi_predefined_datatype_t MPI_DOUBLE __asm__("ompi_mpi_double");
extern struct ompi_predefined_datatype_t MPI_BYTE __asm__("ompi_mpi_byte");

typedef struct ompi_op_t *MPI_Op;
extern struct ompi_predefined_op_t MPI_MIN __asm__("ompi_mpi_op_min");
extern struct ompi_predefined_op_t MPI_SUM __asm__("ompi_mpi_op_sum");

typedef struct ompi_request_t *MPI_Request;

int MPI_Init(int *, char ***);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm, int *);
int MPI_Comm_rank(MPI_Comm, int *);
int MPI_Iprobe(int, int, MPI_Comm, int *, MPI_Status *);
int MPI_Probe(int, int, MPI_Comm, MPI_Status *);
int MPI_Send(const void *, int, MPI_Datatype, int, int, MPI_Comm);
int MPI_Isend(const void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int MPI_Recv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
int MPI_Get_count(MPI_Status *, MPI_Datatype, int *);
int MPI_Get_elements(MPI_Status *, MPI_Datatype, int *);
int MPI_Allreduce(void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
int MPI_Barrier(MPI_Comm);
]]

-- Dynamically link the MPI library into the global namespace
if not pcall(function() return ffi.C.MPI_Init end) then
    return ffi.load("mpi", true)
end
