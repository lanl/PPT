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
--  Common MPI wrapper for MPICH3 and Open-MPI
--  NOTE: Currently, MPICH-v3.1.3 and Open-MPI-v1.6.5 work
--      There are some severe bugs in Open-MPI-v1.8.3
--]]
--local mpi = require "OMPI"
local mpi = require "MPICH"
local msg = require "MessagePack"
local ffi = require "ffi"

local MPI = {}

--Cache variables
local ANYSRC, ANYTAG, SUCCESS, COMMWORLD, BYTE, DOUBLE, MIN, SUM
    = mpi.MPI_ANY_SOURCE, mpi.MPI_ANY_TAG, mpi.MPI_SUCCESS,
        mpi.MPI_COMM_WORLD, mpi.MPI_BYTE, mpi.MPI_DOUBLE,
        mpi.MPI_MIN, mpi.MPI_SUM
local MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size,
    MPI_Iprobe, MPI_Probe, MPI_Send, MPI_Isend, MPI_Recv, MPI_Get_count,
    MPI_Get_elements, MPI_Allreduce, MPI_Barrier
        = mpi.MPI_Init, mpi.MPI_Finalize, mpi.MPI_Comm_rank,
        mpi.MPI_Comm_size, mpi.MPI_Iprobe, mpi.MPI_Probe,
        mpi.MPI_Send, mpi.MPI_Isend, mpi.MPI_Recv, mpi.MPI_Get_count,
        mpi.MPI_Get_elements, mpi.MPI_Allreduce, mpi.MPI_Barrier

function MPI.init(self)
    if MPI_Init(nil, nil) == SUCCESS then
        self.CBUF_LEN = 4096 --4kB

        self.comm = ffi.cast("MPI_Comm", COMMWORLD)
        self.BYTE = ffi.cast("MPI_Datatype", BYTE)
        self.DOUBLE = ffi.cast("MPI_Datatype", DOUBLE)
        self.MIN = ffi.cast("MPI_Op", MIN)
        self.SUM = ffi.cast("MPI_Op", SUM)
        self.request = ffi.new("MPI_Request[1]")
        self.status = ffi.new("MPI_Status[1]")
        self.itemp = ffi.new("int[5]")
        self.dtemp = ffi.new("double[5]")
        self.ctemp = ffi.new("char[?]", self.CBUF_LEN) --Preallocate
        return false
    end
    error("Could not initialize MPI")
end

function MPI.finalize(self)
    if MPI_Finalize() == SUCCESS then
        return false
    end
    error("Could not finalize MPI")
end

function MPI.rank(self)
    if MPI_Comm_rank(self.comm, self.itemp) == SUCCESS then
        return self.itemp[0]
    end
    error("Could not get rank in MPI")
end

function MPI.size(self)
    local size = ffi.new("int[1]")
    if MPI_Comm_size(self.comm, self.itemp) == SUCCESS then
        return self.itemp[0]
    end
    error("Could not get size in MPI")
end

function MPI.iprobe(self, src, tag) --Non-blocking asynch
    --Probe trial number of times (to catch end cases)
    local src = src or ANYSRC
    local tag = tag or ANYTAG
    if MPI_Iprobe(src, tag, self.comm, self.itemp, self.status) == SUCCESS then
        return (self.itemp[0] > 0)
    end
    error("Could not Iprobe in MPI")
end

function MPI.iprobeTrials(self, trials, src, tag) --Non-blocking asynch
    local src = src or ANYSRC
    local tag = tag or ANYTAG
    local trials = trials or 0
    for i=1,trials do --Probe trial number of times (to catch more difficult cases)
        if MPI_Iprobe(src, tag, self.comm, self.itemp, self.status) == SUCCESS then
            if (self.itemp[0] > 0) then
                return true
            else
                return self:iprobe(src, tag)
            end
        end
        error("Could not Iprobe in MPI")
    end
    return true
end

function MPI.probe(self, src, tag) --Blocking synch
    local src = src or ANYSRC
    local tag = tag or ANYTAG
    return (MPI_Probe(src, tag, self.comm, self.status) == SUCCESS)
end

function MPI.send(self, x, dst, tag) --Blocking
    local m = msg.pack(x)
    local tag = tag or #m --Set to message length if nil
    if MPI_Send(m, #m, self.BYTE, dst, tag, self.comm) ~= SUCCESS then
        error("Could not Send in MPI")
    end
end

function MPI.isend(self, x, dst, tag) --Non-Blocking
    local m = msg.pack(x)
    local tag = tag or #m --Set to message length if nil
    if MPI_Isend(m, #m, self.BYTE, dst, tag, self.comm, self.request) ~= SUCCESS then
        error("Could not Isend in MPI")
    end
end

function MPI.recv(self, maxSize, src, tag) --Blocking
    local src = src or ANYSRC
    local tag = tag or ANYTAG
    local m = self.ctemp
    if maxSize > self.CBUF_LEN then --Temporary buffer is too small
        self.ctemp = ffi.new("char[?]", maxSize)
        self.CBUF_LEN = maxSize
    end
    if MPI_Recv(m, maxSize, self.BYTE, src, tag, self.comm, self.status) == SUCCESS then
        return msg.unpack(ffi.string(m, maxSize))
    end
    error("Could not Recv in MPI")
end

function MPI.getCount(self) --Non-blocking
    if MPI_Get_count(self.status, self.BYTE, self.itemp) == SUCCESS then
        return self.itemp[0]
    end
    error("Could not Get_count in MPI")
end

function MPI.getElements(self) --Non-blocking
    if MPI_Get_elements(self.status, self.BYTE, self.itemp) == SUCCESS then
        return self.itemp[0]
    end
    error("Could not Get_elements in MPI")
end

function MPI.recvAnySize(self, src, tag)
    return self:recv(self:getCount(), src, tag)
end

function MPI.allreduce(self, partial, op)
    self.dtemp[0] = partial
    if (MPI_Allreduce(self.dtemp, self.dtemp+1,
                1, self.DOUBLE, --Single double operand
                op, self.comm) ~= SUCCESS) then
        error("Could not Allreduce in MPI")
    end
    return self.dtemp[1]
end

function MPI.barrier(self)
    if (MPI_Barrier(self.comm) ~= SUCCESS) then
        error("Could not Barrier in MPI")
    end
end

return MPI
