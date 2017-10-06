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
    MPI_Iprobe, MPI_Probe, MPI_Send, MPI_Recv, MPI_Get_count,
    MPI_Get_elements, MPI_Allreduce
        = mpi.MPI_Init, mpi.MPI_Finalize, mpi.MPI_Comm_rank,
        mpi.MPI_Comm_size, mpi.MPI_Iprobe, mpi.MPI_Probe,
        mpi.MPI_Send, mpi.MPI_Recv, mpi.MPI_Get_count,
        mpi.MPI_Get_elements, mpi.MPI_Allreduce

function MPI.init(self)
    if MPI_Init(nil, nil) == SUCCESS then
        self.CBUF_LEN = 4096 --4kB

        self.comm = ffi.cast("MPI_Comm", COMMWORLD)
        self.BYTE = ffi.cast("MPI_Datatype", BYTE)
        self.DOUBLE = ffi.cast("MPI_Datatype", DOUBLE)
        self.MIN = ffi.cast("MPI_Op", MIN)
        self.SUM = ffi.cast("MPI_Op", SUM)
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
    local src = src or ANYSRC
    local tag = tag or ANYTAG
    if MPI_Iprobe(src, tag, self.comm, self.itemp, self.status) == SUCCESS then
        return (self.itemp[0] ~= 0)
    end
    error("Could not Iprobe in MPI")
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

function MPI.recv(self, maxSize, src, tag) --Blocking
    local src = src or ANYSRC
    local tag = tag or ANYTAG
    local m = self.ctemp
    if maxSize > self.CBUF_LEN then --Temporary buffer is too small
        m = ffi.new("char[?]", maxSize)
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

return MPI
