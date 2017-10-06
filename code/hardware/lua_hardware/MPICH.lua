--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Wrapper for MPICH-v3.1.3
--]]
local ffi = require("ffi")

ffi.cdef[[
typedef struct MPI_Status MPI_Status;
typedef struct MPI_Status {
    int count_lo;
    int count_hi_and_cancelled;
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
};

static const int MPI_SUCCESS = 0;
static const int MPI_ANY_TAG = -1;
static const int MPI_ANY_SOURCE = -2;

typedef int MPI_Comm;
static const MPI_Comm MPI_COMM_WORLD = (MPI_Comm)0x44000000;

typedef int MPI_Datatype;
static const MPI_Datatype MPI_BYTE = (MPI_Datatype)0x4c00010d;
static const MPI_Datatype MPI_DOUBLE = (MPI_Datatype)0x4c00080b;

typedef int MPI_Op;
static const MPI_Op MPI_MIN = (MPI_Op)0x58000002;
static const MPI_Op MPI_SUM = (MPI_Op)0x58000003;

int MPI_Init(int *argc, char ***argv);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm comm, int *size);
int MPI_Comm_rank(MPI_Comm comm, int *rank);
int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MPI_Get_elements(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
]]

--Dynamically link the MPICH library into the global namespace
if not pcall(function() return ffi.C.MPI_Init end) then
    return ffi.load("mpich", true)
end
