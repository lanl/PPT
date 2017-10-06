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

int MPI_Init(int *, char ***);
int MPI_Finalize(void);
int MPI_Comm_size(MPI_Comm, int *);
int MPI_Comm_rank(MPI_Comm, int *);
int MPI_Iprobe(int, int, MPI_Comm, int *, MPI_Status *);
int MPI_Probe(int, int, MPI_Comm, MPI_Status *);
int MPI_Send(void *, int, MPI_Datatype, int, int, MPI_Comm);
int MPI_Recv(void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
int MPI_Get_count(MPI_Status *, MPI_Datatype, int *);
int MPI_Get_elements(MPI_Status *, MPI_Datatype, int *);
int MPI_Allreduce(void *, void *, int, MPI_Datatype, MPI_Op, MPI_Comm);
]]

-- Dynamically link the MPI library into the global namespace
if not pcall(function() return ffi.C.MPI_Init end) then
    return ffi.load("mpi", true)
end
