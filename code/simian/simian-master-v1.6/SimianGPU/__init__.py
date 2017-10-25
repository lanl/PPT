"""
This python module, SimianPie implements a subset of the functionality in Simian JIT-PDES.
Currently unsupported features of Simian(Lua) vs SimianPie are (SimianPie is a subset of Simian(Lua)):
    (1) Any kind of JITing. However, interpreter loop of Python can be JITed by resorting to PyPy.
    (2) FFI calls to external C/C++ functions
    (3) The messages are packed/unpacked internally by mpi4py. So, msgpack protocol is not used.
    (4) mpi4py depends on MPICH, so OpenMPI is not supported.
"""
