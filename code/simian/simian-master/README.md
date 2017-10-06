# Simian JIT PDES

=================================================================================
Simian Process Oriented Conservative Parallel Discrete Event Simulator from LANL
=================================================================================

Nandakishore Santhi (nsanthi at lanl dot gov)

Simian contains the Lua implementation and it needs luajit-2.1. SimianPie contains the Python implementation, which needs Python 2.7.x with greenlets (optional) or Pypy 2.4.x. MPICH 3.1.4 or OpenMPI 1.6.x are optionally needed if using MPI.

See Docs for API documentation. Example.Lua has examples of Simian (Lua) usage. Example.Py has examples of SimianPie usage.

##If not using MPI:
    Set useMPI flag to false when initializing the Simian PDES Engine

##To use MPI with Simian (Lua):

###If using MPICH:
    (tested with 3.1.3)
    Set useMPI flag to true. Set a link to libmpich.[dylib/so/dll] in the top directory

###If using OpenMPI:
    (some later versions such as 1.8.3 have a serious bug in message size reporting; use 1.6.x)
    Set useMPI flag to true. Set a link to libmpi.[dylib/so/dll] in the top directory, and then comment within file Simian/MPI.lua line 'require "MPICH"' and uncomment in file Simian/MPI.lua line 'require "MPI"'

##To use the Python version SimianPie:
    SimianPie is tested to work with CPython 2.7.x and PyPy 2.4.x
    At present when needing MPI, SimianPie works with either MPICH2 using CTypes or using MPI4Py module - if only OpenMPI is available use Simian (Lua) or if Python version is unavoidable, then user should write a CTypes wrapper for OpenMPI similar to the distributed CTyped wrapper for MPICH 3.1.3
        Set useMPI flag to true. Set a link to libmpich.[dylib/so/dll] in the top directory or pass absolute path to the library to Simian() when creating the engine.

##Test with pHold app without MPI:
    luajit Examples.Lua/phold-noop-noMPI.lua

##Testing with MPI on LANL PDES benchmark app:
    (on a medium sized cluster with more than 1000 cores)
    mpirun -np 1000 luajit-2.1.0-alpha Examples.Lua/pdes_lanl_benchmarkV8.lua 1000 100 1 0 0 false 1 0 100000 0 0.5 1 10 1000 1 true LANL_PDES.log

##LANL internal reference:
CODE Title: Simian, version 1.5 (OSS)

LACC #:  LA-CC-15-015

Copyright Number Assigned: C15036

Funding source: Laboratory-Directed Research and Development (LDRD)
