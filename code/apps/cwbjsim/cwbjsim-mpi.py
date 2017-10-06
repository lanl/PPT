"""
 CWBJSim, an application simulator of the CWBJ radiation transport method.
 
 Date: May 6, 2016  Max Rosa (MR)
"""

# To run: python [CWBJSim py file name]
# Output: To screen and to file cwbjsim-mpi.#.out

version = "mpi-2016.07.21.1"

# Set up path  variables; PPT applications expect it in this fashion.
from sys import path
path.append('../..')
from ppt import *

#path.append('../../simian/simian-master/SimianPie')
#path.append('../../hardware')
#path.append('../../hardware/interconnect')
#path.append('../../hardware/interconnect/configs')
#path.append('../../middleware/mpi')
#path.append('../../middleware/threading')

#sys.dont_write_bytecode = True

import math

###############################################################################
###############################################################################


# Problem Parameters
###############################################################################

nx = 8        # discretization units (aka cells) in x-dimension >0
ny = 8        # discretization units (aka cells) in y-dimension >=0
nz = 0        # discretization units (aka cells) in z-dimension >=0
ichunk = 4    # number of cells per chunk in x direction >=1 <=nx
jchunk = 4    # number of cells per chunk in y direction >=1 <=ny
kchunk = 1    # number of cells per chunk in z direction >=1 <=nz
nmom = 1      # anisotropic scattering order
nang = 1   # number of discrete ordinates per octant(3D), quadrant(2D)
              # or hemisphere(1D). #MR: If q is the sn quadrature order:
              # nang = q*(q+2)/8 for 2D/3D Level Symmetric quadrature
              # nang = q/2 for 1D Level Symmetric quadrature
              # nang = q*q/4 for 2D Square Chebychev-Legendre quadrature
ng = 1        # number of energy groups
ng_bndls = 1  # number of energy group bumdles #MR: add
nsteps = 1    # number of time-steps to be simulated
iitm = 4      # number of inner iterations before convergence
oitm = 1      # number of outer iterations before convergence

# set 1D-3D problem subcells, octants and moments
# MR: Need 3D potentially for comapring with SNAPSim
# MR: Need 2D for comapring with Capsaicin
# MR: No need for 1D at present
noct = 8        # number of octants in 3D
cmom = nmom**2  # number of flux moments
subcells = 1    # 1 (Hex), 2 (Prisms), 5, 6, 24 (Tets)
if nz == 0:
  noct = 4                  # number of quadrants in 2D
  cmom = nmom*(nmom + 1)/2  # number of flux moments
  subcells = 4              # 1 (Quad), 2, 4 (Triangles)
  if ny == 0:
    #noct = 2      # number of hemispheres in 1D
    #cmom = nmom   # number of flux moments
    #subcells = 1  # 1 (Segment)
    print "Warning: At present 1D problems are not supported!"
    sys.exit(1)

total_angles = noct*nang  # total number of discrete ordinates

nxchunks = int(math.ceil(float(nx)/ichunk))  # number of spatial chunks in x
nychunks = int(math.ceil(float(ny)/jchunk))  # number of spatial chunks in y
nzchunks = int(math.ceil(float(nz)/kchunk))  # number of spatial chunks in z

if nzchunks > 0:
  total_ranks = nxchunks*nychunks*nzchunks  # number of MPI ranks for 3D
else:
  total_ranks = nxchunks*nychunks  # number of MPI ranks for 2D

total_chunks_per_rank = 1  #MR: total chunks per rank
                           #MR: for CWBJ it is assumed that 'chunk' is a
                           #MR: 'spatial chunk' and there can only be one
                           #MR: 'chunk' per rank, while for SNAP a 'chunk'
                           #MR: is a 'task' in KBA's directed graph tasks' 
                           #MR: list along an x pencil and there can be
                           #MR: more than one chunk per rank. 

if total_chunks_per_rank > 1:
  print "Warning: total_chunks_per_rank > 1!\n" \
        "For CWBJ a 'chunk' is assumed to be a 'spatial chunk',\n" \
        "and each spatial chunk is assigned to an MPI rank,\n" \
        "hence total_chunks_per_rank should not exceed 1."
  sys.exit(1)

# Helper constants (one only used right now)
###############################################################################

U_SEC = 1
M_SEC = 1000*U_SEC
SEC = 1000*M_SEC

SYNC_INTERVAL = 1   # wait time for thread efficiency
PSI_SIZE = 8        #MR: number of data bytes per angular flux component (double),
                    #MR: determines size of MPI message
STAT_FREQUENCY = 1  # Statistics frequency

###############################################################################


# compute_neighbors
###############################################################################

def compute_neighbors(chunk_id):
  """
  Returns the spatial chunks (neighbors_list) which are neighbors of chunk_id.

  Incoming: chunk_id=(z, y, x)
  Outgoing: neighbors_list
  """

  (z, y, x) = chunk_id

  neighbors_list = []

  if noct == 4: # 2D

    if nxchunks > 1 and nychunks > 1:
      # chunks with four neighbors
      if x > 0 and x < nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))

      # chunks with three neighbors
      if y == 0 and x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y + 1, x))

      if y == nychunks - 1 and x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))

      if x == 0 and y > 0 and y < nychunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      if x == nxchunks - 1 and y > 0 and y < nychunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      # chunks with two neighbors 
      if x == 0 and y == 0:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y + 1, x))

      if x == nxchunks - 1 and y == 0:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y + 1, x))

      if x == 0 and y == nychunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))

      if x == nxchunks - 1 and y == nychunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y - 1, x))

    if nxchunks > 1 and nychunks == 1:
      # chunks with two neighbors along x
      if x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))

      # chunks with one neighbor along x 
      if x == 0:
        neighbors_list.append((z, y, x + 1))

      if x == nxchunks - 1:
        neighbors_list.append((z, y, x - 1))

    if nxchunks == 1 and nychunks > 1:
      # chunks with two neighbors along y
      if y > 0 and y < nychunks - 1:
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      # chunks with one neighbor along y 
      if y == 0:
        neighbors_list.append((z, y + 1, x))

      if y == nychunks - 1:
        neighbors_list.append((z, y - 1, x))

  elif noct == 8: # 3D

    if nxchunks > 1 and nychunks > 1 and nzchunks > 1:
      # chunks with six neighbors
      if x > 0 and x < nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          if z > 0 and z < nzchunks - 1:
            neighbors_list.append((z, y, x - 1))
            neighbors_list.append((z, y, x + 1))
            neighbors_list.append((z, y - 1, x))
            neighbors_list.append((z, y + 1, x))
            neighbors_list.append((z - 1, y, x))
            neighbors_list.append((z + 1, y, x))
    
      # chunks with five neighbors
      if z == 0 and x > 0 and x < nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z + 1, y, x))

      if z == nzchunks - 1 and x > 0 and x < nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))

      if y == 0 and x > 0 and x < nxchunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      if y == nychunks - 1 and x > 0 and x < nxchunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      if x == 0 and y > 0 and y < nychunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and y > 0 and y < nychunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      # chunks with four neighbors
      if z == 0 and y == 0:
        if x > 0 and x < nxchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z + 1, y, x))

      if z == 0 and y == nychunks - 1:
        if x > 0 and x < nxchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z + 1, y, x))

      if z == nzchunks - 1 and y == 0:
        if x > 0 and x < nxchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))

      if z == nzchunks - 1 and y == nychunks - 1:
        if x > 0 and x < nxchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z - 1, y, x))

      if z == 0 and x == 0:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z + 1, y, x))

      if z == 0 and x == nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z + 1, y, x))

      if z == nzchunks - 1 and x == 0:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))

      if z == nzchunks - 1 and x == nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))

      if x == 0 and y == 0:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      if x == 0 and y == nychunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and y == 0:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y + 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and y == nychunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      # chunks with three neighbors 
      if x == 0 and y == 0 and z == 0:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y + 1, x))
        neighbors_list.append((z + 1, y, x))

      if x == 0 and y == nychunks - 1 and z == 0:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and y == 0 and z == 0:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y + 1, x))
        neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and y == nychunks - 1 and z == 0:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z + 1, y, x))

      if x == 0 and y == 0 and z == nzchunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y + 1, x))
        neighbors_list.append((z - 1, y, x))

      if x == 0 and y == nychunks - 1 and z == nzchunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z - 1, y, x))

      if x == nxchunks - 1 and y == 0 and z == nzchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y + 1, x))
        neighbors_list.append((z - 1, y, x))

      if x == nxchunks - 1 and y == nychunks - 1 and z == nzchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z - 1, y, x))

    if nxchunks > 1 and nychunks > 1 and nzchunks == 1:
      # chunks with four neighbors
      if x > 0 and x < nxchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))

      # chunks with three neighbors
      if y == 0 and x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y + 1, x))

      if y == nychunks - 1 and x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))

      if x == 0 and y > 0 and y < nychunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      if x == nxchunks - 1 and y > 0 and y < nychunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      # chunks with two neighbors 
      if x == 0 and y == 0:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y + 1, x))

      if x == nxchunks - 1 and y == 0:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y + 1, x))

      if x == 0 and y == nychunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z, y - 1, x))

      if x == nxchunks - 1 and y == nychunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y - 1, x))

    if nxchunks > 1 and nychunks == 1 and nzchunks > 1:
      # chunks with four neighbors
      if x > 0 and x < nxchunks - 1:
        if z > 0 and z < nzchunks - 1:
          neighbors_list.append((z, y, x - 1))
          neighbors_list.append((z, y, x + 1))
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))

      # chunks with three neighbors
      if z == 0 and x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z + 1, y, x))

      if z == nzchunks - 1 and x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z - 1, y, x))

      if x == 0 and z > 0 and z < nzchunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and z > 0 and z < nzchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z + 1, y, x))

      # chunks with two neighbors 
      if x == 0 and z == 0:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z + 1, y, x))

      if x == nxchunks - 1 and z == 0:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z + 1, y, x))

      if x == 0 and z == nzchunks - 1:
        neighbors_list.append((z, y, x + 1))
        neighbors_list.append((z - 1, y, x))

      if x == nxchunks - 1 and z == nzchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z - 1, y, x))

    if nxchunks == 1 and nychunks > 1 and nzchunks > 1:
      # chunks with four neighbors
      if z > 0 and z < nzchunks - 1:
        if y > 0 and y < nychunks - 1:
          neighbors_list.append((z - 1, y, x))
          neighbors_list.append((z + 1, y, x))
          neighbors_list.append((z, y - 1, x))
          neighbors_list.append((z, y + 1, x))

      # chunks with three neighbors
      if y == 0 and z > 0 and z < nzchunks - 1:
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z + 1, y, x))
        neighbors_list.append((z, y + 1, x))

      if y == nychunks - 1 and z > 0 and z < nzchunks - 1:
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z + 1, y, x))
        neighbors_list.append((z, y - 1, x))

      if z == 0 and y > 0 and y < nychunks - 1:
        neighbors_list.append((z + 1, y, x))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      if z == nzchunks - 1 and y > 0 and y < nychunks - 1:
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      # chunks with two neighbors 
      if z == 0 and y == 0:
        neighbors_list.append((z + 1, y, x))
        neighbors_list.append((z, y + 1, x))

      if z == nzchunks - 1 and y == 0:
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z, y + 1, x))

      if z == 0 and y == nychunks - 1:
        neighbors_list.append((z + 1, y, x))
        neighbors_list.append((z, y - 1, x))

      if z == nzchunks - 1 and y == nychunks - 1:
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z, y - 1, x))

    if nxchunks > 1 and nychunks == 1 and nzchunks == 1:
      # chunks with two neighbors along x
      if x > 0 and x < nxchunks - 1:
        neighbors_list.append((z, y, x - 1))
        neighbors_list.append((z, y, x + 1))

      # chunks with one neighbor along x 
      if x == 0:
        neighbors_list.append((z, y, x + 1))

      if x == nxchunks - 1:
        neighbors_list.append((z, y, x - 1))

    if nxchunks == 1 and nychunks > 1 and nzchunks == 1:
      # chunks with two neighbors along y
      if y > 0 and y < nychunks - 1:
        neighbors_list.append((z, y - 1, x))
        neighbors_list.append((z, y + 1, x))

      # chunks with one neighbor along y 
      if y == 0:
        neighbors_list.append((z, y + 1, x))

      if y == nychunks - 1:
        neighbors_list.append((z, y - 1, x))

    if nxchunks == 1 and nychunks == 1 and nzchunks > 1:
      # chunks with two neighbors along z
      if z > 0 and z < nzchunks - 1:
        neighbors_list.append((z - 1, y, x))
        neighbors_list.append((z + 1, y, x))

      # chunks with one neighbor along z 
      if z == 0:
        neighbors_list.append((z + 1, y, x))

      if z == nzchunks - 1:
        neighbors_list.append((z - 1, y, x))

  else:
    print "WARNING: Unsupported spatial dimension!"
    sys.exit(1)

  # return the neighbors list
  return neighbors_list

###############################################################################


# compute_chunk_time
###############################################################################

def compute_chunk_time(core):
  """
  Put together the task list to send to hardware. Contains all the formulas for
  the flops, iops, and memory ops performed per chunk.

  Incoming: core -- core id, each core will perform this calc for one chunk
  Outgoing: chunk_time -- predicted time to solve one work chunk, which is
                          a set of cells, all angles and all group bundles 
  """

  nijk = ichunk*jchunk*kchunk #MR: total computational cells per chunk

  #MR: vertices per subcell
  if noct == 4: # 2D
    if subcells == 1: # Quad
      vertices = 4;
    elif (subcells == 2 or subcells == 4): # Triangles
      vertices = 3;
    else:
      print "Warning: Unknown number of 2D subcells!"
      exit()
  elif noct == 8: # 3D
    if subcells == 1: # Hex
      vertices = 8;
    elif subcells == 2: # Prisms
      vertices = 6;
    elif (subcells == 5 or subcells == 6 or subcells == 24): # Tets
      vertices = 4;
    else:
      print "Warning: Unknown number of 3D subcells!"
      exit()
  else:
    print "Warning: Supported dimensions are 2 and 3!"
    sys.exit(1)

  #MR: energy group check
  if (ng % ng_bndls) != 0:
    print "Warning: For ease of implementation assume ng multiple of ng_bndls!"
    sys.exit(1)

  gbndl = ng / ng_bndls #MR: number of groups per energy group bundle

  N = vertices * total_angles * gbndl #MR: size of linear system per 
                                      #MR: subcell and energy group bundle

  mem_access_model = 'HITRATES' #'MEM_ACCESS' or 'HITRATES' 

  if mem_access_model == 'MEM_ACCESS':
    num_index_vars = nijk*subcells*ng_bndls*N          # number of index variables
    num_float_vars = nijk*subcells*ng_bndls*N*(N + 1)  # number of float variables

    index_loads = nijk*subcells*ng_bndls*0.75*N**3  # all integer loads, ignoring logical
    float_loads = nijk*subcells*ng_bndls*(2.0/3.0)*N**3  # float plus float vector loads

    avg_dist = 8.08             # average distance in arrays between accessed
                                #   elements in bytes; assume stride 1 word with
                                #   a little extra cost

    avg_reuse_dist = 5.5*N          # avg number of unique loads between two
                                #   consecutive accesses of the same element
                                #   (ie use different weights for int and float
                                #   loads); assume about 10 variables between
                                #   reuse
    stdev_reuse_dist = 7.5*N        # stddev number of unique loads between two
                                #   consecutive accesses of the same element;
                                #   assume our reuse guess above is pretty good

    int_alu_ops      = nijk*subcells*ng_bndls*1.25*N**3 # includes logical ops

    float_alu_ops    = nijk*subcells*ng_bndls*(2.0/3.0)*N**3
                                            # cost of LU solve from literature

    float_DIV_ops    = nijk*subcells*ng_bndls*(1.0/2.0)*N**2

    int_vector_ops   = 0

    float_vector_ops = 0

    tasklist_per_chunk = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                           ['fDIV', float_DIV_ops],
                           ['INTVEC', int_vector_ops, 1],
                           ['VECTOR', float_vector_ops, 1],
                           ['MEM_ACCESS', num_index_vars, num_float_vars,
                            avg_dist, avg_reuse_dist, stdev_reuse_dist,
                            index_loads, float_loads, False] ]
    # TODO: put memory in at some point. Probably at the beginning.
  elif  mem_access_model == 'HITRATES':
    num_index_vars = nijk*subcells*ng_bndls*N          # number of index variables
    num_float_vars = nijk*subcells*ng_bndls*N*(N + 1)  # number of float variables

    alpha = (2.0/3.0)*21.0
    beta = (2.0/3.0)*210.0
    gamma = (2.0/3.0)*18330.0

    index_loads = nijk*subcells*ng_bndls*((3.2/3.0)*N**3 + alpha*N**2 + beta*N + gamma)  # all integer loads, ignoring logical
    float_loads = nijk*subcells*ng_bndls*((2.0/3.0)*N**3 + alpha*N**2 + beta*N + gamma)  # float plus float vector loads

    avg_dist = 8.08             # average distance in arrays between accessed
                                #   elements in bytes; assume stride 1 word with
                                #   a little extra cost

#    L1_hitrate = 0.65 + (0.35*math.log(12)**(1./1.7))/math.log(N)**(1./1.7)
#    L1_hitrate = 0.65 + (0.35*12**(1./1))/N**(1./1)
    L1_hitrate = 1.000
    L2_hitrate = 1.000
    L3_hitrate = 1.000

    int_alu_ops      = nijk*subcells*ng_bndls*((3.2/3.0)*N**3 + alpha*N**2 + beta*N + gamma) # includes logical ops

    float_alu_ops    = nijk*subcells*ng_bndls*((2.0/3.0)*N**3 + alpha*N**2 + beta*N + gamma)
                                            # cost of LU solve from literature

    float_DIV_ops    = nijk*subcells*ng_bndls*(1.0/2.0)*(N**2 + N)

    int_vector_ops   = 0

    float_vector_ops = 0

    tasklist_per_chunk = [ ['iALU', int_alu_ops], ['fALU', float_alu_ops],
                           ['fDIV', float_DIV_ops],
                           ['INTVEC', int_vector_ops, 1],
                           ['VECTOR', float_vector_ops, 1],
                           ['HITRATES', L1_hitrate, L2_hitrate, L3_hitrate,
                            num_index_vars, num_float_vars, avg_dist,
                            index_loads, float_loads, False] ]
  else:
    print "Warning: unsupported mem_access_model!"
    sys.exit(1)   

  # Difference in tasklists is the Init_file set to True. In border cells, more
  #   float elements will need to be loaded from main

  # Compute time spent for entire chunk
  print "\ntime_compute for sweeping chunk"
  chunk_time, stats_chunk = core.time_compute(tasklist_per_chunk, True)

  # Return
  return chunk_time

###############################################################################


# cwbj_process
###############################################################################

def cwbj_process(mpi_comm_world): #MR
  """
  Driver for the SNAP nested loop structure. Computes the time for a work
  chunk Each core then starts the inner loop process. The cores know their own
  first work chunk, and uses that to determine what chunks it requires to be
  finished and which chunks depend on it from compute_dependencies. The process
  waits for required chunks, computes (sleeps, really) its current chunk's time,
  then passes a message to the dependent cores/chunks. Add in the time to
  compute the cross-group scatter (outer source). Once the process is laid out
  for a single inner iteration, perform a generic looping over
  the time steps-->outers-->inners, accumulating the appropriate time.

  Input: mpi_comm_world -- communicator from the MPI model
  Output: nothing returned, but total simulation time printed
  """

  n = mpi_comm_size(mpi_comm_world)                     # total # ranks
  p = global_core_id = mpi_comm_rank(mpi_comm_world)    # rank of this process

  # Compute time to sweep a single work chunk
  timestep_i = 0
  host = mpi_ext_host(mpi_comm_world)
  core_id = p % cores_per_host
  core = host.cores[core_id]

  chunk_time = compute_chunk_time(core)

  # Start main loop: Wait for required chunks to be completed, complete own
  #   chunk, send to dependents, determine next chunk on core, repeat.
  ############################

  # Initialize timing and the received chunk dictionary
  time_before_inner = mpi_wtime(mpi_comm_world)
  recvd_chunks = {}

  # Determine starting chunk for the core
  x = global_core_id%nxchunks
  y = int((global_core_id%(nxchunks*nychunks))/nxchunks)
  z = int(global_core_id/(nxchunks*nychunks))
#  print "x, y, z ", x, y, z
  num_chunks_in_core = 0
  cur_chunk_id = (z, y, x)

  # Start the loop over core's chunks. cur_chunk_id is initialized to value
  #   above. Later reset until the core has no more work, where loop ends.
  while cur_chunk_id != None:

    #print "Core ", global_core_id, " with chunk", cur_chunk_id

    # Determine neighbors
    num_chunks_in_core += 1
    neighbors = compute_neighbors(cur_chunk_id)

    #print "current chunk id ", cur_chunk_id, "neighbors", neighbors 

    #print "neighbors:", neighbors

    # We have all neighbors, so we can mimic the computation time--i.e., just
    #   sleep for the time it takes to compute over the work chunk. This model
    #   assumes the cache effects across chunks are minimal, probably a safe
    #   assumption.
    #print p, " now sleeping for chunk ", cur_chunk_id
    mpi_ext_sleep(chunk_time, mpi_comm_world)

    #print "Core ", global_core_id," executed chunk ", cur_chunk_id, \
    #      " at time ", mpi_wtime(mpi_comm_world)

    # Communicate to the neighbors
    for neigh_id in neighbors:

      # Set tuple for sending chunk and destination chunk from neighbors. Use
      #   receiving chunk info to determine destination rank.
      (myz, myy, myx) = cur_chunk_id
      (z, y, x) = neigh_id

      dest_rank = x + nxchunks*(y + nychunks*z) 

      #print "current chunk id ", cur_chunk_id, "neigh id ", neigh_id, "dest rank ", dest_rank

      # Set data size according to grid direction
      if dest_rank == p:
        print "ERROR: cwbj_process: dest_rank == p. Should not be here."
        sys.exit(1)
      else:
        if noct == 4: # 2D
          svertices = 2 # vertices on a cell's side
          if myy == y and myx != x:
            data_size = jchunk*svertices*ng*(total_angles/2)*PSI_SIZE
          elif myy != y and myx == x:
            data_size = ichunk*svertices*ng*(total_angles/2)*PSI_SIZE
          else:
            print "ERROR: cwbj_process: myy!=y and myx!=x. Should not be here."
            sys.exit(1)
        elif noct == 8: # 3D
          Area_v = 0
          if subcells == 1: # Hex
            fvertices_yz = 4; # Cell's left or right faces 
            fvertices_xz = 4; # Cell's front or back faces
            fvertices_xy = 4; # Cell's bottom or top faces
          elif subcells == 2: # Prisms
            fvertices_yz = 4; # Cell's left or right faces 
            fvertices_xz = 6; # Cell's front or back faces
            fvertices_xy = 4; # Cell's bottom or top faces
          elif (subcells == 5 or subcells == 6): # Tets
            fvertices_yz = 6; # Cell's left or right faces 
            fvertices_xz = 6; # Cell's front or back faces
            fvertices_xy = 6; # Cell's bottom or top faces
          elif subcells == 24: # Tets
            fvertices_yz = 12; # Cell's left or right faces 
            fvertices_xz = 12; # Cell's front or back faces
            fvertices_xy = 12; # Cell's bottom or top faces
          else:
            print "Warning: Unknown number of 3D subcells!"
            sys.exit(1)
          if myz == z and myy == y and myx != x:
            Area_yz = jchunk*kchunk # left or right proc interface
            Area_v = Area_yz*fvertices_yz 
          elif myz == z and myy != y and myx == x:
            Area_xz = ichunk*kchunk # front or back proc interface
            Area_v = Area_xz*fvertices_xz
          elif myz != z and myy == y and myx == x:
            Area_xy = ichunk*jchunk # bottom or top proc interface
            Area_v = Area_xy*fvertices_xy
          else:
            print "ERROR: cwbj_process: myz!=z and myy!=y and myx!=x. Should not be here."
            sys.exit(1)
          data_size = Area_v*ng*(total_angles/2)*PSI_SIZE        
        else:
          print "Warning: Supported dimensions are 2 and 3!"
          sys.exit(1) 

      # Send message
      if dest_rank != p:
        mpi_send(dest_rank, cur_chunk_id, data_size, mpi_comm_world)

    # Wait to receive from neighbors. Once all data are received,
    #   clear the recvd_chunks dictionary.
    #   ('set' makes this unordered for comparison purposes)
    while set(neighbors) > set(recvd_chunks):
      r = mpi_recv(mpi_comm_world)
      recvd_chunks[r["data"]] = "rec"
    for neigh_id in neighbors:
      del recvd_chunks[neigh_id]

    cur_chunk_id = None

  # This rank is done, print the finished status
  print "\nCore ", global_core_id, \
        " done at time ", mpi_wtime(mpi_comm_world), \
        " after executing ", num_chunks_in_core, " chunks"

  if num_chunks_in_core != total_chunks_per_rank:
    print "ERROR: cwbj_process: incorrect number of chunks swept."
    sys.exit(1)

  # We synchronize to let all ranks finish inner loops
  mpi_barrier(mpi_comm_world)
  time_after_inner = mpi_wtime(mpi_comm_world)
  time_per_inner_loop = time_after_inner - time_before_inner

  # Compute cross-group scattering (outer source)
#  cells_on_core = nx*jchunk*kchunk
#  cross_compute_tasklist = [ ['fALU', 2*cells_on_core*cmom*ng**2],
#                             ['MEM_ACCESS', 5,
#                              2*cells_on_core*cmom*ng+cells_on_core*cmom*ng**2,
#                              8.0, cells_on_core*cmom*ng**2*8.0, 8.0, 5.0,
#                              2*cells_on_core*cmom*ng+cells_on_core*cmom*ng**2,
#                              True] ]
#  cross_compute_time = core.time_compute(cross_compute_tasklist)
#  mpi_ext_sleep(cross_compute_time, mpi_comm_world)
#  mpi_barrier(mpi_comm_world)
#  time_after_cross = mpi_wtime(mpi_comm_world)
#  time_per_scatter= time_after_cross - time_after_inner

  # Compute the entire duration (only on one rank)
  if p == 0:

    time = 0.0
    for timestep_i in range(nsteps):
      for outer_i in range(oitm):
        for inner_i in range(iitm):
          time += time_per_inner_loop
#        time += time_per_scatter

    # Process 0 has already advanced time by one inner and one scatter, so
    #   we deduct that from time and put mpi to sleep to simulate that time
    time -= time_per_inner_loop # + time_per_scatter
    mpi_ext_sleep(time, mpi_comm_world)

    # Print the results to screen
    print "\nEnd results:"
    print "============"
    print "Total time (sec):", mpi_wtime(mpi_comm_world)
    print "Time per inner loop (sec): ", time_per_inner_loop
#    print "Time for crossgroup scatter (sec): ", time_per_scatter

  # Finalize mpi and the simulation
  mpi_finalize(mpi_comm_world)

###############################################################################


# MAIN
###############################################################################

modeldict = { "model_name"    : "cwbjsim-mpi",
              "sim_time"      : 1000000,
              "use_mpi"       : False,
              #"use_mpi"       : True,
              "mpi_path"      : "/projects/opt/centos7/mpich/3.1.3_gcc-4.8.5/lib/libmpi.so.12.0.4",
#              "node_type"     : "CieloNode", #"MLIntelNode", #"CieloNode",
#              "host_type"     : "mpihost",
#              "intercon_type" : "crossbar",
#              "crossbar"      : { "nhosts" : 50 },
#              "mpiopt"        : { "min_pktsz"     : 2048,
#                                  "max_pktsz"     : 10000,
#                                  "resend_intv"   : 0.01,
#                                  "resend_trials" : 10,
#                                  "call_time"     : 1e-6,
#                                  "max_injection" : 1e9 } }
              "intercon_type" : "Gemini",
              "torus"         : configs.cielo_intercon,
              "host_type"     : "CieloNode",
              "load_libraries": set(["mpi"]),
              "mpiopt"        : configs.gemini_mpiopt,
            }

print "\nCWBJSim run with Simian PDES Engine\nVersion = ", version #MR

# We use __builtin__ to create a truly global variable that the host module
#   can inherit from.
#import __builtin__
#__builtin__.nodeType = "default"
#if "node_type" in modeldict:
#  __builtin__.nodeType = modeldict["node_type"]

cluster = Cluster(modeldict)

total_hosts = cluster.intercon.num_hosts()

cores_per_host = 16
total_cores = total_hosts*cores_per_host

if total_ranks >= total_cores:
  print "ERROR: the cluster doesn't have enough cores (%d*%d=%d) to run this \
         job (p=%d)" % total_hosts, cores_per_host, total_cores, total_ranks
  sys.exit(2)

# each compute node has multiple cores; we try to put an mpi process
# on each core and put neighboring processes on the same node if at
# all possible
hostmap = [(i/cores_per_host)%total_hosts for i in range(total_ranks)]

cluster.start_mpi(hostmap, cwbj_process)
cluster.run()

###############################################################################
###############################################################################
