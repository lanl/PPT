import os,sys
pptpath = os.path.dirname(__file__)
#sys.path.append('%s/simian/simian-master/SimianPie.MPI4Py'%pptpath)
sys.path.append('%s/simian/simian-master/SimianPie'%pptpath)
sys.path.append('%s/hardware'%pptpath)
sys.path.append('%s/middleware/mpi'%pptpath)
sys.path.append('%s/middleware/threading'%pptpath)
from cluster import *
from mpi import *
