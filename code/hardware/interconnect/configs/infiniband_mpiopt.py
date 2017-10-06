# MPI configuration for Infiniband network

# From "MVAPICH2-MIC: A High Performance MPI Library for Xeon Phi Clusters with Infiniband"
# by Potluri, Sreeram and Hamidouche, Khaled and Bureddy, Devendar and Panda, Dhabaleswar K.

# MVAPICH2 uses a chunk size of 8KByte for most host platform

infiniband_mpiopt = {
    "min_pktsz" : 0,
    "max_pktsz" : 8*1024,
}



