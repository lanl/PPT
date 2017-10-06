# MPI configuration for Blue Gene/Q network

# From "IBM Blue Gene/Q Interconnection Fabric"
# by Chen, Dong and Eisley, Noel and Heidelberger, Philip and Senger, 
# Robert M and Sugawara, Yutaka and Kumar, Sameer and Salapura, Valentina and Satterfield, David L.

# Data packets on BG/Q include a 32-byte header, a 0- to 512-byte data payload in
# increments of 32 bytes, and an 8-byte trailer for link-level packet checks.


# From "The IBM Blue Gene/Q interconnection network and message unit"
# The data portion of packet is from 0 to 512B, in increments of 32B chunks
# Data packets on BG/Q include a 32 byte header; 12B for the network and 20B for the MU
bluegeneq_mpiopt = {
    "min_pktsz" : 0,
    "max_pktsz" : 32,
    "data_overhead" : 32,
}



