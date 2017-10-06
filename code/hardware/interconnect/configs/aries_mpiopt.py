# MPI configuration for Cray's Aries network

# From "Cray XC series network" by Bob Alverson, Edwin Froese, 
# Larry Kaplan and Duncan Roweth. Cray Inc., White Paper WP-Aries01-1112.

# In Cray XC, remote references are performed as gets/puts and atomic memory operations.
# A put causes data to flow directly across the network to target node. When node issues
# commands, NIc packetizes these requests and issues the packets to the network. When
# packet reaches its destination, the destination node returns a response to the source.
# Packet contains up to 64 bytes of data.

# Each 64-byte write (put) requires 14 request phits and 1 response phit.
# Each 64-byte read (get) requires three request phit and 12 response phits.

# A phit is 24 bits (3 bytes)

# The Aries NIC can perform a 64-byte read or write every five cycles (10.2GB/s at
# 800MHz). This number represents the peak injection rate achieveable by user processes. 

# Aries offers sophisticated reliability, availability, and serviceability (RAS)
# capabilities with comprehensive hardware error correction coverage, resulting in
# "fewer" retransmissions.
# The Link Control Block (LCB) inside router provides automatic retransmission
# in the event of error.

# Assumption: only put transactions considered

aries_mpiopt = {
    "min_pktsz" : 0,
    "max_pktsz" : 64,
    "put_data_overhead" : 42, # 14 phits (14X3 = 42 bytes)
    "put_ack_overhead" : 3, # 1 phit (1X3 = 3 bytes)
    "get_data_overhead" : 36, # 12 phits (12X3 = 36 bytes)
    "get_ack_overhead" : 9, # 3 phit3 (3X3 = 9 bytes)
}



