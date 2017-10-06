# MPI configuration for Cray's Gemini network

#
# From "Using the Cray Gemini Performance Counters", by Kevin
# Pedretti, Courtenay Vaughan, Richard Barrett, Karen Devine, K. Scott
# Hemmert. Sandia National Labs, Tech Report, SAND2013-0216C. 

#
# In Cray's Gemini, a large MPI message will get broken down into many
# individual 64 byte transactions.  A typical PUT transaction that
# sends 64 bytes of data from a source to a destination consists of a
# 32 phit request packet (96 bytes) followed by a 3 phit response
# packet (9 bytes) from destination to source. The total traffic on
# the network is 96 + 9 = 105 bytes. A typical GET transaction
# requesting 64 bytes of data from a remote node consists of a 8 phit
# request packet (24 bytes) followed by a 27 phit response packet (81
# bytes). Total traffic on the network is 24 + 81 = 105 bytes, which
# is the same as for PUT transactions.  
#
# A phit is 24 bits (3 bytes).
#
# From "Gemini System Interconnect", by Bob Alverson, Duncan
# Roweth, Larry Kaplan, in Hot Interconnects. 
#
# User data injection rate can sustain greater than 6 GB/s.
#

gemini_mpiopt = { 
    "min_pktsz" : 0,
    "max_pktsz" : 64,
    "put_data_overhead" : 32,
    "put_ack_overhead" : 9,
    "get_data_overhead" : 17,
    "get_ack_overhead" : 24,
    #"max_injection" : 1e11,
    "max_injection" : 6e9,
    #"resend_intv" : 1e-3,
    "resend_intv" : 0.1,
}
