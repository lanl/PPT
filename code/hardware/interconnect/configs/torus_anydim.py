# torus_anydim.py :- create torus of arbitrary dimensions

def gemini_anydim(dimx, dimy, dimz):
    return {
        "dimx" : dimx,
        "dimy" : dimy,
        "dimz" : dimz,
    }

def torus_anydim(dims, dimh=2, dups=None):
    if dups is None: dups = (1,)*len(dims)
    if len(dims) != len(dups):
        raise Exception("invalid dims=%r or dups=%r" % (dims, dups))
    return {
        "dims" : dims, # this is a tuple
        "attached_hosts_per_switch" : dimh,
        "dups" : dups, # this is also a tuple of the same dimension
    }
