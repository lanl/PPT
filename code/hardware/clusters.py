# Copyright (c) 2014. Los Alamos National Security, LLC. 

# This material was produced under U.S. Government contract DE-AC52-06NA25396
# for Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
# National Security, LLC for the U.S. Department of Energy. The U.S. Government 
# has rights to use, reproduce, and distribute this software.  

# NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
# WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
# SOFTWARE.

# If software is modified to produce derivative works, such modified software
# should be clearly marked, so as not to confuse it with the version available
# from LANL.

# Additionally, this library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License v 2.1 as
# published by the Free Software Foundation. Accordingly, this library is
# distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE.txt for more details.

"""
*********** Performance Prediction Toolkit PPT *********

File: clusters.py
Description: main library of cluster definitions.
"""

# MR: add begin
from sys import path
path.append('../..')
from ppt import *
# MR: add end

import nodes


class Cluster(object):
    """
    Class that represents a supercomputer cluster. It consists of a number of
    nodes. This is a dummy cluster and nodes.Node does not actually exist.
    """
    def __init__(self, simianEngine):
        self.num_nodes = 100 # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.Node, i, {})
            simianEngine.addEntity("Node", nodes.Node, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add


class Mustang(object):
    """
    Class that represents a supercomputer cluster: LANL's Mustang.
    """
    def __init__(self, simianEngine, n):
        self.num_nodes = n # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.AMDOpteron, i, {})
            simianEngine.addEntity("Node", nodes.AMDOpteron, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = Mustang\n"


class HalfTrinity(object):
    """
    Class that represents a supercomputer cluster: LANL's Trinity.
    This is supposed to represent a user-specified number of Trinity Phase 2
    KNL nodes.
    """
    def __init__(self, simianEngine, n):
        self.num_nodes = n # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.KNLNode, i, {})
            simianEngine.addEntity("Node", nodes.KNLNode, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = HalfTrinity\n"


class MiniTrinity(object):
    """
    Class that represents a supercomputer cluster: LANL's Trinity.
    This is supposed to represent a number of Trinity Phase 2 KNL nodes.
    """
    def __init__(self, simianEngine):
        self.num_nodes = 50 # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.KNLNode, i, {})
            simianEngine.addEntity("Node", nodes.KNLNode, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = MiniTrinity\n"


class SingleCielo(object):
    """
    Class that represents a supercomputer cluster: LANL's Cielo.
    """
    def __init__(self, simianEngine):
        self.num_nodes = 50 # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.CieloNode, i, {})
            simianEngine.addEntity("Node", nodes.CieloNode, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = SingleCielo\n"


class Moonlight(object):
    """
    Class that represents a supercomputer cluster: LANL's Moonlight.
    """
    def __init__(self, simianEngine):
        self.num_nodes = 50 # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.MLIntelNode, i, {}) # MR: comment out
            simianEngine.addEntity("Node", nodes.MLIntelNode, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = Moonlight\n"


class SingleMBP(object):
    """
    Class that represents a single MacBookPro.
    """
    def __init__(self, simianEngine):
        self.num_nodes = 1 # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.MBPNode, i, {})
            simianEngine.addEntity("Node", nodes.MBPNode, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = SingleMBP\n"


class Titan(object):
    """
    Class that represents a supercomputer cluster: ORNL's Titan.
    """
    def __init__(self, simianEngine):
        self.num_nodes = 5 #18688 # number of compute nodes
        # MR: add begin
        modeldict = { "model_name"    : "n01",
                      "sim_time"      : 1000000,
                      "use_mpi"       : False,
                      "intercon_type" : "Bypass",
                      "torus"         : configs.cielo_intercon,
                      "host_type"     : "CieloNode",
                      "load_libraries": set(["mpi"]),
                      "mpiopt"        : configs.gemini_mpiopt,
                      "debug_options" : []
                    }
        # MR: add end
        for i in xrange(self.num_nodes):
            #simianEngine.addEntity("Node", nodes.TTNNode, i, {})
            simianEngine.addEntity("Node", nodes.TTNNode, i, modeldict, 1,1,1,1,1,1,1,1,1,1) # MR: add
        print "Cluster model = Titan\n"


