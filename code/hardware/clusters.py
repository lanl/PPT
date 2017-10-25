# Copyright (c) 2017, Los Alamos National Security, LLC
# All rights reserved.
# Copyright 2017. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
# Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
