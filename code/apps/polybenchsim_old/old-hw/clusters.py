# Copyright (c) 2014. Los Alamos National Security, LLC. 

# This material was produced under U.S. Government contract DE-AC52-06NA25396
# for Los Alamos National Laboratory (LANL), which is operated by Los Alamos 
# National Security, LLC for the U.S. Department of Energy. The U.S. Government 
# has rights to use, reproduce, and distribute this software.  

# NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, 
# EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  
# If software is modified to produce derivative works, such modified software should
# be clearly marked, so as not to confuse it with the version available from LANL.

# Additionally, this library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License v 2.1 as published by the 
# Free Software Foundation. Accordingly, this library is distributed in the hope that 
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE.txt for more details.

"""
*********** Performance Prediction Toolkit PPT *********

File: clusters.py
Description: main library of cluster definitions.

Comments:
 2015-11-12: included into repository
"""




import nodes

class Cluster(object):
    """
    Class that represents a supercomputer cluster. It consists of a number of nodes
    """
    def __init__(self, simianEngine, num_nodes=100):
        self.num_nodes = num_nodes # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.Node, i)


class HalfTrinity(object):
    """
    Class that represents a supercomputer scluster. It consists of a number of nodes
    """
    def __init__(self, simianEngine, num_nodes):
        self.num_nodes = num_nodes # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.KNLNode, i)
        print "Cluster model = HalfTrinity\n"
            
class MiniTrinity(object):
    """
    Class that represents a supercomputer scluster. It consists of a number of nodes
    """
    def __init__(self, simianEngine, num_nodes=50):
        self.num_nodes = num_nodes # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.KNLNode, i)

class SingleCielo(object):
    """
    Class that represents a supercomputer scluster. It consists of a number of nodes
    """
    def __init__(self, simianEngine):
        self.num_nodes = 1 # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.CieloNode, i)

class Cielo(object):
    """
    Class that represents a supercomputer scluster. It consists of a number of nodes
    """
    def __init__(self, simianEngine, num_nodes=50):
        self.num_nodes = num_nodes # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.CieloNode, i)


class SingleMBP(object):
    """
    Class that represents a single MBP, my own machine, MacBookPro
    """
    def __init__(self, simianEngine):
        self.num_nodes = 1 # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.MBPNode, i)
                        
class Titan(object):
    """
    Class that represents a supercomputer cluster. It consists of a number of nodes
    """
    def __init__(self, simianEngine, num_nodes=5):
        self.num_nodes = num_nodes #18688 # number of compute nodes
        for i in xrange(self.num_nodes):
            simianEngine.addEntity("Node", nodes.TTNNode, i)
