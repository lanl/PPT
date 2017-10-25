
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

#
# node.py :- a node can be a switch node or a compute node
#

from pickle import loads
from simian import *

class Node(Entity):
    """Represents an entity equipped with network interfaces."""

    # local variables:
    #   hpcsim_dict: model parameters as a dictionary
    #   node_id: unique id (switches and hosts have different set of identifiers)
    #   interfaces: map from string name to interface object
    #   fast_queue: used for fast message delivery (for now)
    #   arrival_semaphore: used for waking up receiver process

    def __init__(self, baseinfo, hpcsim_dict):
        """Initializes the compute node."""

        super(Node, self).__init__(baseinfo)
        # hosts and switches have separate set of node ids
        self.hpcsim_dict = hpcsim_dict
        self.node_id = baseinfo["num"]

        # interfaces is a map from interface name to the object; the
        # derived class must fill it in later
        self.interfaces = dict()

        # received prioritized messages are put in fast_queue; they
        # are handled first
        import interconnect
        self.fast_queue = interconnect.Inport(None, 0, self)

        # derived class must have a process named "packet_receiver";
        # if the semaphore is zero, upon a packet arrival, the
        # "packet_receiver" process needs to wake up
        self.arrival_semaphore = 0

    def __str__(self):
        """Returns the string name of this compute node."""
        return "%s[%d]" % (self.__class__.__name__.lower(), self.node_id)

    def get_now(self):
        """Conveniently returns the current simulation time."""
        return self.engine.now

    def handle_packet_arrival(self, *args):
        """A service handler to handle packet arrivals.

        A service handler is called when a packet arrives from another
        node: the packet needs to be inserted to the corresponding
        input port buffer and a signal sent if necessary to wake up
        the "packet receiver" process, which is waiting for arrivals
        """

        pkt = loads(args[0])
        pkt.add_to_path(str(self))

        if pkt.is_prioritized():
            if "switch" in self.hpcsim_dict["debug_options"]:
                print("%f: %s handle_packet_arrival puts %s into fast queue" %
                      (self.get_now(), self, pkt))
            self.fast_queue.enqueue(pkt)
        else:
            i, p = pkt.get_nexthop()
            if "switch" in self.hpcsim_dict["debug_options"]:
                print("%f: %s handle_packet_arrival puts %s into iface(%s,%d)" %
                      (self.get_now(), self, pkt, i, p))
            self.interfaces[i].deposit_pkt(p, pkt)

        # wake up the process who'd be waiting for processing packet arrivals
        #print("arrival_semaphore: %d" % self.arrival_semaphore)
        self.arrival_semaphore += 1
        if self.arrival_semaphore == 1:
            self.wakeProcess("packet_receiver")
