#
# intercon.py :- basic classes for an interconnection network
#

from collections import deque
from pickle import dumps
from node import *

class Interconnect(object):
    """Base class for all interconnect models."""

    # local variables:
    #   nswitches: number of switch nodes of the interconnection network
    #   nhosts: number of compute nodes connected by the interconnection network

    def __init__(self, hpcsim_dict):
        self.hpcsim_dict = hpcsim_dict
        self.nswitches = 0
        self.nhosts = 0

    def num_switchs(self):
        return self.nswitches

    def num_hosts(self):
        return self.nhosts

    def network_diameter(self):
        """Returns the network diameter in hops (for setting time-to-live)."""
        raise Exception("derived class must override this method")

    def network_diameter_time(self):
        """Returns the network diameter in time (only the propagation delay)."""
        raise Exception("derived class must override this method")

    @staticmethod
    def calc_min_delay(hpcsim_dict):
        """Calculates the min delay between hosts and switches.

        The min delay value is calculated according to the
        interconnect type and configuration. The function is static
        and is expected to be called before the interconnect model is
        instantiated. That is, the min delay is derived only from the
        model parameters (hpcsim_dict).
        """
        raise Exception("derived interconnect model must implement this")


class Packet(object):
    """Base class for messages sent between nodes."""

    # local variables:
    #   srchost: id of the source host
    #   dsthost: id of the destination host
    #   type: message type (data or ack)
    #   seqno: every message from a source to a destination has its sequence number
    #   msglen: length of the message in bytes
    #   ttl: time to live
    #   prioritized: message priority is either set or unset
    #   return_data: the data to be returned verbatim by ack
    #   nonreturn_data: data transferred in only one way, not ack
    #   sendtime: recorded for measuring end-to-end delay
    #   path: a sequence of nodes traversed by the packet if blaze_trail is set
    #   nexthop_name: next hop interface name
    #   nexthop_id: next hop interface port number

    def __init__(self, from_host, to_host, type, seqno, msglen, 
                 return_data=None, nonreturn_data=None,
                 ttl=-1, prio=False, blaze_trail=True):
        # ttl should be default to the diameter of the network, but
        # since we don't really have the information here we default
        # it to be infinite (represented by -1)
        self.srchost = from_host
        self.dsthost = to_host
        self.type = type
        self.seqno = seqno
        self.msglen = msglen
        self.ttl = ttl
        self.prioritized = prio
        self.return_data = return_data
        self.nonreturn_data = nonreturn_data
        self.sendtime = 0 # will set upon send
        self.path = [] if blaze_trail else None

    def __str__(self):
        return "%s[src=%d, dst=%d, seqno=%d, sz=%d, ttl=%d%s]" % \
            (self.type.upper(), self.srchost, self.dsthost, self.seqno,
             self.msglen, self.ttl, ' [*]' if self.prioritized else '')

    def set_nexthop(self, nxtname, nxtid):
        """Sets next hop, including the interface name and port number."""
        
        self.nexthop_name = nxtname
        self.nexthop_id = nxtid

    def get_nexthop(self):
        """Gets the next hop as a tuple (interface/port)."""
        return self.nexthop_name, self.nexthop_id

    def size(self): 
        """Returns the packet size in bytes."""
        return self.msglen

    def is_prioritized(self, thresh=0):
        """Is the message prioritized?

        A message is prioritized if it's been explicitly set so or the
        size of the message is no larger than the given threshold.
        """
        return self.prioritized or self.msglen<=thresh

    # a measurement of end-to-end delay
    def set_sendtime(self, t): self.sendtime = t
    def get_sendtime(self): return self.sendtime

    # record the path traversed through the network
    def add_to_path(self, rid): 
        if self.path is not None: 
            self.path.append(rid)
    def get_path(self): 
        return self.path


class Outport(object):
    """outgoing portal of a network interface"""

    # local variables:
    #   iface: the network interface this portal belongs to (or None if it's memory queue)
    #   node: node of this output portal
    #   port: the port number
    #   peer_node_name: name of the node this port connects to (or None if it's memory queue)
    #   peer_node_id: id of the node this port connects to (or None if it's memory queue)
    #   peer_iface_name: name of the interface this port connects to
    #   peer_iface_port: port number of the interface this port connects to
    #   bdw: link bandwidth (in bits per second)
    #   max_delay: max queuing delay to send a message (when buffer is full)
    #   link_delay: link propagation delay (in seconds)
    #   last_sent_time: time to complete sending of the previous message (in seconds)
    #   stats: statistics kept in a dictionary, including:
    #          "sent_types", "sent_pkts", "dropped_bytes", "dropped_pkts"

    def __init__(self, iface, port, node, peer_node_name, peer_node_id, 
                 peer_iface_name, peer_iface_port, bdw, bufsz, link_delay):
        self.iface = iface # None for memory queue
        self.node = node
        self.port = port
        self.peer_node_name = peer_node_name
        self.peer_node_id = peer_node_id
        #print("node=%r, port=%r, peer-node=%r peer-id=%r" % (node, port, peer_node_name, peer_node_id))
        self.peer_iface_name = peer_iface_name
        self.peer_iface_port = peer_iface_port
        self.bdw = bdw
        self.max_delay = bufsz*8/bdw
        self.link_delay = link_delay
        self.last_sent_time = 0
        self.stats = dict()
        self.stats["sent_bytes"] = 0
        self.stats["sent_pkts"] = 0
        self.stats["dropped_bytes"] = 0
        self.stats["dropped_pkts"] = 0

    def get_qlen_in_bits(self):
        """Returns the current queue length in bits.

        Since we don't really model this queue by buffering the
        packets, the information about the number of packets currently
        in queue is lost. We can recover this information using the
        queuing delay and the link bandwidth.
        """
        return self.get_qdelay()*self.bdw

    def get_qdelay(self):
        """Returns the instantaneous queuing delay."""
        now = self.node.get_now()
        if self.last_sent_time <= now:
            return 0
        else:
            return self.last_sent_time-now

    def send_pkt(self, pkt):
        """Sends a packet; drops it if buffer's overflown."""
        
        xmit_delay = pkt.size()*8/self.bdw
        flush_time = self.get_qdelay()+xmit_delay
        #print("x=%.9f q=%.9f l=%.9f" % (xmit_delay, self.get_qdelay(), self.link_delay))
        if flush_time > self.max_delay:
            # packet dropped due to buffer overflow
            if "interface" in self.node.hpcsim_dict["debug_options"]:
                print("%f: %s iface(%s) outport %d drops %s" % 
                      (self.node.get_now(), self.node,
                       self.iface.name if self.iface is not None else "memory",
                       self.port, pkt))
            self.stats["dropped_bytes"] += pkt.size()
            self.stats["dropped_pkts"] += 1
        else:
            current = self.node.get_now()
            # schedule arrival of the packet at destination
            self.last_sent_time = current+flush_time
            if "interface" in self.node.hpcsim_dict["debug_options"]:
                print("%f: %s iface(%s) outport %d transmits %s until %0.9f qlen=%d (bits)" % 
                      (current, self.node,
                       self.iface.name if self.iface is not None else "memory",
                       self.port, pkt, self.last_sent_time, self.get_qlen_in_bits()))

            if self.peer_node_id is None or self.peer_node_id >= 0:
                pkt.set_nexthop(self.peer_iface_name, self.peer_iface_port)
                self.node.reqService(flush_time+self.link_delay, "handle_packet_arrival", dumps(pkt), 
                                 self.peer_node_name, self.peer_node_id)
            else:
                # this is a hack for bypassing the interconnect
                pkt.set_nexthop(self.peer_iface_name, self.peer_iface_port)
                self.node.reqService(flush_time+self.link_delay, "handle_packet_arrival", dumps(pkt), 
                                     "Host", pkt.dsthost) # directly!!
            self.stats["sent_bytes"] += pkt.size()
            self.stats["sent_pkts"] += 1
            '''
            # statistics for number of hops for each packet
            #TODO: couting # of hops for data packet at the moment
            if pkt.type[:4] == 'data':
                pkt.num_hops += 1
                print pkt.num_hops
                if "num_hops" not in pkt.nonreturn_data:
                    pkt.nonreturn_data["num_hops"] = 1
                else:
                    pkt.nonreturn_data["num_hops"] += 1
            '''


class Inport(object):
    """incoming portal of a network interface"""

    # local variables:
    #   iface: owner network interface (or None if fast queue)
    #   port: the port number
    #   node: node of this input portal
    #   qlen: current queue length (in bytes)
    #   queue: input queue implemented as a deque
    #   stats: statistics, including "rcvd_bytes", "rcvd_pkts"

    def __init__(self, iface, port, node):
        self.iface = iface # possibly None if this inport is fast_queue
        self.port = port
        self.node = node
        self.qlen = 0 # in bytes
        self.queue = deque()
        self.stats = dict()
        self.stats["rcvd_bytes"] = 0
        self.stats["rcvd_pkts"] = 0

    def enqueue(self, pkt):
        """Enqueues a packet."""

        self.qlen += pkt.size()
        self.queue.append(pkt)
        self.stats["rcvd_bytes"] += pkt.size()
        self.stats["rcvd_pkts"] += 1
        if "interface" in self.node.hpcsim_dict["debug_options"]:
            print ("%f: %s iface(%s) inport %d enque %s (qlen=%d bytes, %d pkts)" % 
                   (self.node.get_now(), self.node, 
                    self.iface.name if self.iface is not None else "fast",
                    self.port, pkt, self.qlen, len(self.queue)))

    def dequeue(self):
        """Returns the packet at the front of the queue."""

        if len(self.queue) > 0:
            pkt = self.queue.popleft()
            self.qlen -= pkt.size()
            if "interface" in self.node.hpcsim_dict["debug_options"]:
                print ("%f: %s iface(%s) inport %d deque %s (qlen=%d bytes, %d pkts)" %
                       (self.node.get_now(), self.node, 
                        self.iface.name if self.iface is not None else "fast",
                        self.port, pkt, self.qlen, len(self.queue)))
            return pkt
        else:
            return None

    def get_qlen_in_bytes(self): 
        return self.qlen

    def get_qlen_in_packets(self):
        return len(self.queue)

    def is_empty(self):
        return len(self.queue) == 0


class Interface(object):
    """network interface has ports for sending and receiving packets."""

    # local variables:
    #   node: parrent node of the network interface
    #   name: name of the interface
    #   nports: number of (duplex) ports of the interface
    #   inports: list of input ports
    #   outports: list of output ports

    def __init__(self, node, name, nports, peer_node_names, peer_node_ids, 
                 peer_iface_names, peer_iface_ports, bdw, bufsz, link_delay):
        self.node = node # parent node (either a switch or a host)
        self.name = name
        self.nports = nports
        self.inports = []
        self.outports = []
        for p in xrange(nports):
            self.inports.append(Inport(self, p, self.node))
            self.outports.append(Outport(self, p, self.node, peer_node_names[p], peer_node_ids[p], 
                                         peer_iface_names[p], peer_iface_ports[p], 
                                         bdw, bufsz, link_delay))

    def __str__(self):
        return "iface(%s)" % self.name

    def get_num_ports(self):
        """Returns the number of ports."""
        return self.nports

    def get_send_qlen_in_bytes(self, port):
        """Returns the queue length of the given output port in bytes."""
        return self.outports[port].get_qlen_in_bits()*8

    def get_min_qdelay(self):
        """Returns the min queuing delay among all output ports.

        The function returns a tuple, including both the min delay
        needed to send a new packet and the output port that has the
        min delay
        """

        # mop is min output port, m is the min queue delay
        mop = self.outports[0]; m = mop.get_qdelay()
        for op in self.outports[1:]:
            d = op.get_qdelay() 
            if d < m: m = d; mop = op
        return m, mop

    def send_pkt(self, pkt, port):
        """Sends a packet from a port.
        
        If port is non-negative, the function sends the packet out
        from the given port. If port is negative, the function sends
        the packet out from a port with the minimal delay.
        """

        # at this point, the packet must not have the same source and
        # destination host (loopback should have been taken care of)
        assert pkt.dsthost != pkt.srchost
        if port < 0:
            m, mop = self.get_min_qdelay()
            mop.send_pkt(pkt)
        else:
            self.outports[port].send_pkt(pkt)
        return 0 # for now, we don't give processing time

    def drop_pkt(self, pkt, port):
        """Records a packet drop at a port (due to ttl)."""
        if port < 0: m, port = self.get_min_qdelay()
        self.outports[port].stats["dropped_pkts"] += 1
        self.outports[port].stats["dropped_bytes"] += pkt.size()

    def is_recv_empty(self, port):
        """Checks whether the recv queue is empty at given port."""
        return self.inports[port].is_empty()

    def recv_pkt(self, port):
        """Receives a packet from a given port."""
        return self.inports[port].dequeue()

    def deposit_pkt(self, port, pkt):
        """Puts a received packet into the input buffer of given port."""
        self.inports[port].enqueue(pkt)

    def get_now(self):
        """Returns current time of the host (for convenience)."""
        return self.node.get_now()

    # get statistics
    def stats_sent_pkts(self, port):
        return self.outports[port].stats["sent_pkts"]
    def stats_total_sent_pkts(self):
        sum = 0
        for p in xrange(self.nports):
            sum = sum + self.outports[p].stats["sent_pkts"]
        return sum
    def stats_sent_bytes(self, port):
        return self.outports[port].stats["sent_bytes"]
    def stats_total_sent_bytes(self):
        sum = 0
        for p in xrange(self.nports):
            sum = sum + self.outports[p].stats["sent_bytes"]
        return sum
    def stats_dropped_pkts(self, port):
        return self.outports[port].stats["dropped_pkts"]
    def stats_total_dropped_pkts(self):
        sum = 0
        for p in xrange(self.nports):
            sum = sum + self.outports[p].stats["dropped_pkts"]
        return sum
    def stats_dropped_bytes(self, port):
        return self.outports[port].stats["dropped_bytes"]
    def stats_total_dropped_bytes(self):
        sum = 0
        for p in xrange(self.nports):
            sum = sum + self.outports[p].stats["dropped_bytes"]
        return sum
    def stats_rcvd_pkts(self, port):
        return self.inports[port].stats["rcvd_pkts"]
    def stats_total_rcvd_pkts(self):
        sum = 0
        for p in xrange(self.nports):
            sum = sum + self.inports[p].stats["rcvd_pkts"]
        return sum
    def stats_rcvd_bytes(self, port):
        return self.inports[port].stats["rcvd_bytes"]
    def stats_total_rcvd_bytes(self):
        sum = 0
        for p in xrange(self.nports):
            sum = sum + self.inports[p].stats["rcvd_bytes"]
        return sum


class Switch(Node):
    """Base class for an interconnect switch/router."""

    # local variables: (class derived from Node)
    #   proc_delay: processing delay in seconds

    def __init__(self, baseinfo, hpcsim_dict, proc_delay, *args):
        super(Switch, self).__init__(baseinfo, hpcsim_dict)
        self.proc_delay = proc_delay

        # the process is responsible for conducting traffic
        self.createProcess("packet_receiver", routing_process)
        self.startProcess("packet_receiver")

    # this should have been taken care of in the derived class
    #def __str__(self):
    #    return "switch[%d]" % self.node_id

    def calc_route(self, pkt):
        """Calculates the route and returns the next hop.

        The next hop is a tuple (interface name, parallel port
        number). This function is abstract here; the derived class is
        responsible for implementing this function.
        """
        raise Exception("default switch doesn't know how to route")

    def forward_packet(self, proc, pkt):
        """Forwards a packet for the routing process."""

        iface, port = self.calc_route(pkt)

        # handle packet time-to-live when we forward the packet
        # (moving from input queue to the output queue); note if ttl
        # is already negative or zero before this function is called,
        # it's considered as infinite
        pkt.ttl -= 1
        if pkt.ttl == 0: 
            if "switch" in self.hpcsim_dict["debug_options"]:
                print("%f: %s drops %s (ttl)" % 
                      (self.get_now(), self, pkt))
                self.interfaces[iface].drop_pkt(pkt, port)
        else:
            if "switch" in self.hpcsim_dict["debug_options"]:
                print("%f: %s forwards %s to iface(%s) %s" % 
                      (self.get_now(), self, pkt, iface, 
                       ("port %d"%port if port>=0 else "")))
            proc_time = self.interfaces[iface].send_pkt(pkt, port)

            # if nodal processing time is there, the process will
            # sleep for the set amount of time
            if proc_time > 0: proc.sleep(proc_time)

def routing_process(self):
    """A switch's process for conducting traffic.

    Note that this is a regular function.
    """

    # self is process; sw is the process' entity
    sw = self.entity

    # in very beginning, no packet arrival, the process hibernates
    assert sw.arrival_semaphore == 0
    self.hibernate()

    # list of interace/port pairs for checking; do this once, then we
    # shuffle it in the loop
    if len(sw.interfaces) == 0:
        raise Exception("zero interface switch %d not allowed" % self.node_id)
    keyportlist = []
    for key, iface in sw.interfaces.iteritems():
        for p in range(iface.get_num_ports()):
            keyportlist.append((key, p))
    nxtchk = keyportlist[0]

    while True:
        # construct the new list of interface/port pairs to be checked
        # for this round
        idx = keyportlist.index(nxtchk)
        kpl = keyportlist[idx:]+keyportlist[:idx]

        for idx in range(len(kpl)):
            # some miniscule nodal processing time may elapse here
            if sw.proc_delay > 0:
                self.sleep(sw.proc_delay)

            # always check the fast queue first
            if not sw.fast_queue.is_empty():
                pkt = sw.fast_queue.dequeue()
                if "switch" in sw.hpcsim_dict["debug_options"]:
                    print("%f: %s routing_process found %s in fast queue" %
                          (sw.get_now(), sw, pkt))
                break
            
            # check whether the next interface port has a packet
            (k,p) = kpl[idx]
            if not sw.interfaces[k].is_recv_empty(p):
                pkt = sw.interfaces[k].recv_pkt(p)
                nxtchk = kpl[(idx+1)%len(kpl)]
                if "switch" in sw.hpcsim_dict["debug_options"]:
                    print("%f: %s routing_process found %s in iface(%s,%d)" %
                          (sw.get_now(), sw, pkt, k, p))
                break
        else: pkt = None
        
        # if we exhausted all input ports, time to go to sleep;
        # otherwise, forward the packet, which would cost a
        # transmission time
        if pkt is None:
            if "switch" in sw.hpcsim_dict["debug_options"]:
                print("%f: %s routing_process sleeps" % (sw.get_now(), sw))
            assert sw.arrival_semaphore == 0
            self.hibernate()
        else:
            sw.arrival_semaphore -= 1
            assert sw.arrival_semaphore >= 0
            sw.forward_packet(self, pkt)


class Host(Node):
    """Base class of a compute node attached to interconnect."""

    # local variables: (class derived from Node)
    #   intercon: the interconnection network
    #   mem_queue: for sending messages on the same host

    def __init__(self, baseinfo, hpcsim_dict, *args):
        # list arguments (args) consist of:
        #   self.intercon: the interconnect object
        #   swid: id of the switch connected with this host
        #   swiface: interface name of the switch connected with this host
        #   swport: port number of the switch connected with this host
        #   bdw: bandwidth in bits per seconds
        #   bufsz: the (send) buffer size in bytes (packets may be dropped due to buffer overflow)
        #   dly: link propagation delay
        #   mthru: memory throughput
        #   mbfsz: memory buffer size
        #   mdly: memory access latency
        self.intercon, swid, swiface, swport, bdw, bufsz, dly, mthru, mbfsz, mdly = args
        super(Host, self).__init__(baseinfo, hpcsim_dict)

        # configure the intefaces; we have the following assumptions:
        # 0) host has only one interface, named 'r', and only one port
        # 1) switch's entity name is "Switch"; it's id is given as 'swid'
        # 2) switch's network interface connecting to the hosts has name 'swiface' and port 'swport'
        self.interfaces['r'] = Interface(self, "r",  1, ("Switch",), (swid,),
                                         (swiface,), (swport,), bdw, bufsz, dly)
        if "host.connect" in hpcsim_dict["debug_options"]:
            print("%s %s[0] connects to switch[%d] iface(%s)[%d]" % 
                  (self, self.interfaces['r'], swid, swiface, swport))

        # configure memory bypass
        self.mem_queue = Outport(None, 0, self, None, None, #peer_node_id=None; peer_node_id=None
                                 "r", 0, mthru, mbfsz, mdly)

        # the process is responsible for receiving packets
        self.createProcess("packet_receiver", receive_process)
        self.startProcess("packet_receiver")

    # taken care of by the base class
    #def __str__(self):
    #    return "host[%d]" % self.node_id

    def notify_data_recv(self, pkt):
        """Callback method when data is received.

        It is expected that the derived class override this function
        to handle more delicate situations. Here we smply return a raw
        ACK packet.
        """
        
        # send back ack regardless of the sequence
        ack = Packet(self.node_id, pkt.srchost, "ack_raw", pkt.seqno, 0, 
                     return_data=pkt.return_data,
                     ttl=self.intercon.network_diameter(), 
                     prio=True, # ack is prioritized
                     blaze_trail = pkt.get_path()) # do the same about path
        ack.set_sendtime(self.get_now())
        ack.add_to_path(str(self))

        # check whether the source and destination hosts are the same
        if self.node_id == ack.dsthost:
            self.mem_queue.send_pkt(ack)
        else:
            self.interfaces['r'].send_pkt(ack, 0)

    def notify_ack_recv(self, ack):
        """Callback method when ack is received.

        It is expected that the derived class shall override this
        function to handle more delicate situations. Here we do
        nothing (a notice of the receive has been printed).
        """
        pass

        #if "host" in self.hpcsim_dict["debug_options"]:
        #    print("%f: %s receives %s, delay=%f" % 
        #          (self.get_now(), self, ack, self.get_now()-ack.get_sendtime()))
        #    path = ack.get_path()
        #    if path is not None: 
        #        for h in path: print("  =>%s" % h)

    def test_raw_xfer(self, *args):
        """A test service for sending messages.

        This is a service handler function. It is used by the
        sched_raw_xfer method in the Cluster class.
        """
        
        myargs = args[0]
        pkt = Packet(self.node_id, myargs['dest'], "data_raw", 0, myargs['sz'], 
                     ttl = self.intercon.network_diameter(),
                     blaze_trail = myargs['blaze'] or ("host" in self.hpcsim_dict["debug_options"]))
        pkt.set_sendtime(self.get_now())
        pkt.add_to_path(str(self))
        if "host" in self.hpcsim_dict["debug_options"]:
            print("%f: %s test_raw_xfer sends %s" % (self.get_now(), self, pkt))

        if self.node_id == pkt.dsthost:
            self.mem_queue.send_pkt(pkt)
        else:
            self.interfaces['r'].send_pkt(pkt, 0)

def receive_process(self):
    """A host's process for receiving packets.

    Note that this is a regular function.
    """

    # self is process; host is the process' entity
    host = self.entity

    # in the beginning, no packet arrival, the process hibernates
    assert host.arrival_semaphore == 0
    self.hibernate()

    # we assume that host has only one interface named 'r' and it has
    # only one port
    while True:
        if not host.fast_queue.is_empty():
            pkt = host.fast_queue.dequeue()
            host.arrival_semaphore -= 1
            assert host.arrival_semaphore >= 0
        elif not host.interfaces['r'].is_recv_empty(0):
            pkt = host.interfaces['r'].recv_pkt(0)
            host.arrival_semaphore -= 1
            assert host.arrival_semaphore >= 0
        else: 
            assert host.arrival_semaphore == 0
            self.hibernate()
            continue

        # handling all packets; we assume all data packets have type
        # starting with 'data' and all acknowledgment packets have
        # type starting with 'ack'
        assert pkt.dsthost==host.node_id, "unmatched host: %d,%d" % (pkt.dsthost, host.node_id)
        if pkt.type[:4] == 'data':
            if "host" in host.hpcsim_dict["debug_options"]:
                print("%f: %s receives %s, delay=%f" %
                      (host.get_now(), host, pkt, host.get_now()-pkt.get_sendtime()))
                path = pkt.get_path()
                #if path is not None:
                #    for h in path: print("  =>%s" % h)
                print("%f: %s receives packet delay=%f #hops: %d"%(host.get_now(), host, host.get_now()-pkt.get_sendtime(), len(path)-1))   
            # notify whoever wants to handle the received packet
            host.notify_data_recv(pkt)
            '''
            #TODO: change it later ("debug_options")
            #TODO: also only data type
            if "calculate_hops" in host.hpcsim_dict["debug_options"] and "num_hops" in pkt.nonreturn_data:
                print("#hops: %d"%pkt.nonreturn_data["num_hops"])   
            if "calculate_hops" in host.hpcsim_dict["debug_options"]:
                print("#hops: %d"%pkt.num_hops)   
            '''
        elif pkt.type[:3] == 'ack':
            if "host" in host.hpcsim_dict["debug_options"]:
                print("%f: %s receives %s, delay=%f" %
                      (host.get_now(), host, pkt, host.get_now()-pkt.get_sendtime()))
                path = pkt.get_path()
                #if path is not None:
                #    for h in path: print("  =>%s" % h)
                #print("#hops: %d"%(len(path)-1))   
            # notify whoever whats to handle the ack
            host.notify_ack_recv(pkt)    
        else:
            raise Exception("unknown packet type: %s" % pkt.type)
        

