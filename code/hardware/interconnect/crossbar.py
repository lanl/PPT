#
# crossbar.py :- a hyperthetical crossbar interconnect
#

from intercon import *

class CrossbarSwitch(Switch):
    """A switch node for the crossbar interconnect."""

    # local variables: (class derived from Switch)
    #   crossbar: the crossbar interconnect

    def __init__(self, baseinfo, hpcsim_dict, proc_delay, *args):
        super(CrossbarSwitch, self).__init__(baseinfo, hpcsim_dict, proc_delay)
        self.crossbar, bdw, bufsz, link_delay = args
        for h in xrange(self.crossbar.nhosts):
            iname = 'h%d'%h
            peer_node_names = ('Host',)
            peer_node_ids = (h,)
            peer_iface_names = ('r',)
            peer_iface_ports = (0,)
            self.interfaces[iname] = Interface(self, iname, 1, peer_node_names, 
                                               peer_node_ids, peer_iface_names, peer_iface_ports,
                                               bdw, bufsz, link_delay)
            if "switch.connect" in hpcsim_dict["debug_options"]:
                print("%s %s connects to host %r iface %r ports %r" %
                      (self, self.interfaces[iname], tuple(peer_node_ids),
                       peer_iface_names, peer_iface_ports))

    def __str__(self):
        return "crossbar_switch[n=%d]"%self.crossbar.nhosts

    def calc_route(self, pkt):
        """Returns interface name and port number (only if next hop is host)
           needed to get to the next hop."""
        
        if not 0 <= pkt.dsthost < self.crossbar.nhosts:
            raise Exception("packet %s destination out of bound" % pkt)
        return 'h%d'%pkt.dsthost, 0


class Crossbar(Interconnect):
    """A crossbar is made of one switch connecting all hosts."""

    # local variables: (class derived from Switch)
    #   link_delay: link delay, for calculating the network diameter 

    def __init__(self, hpcsim, hpcsim_dict):
        super(Crossbar, self).__init__(hpcsim_dict)

        self.nswitches = 1
        if "crossbar" not in hpcsim_dict:
            raise Exception("'crossbar' must be specified for crossbar interconnect")
        if "nhosts" not in hpcsim_dict["crossbar"]:
            raise Exception("'nhosts' must be specified for crossbar config") 
        self.nhosts = hpcsim_dict["crossbar"]["nhosts"]
        if "hpcsim" in hpcsim_dict["debug_options"] or \
           "intercon" in hpcsim_dict["debug_options"] or \
           "crossbar" in hpcsim_dict["debug_options"]:
            print("crossbar: %d hosts" % self.nhosts)

        # pick out config parameters
        bdw = hpcsim_dict["crossbar"].get("bdw", \
            hpcsim_dict["default_configs"]["intercon_bandwidth"])
        bufsz = hpcsim_dict["crossbar"].get("bufsz", \
            hpcsim_dict["default_configs"]["intercon_bufsz"])
        self.link_delay = hpcsim_dict["crossbar"].get("link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        proc_delay = hpcsim_dict["crossbar"].get("proc_delay", \
            hpcsim_dict["default_configs"]["intercon_proc_delay"])
        mem_bandwidth = hpcsim_dict["crossbar"].get("mem_bandwidth", \
            hpcsim_dict["default_configs"]["mem_bandwidth"])
        mem_bufsz = hpcsim_dict["crossbar"].get("mem_bufsz", \
            hpcsim_dict["default_configs"]["mem_bufsz"])
        mem_delay = hpcsim_dict["crossbar"].get("mem_delay", \
            hpcsim_dict["default_configs"]["mem_delay"])

        if "hpcsim" in hpcsim_dict["debug_options"] or \
           "intercon" in hpcsim_dict["debug_options"] or \
           "crossbar" in hpcsim_dict["debug_options"]:
            print("crossbar: bdw=%f (bits per second)" % bdw)
            print("crossbar: bufsz=%d (bytes)" % bufsz)
            print("crossbar: link_delay=%f (seconds)" % self.link_delay)
            print("crossbar: proc_delay=%f (seconds)" % proc_delay)
            print("crossbar: mem_bandwidth=%f (bits per second)" % mem_bandwidth)
            print("crossbar: mem_bufsz=%d (bytes)" % mem_bufsz)
            print("crossbar: mem_delay=%f (seconds)" % mem_delay)

        # add switch and hosts as entities
        simian = hpcsim_dict["simian"]
        simian.addEntity("Switch", CrossbarSwitch, 0, hpcsim_dict, proc_delay, 
                         self, bdw, bufsz, self.link_delay)
        for h in xrange(self.nhosts):
            simian.addEntity("Host", hpcsim.get_host_typename(hpcsim_dict), h,
                             hpcsim_dict, self, 0, 'h%d'%h, 0,
                             bdw, bufsz, self.link_delay,
                             mem_bandwidth, mem_bufsz, mem_delay)

    # the network diameter (override the same in Interconnect)
    def network_diameter(self):
        """Returns the network diameter in hops."""
        return 2

    def network_diameter_time(self):
        """Returns the network diameter in time."""
        return 2*self.link_delay

    @staticmethod
    def calc_min_delay(hpcsim_dict):
        """Calculates and returns the min delay value from config parameters."""
        
        if "crossbar" not in hpcsim_dict:
            raise Exception("'crossbar' must be specified for crossbar interconnect")
        return hpcsim_dict["crossbar"].get("link_delay", \
                    hpcsim_dict["default_configs"]["intercon_link_delay"])
