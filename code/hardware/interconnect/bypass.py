#
# bypass.py :- a hypothetical all-to-all interconnect without congestion
#

from intercon import *

# block decomposition for the hosts
def bypass_partition(entname, entid, nranks, bypass):
    # the user may specify that we partition up to a given number of
    # hosts (that really run the user application)
    nh = bypass.hpcsim_dict.get('partition_hosts', bypass.nhosts)
    if nh < nranks: chunk = 1.0
    else: chunk = 1.0*nh/nranks
    r = int(entid/chunk)%nranks
    #print("%d entity %s[%d] mapped to rank %d/%d, chunk=%f" % (entid, entname, entid, r, nranks, chunk))
    return r

class Bypass(Interconnect):
    """A bypass have n switch; all hosts connect all-to-all."""

    # local variables: (class derived from Interconnect)
    #   link_delay: link delay, for calculating the network diameter 

    def __init__(self, hpcsim, hpcsim_dict):
        super(Bypass, self).__init__(hpcsim_dict)

        self.nswitches = 0
        if "bypass" not in hpcsim_dict:
            raise Exception("'bypass' must be specified for bypass interconnect")
        if "nhosts" not in hpcsim_dict["bypass"]:
            raise Exception("'nhosts' must be specified for bypass config") 
        self.nhosts = hpcsim_dict["bypass"]["nhosts"]
        if hpcsim_dict['simian'].rank == 0 and \
           ("hpcsim" in hpcsim_dict["debug_options"] or \
            "intercon" in hpcsim_dict["debug_options"] or \
            "bypass" in hpcsim_dict["debug_options"]):
            print("bypass: %d hosts" % self.nhosts)

        # pick out config parameters
        bdw = hpcsim_dict["bypass"].get("bdw", \
            hpcsim_dict["default_configs"]["intercon_bandwidth"])
        self.link_delay = hpcsim_dict["bypass"].get("link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        mem_bandwidth = hpcsim_dict["bypass"].get("mem_bandwidth", \
            hpcsim_dict["default_configs"]["mem_bandwidth"])
        mem_bufsz = hpcsim_dict["bypass"].get("mem_bufsz", \
            hpcsim_dict["default_configs"]["mem_bufsz"])
        mem_delay = hpcsim_dict["bypass"].get("mem_delay", \
            hpcsim_dict["default_configs"]["mem_delay"])

        if hpcsim_dict['simian'].rank == 0 and \
           ("hpcsim" in hpcsim_dict["debug_options"] or \
            "intercon" in hpcsim_dict["debug_options"] or \
            "bypass" in hpcsim_dict["debug_options"]):
            print("bypass: bdw=%f (bits per second)" % bdw)
            print("bypass: link_delay=%f (seconds)" % self.link_delay)
            print("bypass: mem_bandwidth=%f (bits per second)" % mem_bandwidth)
            print("bypass: mem_bufsz=%d (bytes)" % mem_bufsz)
            print("bypass: mem_delay=%f (seconds)" % mem_delay)

        # add switch and hosts as entities
        simian = hpcsim_dict["simian"]
        for h in xrange(self.nhosts):
            simian.addEntity("Host", hpcsim.get_host_typename(hpcsim_dict), h,
                             hpcsim_dict, # simulation configuration
                             self, # interconnect
                             -1, # switch id (-1 means host to host),
                             'r', 0, # switch interface, and switch port
                             bdw, # bandwidth
                             1e38, # buffer size (big enough to be considered infinite) 
                             self.link_delay, # link delay
                             mem_bandwidth, mem_bufsz, mem_delay, # memory bypass configs
                             partition=bypass_partition, partition_arg=self)

    # the network diameter (override the same in Interconnect)
    def network_diameter(self):
        """Returns the network diameter in hops."""
        return 1

    def network_diameter_time(self):
        """Returns the network diameter in time."""
        return self.link_delay

    @staticmethod
    def calc_min_delay(hpcsim_dict):
        """Calculates and returns the min delay value from config parameters."""
        
        if "bypass" not in hpcsim_dict:
            raise Exception("'bypass' must be specified for bypass interconnect")
        return hpcsim_dict["bypass"].get("link_delay", \
                    hpcsim_dict["default_configs"]["intercon_link_delay"])
