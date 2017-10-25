#
# torus.py :- a torus interconnect, including gemini, bluegene/q
#

from intercon import *

# block decomposition for both hosts and switches
def torus_partition(entname, entid, nranks, torus):
    # the user may specify that we partition up to a given number of
    # hosts (that really run the user application)
    nh = torus.hpcsim_dict.get('partition_hosts', torus.nhosts)
    if nh < nranks: chunk = 1.0
    else: chunk = 1.0*nh/nranks
    if entname == 'Switch': id = entid*torus.dimh
    else: id = entid
    r = int(id/chunk)%nranks
    #print("%d entity %s[%d] mapped to rank %d/%d" % (entid, entname, entid, r, nranks))
    return r

class TorusSwitch(Switch):
    """A switch node for the torus interconnect."""

    # local variables: (class derived from Switch)
    #   torus: the torus interconnect
    #   coords: the coordiates of this switch
    #   route_method: the routing method

    def __init__(self, baseinfo, hpcsim_dict, proc_delay, *args):
        super(TorusSwitch, self).__init__(baseinfo, hpcsim_dict, proc_delay)
        self.torus, self.coords, dups, bdws, bdwh, bufsz, \
            switch_link_delay, host_link_delay, self.route_method = args
        ndims = len(self.torus.dims)
        dimh = self.torus.dimh

        # for each torus dimension
        for d in xrange(ndims):
            peer_node_names = ("Switch",)*dups[d]
            peer_iface_ports = tuple(range(dups[d]))

            plus_dir = '+%d'%d
            minus_dir = '-%d'%d

            peer_node_ids = (self.torus.neighbor_swid(self.node_id, plus_dir),)*dups[d]
            peer_iface_names = (minus_dir,)*dups[d]
            self.interfaces[plus_dir] = Interface(self, plus_dir, dups[d], 
                                                  peer_node_names, peer_node_ids, 
                                                  peer_iface_names, peer_iface_ports, 
                                                  bdws[d], bufsz, switch_link_delay)
            if "switch.connect" in hpcsim_dict["debug_options"]:
                print("%s %s connects to switch %r iface %r ports %r" % 
                      (self, self.interfaces[plus_dir], peer_node_ids, 
                       peer_iface_names, peer_iface_ports))

            peer_node_ids = (self.torus.neighbor_swid(self.node_id, minus_dir),)*dups[d]
            peer_iface_names = (plus_dir,)*dups[d]
            self.interfaces[minus_dir] = Interface(self, minus_dir, dups[d], 
                                                   peer_node_names, peer_node_ids, 
                                                   peer_iface_names, peer_iface_ports, 
                                                   bdws[d], bufsz, switch_link_delay)
            if "switch.connect" in hpcsim_dict["debug_options"]:
                print("%s %s connects to switch %r iface %r ports %r" % 
                      (self, self.interfaces[minus_dir], peer_node_ids, 
                       peer_iface_names, peer_iface_ports))

        # connect to hosts
        dir = "h"; 
        peer_node_names = ("Host",)*dimh
        peer_node_ids = []
        for p in xrange(dimh):
            peer_node_ids.append(self.torus.coords_to_hid(self.coords, p))
        peer_iface_names = ("r",)*dimh
        peer_iface_ports = (0,)*dimh
        self.interfaces[dir] = Interface(self, dir, dimh, 
                                         peer_node_names, tuple(peer_node_ids),
                                         peer_iface_names, peer_iface_ports, 
                                         bdwh, bufsz, host_link_delay)
        if "switch.connect" in hpcsim_dict["debug_options"]:
            print("%s %s connects to host %r iface %r ports %r" %
                  (self, self.interfaces[dir], tuple(peer_node_ids),
                   peer_iface_names, peer_iface_ports))

    def __str__(self):
        return "switch[%d] %r" % (self.node_id, self.coords)

    def calc_route(self, pkt):
        """Returns interface name and port number (only if next hop is host)
           needed to get to the next hop."""

        if self.route_method is "deterministic_dimension_order" or \
           self.route_method is "hashed_dimension_order" or \
           self.route_method is "adaptive_dimension_order":
            #     - deterministic_dimension_order: dimension-order routing with predetermined links within each dimension
            #     - hashed_dimension_order: dimension-order routing with flexibility in selecting links
            #     - adaptive_dimension_order (default): dimension-order routing but select lightly loaded links

            c,p = self.torus.hid_to_coords(pkt.dsthost)

            # first, dimension order routing
            dirs = []
            for d in xrange(len(self.torus.dims)):
                if self.coords[d] != c[d]:
                    diff = c[d]-self.coords[d]
                    if diff < 0: diff = diff + self.torus.dims[d]
                    if diff <= self.torus.dims[d]/2: dirs.append("+%d"%d)
                    else: dirs.append("-%d"%d)
                    break # stop when early dimension is found

            if len(dirs) == 0:
                # the packet arrived at the destination switch; send out to host
                #if "switch" in self.hpcsim_dict["debug_options"]:
                #    print("%s calc_route: to %d via iface(h)" % (self, pkt.dsthost))
                return "h", p
            else:
                md = dirs[0] # we have only one dimension
                if self.route_method is "adaptive_dimension_order":
                    # find one with the min delay: just don't name the port
                    #m = self.interfaces[md].get_min_qdelay()
                    #for dir in dirs[1:]:
                    #    d = self.interfaces[dir].get_min_qdelay()
                    #    if d < m: md = dir; m = d
                    return md, -1
                elif self.route_method is "hashed_dimension_order":
                    # there's no detail how hash is applied in cray's
                    # gemini; we select the port according to
                    # destination host id and sequence number
                    m = self.interfaces[md].get_num_ports()
                    port = (pkt.dsthost+self.seqno)%m
                    return md, port
                else: # self.route_method is "deterministic_dimension_order"
                    # for deterministic, we select the port according
                    # to the destination host id
                    m = self.interfaces[md].get_num_ports()
                    port = pkt.dsthost%m
                    return md, port

        else:
            raise Exception("route method %d has not been implemented" % self.route_method)


class Torus(Interconnect):
    """A generic torus network."""

    # local variables: (class derived from Interconnect)
    #   dims: the dimension of the torus interconnect
    #   dimh: the number of hosts attached to each torus switch
    #   hostmap_h2sw: map from host to switch (None if default map)
    #   hostmap_sw2h: map from switch to host (None if default map)
    #   cm: coordinate multiplier (for calculating switch id and its coordiates)
    #   switch_link_delay: link delay between two switches
    #   host_link_delay: link delay between switch and its attached host

    def __init__(self, hpcsim, hpcsim_dict):
        super(Torus, self).__init__(hpcsim_dict)

        if "torus" not in hpcsim_dict:
            raise Exception("'torus' must be specified for torus interconnect")
        if "dims" not in hpcsim_dict["torus"]:
            raise Exception("'dims' must be specified for torus config") 
        self.dims = hpcsim_dict["torus"]["dims"]
        if "attached_hosts_per_switch" not in hpcsim_dict["torus"]:
            raise Exception("'attached_hosts_per_switch' must be specified for torus config") 
        self.dimh = hpcsim_dict["torus"]["attached_hosts_per_switch"]

        # compute the total number of switches and hosts
        self.nswitches = 1
        for x in self.dims: self.nswitches *= x
        self.nhosts = self.nswitches*self.dimh
        #print("dimh: %d"%self.dimh)        
        #print("num of hosts: %d"%self.nhosts)
        # a map can be used optionally to contain host to switch mapping 
        if "hostmap" in hpcsim_dict["torus"]:
            self.hostmap_h2sw = dict()
            self.hostmap_sw2h = dict()
            if len(hpcsim_dict["torus"]["hostmap"]) != self.nhosts:
                raise Exception("incorrect number of hosts in torus hostmap")
            for hmap in hpcsim_dict["torus"]["hostmap"]:
                if len(hmap) != len(self.dims)+2: # host, switch coords, switch port
                    raise Exception("invalid torus hostmap entry")
                h = hmap[0]
                c = tuple(hmap[1:-1])
                p = hmap[-1]
                #print("hostmap: h=%d c=%r p=%d" % (h, c, p))
                if h in self.hostmap_h2sw:
                    raise Exception("duplicate host %d found in torus hostmap" % h)
                self.hostmap_h2sw[h] = (c, p)
                if not (0 <= p < self.dimh):
                    raise Exception("switch port %r in torus hostmap out of range" % p)
                if (c, p) in self.hostmap_sw2h:
                    raise Exception("duplicate switch %r port %d found in torus hostmap" % (c, p))
                self.hostmap_sw2h[(c,p)] = h
            #print(self.hostmap_h2sw)
        else:
            self.hostmap_h2sw = None
            self.hostmap_sw2h = None

        if hpcsim_dict['simian'].rank == 0 and \
           ("hpcsim" in hpcsim_dict["debug_options"] or \
            "intercon" in hpcsim_dict["debug_options"] or \
            "torus" in hpcsim_dict["debug_options"]):
            print("%d-d torus interconnect: %r" % (len(self.dims), self.dims))
            print("torus: %d hosts attached to each switch" % self.dimh)
        
        # calc once: coordinate multiplier (used for translation
        # between switch id and switch coordinates)
        self.cm = [1]*len(self.dims)
        for d in xrange(1, len(self.cm)):
            self.cm[d] = self.cm[d-1]*self.dims[d-1]
        #print(self.cm)

        # pick out config parameters
        if 'bdws' in hpcsim_dict["torus"]:
            bdws = hpcsim_dict["torus"]["bdws"]
            if len(bdws) != len(self.dims):
                raise Exception("invalid bdws %r for torus %r" % (bdws, self.dims))
        else:
            bdws = tuple([hpcsim_dict["default_configs"]["intercon_bandwidth"] for d in self.dims])

        if 'dups' in hpcsim_dict["torus"]:
            dups = hpcsim_dict["torus"]["dups"]
            if len(dups) != len(self.dims):
                raise Exception("invalid dups %r for torus %r" % (dups, self.dims))
        else:
            dups = (1)*len(self.dims)

        bdwh = hpcsim_dict["torus"].get("bdwh", \
            hpcsim_dict["default_configs"]["intercon_bandwidth"])

        bufsz = hpcsim_dict["torus"].get("bufsz", \
            hpcsim_dict["default_configs"]["intercon_bufsz"])

        self.switch_link_delay = hpcsim_dict["torus"].get("switch_link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        self.host_link_delay = hpcsim_dict["torus"].get("host_link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])

        proc_delay = hpcsim_dict["torus"].get("proc_delay", \
            hpcsim_dict["default_configs"]["intercon_proc_delay"])

        route_method = hpcsim_dict["torus"].get("route_method", \
            hpcsim_dict["default_configs"]["torus_route_method"])

        mem_bandwidth = hpcsim_dict["torus"].get("mem_bandwidth", \
            hpcsim_dict["default_configs"]["mem_bandwidth"])
        mem_bufsz = hpcsim_dict["torus"].get("mem_bufsz", \
            hpcsim_dict["default_configs"]["mem_bufsz"])
        mem_delay = hpcsim_dict["torus"].get("mem_delay", \
            hpcsim_dict["default_configs"]["mem_delay"])

        if hpcsim_dict['simian'].rank == 0 and \
           ("hpcsim" in hpcsim_dict["debug_options"] or \
            "intercon" in hpcsim_dict["debug_options"] or \
            "torus" in hpcsim_dict["debug_options"]):
            print("torus: dups=%r" % (dups,))
            print("torus: bdws=%r (bits per second)" % (bdws,))
            print("torus: bdwh=%f (bits per second)" % bdwh)
            print("torus: bufsz=%d (bytes)" % bufsz)
            print("torus: switch_link_delay=%f (seconds)" % self.switch_link_delay)
            print("torus: host_link_delay=%f (seconds)" % self.host_link_delay)
            print("torus: proc_delay=%f (seconds)" % proc_delay)
            print("torus: mem_bandwidth=%f (bits per second)" % mem_bandwidth)
            print("torus: mem_bufsz =%d (bytes)" % mem_bufsz)
            print("torus: mem_delay=%f (seconds)" % mem_delay)
            print("torus: route_method=%s" % route_method)

        # add switches as entities
        simian = hpcsim_dict["simian"]
        allcoords = [(i,) for i in range(self.dims[-1])]
        for d in self.dims[-2::-1]:
            allcoords = [(j,)+i for i in allcoords for j in xrange(d)]
        for s in xrange(self.nswitches):
            #print("creating switch: id=%d coords=%r" % (s, allcoords[s]))
            simian.addEntity("Switch", TorusSwitch, s, hpcsim_dict, proc_delay, 
                             self, allcoords[s], dups, bdws, bdwh, bufsz, 
                             self.switch_link_delay, self.host_link_delay, route_method)#,
                             #partition=torus_partition, partition_arg=self)

        # add hosts and entities
        for h in xrange(self.nhosts):
            c, p = self.hid_to_coords(h)
            #print("creating host: id=%d coords=%r:%d" % (h, c, p))
            swid = self.coords_to_swid(c)
            simian.addEntity("Host", hpcsim.get_host_typename(hpcsim_dict), h,
                             hpcsim_dict, self, swid, 'h', p,
                             bdwh, bufsz, self.host_link_delay,
                             mem_bandwidth, mem_bufsz, mem_delay)#,
                             #partition=torus_partition, partition_arg=self)


    def network_diameter(self):
        """Returns the network diameter in hops."""

        r = 2+self.dims[0]/2 # 2 extra for host connection
        for d in self.dims[1:]: r += int(d/2)
        return r

    def network_diameter_time(self):
        """Returns the network diameter in time."""
    
        d = self.network_diameter()
        return (d-2)*self.switch_link_delay+2*self.host_link_delay

    def coords_to_swid(self, c):
        """Converts from switch coordinates to switch id."""
    
        if len(c) != len(self.dims):
            raise Exception("invalid switch coordinates %r for torus %r" % (c, self.dims))
        r = 0
        for d in xrange(len(c)):
            if not (0 <= c[d] < self.dims[d]):
                raise Exception("invalid switch coordinates %r for torus %r" % (c, self.dims))
            r += c[d]*self.cm[d]
        #print("switch coords %r => id %d" % (c, r))
        return r

    def swid_to_coords(self, swid):
        """Converts from switch id to switch coordinates."""
    
        #_swid = swid
        if not (0 <= swid < self.nswitches):
            raise Exception("invalid switch id=%d for torus %r" % (swid, self.dims))
        c = []
        for d in xrange(len(self.dims)-1, -1, -1): # dimension in reverse order
            c.insert(0, swid/self.cm[d])
            swid %= self.cm[d]
        c = tuple(c)
        #print("switch id %d => coords %r" % (_swid, c))
        return c

    def hid_to_coords(self, hid):
        """Converts from host id to switch coordinates and switch-to-host port."""
    
        if not (0 <= hid < self.nhosts):
            raise Exception("invalid host id=%d for for torus %r" % (hid, self.dims))
        if self.hostmap_h2sw is None:
            p = hid%self.dimh
            c = self.swid_to_coords(hid/self.dimh)
        else:
            c, p = self.hostmap_h2sw[hid]
        #print("host id %d => coords %r:%d" % (hid, c, p))
        return (c, p)

    def coords_to_hid(self, c, p):
        """Converts from switch coordinates and switch-to-host port to host id."""
    
        if not (0 <= p < self.dimh):
            raise Exception("invalid switch-to-host port %d" % p)
        if  self.hostmap_sw2h is None:
            h = self.coords_to_swid(c)*self.dimh+p
        else:
            if (c, p) not in self.hostmap_sw2h:
                raise Exception("invalid switch coords %r" % c)
            h = self.hostmap_sw2h[(c,p)]
        #print("host coords %r:%d => id %d" % (c, p, h))
        return h

    def neighbor_coords(self, c, dir):
        """Return the coordinates of the neighboring switch in the given direction."""
    
        if len(c) != len(self.dims):
            raise Exception("invalid switch coordinates %r for torus %r" % (c, self.dims))
        if (dir[0] != '+' and dir[0] != '-') or \
           not (0 <= int(dir[1:]) < len(self.dims)):
            raise Exception("invalid direction: %s" % dir)
        r = []
        for d in xrange(len(c)):
            if not (0 <= c[d] < self.dims[d]):
                raise Exception("invalid switch coordinates %r for torus %r" % (c, self.dims))
            if d == int(dir[1:]):
                if dir[0] == '+': r.append((c[d]+1)%self.dims[d])
                else: r.append((c[d]+self.dims[d]-1)%self.dims[d])
            else: r.append(c[d])
        return tuple(r)

    def neighbor_swid(self, swid, dir):
        """Returns the id of the neighboring switch in the given direction."""

        c = self.swid_to_coords(swid)
        cn = self.neighbor_coords(c, dir)
        return self.coords_to_swid(cn)

    @staticmethod
    def calc_min_delay(hpcsim_dict):
        """Calculates and returns the min delay value from config parameters."""
    
        if "torus" not in hpcsim_dict:
            raise Exception("'torus' must be specified for torus interconnect")
        d1 = hpcsim_dict["torus"].get("switch_link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        d2 = hpcsim_dict["torus"].get("host_link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        return d1 if d1<d2 else d2


class Gemini(Torus):
    """cray's gemini is a 3D torus."""

    def __init__(self, hpcsim, hpcsim_dict):
        # we'd need to do some translation
        if "torus" not in hpcsim_dict:
            raise Exception("'torus' must be specified for gemini interconnect")
        dimx = hpcsim_dict["torus"].get("dimx", 1)
        dimy = hpcsim_dict["torus"].get("dimy", 1)
        dimz = hpcsim_dict["torus"].get("dimz", 1)
        hpcsim_dict["torus"]["dims"] = (dimx, dimy, dimz)
        
        hpcsim_dict["torus"]["attached_hosts_per_switch"] = 2

        dbw = hpcsim_dict["default_configs"]["intercon_bandwidth"]
        bdwx = hpcsim_dict["torus"].get("bdwx", dbw)
        bdwy = hpcsim_dict["torus"].get("bdwy", dbw)
        bdwz = hpcsim_dict["torus"].get("bdwz", dbw)
        hpcsim_dict["torus"]["bdws"] = (bdwx, bdwy, bdwz)

        hpcsim_dict["torus"]["dups"] = (2, 1, 2)

        super(Gemini, self).__init__(hpcsim, hpcsim_dict)

class BlueGeneQ(Torus):
    """IBM's Blue Gene/Q is a 5D torus."""

    def __init__(self, hpcsim, hpcsim_dict):
        # we'd need to do some translation
        if "torus" not in hpcsim_dict:
            raise Exception("'torus' must be specified for bluegen/q interconnect")
        dima = hpcsim_dict["torus"].get("dima", 1)
        dimb = hpcsim_dict["torus"].get("dimb", 1)
        dimc = hpcsim_dict["torus"].get("dimc", 1)
        dimd = hpcsim_dict["torus"].get("dimd", 1)
        dime = hpcsim_dict["torus"].get("dime", 1)
        hpcsim_dict["torus"]["dims"] = (dima, dimb, dimc, dimd, dime)
        
        hpcsim_dict["torus"]["attached_hosts_per_switch"] = 1

        dbw = hpcsim_dict["default_configs"]["intercon_bandwidth"]
        bdwa = hpcsim_dict["torus"].get("bdwa", dbw)
        bdwb = hpcsim_dict["torus"].get("bdwb", dbw)
        bdwc = hpcsim_dict["torus"].get("bdwc", dbw)
        bdwd = hpcsim_dict["torus"].get("bdwd", dbw)
        bdwe = hpcsim_dict["torus"].get("bdwe", dbw)
        hpcsim_dict["torus"]["bdws"] = (bdwa, bdwb, bdwc, bdwd, bdwe)

        hpcsim_dict["torus"]["dups"] = (2, 2, 2, 2, 2)

        super(BlueGeneQ, self).__init__(hpcsim, hpcsim_dict)

