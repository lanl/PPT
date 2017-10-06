#
# fat_tree.py :- a fat-tree interconnect
#

from intercon import *
import random
import itertools
import math
import os

# The considered fat-tree is an m-port n-tree fat-tree.
# The details of this type of tree is given in 
# "multiple LID routing scheme for fat-tree-based InfiniBand networks" by
# Xuan-Yi Lin, Yeh-Ching Chung, and Tai-Yi Huang. 

class InfinibandSwitch(Switch):
    """A switch for the fat-tree interconnect."""
    
    def __init__(self, baseinfo, hpcsim_dict, intercon, levelid, swid):
        self.fattree = intercon
        self.lid = levelid
        self.swid = swid # this is a list containing the switch id (excluding level info)
        super(InfinibandSwitch, self).__init__(baseinfo, hpcsim_dict, self.fattree.proc_delay)
        h = self.fattree.num_ports_per_switch
        peer_host_ids = []
        for k in xrange(h):
            peer_dict, con_type = self.connect(self.swid, self.lid, k, None)
            if con_type == 1 or con_type == 2:  # connection to a switch
                peer_lid = peer_dict["level"]
                peer_port = peer_dict["port"]
                peer_swid = peer_dict["s_id"]
                peer_id = self.fattree.sw_label_to_id(peer_lid, peer_swid)
                if con_type == 1: # uplink
                    iface_name = 'u%d'%k            # u: up connection
                    peer_iface = 'd%d'%peer_port    # d: down connection
                    peer_node_names = ("Switch",)*self.fattree.switch_link_dups
                    peer_node_ids = (peer_id,)*self.fattree.switch_link_dups
                    peer_iface_names = (peer_iface,)*self.fattree.switch_link_dups
                    peer_iface_ports = tuple(range(self.fattree.switch_link_dups))
                    self.interfaces[iface_name] = Interface(self, iface_name, self.fattree.switch_link_dups, 
                                                            peer_node_names, peer_node_ids, 
                                                            peer_iface_names, peer_iface_ports, 
                                                            self.fattree.switch_link_up_bdw, 
                                                            self.fattree.bufsz, 
                                                            self.fattree.switch_link_up_delay)
                else: # downlink
                    iface_name = 'd%d'%k    
                    peer_iface = 'u%d'%peer_port
                    peer_node_names = ("Switch",)*self.fattree.switch_link_dups
                    peer_node_ids = (peer_id,)*self.fattree.switch_link_dups
                    peer_iface_names = (peer_iface,)*self.fattree.switch_link_dups
                    peer_iface_ports = tuple(range(self.fattree.switch_link_dups))
                    self.interfaces[iface_name] = Interface(self, iface_name, self.fattree.switch_link_dups, 
                                                            peer_node_names, peer_node_ids, 
                                                            peer_iface_names, peer_iface_ports, 
                                                            self.fattree.switch_link_down_bdw, 
                                                            self.fattree.bufsz, 
                                                            self.fattree.switch_link_down_delay)
                if "switch.connect" in hpcsim_dict["debug_options"]:    
                    print("%s %s connects to switch %r iface %r ports %r" % 
                          (self, self.interfaces[iface_name], peer_node_ids, 
                           peer_iface_names, peer_iface_ports))
            else: # connection to a host
                peer_host_label = peer_dict["host"] 
                peer_id = self.fattree.host_label_to_id(peer_host_label)    
                peer_host_ids.append(peer_id)
        n = len(peer_host_ids)
        if n > 0:   # initiliaze hosts attached to the switch
            dir = "h"
            peer_node_names = ("Host",)*n 
            peer_iface_names = ("r",)*n
            peer_iface_ports = (0,)*n
            self.interfaces[dir] = Interface(self, dir, n, 
                                             peer_node_names, tuple(peer_host_ids),
                                             peer_iface_names, peer_iface_ports, 
                                             self.fattree.host_link_bdw, 
                                             self.fattree.bufsz,  
                                             self.fattree.host_link_delay)
            if "switch.connect" in hpcsim_dict["debug_options"]:    
                print("%s %s connects to host %r iface %r ports %r" % 
                      (self, self.interfaces[dir], tuple(peer_host_ids), 
                       peer_iface_names, peer_iface_ports))
        # caching data structure: map to store src-dest pairs' port info into the cache
        self.map_src_dest = dict()

    def __str__(self):
        swid_str = ':'.join(str(e) for e in self.swid)
        return "switch[%d]"%(self.fattree.sw_label_to_id(self.lid, swid_str))
    
    def connect(self, sid, level, port, nid):
        """Returns connected switch or processing node info"""
        
        # Local variables
        #   sid: switch id, level: level of switch, port: connection port id; node info should be None
        #   nid: node id (for which we need to find connected switch info); other parameters should be None
        connection_type = 0 # 0 means connection goes to node, 1 means uplink, 2 means downlink
        n = self.fattree.num_levels
        m = self.fattree.num_ports_per_switch
        dest = dict()
        if nid != None: # means connection info for a processing node is required
            dest["level"] = n-1 # only last level's switches are connected to the nodes
            dest["port"] = int(nid[n-1])
            dest["s_id"] = nid[0:n-1]
            return dest, connection_type
        else: # connection to be found for a switch 
            if level == n-1 and port < m/2:
                # this is for last level's connection to processing node
                sid_str = ':'.join(str(e) for e in sid)
                dest["host"] = sid_str + ':' + str(port)
                dest["host_list"] = sid + [port]
                return dest, connection_type
            elif level > 0 and port >= m/2:
                # this is for connections going up
                connection_type = 1
                dest["level"] = level-1
                dest["port"] = int(sid[level-1])
                sub_sid_1 = sid[0:level-1] 
                sub_sid_1_str = ':'.join(str(e) for e in sub_sid_1)
                sub_sid_2 = sid[level:n-2+1]
                sub_sid_2_str = ':'.join(str(e) for e in sub_sid_2)
                if not sub_sid_1_str and not(not sub_sid_2_str):    # first substring empty
                    dest["s_id"] = sub_sid_2_str + ":" + str(port-m/2)
                elif not sub_sid_2_str and not(not sub_sid_1_str):  # second substring empty
                    dest["s_id"] = sub_sid_1_str + ":" + str(port-m/2)
                elif not sub_sid_1_str and not sub_sid_2_str: # both substrings empty
                    dest["s_id"] = str(port-m/2)
                else:   # none of the substrings is empty
                    dest["s_id"] = sub_sid_1_str + ":"+ sub_sid_2_str + ":" + str(port-m/2)
                dest["s_id_list"] = sub_sid_1 + sub_sid_2 + [port-m/2]
                return dest, connection_type
            else:
                # this is for connections going down
                connection_type = 2
                dest["level"] = level+1
                dest["port"] = sid[n-2] + m/2
                sub_sid = sid[0:n-3+1]
                sub_sid_1 = sub_sid[0:level-1+1]
                sub_sid_1_str = ':'.join(str(e) for e in sub_sid_1)
                sub_sid_2 = sub_sid[level:n-2+1]
                sub_sid_2_str = ':'.join(str(e) for e in sub_sid_2)
                if not sub_sid_1_str and not(not sub_sid_2_str):    # first substring empty
                    dest["s_id"] = str(port) + ":" + sub_sid_2_str 
                elif not sub_sid_2_str and not(not sub_sid_1_str):  # second substring empty  
                    dest["s_id"] = sub_sid_1_str + ":" + str(port)
                elif not sub_sid_1_str and not sub_sid_2_str: # both substrings empty
                    dest["s_id"] = str(port)
                else:   # none of the substrings is empty
                    dest["s_id"] = sub_sid_1_str + ":" + str(port) + ":" + sub_sid_2_str
                dest["s_id_list"] = sub_sid_1 + [port] + sub_sid_2
                return dest, connection_type

    def path_selection(self, src_id_list, dst_id_list):
        """Returns a LID to be used from a choice of multiple LIDs"""

        # Local variables
        #   src_id_list: soruce id in list format
        #   dst_id_list: dst id in list format
        n = self.fattree.num_levels
        m = self.fattree.num_ports_per_switch
        x = os.path.commonprefix([src_id_list, dst_id_list])
        if len(x) > 0:  # check if "x" is in the group gcpg(x, alpha)
            x_str = ':'.join(str(e) for e in x)
            x_str_src = x_str + ":" + str(src_id_list[len(x)])
            x_str_dst = x_str + ":" + str(dst_id_list[len(x)])
            if src_id_list in self.fattree.grp_members_dict[x_str_src] and \
                    dst_id_list in self.fattree.grp_members_dict[x_str_dst] and \
                    src_id_list in self.fattree.grp_members_dict[x_str] and \
                    dst_id_list in self.fattree.grp_members_dict[x_str]:
                    alpha = len(x)
            else:
                raise Exception("no alpha could be found")   
        else:
            #|x| = 0; meaning the nodes are not in any common grp (all processing nodes are in this grp) 
            alpha = len(x)
        # find the BaseLID(P(p'))
        lmc = math.log((m/2)**(n-1), 2) # LID mask control
        base_sum = 0
        for i in xrange(n):
            base_sum += dst_id_list[i]*((m/2)**(n-(i+1)))
        base_lid = int((2**lmc*base_sum)+1)
        # find the rank of the host node inside the grp
        rank = 0
        for alpha_idx in xrange(alpha+1, n):
            rank += src_id_list[alpha_idx]*(m/2)**((n-1)-alpha_idx)
        lid = base_lid + rank
        return lid

    def find_port(self, sw_id, level_id, dst_host, lid, PID):
        """Returns the port number"""

        n = self.fattree.num_levels
        m = self.fattree.num_ports_per_switch
        if PID == int(math.floor((lid-1)/((m/2)**(n-1)))) and \
                sw_id[0:level_id-1+1] == dst_host[0:level_id-1+1]:
            # downward direction
            k = dst_host[level_id]
        else:
            # upward direction
            k = ((math.floor((lid-1)/((m/2)**((n-1)-level_id))))%(m/2))+(m/2)
        return k

    def calc_route(self, pkt):
        """Returns interface name and port number"""

        dst_host_list = self.fattree.host_id_to_label_list(pkt.dsthost)
        src_host_list = self.fattree.host_id_to_label_list(pkt.srchost)
        n = self.fattree.num_levels
        m = self.fattree.num_ports_per_switch
        if self.fattree.route_method == "multiple_lid_nca": # multiple LID nearest common ancestor
            # caching of src-dest pair
            key_idx = str(pkt.srchost) + ":" + str(pkt.dsthost) 
            if key_idx in self.map_src_dest:
                port = self.map_src_dest[key_idx] 
            else:
                # first, check if current switch is directly connected to the dest host
                dest_switch, con_type = self.connect(None, None, None, dst_host_list)
                if dest_switch["level"] == self.lid and dest_switch["s_id"] == self.swid:
                    return "h", dest_switch["port"]
                # next, find the port to send packet through
                lid = self.path_selection(src_host_list, dst_host_list)
                port = self.find_port(self.swid, self.lid, 
                        dst_host_list, lid, pkt.dsthost) # pkt.dsthost denotes PID
                self.map_src_dest[key_idx] = port
            if self.lid > 0 and port >= m/2:    # connection going up
                md = 'u%d'%port
                m = self.interfaces[md].get_num_ports()
                port = random.randint(0,m-1)
                return md, port
            else:                               # connection going down
                md = 'd%d'%port
                m = self.interfaces[md].get_num_ports()
                port = random.randint(0,m-1)
                return md, port
        else:
            raise Exception("route method %d has not been implemented" % self.route_method)
            
class Fattree(Interconnect):
    def __init__(self, hpcsim, hpcsim_dict):
        if "fattree" not in hpcsim_dict:
            raise Exception("'fattree' must be specified for fattree interconnect")
        
        if "num_ports_per_switch" not in hpcsim_dict["fattree"]:
            raise Exception("'num_ports_per_switch' must be specified for fattree config") 
        self.num_ports_per_switch = hpcsim_dict["fattree"]["num_ports_per_switch"]
        
        if "num_levels" not in hpcsim_dict["fattree"]:
            raise Exception("'num_levels' must be specified for fattree config") 
        self.num_levels = hpcsim_dict["fattree"]["num_levels"]
        
        if "switch_link_up_delay" not in hpcsim_dict["fattree"]:
            raise Exception("'switch_link_up_delay' must be specified for fattree config") 
        self.switch_link_up_delay = hpcsim_dict["fattree"]["switch_link_up_delay"]
        
        if "switch_link_down_delay" not in hpcsim_dict["fattree"]:
            raise Exception("'switch_link_down_delay' must be specified for fattree config") 
        self.switch_link_down_delay = hpcsim_dict["fattree"]["switch_link_down_delay"]
        
        if "host_link_delay" not in hpcsim_dict["fattree"]:
            raise Exception("'host_link_delay' must be specified for fattree config") 
        self.host_link_delay = hpcsim_dict["fattree"]["host_link_delay"]

        if "switch_link_up_bdw" not in hpcsim_dict["fattree"]:
            raise Exception("'switch_link_up_bdw' must be specified for fattree config") 
        self.switch_link_up_bdw = hpcsim_dict["fattree"]["switch_link_up_bdw"]
        
        if "switch_link_down_bdw" not in hpcsim_dict["fattree"]:
            raise Exception("'switch_link_down_bdw' must be specified for fattree config") 
        self.switch_link_down_bdw = hpcsim_dict["fattree"]["switch_link_down_bdw"]

        if "host_link_bdw" not in hpcsim_dict["fattree"]:
            raise Exception("'host_link_bdw' must be specified for fattree config") 
        self.host_link_bdw = hpcsim_dict["fattree"]["host_link_bdw"]
        
        if "switch_link_dups" not in hpcsim_dict["fattree"]:
            raise Exception("'switch_link_dups' must be specified for fattree config") 
        self.switch_link_dups = hpcsim_dict["fattree"]["switch_link_dups"]
        
        self.bufsz = hpcsim_dict["fattree"].get("bufsz", \
            hpcsim_dict["default_configs"]["intercon_bufsz"])

        self.proc_delay = hpcsim_dict["fattree"].get("proc_delay", \
            hpcsim_dict["default_configs"]["intercon_proc_delay"])

        self.route_method = hpcsim_dict["fattree"].get("route_method", \
            hpcsim_dict["default_configs"]["fattree_route_method"])
        
        # num_ports_per_switch needs to be a power of 2
        if self.num_ports_per_switch & self.num_ports_per_switch-1 != 0:
            raise Exception("num_ports_per_switch needs to be a power of 2.")
        
        # added for intra-node communication
        mem_bandwidth = hpcsim_dict["fattree"].get("mem_bandwidth", \
            hpcsim_dict["default_configs"]["mem_bandwidth"])
        mem_bufsz = hpcsim_dict["fattree"].get("mem_bufsz", \
            hpcsim_dict["default_configs"]["mem_bufsz"])
        mem_delay = hpcsim_dict["fattree"].get("mem_delay", \
            hpcsim_dict["default_configs"]["mem_delay"])
        
        if "hpcsim" in hpcsim_dict["debug_options"] or \
           "intercon" in hpcsim_dict["debug_options"] or \
           "fattree" in hpcsim_dict["debug_options"]: 
            print("fattree: num_ports_per_switch=%d" % self.num_ports_per_switch)
            print("fattree: num_levels=%d" % self.num_levels)
            print("fattree: switch_link_dups=%d" % self.switch_link_dups)
            print("fattree: switch_link_up_bdw=%f (bits per second)" % self.switch_link_up_bdw)
            print("fattree: switch_link_down_bdw=%f (bits per second)" % self.switch_link_down_bdw)
            print("fattree: switch_link_up_delay=%f (seconds)" % self.switch_link_up_delay)
            print("fattree: switch_link_down_delay=%f (seconds)" % self.switch_link_down_delay)
            print("fattree: host_link_bdw=%f (bits per second)" % self.host_link_bdw)
            print("fattree: host_link_delay=%f (seconds)" % self.host_link_delay)
            print("fattree: proc_delay=%f (seconds)" % self.proc_delay)
            print("fattree: mem_bandwidth=%f (bits per second)" % mem_bandwidth)
            print("fattree: mem_bufsz =%d (bytes)" % mem_bufsz)
            print("fattree: mem_delay=%f (seconds)" % mem_delay)
            print("fatree: route_method=%s" % self.route_method)

        # example self.switch_ids format:
        #   [str(level)+switch_label] = [switch_id, level_idx, [switch_label_list]]
        #   for a 4-port 3-tree: switch at second level with id "10" (i.e., <10, 2>)
        #   self.switch_ids["2:1:0"] = [14, 2, [1, 0]]
        self.switch_ids = dict()
        # example self.host_ids format:
        #   [host_label] = [host_id, [host_label]]
        #   for a 4-port 3-tree: processing node with id "101"
        #   self.host_ids["1:0:1"] = [5, [1, 0, 1]]
        self.host_ids = dict()
        # the following dictionary holds the host id lists against the actual host ids
        #   e.g., self.host_lists[5] = [[1, 0, 1]]
        # this dictionary is used to find the list for corresponding host id
        self.host_lists = dict() 
        # the following dictionary holds the grp members for the grp gcpg(x, alpha)
        #   e.g., self.grp_members_dict['0:0'] = [[0, 0, 0], [0, 0, 1]]
        self.grp_members_dict = dict()
        # add switches and hosts as entities
        simian = hpcsim_dict["simian"]
        
        # switches are denoted as switch id and level id pair;
        # specifically, SW<w=w_0w_1...w_{n-2}, l>, where l={0,1,...,n-1}
        # and w={0,1,...,(m/2)-1}^n-1 for l=0; w={0,1,...m-1}X{0,1,...,(m/2)-1}^n-2 otherwise 
        m = self.num_ports_per_switch
        n = self.num_levels
        swid = 0 
        for level_idx in xrange(n):
            if level_idx == 0:
                #{0,1,...,(m/2)-1}^n-1
                sid_temp = [list(xrange(0, m/2))]*(n-1)
                for sid_prod in itertools.product(*sid_temp):
                    sw_label = ':'.join(str(element) for element in sid_prod)
                    sw_label_list = []
                    for element in sid_prod:
                        sw_label_list.append(element)
                    # each switch is identified by a level id and switch label pair
                    self.switch_ids[str(level_idx)+":"+sw_label] = [swid, level_idx, sw_label_list]
                    swid += 1
            else:
                #{0,1,...,m-1}X{0,1,...,(m/2)-1}^n-2
                sid_part2 = [list(xrange(0, m/2))]*(n-2) 
                sid_temp = []; sid_temp.append(list(xrange(0, m)))
                for i in xrange(len(sid_part2)): sid_temp.append(sid_part2[i])
                for sid_prod in itertools.product(*sid_temp):
                    sw_label = ':'.join(str(element) for element in sid_prod)
                    sw_label_list = []
                    for element in sid_prod:
                        sw_label_list.append(element)
                    # each switch is identified by a level id and switch label pair
                    self.switch_ids[str(level_idx)+":"+sw_label] = [swid, level_idx, sw_label_list]
                    swid += 1
        
        # hosts are denoted as P(p=p_0p_1...p_{n-1})
        # where p = {0,1,...,m-1}X{0,1,...,m/2-1}^n-1
        hid = 0
        hid_part2 = [list(xrange(0, m/2))]*(n-1) 
        hid_temp = []; hid_temp.append(list(xrange(0, m)))
        for i in xrange(len(hid_part2)): hid_temp.append(hid_part2[i])
        for hid_prod in itertools.product(*hid_temp):
            host_id = ':'.join(str(element) for element in hid_prod)
            host_id_list = []
            for element in hid_prod:
                host_id_list.append(element)
            self.host_ids[host_id] = [hid, host_id_list]
            self.host_lists[hid] = host_id_list
            hid += 1
        if "id_lists" in hpcsim_dict["debug_options"]:
            print("Switch IDs:")
            for idx in sorted(self.switch_ids.iterkeys()):
                print("%s --> %r"%(idx, self.switch_ids[idx]))
            print("Host IDs:")
            for idx in sorted(self.host_lists.iterkeys()):
                print("%s --> %r"%(idx, self.host_lists[idx]))
        
        # initialize each switch as entity
        for key in sorted(self.switch_ids.iterkeys()):
            swid = self.switch_ids[key][0]
            level_idx = self.switch_ids[key][1]
            sw_label_list = self.switch_ids[key][2]
            simian.addEntity("Switch", InfinibandSwitch, swid, hpcsim_dict, self, level_idx, sw_label_list)
        
        # initialize each host as entity
        for host_key in sorted(self.host_lists.iterkeys()):  # just to sort the keys in ascending order
            hid = host_key
            # find the attached switch ID
            host_label = self.host_lists[hid] # this is a list denoting the host id label
            dest = dict()
            dest["level"] = n-1 # only last level's switches are connected to the nodes
            dest["port"] = host_label[n-1] # ports start indexing from 1, not 0 (ref: fat-tree, fig 11).
            sub_sid_2 = host_label[0:n-1]
            dest["s_id"] = ':'.join(str(e) for e in sub_sid_2)      
            swid = self.sw_label_to_id(dest["level"], dest["s_id"])

            h = dest["port"]
            simian.addEntity("Host", hpcsim.get_host_typename(hpcsim_dict), hid,
                    hpcsim_dict, self, swid, "h", 
                    h, self.host_link_bdw, self.bufsz,
                    self.host_link_delay,
                    mem_bandwidth, mem_bufsz, mem_delay)
            
            # create the groups and their members here
            for i in xrange(len(host_label)-1):
                x_list = []
                for x_len in xrange(i+1):
                    x_list.append(host_label[x_len])
                x = ':'.join(str(element) for element in x_list)
                if x not in self.grp_members_dict:
                    alpha = len(x_list)
                    grp_members = self.find_gcpg_members(x_list, alpha)
                    self.grp_members_dict[x] = grp_members
        # compute the total number of switches and hosts
        self.nhosts = 2*(m/2)**n
        self.nswitches = (2*n-1)*(m/2)**(n-1)
    
    def find_gcpg_members(self, x_list, alpha):
        """Returns all the processing nodes inside the grp gcpg(x_list, alpha)"""
        
        # Local variables
        #   x_list: greatest common prefix
        #   alpha: length of the greatest common prefix
        # NOTE: grp members are created through appending the cartesian products after x_list
        #   e.g., if x_list = [0, 0], host_lists = [[0, 0, 0], [0, 0, 1]]

        m = self.num_ports_per_switch
        n = self.num_levels
        host_lists = [] # final list containing all the hosts in the grp
        hid_cart = [list(xrange(0, m/2))]*(n-alpha) 
        hid_temp = []
        for i in xrange(len(hid_cart)): hid_temp.append(hid_cart[i])
        for hid_prod in itertools.product(*hid_temp):
            host_id_list = []
            for element in x_list:      # first, insert the x_list
                host_id_list.append(element)
            for element in hid_prod:    # next, append the cartesian products 
                host_id_list.append(element)
            host_lists.append(host_id_list)
        return host_lists

    def network_diameter(self):
        """Retruns the network diameter in hops."""

        # uplink traversals + downlink traversals + 2 hops to hosts
        return 2*self.num_levels + 2

    def network_diameter_time(self):
        """Retruns the network diameter in time."""
        
        return 2*self.host_link_delay+self.switch_link_up_delay*self.num_levels+ \
                self.switch_link_down_delay*self.num_levels

    def sw_label_to_id(self, lid, sw_label):
        """Returns the switch id"""
        
        # Local variables:
        #   lid: level of the switch
        #   sw_label: switch label (in string format)
        sw_id = self.switch_ids[str(lid)+":"+sw_label][0] 
        return sw_id
    
    def host_label_to_id(self, host_label):
        """Returns the host id"""
  
        # Local variables:
        #   host_label: host label (in string format)
        h_id = self.host_ids[host_label][0]
        return h_id 
    
    def host_id_to_label_list(self, hid):
        """Returns the host id (in list format)"""
       
        # Local variables:
        #   hid: host id
        host_list = self.host_lists[hid]
        return host_list

    # calculate the min delay value from config parameters (before the
    # interconnect model is instantiated!)
    @staticmethod
    def calc_min_delay(hpcsim_dict):
        if "fattree" not in hpcsim_dict:
            raise Exception("'fattree' must be specified for fattree interconnect")
        d1 = hpcsim_dict["fattree"].get("switch_link_up_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        d2 = hpcsim_dict["fattree"].get("switch_link_down_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        d3 = hpcsim_dict["fattree"].get("host_link_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        min_d = min([d1, d2, d3])
        return min_d

