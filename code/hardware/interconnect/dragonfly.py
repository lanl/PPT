#
# dragonfly.py :- a dragonfly interconnect; including aries
#

from intercon import *
import random

class DragonflySwitch(Switch):
    """A switch for the dragonfly and aries interconnect."""
    
    def __init__(self, baseinfo, hpcsim_dict, intercon, gid, swid):
        self.dragonfly = intercon
        self.gid = gid
        self.swid = swid
        self.hpcsim_dict = hpcsim_dict
        super(DragonflySwitch, self).__init__(baseinfo, hpcsim_dict, self.dragonfly.proc_delay)
        h = self.dragonfly.num_inter_links_per_switch
        i = self.swid
        j = self.gid
        # establish inter-group connections
        if self.dragonfly.inter_group_topology is 'consecutive':
            for k in xrange(h):
                peer_dict = self.connect_inter_link(j, i, k) # j:group id, i:switch id, k:port
                peer_gid = peer_dict["g_id"]
                peer_swid = peer_dict["s_id"]  # switch id inside group
                peer_port = peer_dict["p_id"]
                peer_id = peer_gid*self.dragonfly.num_switches_per_group+peer_swid 
                iface_name = 'g%d'%k
                peer_iface = 'g%d'%peer_port 
                peer_node_names = ("Switch",)*self.dragonfly.inter_link_dups
                peer_node_ids = (peer_id,)*self.dragonfly.inter_link_dups
                peer_iface_names = (peer_iface,)*self.dragonfly.inter_link_dups
                peer_iface_ports = tuple(range(self.dragonfly.inter_link_dups)) 
                self.interfaces[iface_name] = Interface(self, iface_name, self.dragonfly.inter_link_dups, 
                                                        peer_node_names, peer_node_ids, 
                                                        peer_iface_names, peer_iface_ports, 
                                                        self.dragonfly.inter_group_bdw, 
                                                        self.dragonfly.bufsz, 
                                                        self.dragonfly.inter_group_delay)
                if "switch.connect" in self.hpcsim_dict["debug_options"]:    
                    print("%s %s connects to switch %r iface %r ports %r" % 
                          (self, self.interfaces[iface_name], peer_node_ids, 
                           peer_iface_names, peer_iface_ports))
        # the inter-links are grouped together in such connection.
        elif self.dragonfly.inter_group_topology is 'consecutive_aries':
            m = self.dragonfly.num_inter_links_grouped
            for k in xrange(h):
                for l in xrange(m): # for individual cable (which are bundled together) 
                    peer_dict = self.connect_inter_link(j, i, k, True, l) # j:grp_id, i:sw_id, k:port 
                    peer_gid = peer_dict["g_id"]
                    peer_swid = peer_dict["s_id"]  # switch id inside group
                    peer_port = peer_dict["p_id"]
                    peer_id = peer_gid*self.dragonfly.num_switches_per_group+peer_swid 
                    iface_name = 'g%d'%(k*m+l)
                    peer_iface = 'g%d'%peer_port 
                    peer_node_names = ("Switch",)*self.dragonfly.inter_link_dups
                    peer_node_ids = (peer_id,)*self.dragonfly.inter_link_dups
                    peer_iface_names = (peer_iface,)*self.dragonfly.inter_link_dups
                    peer_iface_ports = tuple(range(self.dragonfly.inter_link_dups)) 
                    self.interfaces[iface_name] = Interface(self, iface_name, self.dragonfly.inter_link_dups, 
                                                            peer_node_names, peer_node_ids, 
                                                            peer_iface_names, peer_iface_ports, 
                                                            self.dragonfly.inter_group_bdw, 
                                                            self.dragonfly.bufsz, 
                                                            self.dragonfly.inter_group_delay)
                    if "switch.connect" in self.hpcsim_dict["debug_options"]:    
                        print("%s %s connects to switch %r iface %r ports %r" % 
                              (self, self.interfaces[iface_name], peer_node_ids, 
                               peer_iface_names, peer_iface_ports))
        # establish intra-group connections
        if self.dragonfly.intra_group_topology == 'all_to_all':
            h = self.dragonfly.num_intra_links_per_switch
            for k in xrange(h):
                peer_dict = self.connect_intra_link(i, k) # i:switch id, k:port
                peer_gid = self.gid
                peer_swid = peer_dict["s_id"]
                peer_port = peer_dict["p_id"]
                peer_id = peer_gid*self.dragonfly.num_switches_per_group+peer_swid 
                iface_name = 'l%d'%k    # l: local   
                peer_iface = 'l%d'%peer_port
                peer_node_names = ("Switch",)*self.dragonfly.intra_link_dups
                peer_node_ids = (peer_id,)*self.dragonfly.intra_link_dups
                peer_iface_names = (peer_iface,)*self.dragonfly.intra_link_dups
                peer_iface_ports = tuple(range(self.dragonfly.intra_link_dups)) 
                self.interfaces[iface_name] = Interface(self, iface_name, self.dragonfly.intra_link_dups, 
                                                        peer_node_names, peer_node_ids, 
                                                        peer_iface_names, peer_iface_ports, 
                                                        self.dragonfly.intra_group_bdw, 
                                                        self.dragonfly.bufsz, 
                                                        self.dragonfly.intra_group_delay)
                if "switch.connect" in self.hpcsim_dict["debug_options"]:    
                    print("%s %s connects to switch %r iface %r ports %r" % 
                          (self, self.interfaces[iface_name], peer_node_ids, 
                           peer_iface_names, peer_iface_ports))
        elif self.dragonfly.intra_group_topology == 'cascade':
            p = self.dragonfly.num_intra_links_for_blades
            q = self.dragonfly.num_intra_links_for_chassis
            m = self.dragonfly.num_intra_links_grouped
            # intra-chassis:              
            for k in xrange(p):
                peer_dict = self.connect_intra_link(i, k, True) # i:switch id, k:port, True: cascade
                peer_gid = self.gid
                peer_swid = peer_dict["s_id"]
                peer_port = peer_dict["p_id"]
                peer_id = peer_gid*self.dragonfly.num_switches_per_group+peer_swid 
                iface_name = 'l%d'%k    # l: local   
                peer_iface = 'l%d'%peer_port
                peer_node_names = ("Switch",)*self.dragonfly.intra_chassis_dups
                peer_node_ids = (peer_id,)*self.dragonfly.intra_chassis_dups
                peer_iface_names = (peer_iface,)*self.dragonfly.intra_chassis_dups
                peer_iface_ports = tuple(range(self.dragonfly.intra_chassis_dups)) 
                self.interfaces[iface_name] = Interface(self, iface_name, self.dragonfly.intra_chassis_dups, 
                                                        peer_node_names, peer_node_ids, 
                                                        peer_iface_names, peer_iface_ports, 
                                                        self.dragonfly.intra_chassis_bdw, 
                                                        self.dragonfly.bufsz, 
                                                        self.dragonfly.intra_chassis_delay)
                if "switch.connect" in self.hpcsim_dict["debug_options"]:    
                    print("%s %s connects to switch %r iface %r ports %r" % 
                          (self, self.interfaces[iface_name], peer_node_ids, 
                           peer_iface_names, peer_iface_ports))
            # inter-chassis: 
            for k in xrange(p, p+q):
                for l in xrange(m):
                    peer_dict = self.connect_intra_link(i, k, True, l) # i:switch id, k:port, True: cascade
                    peer_gid = self.gid
                    peer_swid = peer_dict["s_id"]
                    peer_port = peer_dict["p_id"]
                    peer_id = peer_gid*self.dragonfly.num_switches_per_group+peer_swid 
                    iface_name = 'l%d'%(p+(k-p)*m+l) # l: local   
                    peer_iface = 'l%d'%peer_port
                    peer_node_names = ("Switch",)*self.dragonfly.inter_chassis_dups
                    peer_node_ids = (peer_id,)*self.dragonfly.inter_chassis_dups
                    peer_iface_names = (peer_iface,)*self.dragonfly.inter_chassis_dups
                    peer_iface_ports = tuple(range(self.dragonfly.inter_chassis_dups)) 
                    self.interfaces[iface_name] = Interface(self, iface_name, self.dragonfly.inter_chassis_dups, 
                                                            peer_node_names, peer_node_ids, 
                                                            peer_iface_names, peer_iface_ports, 
                                                            self.dragonfly.inter_chassis_bdw, 
                                                            self.dragonfly.bufsz, 
                                                            self.dragonfly.inter_chassis_delay)
                    if "switch.connect" in self.hpcsim_dict["debug_options"]:    
                        print("%s %s connects to switch %r iface %r ports %r" % 
                              (self, self.interfaces[iface_name], peer_node_ids, 
                               peer_iface_names, peer_iface_ports))
        # connect to hosts
        dir = "h"
        n = self.dragonfly.num_hosts_per_switch
        s = self.dragonfly.num_switches_per_group
        peer_node_names = ("Host",)*n 
        peer_node_ids = []
        for p in xrange(n):
            hid = self.gid*s*n + self.swid*n + p 
            peer_node_ids.append(hid)
        peer_iface_names = ("r",)*n
        peer_iface_ports = (0,)*n
        self.interfaces[dir] = Interface(self, dir, n, 
                                         peer_node_names, tuple(peer_node_ids),
                                         peer_iface_names, peer_iface_ports, 
                                         self.dragonfly.switch_host_bdw, 
                                         self.dragonfly.bufsz,
                                         self.dragonfly.switch_host_delay)
        if "switch.connect" in self.hpcsim_dict["debug_options"]:
            print("%s %s connects to host %r iface %r ports %r" %
                  (self, self.interfaces[dir], tuple(peer_node_ids),
                   peer_iface_names, peer_iface_ports))

    def __str__(self):
        return "switch[%d](%d,%d)"%(self.gid*self.dragonfly.num_switches_per_group+self.swid,
                self.gid, self.swid)

    def connect_inter_link(self, gid, sid, port, aries = False, l = None):
        """Returns inter-link connection info (grp id, switch id, port)"""
        
        # Local variables:
        #   gid: grp id of the switch, sid: id of the switch, port: id of the port
        # Optional local variables:
        #   aries: default (False) is dragonfly; True is aries (multiple cables bundled together) 
        #   l: default (None) is dragonfly; otherwise value represents # of cables bundled together         
        # According to "Topological Characterization of Hamming
        # and Dragonfly Networks and Routing" paper, by Cristobal
        # Camarero, Enrique Vallejo, and Ramon Beivide, ACM TACO,
        # 11(4), 2015, "The consecutive allocation of global links
        # consists of connecting the routers in each group in
        # consecutive order, with the groups in the network also
        # in consecutive order, starting always from group 0 and
        # skipping those links with source and destination being
        # in the same group. Specifically, the vertex i in group j
        # is connected for every integer k = 0, ... , h-1 with the
        # vertex floor((j-1)/h) of the group g=i*h+k if g<j and
        # with the vertex floor(j/h) of the group g+1 otherwise",
        # where h is the number of inter-group links at each
        # switch.
        if(gid > self.dragonfly.num_groups - 1):
            raise Exception("Group id should be between 0 and num_of_group-1.")
        if(sid > self.dragonfly.num_switches_per_group - 1):
            raise Exception("Switch id should be between 0 and num_of_switches_per_group-1.")
        if(port > self.dragonfly.num_inter_links_per_switch - 1):
            raise Exception("Port id should be between 0 and num_of_global_links_per_switch-1.")
        dest = dict()
        h = self.dragonfly.num_inter_links_per_switch
        g = sid*h+port
        if g >= gid:
            dest["g_id"] = g+1
            dest["s_id"] = int(gid/h)
            if aries == False: 
                dest["p_id"] = gid-int(gid/h)*h
            else: # consecutive Aries connection
                m = self.dragonfly.num_inter_links_grouped
                dest["p_id"] = (gid-int(gid/h)*h)*m+l
        else:
            dest["g_id"] = g
            dest["s_id"] = int((gid-1)/h)
            if aries == False:
                dest["p_id"] = (gid-1)-int((gid-1)/h)*h
            else: # consecutive Aries connection  
                m = self.dragonfly.num_inter_links_grouped
                dest["p_id"] = ((gid-1)-int((gid-1)/h)*h)*m+l
        return dest

    def connect_intra_link(self, sid, port, cascade = False, l = None):
        """Returns intra-link connection info (switch id, port)"""

        # Local variables:
        #   sid: id of the switch, port: id of the port
        #   cascade: type of intra-connect, default (false) dragonfly topology, true=cascade
        # Optional local variables:
        #   cascade: default (False) is dragonfly; True is cascade (multiple cables bundled together) 
        #   l: default (None) is dragonfly; otherwise value represents # of cables bundled together         
        if(sid > self.dragonfly.num_switches_per_group - 1):
            raise Exception("Switch id should be between 0 and num_of_switches_per_group-1.")
        if cascade == False:
            if(port > self.dragonfly.num_intra_links_per_switch - 1):
                raise Exception("Port id should be between 0 and num_of_intra_links_per_switch-1.")
        else: # cascade
            if(port > (self.dragonfly.num_intra_links_for_blades+ \
                    self.dragonfly.num_intra_links_for_chassis-1)):
                raise Exception("Port id should be between 0 and num_of_intra_links_per_switch-1.")

        dest = dict()
        if cascade == False:
            if port >= sid:
                dest["s_id"] = port+1
                dest["p_id"] = sid
            else:
                dest["s_id"] = port
                dest["p_id"] = sid-1
        else:   # cascade intra-connect
            # Note: blades within a chassis form a subgroup
            n_b = self.dragonfly.num_blades_per_chassis 
            l_b = self.dragonfly.num_intra_links_for_blades# num_of_links for diff blades among a chassis
            sid_rel = sid%n_b   # relative switch id within the subgroup
            sg_id = sid/n_b     # subgroup id where the switch is located 
            m = self.dragonfly.num_intra_links_grouped
            if port < l_b:      # "all_to_all" connection among blades for same chassis
                if port >= sid_rel:
                    dest_sid_rel = port+1   # destination "relative" switch id within subgroup
                    dest["p_id"] = sid_rel
                else:
                    dest_sid_rel = port
                    dest["p_id"] = sid_rel-1
                dest["s_id"] = sg_id*n_b+dest_sid_rel
            else:   # connection to other chassis in same group
                u_or_d = port - l_b # to decide whether the link is going up or down.
                if u_or_d >= sg_id: # link going down
                    dest["s_id"] = (u_or_d+1)*n_b+sid_rel
                    dest["p_id"] = sg_id*m+l_b+l
                else:   # link going up
                    dest["s_id"] = u_or_d*n_b+sid_rel
                    dest["p_id"] = (sg_id-1)*m+l_b+l
        return dest
    
    def min_forward(self, src_gid, src_sid, dest_gid, dest_sid):
        """Returns interface for minimal (MIN) routing"""

        # Local vairables    
        #   src_gid: current group id, src_sid: current switch id 
        #   dest_gid: destination group id, dest_sid: destination switch id
        num_switch = self.dragonfly.num_switches_per_group
        num_inter_link = self.dragonfly.num_inter_links_per_switch
        if src_gid == dest_gid:
            # step 1: route within destination group
            if src_sid < dest_sid: 
                port = (dest_sid%num_switch)-1
            else:
                port = dest_sid%num_switch
            return 'l%d'%port  
        else:
            if src_gid > dest_gid:
                grp_output = dest_gid
            else:
                grp_output = dest_gid-1
            grp_rid = grp_output/num_inter_link 
            if grp_rid == src_sid:  # using global connection
                # step 2: route among different groups (i.e., optical links)
                port = grp_output%num_inter_link
                return 'g%d'%port
            else:
                # step 3: route within current group
                if src_sid < grp_rid:
                    port = (grp_rid%num_switch)-1
                else:
                    port = grp_rid%num_switch
                return 'l%d'%port
        #raise Exception("No port could be found during MIN routing")
    
    def non_min_forward(self, src_gid, src_sid, int_gid, dest_gid, dest_sid): 
        """Returns interface for non_minimal (VAL) routing"""

        # Local vairables    
        #   src_gid: current group id, src_sid: current switch id
        #   int_gid: intermediate group id
        #   dest_gid: destination group id, dest_sid: destination switch id
        num_switch = self.dragonfly.num_switches_per_group
        num_inter_link = self.dragonfly.num_inter_links_per_switch
        if src_gid == dest_gid:
            # step 1: route within destination group
            if src_sid < dest_sid: 
                port = (dest_sid%num_switch)-1
            else:
                port = dest_sid%num_switch
            return 'l%d'%port  
        # steps 2 & 3: the packet is at the intermediate group
        elif src_gid == int_gid:
            if src_gid > dest_gid:
                grp_output = dest_gid   # destination group id?
            else:
                grp_output = dest_gid-1
            grp_rid = grp_output/num_inter_link # router id within destination group?  
            if grp_rid == src_sid:  # using global connection
                # step 2: route among different groups
                port = grp_output%num_inter_link
                return 'g%d'%port
            else:
                # step 3: route within intermediate group
                if src_sid < grp_rid:
                    port = (grp_rid%num_switch)-1
                else:
                    port = grp_rid%num_switch
                return 'l%d'%port
        # steps 4 & 5: the packet is neither at intermediate group nor at destination group  
        else:
            if src_gid > int_gid:
                grp_output = int_gid   # intermediate group id?
            else:
                grp_output = int_gid-1
            grp_rid = grp_output/num_inter_link # router id within intermediate group?  
            if grp_rid == src_sid:  # using global connection
                # step 4: route among different groups
                port = grp_output%num_inter_link
                return 'g%d'%port
            else:
                # step 5: route within curernt group
                if src_sid < grp_rid:
                    port = (grp_rid%num_switch)-1
                else:
                    port = grp_rid%num_switch
                return 'l%d'%port
        #raise Exception("No port could be found during VAL routing")
    
    # following implements minimal routing inside a group 
    # (source grp/dest grp/intermediate grp) for cascade connection
    def route_inside_grp_min(self, cur_sid, dest_sid):
        """Returns interface for minimal (MIN) routing inside a grp (cascade connection)."""
        
        # Local vairables    
        #   cur_sid: current switch id, dest_sid: destination switch
        n_b = self.dragonfly.num_blades_per_chassis 
        l_b = self.dragonfly.num_intra_links_for_blades # num_of_links for diff blades among a chassis
        cur_sid_rel = cur_sid%n_b       # src relative switch id within the subgroup
        cur_sgid = cur_sid/n_b          # src subgrp no. where the switch is located
        dest_sid_rel = dest_sid%n_b     # dest relative switch id within the subgroup
        dest_sgid = dest_sid/n_b        # dest subgrp no. where the switch is located
        # first, check if the switches are in the same column 
        # (i.e., directly reachable through chassis connection/vertical connection)
        if cur_sid%n_b == dest_sid%n_b:
            # step 1a. route among different chassis.
            # equivalent to routing "globally" among chassis
            if cur_sgid > dest_sgid:
                subgrp_output = dest_sgid
            else:
                subgrp_output = dest_sgid-1
            port = subgrp_output + l_b 
            # select a random port from "bundled" intra-links:
            m = self.dragonfly.num_intra_links_grouped
            port_range = [l_b+(port-l_b)*m, l_b+(port-l_b)*m+m-1]
            port_rand = self.find_port_aries(port_range)
            port = port_rand
            return port  
        # next, check if the switches are in the same row
        # (i.e., directly reachable through blade connection/horizontal connection)
        elif cur_sid/n_b == dest_sid/n_b:
            # step 1b. route among blades.
            # equivalent to routing "locally" among blades in a chassis 
            if cur_sid_rel < dest_sid_rel:
                port = dest_sid_rel%n_b-1
            else:
                port = dest_sid_rel%n_b
            return port  
        else:
            # step 1c. go to router "M" to reach destination or intermediate router.
            dest_sid_temp = cur_sgid*n_b+dest_sid_rel  # switch id at intersection of src and dest switch     
            dest_sid_temp_rel = dest_sid_temp%n_b      # dest relative switch id within the subgroup
            if cur_sid < dest_sid_temp: 
                port = dest_sid_temp_rel%n_b-1
            else:
                port = dest_sid_temp_rel%n_b
            return port  
        #raise Exception("No port could be found during MIN routing inside grp for Aries")
   
    def find_port_aries(self, port_range):
        """Returns a randomly selected port for inter- or intra-link communication in Aries"""

        # Local variable
        #   port_range: the port id range, from which a random port number is selected
        a = port_range[0]
        b = port_range[1]
        return random.randint(a, b)

    def min_forward_cascade(self, dest_gid, dest_sid):
        """Returns interface for minimal (MIN) routing (cascade connection)."""
        # The routing algorithm is at: "Cray XC30 System: Overview" by Nathan Wichmann(slide 20).
        # Routing algorithm inside the group is also "adaptive".
        # Chooses between two minimal (global and local) and two non_minimal (global and local) paths.
        
        # Local vairables    
        #   dest_gid: destination group id, dest_sid: destination switch id
        cur_gid = self.gid; cur_sid = self.swid
        k = self.dragonfly.num_inter_links_per_switch
        # step 1: route within destination group
        if cur_gid == dest_gid:
            port = self.route_inside_grp_min(cur_sid, dest_sid)
            return 'l%d'%port  
        else:
            if cur_gid > dest_gid:
                grp_output = dest_gid
            else:
                grp_output = dest_gid-1
            grp_rid = (grp_output/k)
            if grp_rid == cur_sid:
                # step 2: route between groups (i.e., optical links)
                port = grp_output%k
                if self.dragonfly.inter_group_topology is 'consecutive_aries':
                    # select "randomly" among multiple ports to reach desired destination (Aries property)
                    m = self.dragonfly.num_inter_links_grouped
                    port_range = [port*m, port*m+m-1]
                    port_rand = self.find_port_aries(port_range)
                    port = port_rand
                return 'g%d'%port
            else:
                # step 3: route within source group
                port = self.route_inside_grp_min(cur_sid, grp_rid)
                return 'l%d'%port  
        #raise Exception("No port could be found during MIN routing for Aries")
    
    def route_inside_grp_non_min(self, src_sid, int_sid, dest_sid, first_time):
        """Returns interface for non_minimal (VAL) routing inside a grp (cascade connection)."""
        
        # Local vairables    
        #   src_sid: source switch id, dest_sid: destination switch,int_sid: intermediate switch id.
        #   first_time: denotes whether this is the first time packet is inside source switch of 
        #   source/intermediate/dest grp. This info is used to avoid intermediate switch
        #   resending to the source switch. 
        n_b = self.dragonfly.num_blades_per_chassis 
        l_b = self.dragonfly.num_intra_links_for_blades # num_of_links for diff blades among a chassis
        src_sid_rel = src_sid%n_b       # src relative switch id within the subgroup
        src_sgid = src_sid/n_b          # src subgrp no. where the switch is located
        int_sid_rel = int_sid%n_b       # intermediate relative switch id within the subgroup
        int_sgid = int_sid/n_b          # intermediate subgrp no. where the int switch is located
        cur_sid_rel = self.swid%n_b     # current relative switch id within the subgroup
        cur_sgid = self.swid/n_b        # current subgrp no. where the int switch is located
        dest_sid_rel = dest_sid%n_b     # dest relative switch id within the subgroup
        dest_sgid = dest_sid/n_b        # dest subgrp no. where the switch is located
        m = self.dragonfly.num_intra_links_grouped

        # case#1: current switch id is the source switch id;
        # then route to the intermediate rotuer
        if self.swid == src_sid and first_time == True:
            # first, check if the switches are in the same column 
            # (i.e., directly reachable through chassis connection/vertical connection)
            if src_sid%n_b == int_sid%n_b:
                # step 1a. route among different chassis.
                # equivalent to routing "globally" among chassis
                if src_sgid > int_sgid:
                    subgrp_output = int_sgid
                else:
                    subgrp_output = int_sgid-1
                port = subgrp_output + l_b
                # select a random port from "bundled" intra-links:
                port_range = [l_b+(port-l_b)*m, l_b+(port-l_b)*m+m-1]
                port_rand = self.find_port_aries(port_range)
                port = port_rand
                return port  
            # next, check if the switches are in the same row
            # (i.e., directly reachable through blade connection/horizontal connection)
            elif src_sid/n_b == int_sid/n_b:
                # step 1b. route among blades.
                # equivalent to routing "locally" among blades in a chassis 
                if src_sid_rel < int_sid_rel:
                    port = int_sid_rel%n_b-1
                else:
                    port = int_sid_rel%n_b
                return port  
            else:
                # step 1c. go to router "M" to reach intermediate router.
                int_sid_temp = src_sgid*n_b+int_sid_rel  # switch id at intersection of source and dest switch    
                int_sid_temp_rel = int_sid_temp%n_b      # dest relative switch id within the subgroup
                if src_sid < int_sid_temp: 
                    port = int_sid_temp_rel%n_b-1
                else:
                    port = int_sid_temp_rel%n_b
                return port  
        # case#2: current switch id is the intermediate switch id;
        # then route to the destination router
        elif self.swid == int_sid:
            # first, check if the switches are in the same column 
            # (i.e., directly reachable through chassis connection/vertical connection)
            if int_sid%n_b == dest_sid%n_b:
                # step 1a. route among different chassis.
                # equivalent to routing "globally" among chassis
                if int_sgid > dest_sgid:
                    subgrp_output = dest_sgid
                else:
                    subgrp_output = dest_sgid-1
                port = subgrp_output + l_b 
                # select a random port from "bundled" intra-links:
                port_range = [l_b+(port-l_b)*m, l_b+(port-l_b)*m+m-1]
                port_rand = self.find_port_aries(port_range)
                port = port_rand
                return port  
            # next, check if the switches are in the same row
            # (i.e., directly reachable through blade connection/horizontal connection)
            elif int_sid/n_b == dest_sid/n_b:
                # step 1b. route among blades.
                # equivalent to routing "locally" among blades in a chassis 
                if int_sid_rel < dest_sid_rel:
                    port = dest_sid_rel%n_b-1
                else:
                    port = dest_sid_rel%n_b
                return port  
            else:
                # step 1c. go to router "M" to reach destination router.
                dest_sid_temp = int_sgid*n_b+dest_sid_rel  # switch id at intersection of source and dest switch    
                dest_sid_temp_rel = dest_sid_temp%n_b      # dest relative switch id within the subgroup
                if int_sid < dest_sid_temp: 
                    port = dest_sid_temp_rel%n_b-1
                else:
                    port = dest_sid_temp_rel%n_b
                return port  
        # case#3: current switch id is neither source switch id 
        # nor intermediate switch id
        else:
            # current switch is "M1"; i.e., "M1" and "intermediate" in same column,
            # "M1" and "source" in same row
            if self.swid%n_b == int_sid%n_b and self.swid/n_b == src_sid/n_b:
                # route vertically to the intermediate switch
                if cur_sgid > int_sgid:
                    subgrp_output = int_sgid
                else:
                    subgrp_output = int_sgid-1
                port = subgrp_output + l_b 
                # select a random port from "bundled" intra-links:
                port_range = [l_b+(port-l_b)*m, l_b+(port-l_b)*m+m-1]
                port_rand = self.find_port_aries(port_range)
                port = port_rand
                return port  
            # current switch is "M2"; i.e., "M2" and "destination" in same column,
            # "M2" and "source" in same row
            if self.swid%n_b == dest_sid%n_b and self.swid/n_b == int_sid/n_b:
                # route vertically to the intermediate switch
                if cur_sgid > dest_sgid:
                    subgrp_output = dest_sgid
                else:
                    subgrp_output = dest_sgid-1
                port = subgrp_output + l_b 
                # select a random port from "bundled" intra-links:
                port_range = [l_b+(port-l_b)*m, l_b+(port-l_b)*m+m-1]
                port_rand = self.find_port_aries(port_range)
                port = port_rand
                return port
        #raise Exception("No port could be found during VAL routing inside grp for Aries")

    def non_min_forward_cascade(self, src_gid, src_sid, int_gid, int_sid, dest_gid, dest_sid, first_time):
        """Returns interface for non_minimal (VAL) routing (cascade connection). 
        Also, returns next source switch id for both intermediate and destination grp."""
        # The routing algorithm is at: "Cray XC30 System: Overview" by Nathan Wichmann (slide 20).
        # NOTE: src_sid is different for the followings: source grp, intermediate grp, dest grp   
        
        # Local variable
        #   src_gid: source group id, src_sid: source switch id 
        #   dest_gid: destination group id, dest_sid: destination switch id
        #   int_gid: intermediate group id, int_sid: int switch id
        #   first_time: the variable described in "route_inside_grp_non_min" method
        k = self.dragonfly.num_inter_links_per_switch
        m = self.dragonfly.num_inter_links_grouped
        # step 1: route within destination group
        if self.gid == dest_gid: 
            port = self.route_inside_grp_non_min(src_sid, int_sid, dest_sid, first_time)
            return 'l%d'%port, None
        # steps 2 & 3: the packet is at the intermediate group
        elif self.gid == int_gid:
            if self.gid > dest_gid:
                grp_output = dest_gid
            else:
                grp_output = dest_gid-1
            grp_rid = grp_output/k  
            if grp_rid == self.swid:  # using global connection
                # step 2: route among different groups
                port = grp_output%k
                dest = self.connect_inter_link(self.gid, self.swid, port) # new src info for dest grp 
                if self.dragonfly.inter_group_topology is 'consecutive_aries':
                    # select "randomly" among multiple ports to reach desired destination
                    port_range = [port*m, port*m+m-1]
                    port_rand = self.find_port_aries(port_range)
                    dest = self.connect_inter_link(self.gid, self.swid, port, True, port_rand-port*m)
                    port = port_rand
                return 'g%d'%port, dest["s_id"]
            else:
                # step 3: route within intermediate group
                port = self.route_inside_grp_non_min(src_sid, int_sid, grp_rid, first_time)
                return 'l%d'%port, None
        # steps 4 & 5: the packet is at the source group
        else:
            if self.gid > int_gid:
                grp_output = int_gid
            else:
                grp_output = int_gid-1
            grp_rid = grp_output/k  
            if grp_rid == self.swid:  # using global connection
                # step 4: route among different groups
                port = grp_output%k
                # select "randomly" among multiple ports to reach desired destination (Aries property)
                port_range = [port*m, port*m+m-1]
                port_rand = self.find_port_aries(port_range)
                dest = self.connect_inter_link(self.gid, self.swid, port, True, port_rand-port*m)
                return 'g%d'%port_rand, dest["s_id"]
            else:
                # step 5: route within source group
                port = self.route_inside_grp_non_min(src_sid, int_sid, grp_rid, first_time)
                return 'l%d'%port, None
        #raise Exception("No port could be found during VAL routing for Aries")
            
    def calc_route(self, pkt): 
        """Returns interface name and port number"""
        
        h = self.dragonfly.num_hosts_per_group
        k = self.dragonfly.num_hosts_per_switch
        intra_grp_topo = self.dragonfly.intra_group_topology
        route_method = self.dragonfly.route_method
        # find destination group and switch id from packet destination info
        dest_gid = int(pkt.dsthost/h)
        dest_sid = (pkt.dsthost%h)/k
        # find source group and switch id from packet source info
        src_gid = int(pkt.srchost/h)
        src_sid = (pkt.srchost%h)/k
        
        if route_method is "minimal" and intra_grp_topo is "all_to_all":
            if self.gid == dest_gid and self.swid == dest_sid:# packet reached dest switch
                port = self.dragonfly.hid_to_port(pkt.dsthost)
                return "h", port 
            else: 
                md = self.min_forward(self.gid, self.swid, dest_gid, dest_sid) 
                m = self.interfaces[md].get_num_ports()
                port = random.randint(0, m-1)
                return md, port
        elif route_method is "non_minimal" and intra_grp_topo is 'all_to_all': 
            if pkt.type[:4] == 'data' and "int_gid" not in pkt.nonreturn_data:# insert int grp id
                    group_ids = list(range(0, self.dragonfly.num_groups))
                    group_ids.remove(src_gid)
                    if dest_gid in group_ids: group_ids.remove(dest_gid)
                    pkt.nonreturn_data["int_gid"] = random.choice(group_ids)
            if self.gid == dest_gid and self.swid == dest_sid:# packet reached dest switch
                port = self.dragonfly.hid_to_port(pkt.dsthost)
                return "h", port 
            else: 
                if pkt.type[:4] == 'data':
                    int_gid = pkt.nonreturn_data["int_gid"]
                    md = self.non_min_forward(self.gid, self.swid, int_gid, dest_gid, dest_sid) 
                    m = self.interfaces[md].get_num_ports()
                    port = random.randint(0, m-1) 
                    return md, port
                else:   # ACK
                    md = self.min_forward(self.gid, self.swid, dest_gid, dest_sid) 
                    m = self.interfaces[md].get_num_ports()
                    port = random.randint(0, m-1)
                    return md, port
        elif route_method is "minimal" and intra_grp_topo == "cascade":
            if self.gid == dest_gid and self.swid == dest_sid:# packet reached dest switch
                port = self.dragonfly.hid_to_port(pkt.dsthost)
                #TODO: change it later ("debug_options")
                if pkt.type[:4] == 'data':
                    if "calculate_hops" in self.hpcsim_dict["debug_options"]:
                        pass
                        #print("#hops: %d"%pkt.nonreturn_data["num_hops"])   
                return "h", port 
            else: 
                md = self.min_forward_cascade(dest_gid, dest_sid) 
                m = self.interfaces[md].get_num_ports()
                port = random.randint(0, m-1) 
                return md, port
        elif route_method is "non_minimal" and intra_grp_topo == 'cascade': 
            ### insert intermediate grp id
            if pkt.type[:4] == 'data' and "int_gid" not in pkt.nonreturn_data:
                grp_ids = list(range(0, self.dragonfly.num_groups))
                grp_ids.remove(src_gid)
                if dest_gid in grp_ids: grp_ids.remove(dest_gid)
                if not grp_ids:
                    raise Exception("# of groups insufficient to choose a random intermediate grp.")
                pkt.nonreturn_data["int_gid"] = random.choice(grp_ids) 
            
            ### find the intermediate switch id for source group
            if pkt.type[:4] == 'data' and self.gid == src_gid and \
                    "sid_src_grp" not in pkt.nonreturn_data:
                switch_ids = list(range(0, self.dragonfly.num_switches_per_group))
                switch_ids.remove(src_sid) 
                pkt.nonreturn_data["sid_src_grp"] = random.choice(switch_ids)
            
            ### find the intermediate switch id for intermediate group
            if pkt.type[:4] == 'data' and self.gid == pkt.nonreturn_data["int_gid"] and \
                    "sid_int_grp" not in pkt.nonreturn_data:
                switch_ids = list(range(0, self.dragonfly.num_switches_per_group))
                switch_ids.remove(self.swid) # removing the source switch inside intermediate grp
                pkt.nonreturn_data["sid_int_grp"] = random.choice(switch_ids)
            
            ### find the intermediate switch id for destination group
            if pkt.type[:4] == 'data' and self.gid == dest_gid and \
                    "sid_dest_grp" not in pkt.nonreturn_data:
                switch_ids = list(range(0, self.dragonfly.num_switches_per_group))
                switch_ids.remove(self.swid) # removing the source switch inside destination grp
                pkt.nonreturn_data["sid_dest_grp"] = random.choice(switch_ids)
            
            if self.gid == dest_gid and self.swid == dest_sid:# packet reached dest switch
                port = self.dragonfly.hid_to_port(pkt.dsthost)
                return "h", port 
            else: 
                if pkt.type[:4] == 'data':
                    int_gid = pkt.nonreturn_data["int_gid"]
                    # Find out whether this is the first time a packet is inside the 
                    # src/intermediate/dest grp. If yes, use that information (i.e., 
                    # value of first_time) to avoid intermediate switch resending the 
                    # packet to the source switch.
                    first_time = False
                    if self.gid == src_gid and \
                            "inside_src_grp" not in pkt.nonreturn_data:     # first_time_in_src_grp
                        pkt.nonreturn_data["inside_src_grp"] = 1
                    elif self.gid == int_gid and \
                            "inside_int_grp" not in pkt.nonreturn_data:     # first_time_in_int_grp
                        pkt.nonreturn_data["inside_int_grp"] = 1
                    elif self.gid == dest_gid and \
                            "inside_dest_grp" not in pkt.nonreturn_data:    # first_time_in_dest_grp 
                        pkt.nonreturn_data["inside_dest_grp"] = 1
                    
                    if "inside_src_grp" in pkt.nonreturn_data and \
                            pkt.nonreturn_data["inside_src_grp"] == 1:
                        first_time = True
                        pkt.nonreturn_data["inside_src_grp"] = 0
                    elif "inside_int_grp" in pkt.nonreturn_data and \
                            pkt.nonreturn_data["inside_int_grp"] == 1:
                        first_time = True
                        pkt.nonreturn_data["inside_int_grp"] = 0
                    elif "inside_dest_grp" in pkt.nonreturn_data and \
                            pkt.nonreturn_data["inside_dest_grp"] == 1:
                        first_time = True
                        pkt.nonreturn_data["inside_dest_grp"] = 0
                    
                    #random switch for each of the group (source, intermediate and destination)
                    if self.gid == src_gid:
                        int_sid = pkt.nonreturn_data["sid_src_grp"]
                    elif self.gid == dest_gid:
                        int_sid = pkt.nonreturn_data["sid_dest_grp"]
                    elif self.gid == int_gid:
                        int_sid = pkt.nonreturn_data["sid_int_grp"]
                    else:
                        raise Exception("The packet can't be outside src, dest or intermediate grp.")
                    
                    # "src_sid" denotes the source switch for both intermediate and dest grp.
                    if "src_sid" not in pkt.nonreturn_data: 
                        md, new_src_sid = self.non_min_forward_cascade(src_gid, src_sid, 
                                int_gid, int_sid, dest_gid, dest_sid, first_time) 
                    else:
                        src_sid = pkt.nonreturn_data["src_sid"]
                        md, new_src_sid = self.non_min_forward_cascade(src_gid, src_sid, 
                                int_gid, int_sid, dest_gid, dest_sid, first_time)
                    if new_src_sid != None:    
                        pkt.nonreturn_data["src_sid"] = new_src_sid
                    
                    m = self.interfaces[md].get_num_ports()
                    port = random.randint(0, m-1)
                    return md, port
                else:   # ACK
                    md = self.min_forward_cascade(dest_gid, dest_sid)
                    m = self.interfaces[md].get_num_ports()
                    port = random.randint(0, m-1)
                    return md, port
        else:
            raise Exception("route method %s has not been implemented yet" % self.dragonfly.route_method)

class Dragonfly(Interconnect):
    """A generic dragonfly network."""

    def __init__(self, hpcsim, hpcsim_dict):
        super(Dragonfly, self).__init__(hpcsim_dict)

        if "dragonfly" not in hpcsim_dict:
            raise Exception("'dragonfly' must be specified for dragonfly interconnect")
        
        if "num_groups" not in hpcsim_dict["dragonfly"]:
            raise Exception("'num_groups' must be specified for dragonfly config") 
        self.num_groups = hpcsim_dict["dragonfly"]["num_groups"]
        
        if "num_switches_per_group" not in hpcsim_dict["dragonfly"]:
            raise Exception("'num_switches_per_group' must be specified for dragonfly config") 
        self.num_switches_per_group = hpcsim_dict["dragonfly"]["num_switches_per_group"]
        
        if "num_hosts_per_switch" not in hpcsim_dict["dragonfly"]:
            raise Exception("'num_hosts_per_switch' must be specified for dragonfly config") 
        self.num_hosts_per_switch = hpcsim_dict["dragonfly"]["num_hosts_per_switch"]
        
        if "num_ports_per_host" not in hpcsim_dict["dragonfly"]:
            raise Exception("'num_ports_per_host' must be specified for dragonfly config") 
        self.num_ports_per_host = hpcsim_dict["dragonfly"]["num_ports_per_host"]
        
        if "num_inter_links_per_switch" not in hpcsim_dict["dragonfly"]:
            raise Exception("'num_inter_links_per_switch' must be specified for dragonfly config") 
        self.num_inter_links_per_switch = hpcsim_dict["dragonfly"]["num_inter_links_per_switch"]
        
        if self.num_switches_per_group*self.num_inter_links_per_switch < self.num_groups - 1:
            raise Exception("not sufficient inter-links to support all group connections")
        
        if "inter_link_dups" not in hpcsim_dict["dragonfly"]:
            raise Exception("'inter_link_dups' must be specified for dragonfly config") 
        self.inter_link_dups = hpcsim_dict["dragonfly"]["inter_link_dups"]
        
        if "inter_group_topology" not in hpcsim_dict["dragonfly"]:
            raise Exception("'inter_group_topology' must be specified for dragonfly config") 
        self.inter_group_topology = hpcsim_dict["dragonfly"]["inter_group_topology"]
        
        #NOTE: consecutive_aries is specific to aries interconnect (where links are bundled together)
        if self.inter_group_topology is "consecutive_aries":
            if "num_inter_links_grouped" not in hpcsim_dict["dragonfly"]:
                raise Exception("'num_inter_links_grouped' must be specified for Aries config") 
            self.num_inter_links_grouped = hpcsim_dict["dragonfly"]["num_inter_links_grouped"]
            self.num_inter_links_per_switch /= self.num_inter_links_grouped # number of "bundled" inter-links
        
        if "intra_group_topology" not in hpcsim_dict["dragonfly"]:
            raise Exception("'intra_group_topology' must be specified for dragonfly config") 
        self.intra_group_topology = hpcsim_dict["dragonfly"]["intra_group_topology"]
        
        if self.intra_group_topology is "all_to_all":
            #print("I'm here: %s"%self.intra_group_topology)
            if "num_intra_links_per_switch" not in hpcsim_dict["dragonfly"]:
                raise Exception("'num_intra_links_per_switch' must be specified for dragonfly config") 
            self.num_intra_links_per_switch = hpcsim_dict["dragonfly"]["num_intra_links_per_switch"]
            
            if "intra_link_dups" not in hpcsim_dict["dragonfly"]:
                raise Exception("'intra_link_dups' must be specified for dragonfly config") 
            self.intra_link_dups = hpcsim_dict["dragonfly"]["intra_link_dups"]
        
            self.intra_group_bdw = hpcsim_dict["dragonfly"].get("intra_group_bdw", \
                hpcsim_dict["default_configs"]["intercon_bandwidth"])
        
            self.intra_group_delay = hpcsim_dict["dragonfly"].get("intra_group_delay", \
                hpcsim_dict["default_configs"]["intercon_link_delay"])
        
        if self.intra_group_topology is "cascade":
            if "num_chassis_per_group" not in hpcsim_dict["dragonfly"]:
                raise Exception("'num_chassis_per_group' must be specified for dragonfly/cascade config") 
            self.num_chassis_per_group = hpcsim_dict["dragonfly"]["num_chassis_per_group"]
            
            if "inter_chassis_dups" not in hpcsim_dict["dragonfly"]:
                raise Exception("'inter_chassis_dups' must be specified for dragonfly/cascade config") 
            self.inter_chassis_dups = hpcsim_dict["dragonfly"]["inter_chassis_dups"]
            
            if "intra_chassis_dups" not in hpcsim_dict["dragonfly"]:
                raise Exception("'intra_chassis_dups' must be specified for dragonfly/cascade config") 
            self.intra_chassis_dups = hpcsim_dict["dragonfly"]["intra_chassis_dups"]
            
            if "intra_chassis_bdw" not in hpcsim_dict["dragonfly"]:
                raise Exception("'intra_chassis_bdw' must be specified for dragonfly/cascade config") 
            self.intra_chassis_bdw = hpcsim_dict["dragonfly"]["intra_chassis_bdw"]
            
            if "inter_chassis_bdw" not in hpcsim_dict["dragonfly"]:
                raise Exception("'inter_chassis_bdw' must be specified for dragonfly/cascade config") 
            self.inter_chassis_bdw = hpcsim_dict["dragonfly"]["inter_chassis_bdw"]
            
            if "intra_chassis_delay" not in hpcsim_dict["dragonfly"]:
                raise Exception("'intra_chassis_delay' must be specified for dragonfly/cascade config") 
            self.intra_chassis_delay = hpcsim_dict["dragonfly"]["intra_chassis_delay"]
            
            if "inter_chassis_delay" not in hpcsim_dict["dragonfly"]:
                raise Exception("'inter_chassis_delay' must be specified for dragonfly/cascade config") 
            self.inter_chassis_delay = hpcsim_dict["dragonfly"]["inter_chassis_delay"]
            
            if "num_chassis_per_group" not in hpcsim_dict["dragonfly"]:
                raise Exception("'num_chassis_per_group' must be specified for dragonfly/cascade config") 
            self.num_chassis_per_group = hpcsim_dict["dragonfly"]["num_chassis_per_group"]
            
            if "num_blades_per_chassis" not in hpcsim_dict["dragonfly"]:
                raise Exception("'num_blades_per_chassis' must be specified for dragonfly/cascade config") 
            self.num_blades_per_chassis = hpcsim_dict["dragonfly"]["num_blades_per_chassis"]
            
            if "num_intra_links_grouped" not in hpcsim_dict["dragonfly"]: # bundled links among chassis
                raise Exception("'num_intra_links_grouped' must be specified for dragonfly/cascade config") 
            self.num_intra_links_grouped = hpcsim_dict["dragonfly"]["num_intra_links_grouped"]
        
            if self.num_chassis_per_group*self.num_blades_per_chassis != self.num_switches_per_group:
                raise Exception("num of switches does not match number of chassis and blades")
             
            self.num_intra_links_for_blades = self.num_blades_per_chassis-1
            # NOTE: each chassis intra link is bundle of "self.num_intra_links_grouped" number of links
            # i.e., total chassis intra links = self.num_intra_links_for_chassis*self.num_intra_links_grouped
            self.num_intra_links_for_chassis = self.num_chassis_per_group-1

        self.inter_group_bdw = hpcsim_dict["dragonfly"].get("inter_group_bdw", \
            hpcsim_dict["default_configs"]["intercon_bandwidth"])


        self.switch_host_bdw = hpcsim_dict["dragonfly"].get("switch_host_bdw", \
            hpcsim_dict["default_configs"]["intercon_bandwidth"])

        self.inter_group_delay = hpcsim_dict["dragonfly"].get("inter_group_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])


        self.switch_host_delay = hpcsim_dict["dragonfly"].get("switch_host_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])

        self.bufsz = hpcsim_dict["dragonfly"].get("bufsz", \
            hpcsim_dict["default_configs"]["intercon_bufsz"])

        self.proc_delay = hpcsim_dict["dragonfly"].get("proc_delay", \
            hpcsim_dict["default_configs"]["intercon_proc_delay"])
        
        self.route_method = hpcsim_dict["dragonfly"].get("route_method", \
            hpcsim_dict["default_configs"]["dragonfly_route_method"])

        # added for intra-node communication
        mem_bandwidth = hpcsim_dict["dragonfly"].get("mem_bandwidth", \
            hpcsim_dict["default_configs"]["mem_bandwidth"])
        mem_bufsz = hpcsim_dict["dragonfly"].get("mem_bufsz", \
            hpcsim_dict["default_configs"]["mem_bufsz"])
        mem_delay = hpcsim_dict["dragonfly"].get("mem_delay", \
            hpcsim_dict["default_configs"]["mem_delay"])

        if "hpcsim" in hpcsim_dict["debug_options"] or \
           "intercon" in hpcsim_dict["debug_options"] or \
           "dragonfly" in hpcsim_dict["debug_options"]:
            print("dragonfly: num_groups=%d" % self.num_groups)
            print("dragonfly: num_switches_per_group=%d" % self.num_switches_per_group)
            print("dragonfly: num_hosts_per_switch=%d" % self.num_hosts_per_switch)
            print("dragonfly: num_ports_per_host=%d" % self.num_ports_per_host)
            print("dragonfly: num_inter_links_per_switch=%d" % self.num_inter_links_per_switch)
            print("dragonfly: inter_link_dups=%d" % self.inter_link_dups)
            print("dragonfly: inter_group_toplogy=%s" % self.inter_group_topology)
            print("dragonfly: intra_group_toplogy=%s" % self.intra_group_topology)
            if self.intra_group_topology is "all_to_all":
                print("dragonfly: num_intra_links_per_switch=%d" % self.num_intra_links_per_switch)
                print("dragonfly: intra_link_dups=%d" % self.intra_link_dups)
                print("dragonfly: intra_group_bdw=%f (bits per second)" % self.intra_group_bdw)
                print("dragonfly: intra_group_delay=%f (seconds)" % self.intra_group_delay)
            if self.intra_group_topology is "cascade":
                print("dragonfly: num_chassis_per_group=%d" % self.num_chassis_per_group)
                print("dragonfly: num_blades_per_chassis=%d" % self.num_blades_per_chassis)
                print("dragonfly: num_intra_links_for_blades=%d" % self.num_intra_links_for_blades)
                print("dragonfly: num_intra_links_for_chassis=%d" % self.num_intra_links_for_chassis)
                print("dragonfly: num_intra_links_grouped=%d" % self.num_intra_links_grouped)
                print("dragonfly: inter_chassis_dups=%d" % self.inter_chassis_dups) # among chassis
                print("dragonfly: intra_chassis_dups=%d" % self.intra_chassis_dups) # among blades
                print("dragonfly: inter_chassis_bdw=%d" % self.inter_chassis_bdw)
                print("dragonfly: intra_chassis_bdw=%d" % self.intra_chassis_bdw)
                print("dragonfly: inter_chassis_delay=%d" % self.inter_chassis_delay)
                print("dragonfly: intra_chassis_delay=%d" % self.intra_chassis_delay)
            print("dragonfly: inter_group_bdw=%f (bits per second)" % self.inter_group_bdw)
            print("dragonfly: inter_group_delay=%f (seconds)" % self.inter_group_delay)
            print("dragonfly: switch_host_bdw=%f (bits per second)" % self.switch_host_bdw)
            print("dragonfly: switch_host_delay=%f (seconds)" % self.switch_host_delay)
            print("dragonfly: proc_delay=%f (seconds)" % self.proc_delay)
            print("dragonfly: mem_bandwidth=%f (bits per second)" % mem_bandwidth)
            print("dragonfly: mem_bufsz =%d (bytes)" % mem_bufsz)
            print("dragonfly: mem_delay=%f (seconds)" % mem_delay)
            print("dragonfly: route_method=%s" % self.route_method)
        
        # compute the total number of switches and hosts
        self.nswitches = self.num_groups*self.num_switches_per_group
        self.nhosts = self.nswitches*self.num_hosts_per_switch
        self.num_hosts_per_group = self.num_hosts_per_switch*self.num_switches_per_group
        
        # add switches and hosts as entities
        simian = hpcsim_dict["simian"]
        swid = hid = 0
        for g in xrange(self.num_groups):
            for a in xrange(self.num_switches_per_group):
                # each switch is identified by a group id and switch id pair
                simian.addEntity("Switch", DragonflySwitch, swid, hpcsim_dict, self, g, a)
                # add host as entities 
                for h in xrange(self.num_hosts_per_switch):
                    p = self.hid_to_port(h)
                    simian.addEntity("Host", hpcsim.get_host_typename(hpcsim_dict), hid,
                            hpcsim_dict, self, swid, 'h', 
                            p, self.switch_host_bdw, self.bufsz,
                            self.switch_host_delay,
                            mem_bandwidth, mem_bufsz, mem_delay)
                    hid += 1
                swid += 1
        
    def network_diameter(self):
        """Returns the network diameter in hops."""
    
        if self.intra_group_topology == "cascade":
            # max four hops within each group(4X3), max three groups traversed (2),
            # plus 2 hops to hosts
            return 16
        elif self.intra_group_topology == "all_to_all":
            # max one hop within each group(3), max three groups traversed(2), 
            # plus 2 hops to hosts
            return 7

    def network_diameter_time(self):
        """Returns the network diameter in time."""
        
        d = self.network_diameter()
        if self.intra_group_topology == "cascade":
            return 2*self.switch_host_delay+(d-2)*self.inter_chassis_delay+ \
                    (d-2)*self.intra_chassis_delay+2*self.inter_group_delay
        elif self.intra_group_topology == "all_to_all":
            return 2*self.switch_host_delay+(d-4)*self.intra_group_delay+ \
                    2*self.inter_group_delay
    
    def hid_to_port(self, hid):
        """Returns switch port (through which the host is connected)."""

        # Local variable
        #   hid: host id 
        if not (0 <= hid < self.nhosts):
            raise Exception("invalid host id=%d for dragonfly"%(hid))
        p = hid%self.num_hosts_per_switch
        return p
    
    @staticmethod
    def calc_min_delay(hpcsim_dict):
        """Calculates and returns the min delay value for config parameters."""
        
        if "dragonfly" not in hpcsim_dict:
            raise Exception("'dragonfly' must be specified for dragonfly interconnect")
        d1 = hpcsim_dict["dragonfly"].get("inter_group_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        d2 = hpcsim_dict["dragonfly"].get("intra_group_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        d3 = hpcsim_dict["dragonfly"].get("switch_host_delay", \
            hpcsim_dict["default_configs"]["intercon_link_delay"])
        min_d = min([d1, d2, d3])
        return min_d

class Aries(Dragonfly):
    """cray's aries is a dragonfly."""

    def __init__(self, hpcsim, hpcsim_dict):
        if "dragonfly" not in hpcsim_dict:
            raise Exception("'dragonfly' must be specified for aries interconnect")

        super(Aries, self).__init__(hpcsim, hpcsim_dict)
        
