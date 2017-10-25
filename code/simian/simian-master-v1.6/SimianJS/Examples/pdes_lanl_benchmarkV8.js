/*
Copyright (c) 2015, Los Alamos National Security, LLC
All rights reserved.

Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
/*Usage example:
    mpirun -np 1 chai Examples.JS/pdes_lanl_benchmarkV8.js 100 100 1 0 0 0 1000 0   1000    0 0.5 743289 10 1000 1 true pdes

PDES LANL BENCHMARK is a benchmark to test parallel discrete event simulation performance
through a combination of communication loads, memory requirements, and computational loads

Overview
==========
Each entity A sends a "request computation" message to another entity B; upon message receipt,
B performs randomly weighted subset sum calculations on its local list data structure.
Each entity A also sends "timer" messages to itself with some delay before it sends another
"request computation" message. The main parameters are as follows:

Communication Parameters
========================
n_ent:        Number of entities
s_ent:        Average number of send events per entity
            Individual entities determine how many events they need to send
            based on p_send and their index and then adjust their local intersend_delay
            using an exponential distribution.
endTime:    Duration of simulation. Note that minDelay = 1.0 always, so
            setting endTime to n_ent*s_ent will result in one event per minDelay
            epoch when running in parallel mode
q_avg:        Average number of events in the event queue per entity
            For individual entities this is made proportional
            the number of total events that the entity needs to send.
            Default value is 1. Higher values will stress-test the event queue
            mechanism of the DES engine
p_receive: Parameter for geometric distribution of destination entities indexed by entity index.
            Entity i receives a fraction of p_receive*(1-p_receive)**(i-1) of all request messages
            Lower-indexed entities receive larger shares
            p_receive = 0: uniform distribution; p_receive = 1: only entity 1 receives messages
p_send:        Parameter for geometric distribution of source entities indexed by entity index
            See p_receive for more details
invert:        Flag to indicate whether receive and sent distribution should be inverted
            If set to True: highest-index entity sends most messages

Memory Parameters
==========================
m_ent:         Average memory footprint per entity,
            modeled as the average linear list size (8 byte units).
            Each entity has a local list as a data structure that  uses up memory
p_list:        Parameter for geometric distribution of linear list sizes
            Set to 0 for uniform distribution
            Set to 1.0 to make entity 0 the only entity with a list

Computation Parameters
==========================
ops_ent:     Average operations per handler per entity.
            Computational cycle use is implemented as a weighted subset sum calculation
            of the first k elements of the list with randomly drawn weights (to eliminate
            the possibility that the calculation gets optimized away).
            Each entity linearly scales down the number of operations based on its local
            list size as determined by p_list.
ops_sigma:     Variance of numer of operations per handler per entity, as a fraction of ops_ent

cache_friendliness:
            Determines how many different list elements are accessed during operations
            traded off with more operations per list element
            Set to p to access the first p fraction of list elements
            Set to 0.0 to access only first list element
            Set to 1.0 to access all list elements
            Set to 0.5 if no other value is known

PDES Parameters
========================
time_bins:    Purely for reporting purposes, this parameter gives the number of equal-size
            time bins in which send events are sent
init_seed:    Initial seed value for random number generation. Built-in Python random number
            generator is used. Seed values are passed along to entities upon creation and
            also as parameters for graph/matrix generation

Output statistics are written into the output file of entity 0.

POC: Stephan Eidenbenz, eidenben@lanl.gov

Ported to Chai JS: Nandakishore Santhi, nsanthi@lanl.gov
Date: June 01, 2016
*/
"use strict";
"use wasm";

masala.io.load("./simian.js");
masala.io.load("./sprintf.js");

// Variables //
var n_ent = 100;
var s_ent = 100;
var q_avg = n_ent*s_ent;
var p_receive = 0;
var p_send = 0;
var invert = false;

var m_ent = 1000;
var p_list = 0;
var ops_ent = 10000;
var ops_sigma = 0;
var cache_friendliness = 0.5;
var init_seed = 743289;
var time_bins = 10;

//var endTime = n_ent*s_ent;
var endTime = 1000;
var minDelay = 1;    // Minimum Delay value for synchronization between MPI ranks (if applicable)
var useMPI = true;

var logName = "pdes";

print(n_ent, s_ent, q_avg, p_receive, p_send, invert, m_ent, p_list, ops_ent, ops_sigma, cache_friendliness, init_seed, time_bins, endTime, minDelay, useMPI, logName);

/*
// Variables //
local n_ent = 10        // Number of entities 10
local s_ent = 10000        // Average number of send events per entity 100
                // Individual entities determine how many events they need to send
                // based on p_send and their index and then adjust their local intersend_delay
                // using an exponential distribution.
//local endTime = n_ent*s_ent        // Duration of simulation. Note that minDelay = 1.0 always, so
local endTime = 5        // Duration of simulation. Note that minDelay = 1.0 always, so
                // setting endTime to n_ent*s_ent will result in one event per minDelay
                // epoch when running in parallel mode
local q_avg = 1 // Average number of events in the event queue per entity
                // For individual entities this is made proportional
                // the number of total events that the entity needs to send.
                // Default value is 1. Higher values will stress-test the event queue
                // mechanism of the DES engine
                // try from 1(default), 0.2*s_ent, 0.5*s_ent, 0.8*s_ent,s_ent
local p_receive = 0            // Parameter to geometric distribution for choosing destination entities
                // Set to 0 for uniform distribution
                // Set to 1.0 to make entity 0 the only destination
                // Lower index entities receive more messages
local p_send = 0            // Parameter for geometric distribution of source entities

                // Set to 0 for uniform distribution
                // Set to 1.0 to make entity 0 the only source
local invert = false    // Flag to indicate whether receive and sent distribution should be inverted
                // If True: entity n_ent sends most  messages

local m_ent = 1000    // Average memory footprint per entity,
                // modeled as the average linear list size (8 byte units)
local p_list    = 0    // Parameter for geometric distribution of linear list sizes
                // Set to 0 for uniform distribution
                // Set to 1.0 to make entity 0 the only entity with a list
local ops_ent = 1000    // Average operations per handler per entity.
local ops_sigma = 0    // Variance of numer of operations per handler per entity, as a fraction of ops_ent
                // drawn from a Gaussian
local cache_friendliness = 0.5
                // Determines how many different list elements are accessed during operations
                // traded off with more operations per list element
                // Set to p to access the first p fraction of list elements
                // Set to 0.0 to access only first list element (cache-friendly)
                // Set to 1.0 to access all list elements (cache-unfriendly)
                // Set to 0.5 if no other value is known

local init_seed = 1    // Initial random seed to be passed around
local time_bins = 10     // Number of bins for time and event reporting (Stats only)
local useMPI = true
*/

////
//  Initialization stuff
const max = Math.max, min = Math.min, floor = Math.floor, inf = 1e100, ceil = Math.ceil, log = Math.log, sqrt = Math.sqrt, uniform = masala.random.uniform, range = masala.random.range, seed = masala.random.seed, rand = masala.random.rand;

function exponential(lambda) { //Lambda := 1/Mean
    return -log(uniform())/lambda;
}

function gauss(mu, sigma) {
    var x1, x2, w, y1;
 
    do {
        x1 = range(2.0) - 1.0;
        x2 = range(2.0) - 1.0;
        w = (x1*x1) + (x2*x2);
    } while (w >= 1.0);

    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;

    return (y1 * sigma) + mu;
}

// Compute the min value for geometric distribution function
var r_min = Math.pow((1-p_receive), n_ent);
// Compute target number of send events
var target_global_sends = n_ent * s_ent;

////
class PDES_LANL_Node extends Entity {
    constructor(...theArgs) {
        super(...theArgs);
        var newSeed = this.num + init_seed; // Initialize random seed with own id
        // 1. Compute number of events that the entity will send out
        var prob;
        if (p_send == 0) prob = 1.0/n_ent; //uniform case
        else {
            if (invert) prob = p_send*Math.pow((1-p_send), (n_ent - self.num)); // Probability that an event gets generated on this entity
            else prob = p_send*Math.pow((1-p_send), self.num);
        }
        var target_sends = floor(prob * target_global_sends);
        if (target_sends > 0) this.local_intersend_delay = endTime/target_sends;
        // if the entity sends zero events, we let it create one that will most likely be after the sim ends
        else this.local_intersend_delay = 10*endTime;

        // 2. Allocate appropriate memory space through list size, and number of ops
        if (p_list == 0) prob = 1.0/n_ent; // uniform case
        else prob = p_list*Math.pow((1-p_list), this.num);

        this.list_size = floor(prob * n_ent * m_ent); // there are n_ent*m_ent list elements in total
        this.ops = floor(prob * n_ent * ops_ent); // there are n_ent*m_ent list elements in total
        this.active_elements = floor(cache_friendliness * this.list_size); // only this many list elements will be accessed
        //this.list = new Float64Array(new ArrayBuffer(this.list_size*8));
        this.list = new Array(this.list_size);
        for (let i=0; i<this.list_size; i++) this.list[i] = uniform(); // create a list of random elements of length list_size

        // 3. Set up queue size
        this.q_target = (q_avg/s_ent) * target_sends;
        this.q_size = 1; // number of send events scheduled ahead of time by this entity
        this.last_scheduled = this.engine.now; // time of last scheduled event

        // 4. Set up statistics
        this.send_count = 0;
        this.receive_count = 0;
        this.opsCount = 0; // for stats
        this.ops_max = 0; // for stats
        this.ops_min = inf; // for stats
        this.ops_mean = 0.0; // for stats
        this.time_sends = []; // for time reporting
        for (let i=0; i<time_bins; i++) this.time_sends[i] = 0;
        if (this.num == 0) { // only for the global statistics entity
            this.stats_received = 0;
            this.gsend_count = 0;
            this.greceive_count = 0;
            this.gopsCount = 0;
            this.gops_max = 0;
            this.gops_min = inf;
            this.gops_mean = 0.0;
            this.gtime_sends = []; // for time reporting
            for (var i=0; i<time_bins; i++) this.gtime_sends[i] = 0;
        }

        // 5. Schedule FinishUp at end of time
        this.reqService(endTime - this.engine.now, "FinishHandler", null, null, null);
        this.SendHandler(this, newSeed);
    }

    SendHandler(self, newSeed, ...theArgs) { // varargs ... is artificial
        seed(newSeed);
        self.send_count++;
        self.q_size--;
        var bin = floor(self.engine.now/(endTime+0.0001)*time_bins); //TODO: Check!
        self.time_sends[bin]++;
        // Generate next event for myself
        // Reschedule myself until q is full or time has run out
        while ((self.q_size < self.q_target) && !(self.last_scheduled > endTime)) {
            let own_delay = exponential(1.0/self.local_intersend_delay);
            self.last_scheduled =  self.last_scheduled + own_delay;
            if (self.last_scheduled < endTime) {
                self.q_size++;
                self.reqService(self.last_scheduled-self.engine.now, "SendHandler", uniform());
            }
        }
        // Generate computation request event to destination entity
        var DestIndex;
        if (p_receive == 1.0) // If p is exactly 1.0, then the only entity 0 is only destination
            DestIndex = 0;
        else if (p_receive == 0) // by convention, p == 0 means we want uniform distribution
            DestIndex = floor(range(n_ent));
        else {
            let U = range(1.0 - r_min) + r_min; // We computed r_min such that the we only get indices less than num_ent
            DestIndex = floor(ceil(log(U) / log(1.0 - p_receive))) - 1; //TODO: Check!
        }
        var new_seed = rand();
        // Send event to destination ReceiveHandler (only if not past reporting time)
        if (self.engine.now+minDelay < endTime) self.reqService(minDelay, "ReceiveHandler", new_seed, "PDES_LANL_Node", DestIndex);
    }

    ReceiveHandler(self, newSeed, ...theArgs) { // varargs ... is artificial
        seed(newSeed);
        var r_ops = max(1, floor(gauss(self.ops, self.ops*ops_sigma))); // number of operations
        var r_active_elements = floor(self.active_elements * (r_ops/self.ops)); // only this many list elements will be accessed
        var r_active_elements = min(r_active_elements, self.list_size); // cannot be more than list size
        var r_active_elements = max(1, r_active_elements); // cannot be less than 1
        var r_ops_per_element = floor(r_ops/r_active_elements);
        // Update stats
        self.receive_count++;
        self.ops_max = max(self.ops_max, r_ops);
        self.ops_min = min(self.ops_min, r_ops);
        self.ops_mean = (self.ops_mean*(self.receive_count-1) +  r_ops)/self.receive_count;
        // Compute loop
        var value = 0;
        for (let i=0; i<r_active_elements; i++) {
            for (let j=0; j<r_ops_per_element; j++) value += range(self.list[i]);
            self.opsCount += r_ops_per_element;
        }
        return value;
    }

    FinishHandler(self, ...theArgs) { // varargs ... is artificial
        // Send stats to entity 0 for outputting of global stats
        var msg = [self.num, self.send_count, self.receive_count, self.ops_min, self.ops_mean, self.ops_max, self.time_sends, self.opsCount];
        self.reqService(minDelay, "OutputHandler", msg, "PDES_LANL_Node", 0);
    }

    OutputHandler(self, msg, ...theArgs) { // varargs ... is artificial
        var header = sprintf("%10s %10s %10s %10s %10s %10s    '%s'\n", "#EntityID", "#Sends", "#Receives", "Ops(Min", "Avg", "Max)", "Time Bin Sends");
        // Write out Stats, only invoked on entity 0
        if (self.stats_received == 0) self.out.write(header); // Only write header line a single time

        self.stats_received++;
        self.gops_mean = (msg[2]*msg[4] + self.gops_mean*self.greceive_count)/max(1, self.greceive_count+msg[2]);
        self.gsend_count = self.gsend_count + msg[1];
        self.greceive_count = self.greceive_count + msg[2];
        self.gopsCount = self.gopsCount + msg[7];
        self.gops_min = min(self.gops_min, msg[3]);
        self.gops_max = max(self.gops_max, msg[5]);
        for (let i=0; i<time_bins; i++) self.gtime_sends[i] = self.gtime_sends[i] + msg[6][i];

        var str_out = sprintf("%10d %10d %10d %10d %10.5g %10d    %s\n", msg[0], msg[1], msg[2], msg[3], msg[4], msg[5], "[" + msg[6] + "]");
        self.out.write(str_out);

        if (self.stats_received == n_ent) { // We can write out global stats
            self.out.write("===================== LANL PDES BENCHMARK  Collected Stats from All Ranks =======================\n");
            header = sprintf("%10s %10s %10s %10s %10s %10s %10s    '%s'\n", "#Entities", "#Sends", "#Receives", "OpsCount", "Ops(Min", "Avg", "Max)", "Time Bin Sends");
            self.out.write(header);
            str_out = sprintf("%10d %10d %10d %10d %10d %10.5g %10d    %s\n", n_ent, self.gsend_count, self.greceive_count, self.gopsCount, self.gops_min, self.gops_mean, self.gops_max, "[" + self.gtime_sends + "]");
            self.out.write(str_out);
            self.out.write("=================================================================================================\n");
        }
    }
}

var simName = "PDES_LANL_Benchmark_" + logName;
var startTime =  0.0;

// Note little trick with endTime setting, as we need to collect statistics in the end
var SimianEngine = new Simian(simName, startTime, max(endTime, 2)+3*minDelay, minDelay, useMPI);

// "MAIN"
for (let i=0; i<n_ent; i++) SimianEngine.addEntity("PDES_LANL_Node", PDES_LANL_Node, i);

// 5. Run Simian
SimianEngine.run();
SimianEngine.exit();
