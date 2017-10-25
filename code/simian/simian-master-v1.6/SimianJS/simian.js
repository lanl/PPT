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
/*
Author: Nandakishore Santhi
Date: 23 May, 2016
Modified: 07 Dec, 2016
Copyright: Open source, must acknowledge original author
Purpose: JITed PDES Engine in MasalaChai JavaScript, a custom dialect of Mozilla Spidermonkey
   Main engine script
*/
"use strict";
"use wasm"; //Webassembly for JIT speedup

masala.io.load("./file.js");
masala.io.load("./hash.js");
masala.io.load("./msgpack.js");
masala.io.load("./eventQ.js");
masala.io.load("./process.js")
masala.io.load("./entity.js");
masala.io.load("./sprintf.js");

const print = masala.io.print, now = masala.time.now;

//print(arguments);
var Simian = class {
    constructor(simName, startTime, endTime, minDelay, useMPI) {
        this.version = "Simian.CHAI JIT PDES Engine\nVersion: v1.62-alpha; 07-Dec-2016\nRunning on customised IonMonkey: MasalaChai\nAuthor: Nandakishore Santhi (C) 2014-\n";

        this.name = simName;
        this.startTime = startTime;
        this.endTime = endTime || 1e100;
        this.minDelay = minDelay || 1;

        this.now = startTime;

        //Status of JITing
        this.jitStatus = true;

        //If simulation is running
        this.running = false;

        //Stores the entities available on this LP
        this.entities = {};

        /*Events are stored in a priority-queue or heap, in increasing
          order of time field. Heap top can be peeked using this.eventList[0]
          event = {time, name, data, tx, txId, rx, rxId}.*/
        this.eventList = [];
        this.eventQueue = new eventQ(this.eventList);

        /*Stores the minimum time of any event sent by this process,
          which is used in the global reduce to ensure global time is set to
          the correct minimum.*/
        this.infTime = endTime + 2*minDelay;

        //Base rank is an integer hash of entity's name
        this.baseRanks = {};

        //Make things work correctly with and without MPI
        this.MPI = masala.mpi;
        if (useMPI) { //Initialize MPI
            this.useMPI = true;
            this.MPI.init();
            this.rank = this.MPI.rank();
            this.size = this.MPI.size();
        } else {
            this.useMPI = false;
            this.rank = 0;
            this.size = 1;
        }

        //One output file per rank
        this.out = new file(this.name + "." + this.rank + ".out", "w");
    }

    toString() {
        return this.version;
    }

    exit() {
        if (this.useMPI) { //Exit only when all engines are ready
            this.out.flush();
            this.out.close();
            this.out = undefined;
            this.MPI.finalize();
        }
    }

    run() { //Run the simulation
        const startClock = now();
        const infTime = this.infTime, startTime = this.startTime, endTime = this.endTime, minDelay = this.minDelay, rank = this.rank, size = this.size, min = Math.min;
        const iprobe = this.MPI.iprobe, probe = this.MPI.probe, iprobetrials = this.MPI.iprobetrials, recv = this.MPI.recv, alltoallSum = this.MPI.alltoallSum, allreduce = this.MPI.allreduce, barrier = this.MPI.barrier, decode = msgpack.decode;

        if (!rank) {
            print(this); //Print the version string
            print("===========================================");
            print("--------SIMIAN.CHAI JIT-PDES ENGINE--------");
            print("===========================================");
            if (this.jitStatus) print("JIT: ON");
            else print("JIT: OFF");
            if (this.useMPI) print("MPI: ON");
            else print("MPI: OFF");
        }
        var numEvents = 0.0, totalEvents = 0.0, synchEvents = 0.0, totalSynchEvents = 0.0;
        var entities = this.entities, eventList = this.eventList, eventQueue = this.eventQueue;

        this.running = true;
        var globalMinLeft = startTime;
        while (globalMinLeft <= endTime) { //Exit loop only when global-epoch is past endTime
            let epoch = globalMinLeft + minDelay;

            while ((eventList.length) && (eventList[0].time < epoch)) {
                let curEvent = eventQueue.pop(); //Next event
                if (this.now > curEvent.time) 
                    throw new Error("Out of order event: now=" + self.now + ", evt=" + event.time);
                this.now = curEvent.time; //Advance time

                //Simulate event
                let entity = entities[curEvent.rx][curEvent.rxId];
                let service = entity[curEvent.name];
                service(entity, curEvent.data, curEvent.tx, curEvent.txId); //Receive

                numEvents++;
            }

            if (size > 1) {
                let toRcvCount = alltoallSum();
                while (toRcvCount > 0) {
                    probe(null, null);
                    eventQueue.push(decode(recv(null, null, null)));
                    toRcvCount--;
                }
                let minLeft = (eventList.length) ? eventList[0].time : infTime;
                globalMinLeft = allreduce(minLeft, 0); //Synchronize minLeft, by AllReduce MIN
                synchEvents += 2;
            } else {
                globalMinLeft = (eventList.length) ? eventList[0].time : infTime;
            }
        }

        if (size > 1) {
            barrier(); //Forcibly synchronize all ranks before counting total events
            totalEvents = allreduce(numEvents, 1); //AllReduce SUM
            totalSynchEvents = allreduce(synchEvents, 1); //AllReduce SUM
            synchEvents += 3;
        }
        else totalEvents = numEvents;
        if (!rank) {
            let elapsedClock = (now() - startClock)/1000; //In seconds
            print("SIMULATION FINISHED IN : " + elapsedClock + " (s)");
            print("SIMULATED EVENTS       : " + totalEvents);
            print("SYNCHRONIZATION EVENTS : " + totalSynchEvents);
            print("MODEL-EVENTS/SECOND    : " + totalEvents/elapsedClock);
            print("TOTAL-EVENTS/SECOND    : " + (totalEvents+totalSynchEvents)/elapsedClock);
            print("===========================================");
        }
        this.running = false;
    }

    schedService(time, eventName, data, rx, rxId) {
        /*Purpose: Add an event to the event-queue.
          For kicking off simulation and waking processes after a timeout*/
        if (time > this.endTime) return; //No need to push this event

        var recvRank = this.getOffsetRank(rx, rxId);

        if (recvRank == this.rank) {
            let e = {
                "tx"   : null, //String (Implictly this.name)
                "txId" : null, //Number (Implictly this.num)
                "rx"   : rx, //String
                "rxId" : rxId, //Number
                "name" : eventName, //String
                "data" : data, //Object
                "time" : time, //Number
            }

            this.eventQueue.push(e);
        }
    }

    getBaseRank(name) { //Can be overridden for more complex Entity placement on ranks
        return hash(name) % this.size;
    }

    getOffsetRank(name, num) { //Can be overridden for more complex Entity placement on ranks
        return (this.baseRanks[name] + num) % this.size;
    }

    getEntity(name, num) { //Returns a reference to a named entity of given serial number
        if (this.entities[name]) {
            let entity = this.entities[name];
            return entity[num];
        }
    }

    attachService(klass, name, fun) { //Attaches a service at runtime to an entity klass type
        klass[name] = fun;
    }

    addEntity(name, entityClass, num, ...theArgs) {
        /*Purpose: Add an entity to the entity-list if Simian is idle
          This function takes a pointer to a class from which the entities can
          be constructed, a name, and a number for the instance.*/
        if (this.running) throw new Error("Cannot add new entity when Simian is running!");

        if (!this.entities[name]) this.entities[name] = {}; //To hold entities of this "name"
        var entity = this.entities[name];

        this.baseRanks[name] = this.getBaseRank(name); //Register base-ranks
        var computedRank = this.getOffsetRank(name, num);

        if (computedRank == this.rank) { //This entity resides on this engine
            //Output log file for this Entity
            this.out.write(name + "(" + num + ") Running on rank " + computedRank + "\n");

            entity[num] = new entityClass(name, this.out, this, num, ...theArgs); //Entity is instantiated
        }
    }
}
