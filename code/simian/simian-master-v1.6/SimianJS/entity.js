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
Copyright: Open source, must acknowledge original author
Purpose: JITed PDES Engine in MasalaChai JavaScript, a custom dialect of Mozilla Spidermonkey
   Named entity class with single inheritence and processes
*/
var Entity = class {
    constructor(name, out, engine, num, ...theArgs) {
        //Constructor to be called as <entityName>(<args>)
        this.name = name; //Explicit instance name
        this.out = out; //Log file for this instance
        this.engine = engine; //Engine
        this.num = num; //Serial Number
        this._procList = {}; //A separate process table for each instance
        this._category = {}; //A map of sets for each kind of process
    }

    toString() {
        return this.name + "(" + this.num + ")";
    }

    reqService(offset, eventName, data, rx, rxId) {
        //Purpose: Send an event if Simian is running
        const engine = this.engine; //Get the engine for this entity
        const sendAndCount = engine.MPI.sendAndCount, encode = msgpack.encode;
        var eventQueue = engine.eventQueue;

        if ((rx != null) && (offset < engine.minDelay)) {
            if (!engine.running) throw new Error("Cannot send an event when Simian is idle");
            //If sending to this, then do not check against min-delay
            throw new Error(this + " attempted to send with too little delay");
        }

        var time = engine.now + offset;
        if (time > engine.endTime) return; //No need to send this event

        var rx = (rx != null) ? rx : this.name;
        var rxId = (rxId != null) ? rxId : this.num;
        var e = {
            "tx"   : this.name, //String
            "txId" : this.num, //Number
            "rx"   : rx, //String
            "rxId" : rxId, //Number
            "name" : eventName, //String
            "data" : data, //Object
            "time" : time, //Number
        }

        var recvRank = engine.getOffsetRank(rx, rxId);

        if (recvRank == engine.rank) eventQueue.push(e); //Send to this
        else {
            sendAndCount(encode(e), recvRank, null); //Send to others
        }
    }

    attachService(name, fun) { //Attaches a service to an entity instance at runtime
        this[name] = fun;
    }

    //Following code is to support coroutine processes on entities:
    //Entity methods to interact with processes
    createProcess(name, fun, kind) { //Creates a named process
        if (name == "*") throw new Error("Reserved name to represent all child processes: " + name);
        var proc = new Process(name, fun, this, null);
        if (!proc) throw new Error("Could not create a valid process named: " + name);
        this._procList[name] = proc;
        if (kind) this.categorizeProcess(kind, name); //Categorize
    }

    startProcess(name, ...theArgs) { //Starts a named process
        var proc = this._procList[name];
        if (proc) {
            if (!proc.started) {
                proc.started = true;
                proc.co = proc.genFun(proc, ...theArgs);
                //When starting, pass process instance as first arg
                //This first arg is in turn passed to the process function
                //which is the factory for the co-routine - as "this" in examples)
                //Through this argument all the members of the process table below become
                //accessible to all the process-instance co-routine functions.
                proc.wake(...theArgs);
            }
            else throw new Error("Starting an already started process: " + proc.name);
        }
    }

    _wakeProcess(self, name) { //Hidden, implicit wake a named process without arguments
        var proc = self._procList[name];
        if (proc) proc.wake(); //If existing and not been killed asynchronously
        else throw new Error("Attempted to wake a non existant process: " + name);
    }

    wakeProcess(name, ...theArgs) { //Wake a named process with arguments
        var proc = this._procList[name];
        if (!proc) throw new Error("Attempted to wake a non existant process: " + name);
        else proc.wake(...theArgs); //NOTE: Here, theArgs goes to LHS of previous hibernate's 'yield*'
                                    //through co.next(...theArgs) call
    }

    killProcess(name) { //Kills named process or all entity-processes
        if (name) { //Kills named child-process
            var proc = this._procList[name];
            proc.kill(); //Kill itself and all subprocesses
        }
        else { //Kills all subprocesses
            for (var pName in this._procList) {
                if (this._procList.hasOwnProperty(pName)) {
                    proc = this._procList[pName]
                        proc.kill(); //Kill itself and all subprocesses
                }
            }
            this._procList = {}; //A new process table
        }
    }

    killProcessKind(kind) { //Kills all @kind processes on entity
        var nameSet = this._category[kind];
        if (!nameSet) throw new Error("killProcessKind: No category of processes on this entity called " + kind);
        for (let name in nameSet) {
            if (nameSet.hasOwnProperty(name)) {
                let proc = this._procList[name];
                if (proc) proc.kill(); //Kill itself and all subprocesses
            }
        }
    }

    statusProcess(name) {
        var proc = this._procList[name];
        if (proc) return proc.status();
        else return "NonExistent";
    }

    categorizeProcess(kind, name) {
        //Check for existing process and then categorize
        var proc = this._procList[name];
        if (proc) { //Categorize both ways for easy lookup
            if (!proc._kindSet) proc._kindSet = {};
            proc._kindSet[kind] = true; //Indicate to proc that it is of this kind to its entity
            //Indicate to entity that proc is of this kind
            let kindList = this._category[kind];
            if (!kindList) this._category[kind] = {name : true}; //New kind
            else kindList[name] = true; //Existing kind
        } else throw new Error("categorize: Expects a proper child to categorize");
    }

    unCategorizeProcess(kind, name) {
        //Check for existing process and then unCategorize
        var proc = this._procList[name];
        if (proc) //unCategorize both ways for easy lookup
            if (!proc._kindSet) proc._kindSet[kind] = null; //Indicate to proc that it is not of this kind to its entity
        //Indicate to entity that proc is not of this kind
        var kindList = this._category[kind];
        if (kindList) kindList[name] = null; //Existing kind
        else throw new Error("unCategorize: Expects a proper child to un-categorize");
    }

    isProcess(name, kind) {
        var proc = this._procList[name];
        if (proc) return proc.is_a(kind);
    }

    getProcess(name) {
        //A reference to a named process is returned if it exists
        //NOTE: User should delete it by setting to null to free its small memory if no longer needed
        var proc = this._procList[name];
        if (proc) return proc;
        else return null;
    }

    getCategoryNames() {
        var kindSet = [];
        if (this._category) {
            let n = 0;
            for (let k in this._category) {
                if (this._category.hasOwnProperty(k)) kindSet[n++] = k;
            }
        }
        return kindSet;
    }

    getProcessNames() {
        var nameSet = [];
        if (this._procList) {
            let n = 0;
            for (let k in this._procList) {
                if (this._procList.hasOwnProperty(k)) nameSet[n++] = k;
            }
        }
        return nameSet;
    }
}
