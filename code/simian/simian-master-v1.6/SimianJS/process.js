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
  Named processes
*/
function _killallChildren(thisProcess) { //Hidden function to kill all children
    var entity = thisProcess.entity;
    if (thisProcess._childList) {
        for (let name in thisProcess._childList) {
            if (thisProcess._childList.hasOwnProperty(name)) {
                let proc = entity._procList[name]; //Get actual process
                if (proc) proc:kill(); //Kill child and all its subprocesses
            }
        }
        thisProcess._childList = null; //Empty child table
    }
}

hibernate = function*(...theArgs) {
    //Arguments "...theArgs" to hibernate => LHS of wake
    //Processes to be woken explicitly by events may return values
    if (this.co) {
        this.state = "suspended";
        yield theArgs; //NOTE: Here, theArgs goes to LHS of co.next(); then to LHS of next proc.wake();
    }
    else throw new Error("Cannot hibernate killed process");
}

sleep = function*(x, ...theArgs) {
    //Processes which are to implicitly wake at set timeouts
    //All return values are passed to __call/wake
    if ((typeof(x) != "number") || (x < 0)) throw new Error("Sleep not given non-negative number argument!");

    //Schedule a local event after x timesteps to wakeup
    if (this.co) {
        let entity = this.entity;
        entity.engine.schedService(entity.engine.now + x, "_wakeProcess", this.name, entity.name, entity.num);
        this.state = "suspended";
        yield theArgs; //NOTE: Here, theArgs goes to LHS of co.next(); then to LHS of next proc.wake();
    }
    else throw new Error("Cannot sleep killed process")
}

/*
Process class:
  ent:createProcess/proc:hibernate <=> proc:wake/ent:wakeProcess
  proc:sleep/proc:compute <=> ent:wakeProcess
*/
class Process {
    constructor(name, genFun, thisEntity, thisParent) {
        if (genFun.constructor.name !== 'GeneratorFunction') throw new Error("Process function has to be a generator function");
        //Instantiates a Simian process
        this.name = name;
        this.genFun = genFun;
        //this.co = null;
        this.started = false;
        this.entity = thisEntity;
        this.parent = thisParent; //Parent is undefined if created by entity

        //These optional members will be created as needed at runtime to conserve memory:
        //At present, on OSX, per process memory need is about 2842 bytes for simple processes
        //this._kindSet = null; //Set of kinds that it belongs to on its entity
        //this._childList = null; //Set of kinds that it belongs to on its entity

        this.state = "suspended"; //"running"/"suspended"

        this.hibernate = hibernate.bind(this);
        this.sleep = sleep.bind(this);
    }

    toString() {
        return "Process '" + this.name + "' running on " + this.entity;
    }

    wake(...theArgs) {
        //Arguments "...theArgs" to __call => function-body
        //Arguments "...theArgs" to wake => LHS of hibernate
        var co = this.co;
        if (co && (this.state == "suspended")) {
            //print("wake: '" + this.name +"': " + co + " Arguments: " + theArgs);
            this.state = "running";
            return co.next(...theArgs); //NOTE: Here, theArgs goes to LHS of previous hibernate's 'yield*'
        }
        else throw new Error("Attempted to wake a process which was either killed or not suspended: " + this.name);
    }

    categorize(kind, name) {
        var entity = this.entity;
        entity.categorize(kind, name); //Also categorize as @kind on entity
    }

    unCategorize(kind, name) {
        var entity = this.entity;
        entity.unCategorize(kind, name); //Also categorize as @kind on entity
    }

    spawn(name, fun, kind) {
        //Create a new named processes as child or @kind
        var entity = this.entity;
        if (entity._procList[name]) throw new Error("spawn: Process by name '" + name + "' already exists in entity " + entity.name + "[" + entity.num + "]");
        entity.createProcess(name, fun, kind); //Creates a named process of kind type
               //Make this a child of "this"
               //NOTE: This is the difference between process:spawn and entity:createProcess
               entity._procList[name].parent = this;
               if (this._childList) this._childList[name] = true;
               else this._childList = {name : true};
    }

    kill(name) { //Kills itself, or named child-process
        //name: One of undefined or process-name
        var entity = this.entity;
        var parent = this.parent;
        if (!name) { //Kill self
            killallChildren(this); //Killall children recursively
            this.co = undefined;
            //Parent process is guaranteed to be alive
            if (parent) parent._childList[this.name] = undefined; //Remove from child-list of parent
            //Parent entity is always guaranteed to be alive
            //Remove references from entity category and process lists
            if (this._kindSet)
                for (let k in this._kindSet)
                    if (this._kindSet.hasOwnProperty(k)) entity._category[k][this.name] = undefined;
            entity._procList[this.name] = undefined; //Remove all references to this process
        }
        else if (name == "*") _killallChildren(this); //Kill every chid-process
        else if (this._childList && this._childList[name])
            entity._procList[name].kill(); //Is this a child process? If yes, kill it
    }

    is_a(kind) {
        var name = this.name;
        var entity = this.entity;
        if (entity._category[kind] && entity._category[kind][name] && entity._procList[name])
            return true; //Is indeed a @kind?
        return false;
    }

    getCategoryNames() {
        var kindSet = [];
        if (this._kindSet) {
            let n = 0;
            for (let k in this._kindSet) {
                if (this._kindSet.hasOwnProperty(k)) kindSet[n++] = k;
            }
        }
        return kindSet;
    }

    getChildNames() {
        var nameSet = [];
        if (this._childList) {
            let n = 0;
            for (let k in this._childList) {
                if (this._childList.hasOwnProperty(k)) nameSet[n++] = k;
            }
        }
        return nameSet;
    }

    status() {
        if (this.started) return this.state;
        else return "NotStarted";
    }
}
