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
  Simple example simulation script for PHOLD with application process
*/
masala.io.load("./simian.js")

const simName="PROC", startTime=0, endTime=10000, minDelay=0.0001, useMPI=false;
//const count = 4500;
const count = 10;
const lookahead = minDelay;

const ln = Math.log, random = masala.random.uniform, range = masala.random.range, floor = Math.floor;
function exponential(lambda) { return -ln(random())/lambda; }

//Example of a process on an entity
function* appProcess(self, data1, data2) { //Here "self" is current process
    var entity = self.entity;
    entity.out.write(self + "' started with data: " + data1 + ", " + data2 + "\n");
    while (true) {
        var x = range(100);
        //Shows how to log outputs
        entity.out.write("Time " + entity.engine.now + ": " + self + ": Process App is sleeping for " + x + "\n");
        yield* self.sleep(x); //Shows how to compute/sleep
        entity.out.write("Time " + entity.engine.now + ": " + self + ": Process App just woke up\n");
    }
}

class Node extends Entity {
    constructor(...theArgs) { //All agents have a constructor
        super(...theArgs);

        this.createProcess("App", appProcess); //Shows how to create "App"
        this.startProcess("App", 78783 + this.num, {"two" : 2}); //Shows how to start "App" process with arbitrary number of data
    }

    generate(self, ...theArgs) {
        var targetId = floor(range(count));
        var offset = exponential(1) + lookahead;

        //Shows how to log outputs
        self.out.write("Time " + self.engine.now + ": Waking " + targetId + " at " + offset + " from now\n");

        self.reqService(offset, "generate", null, "Node", targetId);
    }
}

//Initialize Simian
var SimianEngine = new Simian(simName, startTime, endTime, minDelay, useMPI);

for (var i=0; i<count; i++) SimianEngine.addEntity("Node", Node, i);
for (var i=0; i<count; i++) SimianEngine.schedService(0, "generate", undefined, "Node", i);

SimianEngine.run()
SimianEngine.exit()
