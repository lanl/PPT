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
  Simple example simulation scipt
*/
masala.io.load("./simian.js");
var random = masala.random.uniform, floor = Math.floor, sqrt = Math.sqrt;

//Initialize Simian
const simName="HELLO", startTime=0, endTime=100000, minDelay=0.0001, useMPI=true;
const count = 10;

class Alice extends Entity {
    constructor(...theArgs) { //All agents have a constructor
        super(...theArgs);
    }

    generate(self, ...theArgs) {
        var targets = [{"entity" : "Alice", "service" : "square"},
                         {"entity" : "Bob", "service" : "sqrt"}];

        var target = targets[floor(random()*targets.length)];
        var targetId = floor(random()*count);

        var data = random()*100;

        self.out.write(self + ": Time "
                    + self.engine.now
                    + ": Sending " + data + " to "
                    + target.entity + "[" + targetId + "]\n");

        self.reqService(10, target.service, data, target.entity, targetId);
        self.reqService(25, "generate");
    }

    square(self, data, tx, txId) {
        self.reqService(10, "result", data*data, tx, txId);
    }

    result(self, data, tx, txId) {
        self.out.write(self + ": Time "
                    + self.engine.now
                    + ": Got " + data
                    + " from " + tx + "[" + txId + "]\n");
    }
}

class Bob extends Entity {
    constructor(...theArgs) { //All agents have a constructor
        super(...theArgs);
    }

    generate(self, ...theArgs) {
        var targets = [{"entity" : "Alice", "service" : "square"},
                         {"entity" : "Bob", "service" : "sqrt"}];

        var target = targets[floor(random()*targets.length)];
        var targetId = floor(random()*count);

        var data = random()*100;

        self.out.write(self + ": Time "
                    + self.engine.now
                    + ": Sending " + data + " to "
                    + target.entity + "[" + targetId + "]\n");

        self.reqService(10, target.service, data, target.entity, targetId);
        self.reqService(25, "generate");
    }

    sqrt(self, data, tx, txId) {
        self.reqService(10, "result", sqrt(data), tx, txId);
    }

    result(self, data, tx, txId) {
        self.out.write(self + ": Time "
                    + self.engine.now
                    + ": Got " + data
                    + " from " + tx + "[" + txId + "]\n");
    }
}

var SimianEngine = new Simian(simName, startTime, endTime, minDelay, useMPI);

for (var i=0; i<count; i++) {
    SimianEngine.addEntity("Alice", Alice, i);
    SimianEngine.addEntity("Bob", Bob, i);
}

for (var i=0; i<count; i++) {
    SimianEngine.schedService(0, "generate", undefined, "Alice", i);
    SimianEngine.schedService(50, "generate", undefined, "Bob", i);
}

SimianEngine.run();
SimianEngine.exit();
