#Copyright (c) 2015, Los Alamos National Security, LLC
#All rights reserved.
#
#Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.
#
#Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
#	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
#	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
#THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Author: Nandakishore Santhi
#Date: 23 November, 2014
#Copyright: Open source, must acknowledge original author
#Purpose: PDES Engine in Python, mirroring a subset of the Simian JIT-PDES
#  Simple example simulation scipt to test runtime attaching of services
from SimianPie.simian import Simian
import random, math

#Initialize Simian
simName, startTime, endTime, minDelay, useMPI = "HELLO-ATTACH", 0, 100000, 0.0001, True
simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

count = 4

class Alice(simianEngine.Entity):
    def __init__(self, baseInfo, *args):
        super(Alice, self).__init__(baseInfo)

    def generate(self, *args):
        targets = [{"entity": "Alice", "service": "square"},
                    {"entity": "Bob", "service": "sqrt"}]

        target = targets[int(random.randrange(len(targets)))]
        targetId = int(random.randrange(count))

        data = random.randrange(100)

        self.out.write("Time "
                    + str(self.engine.now)
                    + ": Sending " + str(data) + " to "
                    + target["entity"] + "[" + str(targetId) + "]\n")

        self.reqService(10, target["service"], data, target["entity"], targetId)
        self.reqService(25, "generate", None, None, None)

    def result(self, data, tx, txId):
        self.out.write("Time "
                    + str(self.engine.now)
                    + ": Got " + str(data)
                    + " from " + tx + "[" + str(txId) + "]\n")

def square(self, data, tx, txId): #Service to attach to class Alice at runtime
    self.reqService(10, "result", data^2, tx, txId)

class Bob(simianEngine.Entity):
    def __init__(self, baseInfo, *args):
        super(Bob, self).__init__(baseInfo)

    def generate(self, *args):
        targets = [{"entity": "Alice", "service": "square"},
                    {"entity": "Bob", "service": "sqrt"}]

        target = targets[int(random.randrange(len(targets)))]
        targetId = int(random.randrange(count))

        data = random.randrange(100)

        self.out.write("Time "
                    + str(self.engine.now)
                    + ": Sending " + str(data) + " to "
                    + target["entity"] + "[" + str(targetId) + "]\n")

        self.reqService(10, target["service"], data, target["entity"], targetId)
        self.reqService(25, "generate", None, None, None)

    def result(self, data, tx, txId):
        self.out.write("Time "
                    + str(self.engine.now)
                    + ": Got " + str(data)
                    + " from " + tx + "[" + str(txId) + "]\n")

def sqrt(self, data, tx, txId):
    self.reqService(10, "result", math.sqrt(data), tx, txId)

for i in xrange(count):
    simianEngine.addEntity("Alice", Alice, i) #Additional arguments, if given are passed to Alice.__init__()
    simianEngine.addEntity("Bob", Bob, i)

#Example showing how to attach services to klasses and instances at runtime
simianEngine.attachService(Alice, "square", square) #Attach square service at runtime to klass Alice
for i in xrange(count): #Attach sqrt service at runtime to all Bob instances
    entity = simianEngine.getEntity("Bob", i)
    if entity:
        entity.attachService("sqrt", sqrt)

for i in xrange(count):
    simianEngine.schedService(0, "generate", None, "Alice", i)
    simianEngine.schedService(50, "generate", None, "Bob", i)

simianEngine.run()
simianEngine.exit()
