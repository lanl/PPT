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
#  Simple example simulation script for PHOLD with application process and child processes
from SimianPie.simian import Simian
import random, math

simName, startTime, endTime, minDelay, useMPI = "CHILD", 0, 10000, 0.0001, True
simianEngine = Simian(simName, startTime, endTime, minDelay, useMPI)

count = 10
lookahead = minDelay

def exponential(mean):
    return -math.log(random.random())/mean

#A slightly convoluted example of a process and its child on an entity
def appProcess(this, child=False): #Here arg(1) "this" is current process
    entity = this.entity
    childStarted, childKilled = False, True #Init local flags
    entity.out.write("Process " + this.name + " started\n")
    while True:
        x = random.randrange(100)

        #Shows how to log outputs
        entity.out.write("Time " + str(entity.engine.now)
            + ": Process " + this.name + " is sleeping for " + str(x) + "\n")

        #Shows how to spawn and start "$child" processes
        #One can do same with an-arbitrary-string as a @kind
        if not (childStarted or child):
            this.spawn("AppChild", appProcess, "AppChildKind") #Create a new child process using this same function ;-)

            #Shows an example of finding status of a process
            entity.out.write("Time " + str(entity.engine.now)
                + ": Process AppChild status: " + entity.statusProcess("AppChild") + "\n")

            entity.startProcess("AppChild", True) #Start "AppChild" process
            entity.out.write("Time " + str(entity.engine.now)
                + ": Process AppChild status: " + entity.statusProcess("AppChild") + "\n")
            childStarted = True #Adjust local flags
            childKilled = False #Adjust local flags

        #Shows how to sleep for specified time periods
        this.sleep(x)
        entity.out.write("Time " + str(entity.engine.now)
            + ": Waking up Process " + this.name + "\n")

        entity.out.write("Time " + str(entity.engine.now)
                + ": Process AppChild status: " + entity.statusProcess("AppChild") + "\n")

        #Shows how to retrieve process/child/category names
        processNames = entity.getProcessNames()
        entity.out.write(entity.name + " Entity's Process Names: ")
        for _,v in processNames.iteritems():
            entity.out.write(v + ", ")
        entity.out.write("\n")

        categoryNames = entity.getCategoryNames()
        entity.out.write(entity.name + " Entity's Category Names: ")
        for _,v in categoryNames.iteritems():
            entity.out.write(v + ", ")
        entity.out.write("\n")

        childNames = this.getChildNames()
        entity.out.write(this.name + " Process's Child Names: ")
        for _,v in childNames.iteritems():
            entity.out.write(v + ", ")
        entity.out.write("\n")

        kindNames = this.getCategoryNames()
        entity.out.write(this.name + " Process's Kind Names: ")
        for _,v in kindNames.iteritems():
            entity.out.write(v + ", ")
        entity.out.write("\n")

        #Shows how to kill child processes
        #One can do same with an-arbitrary-string as a @kind
        if (entity.engine.now > 100) and not childKilled:
            entity.out.write("Time " + str(entity.engine.now)
                + ": Killing Child Process AppChild in " + this.name + "\n")
            this.kill("AppChild")
            childKilled = True #Adjust local flags

class Node(simianEngine.Entity):
    def __init__(self, baseInfo, *args):
        super(Node, self).__init__(baseInfo)

        self.createProcess("App", appProcess, "AppKind") #Create "App"
        self.startProcess("App") #Start "App" process

    def generate(self, *args):
        targetId = random.randrange(count)
        offset = exponential(1) + lookahead

        self.out.write("Time "
                + str(self.engine.now)
                + ": Waking " + str(targetId)
                + " at " + str(offset) + " from now\n")

        self.reqService(offset, "generate", None, "Node", targetId)

for i in xrange(count):
    simianEngine.addEntity("Node", Node, i)

for i in xrange(count):
    simianEngine.schedService(0, "generate", None, "Node", i)

simianEngine.run()
simianEngine.exit()
