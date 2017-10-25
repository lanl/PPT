#Author: Nandakishore Santhi
#Date: 23 November, 2014
#Copyright: Open source, must acknowledge original author
#Purpose: PDES Engine in Python, mirroring a subset of the Simian JIT-PDES
#  Named entity class with inheritence and processes
from utils import SimianError
import heapq
from process import Process
import types #Used to bind Service at runtime to specific instances

#Making this pythonic - this is a base class that all derived Entity classes will inherit from
class Entity(object):
    def __init__(self, initInfo):
        #Constructor of derived entity to be called as <entityName>(name, out, engine, num, reqServiceProxy, <args>)
        #Here <args> are any additional arguments needed in the derived entity-class's __init__() method
        self.name = initInfo["name"]
        self.out = initInfo["out"] #Log file for this instance
        self.engine = initInfo["engine"] #Engine
        self.num = initInfo["num"] #Serial Number
        self._procList = {} #A separate process table for each instance
        self._category = {} #A map of sets for each kind of process

    def reqService(self, offset, eventName, data, rx=None, rxId=None):
        #Purpose: Send an event if Simian is running.
        engine = self.engine #Get the engine for this entity

        if rx != None and offset < engine.minDelay:
            if not engine.running: raise SimianError("Sending event when Simian is idle!")
            #If sending to self, then do not check against min-delay
            raise SimianError(self.name + "[" + str(self.num) + "]"
                + " attempted to send with too little delay")

        if rx == None: rx = self.name
        if rxId == None: rxId = self.num
        time = engine.now + offset

        e = {
                "tx": self.name, #String
                "txId": self.num, #Number
                "rx": rx, #String
                "rxId": rxId, #Number
                "name": eventName, #String
                "data": data, #Object
                "time": time, #Number
            }

        recvRank = engine.getOffsetRank(rx, rxId)

        if recvRank == engine.rank: #Send to self
            heapq.heappush(engine.eventQueue, (time, e))
        else:
            if time < engine.minSent: engine.minSent = time
            engine.comm.send(e, recvRank) #Send to others

    def attachService(self, name, fun):
        #Attaches a service at runtime to instance
        setattr(self, name, types.MethodType(fun, self))

    #Following code is to support coroutine processes on entities:
    #Entity methods to interact with processes
    def createProcess(self, name, fun, kind=None): #Creates a named process
        if name == "*":
            raise SimianError("Reserved name to represent all child processes: " + name)
        proc = Process(name, fun, self, None) #No parent means, entity is parent
        if not proc:
            raise SimianError("Could not create a valid process named: " + name)
        self._procList[name] = proc
        if kind != None:
            self.categorizeProcess(kind, name) #Categorize

    def startProcess(self, name, *args): #Starts a named process
        if name in self._procList:
            proc = self._procList[name]
            if not proc.started:
                proc.started = True
                #When starting, pass process instance as first arg, which can be accessed inside the "fun"
                return proc.wake(proc, *args)
            else:
                raise SimianError("Starting an already started process: " + proc.name)

    def _wakeProcess(self, name, tx=None, txId=None): #Hidden: implicit wake a named process without arguments
        if name in self._procList:
            proc = self._procList[name]
            proc.wake()

    def wakeProcess(self, name, *args): #Wake a named process with arguments
        if not (name in self._procList):
            raise SimianError("Attempted to wake a non existant process: " + name)
        else: #If existing and not been killed asynchronously
            proc = self._procList[name]
            proc.wake(*args)

    def killProcess(self, name): #Kills named process or all entity-processes
        if name: #Kills named child-process
            proc = self._procList[name]
            proc.kill() #Kill itself and all subprocesses
        else: #Kills all subprocesses
            for _,proc in self._procList.items(): #So we can delete while iterating
                proc.kill() #Kill itself and all subprocesses
            self._procList = {} #A new process table

    def killProcessKind(self, kind): #Kills all @kind processes on entity
        if not (kind in self._category):
            raise SimianError("killProcessKind: No category of processes on this entity called " + str(kind))
        else:
            nameSet = self._category[kind]
            for name,_ in nameSet.items(): #So we can delete while iterating
                proc = self._procList[name]
                proc.kill() #Kill itself and all subprocesses

    def statusProcess(self, name):
        if not (name in self._procList):
            return "NonExistent"
        else:
            proc = self._procList[name]
            return proc.status()

    def categorizeProcess(self, kind, name): #Check for existing process and then categorize
        if name in self._procList:
            proc = self._procList[name]
            #Categorize both ways for easy lookup
            proc._kindSet[kind] = True #Indicate to proc that it is of this kind to its entity
            #Indicate to entity that proc is of this kind
            if not kind in self._category: self._category[kind] = {name: True} #New kind
            else: self._category[kind][name] = True #Existing kind
        else: raise SimianError("categorize: Expects a proper child to categorize")

    def unCategorizeProcess(self, kind, name):
        #Check for existing process and then unCategorize
        if name in self._procList:
            proc = self._procList[name]
            #unCategorize both ways for easy lookup
            proc._kindSet.pop(kind) #Indicate to proc that it is not of this kind to its entity
            #Indicate to entity that proc is not of this kind
            if kind in self._category:
                self._category[kind].pop(name) #Existing kind deleted
        else: raise SimianError("unCategorize: Expects a proper child to un-categorize")

    def isProcess(self, name, kind):
        if name in self._procList:
            proc = self._procList[name]
            return proc.is_a(kind)
        else:
            return False

    def getProcess(self, name):
        #A reference to a named process is returned if it exists
        #NOTE: User should delete it to free its small memory when no longer needed
        if name in self._procList:
            proc = self._procList[name]
            return proc
        else: return None

    def getCategoryNames(self):
        kindSet = {}
        n = 1
        for k in self._category:
            kindSet[n] = k
            n = n + 1
        return kindSet

    def getProcessNames(self):
        nameSet = {}
        n = 1
        for k in self._procList:
            nameSet[n] = k
            n = n + 1
        return nameSet
