#Author: Nandakishore Santhi
#Date: 23 November, 2014
#Copyright: Open source, must acknowledge original author
#Purpose: PDES Engine in Python, mirroring a subset of the Simian JIT-PDES
#  Named process class
from utils import SimianError
from greenlet import greenlet
try:
    from greenlet import greenlet
except:
    raise SimianError("Please install greenlets before using SimianPie to run simulations")

#Making this pythonic - this is a base class that all derived Processes will inherit from
class Process(object):
    #ent.createProcess/proc.hibernate <=> proc.wake/ent.wakeProcess
    #proc.sleep/proc.compute <=> ent.wakeProcess
    def __init__(self, name, fun, thisEntity, thisParent):
        self.name = name
        self.co = greenlet(run=fun)
        self.started = False
        self.suspended = False
        self.main = greenlet.getcurrent() #To hold the main process for to/from context-switching within sleep/wake/hibernate

        self.entity = thisEntity
        self.parent = thisParent #Parent is None if created by entity

        self._kindSet = {} #Set of kinds that it belongs to on its entity
        self._childList = {}

    def wake(thisProcess, *args):
        #Arguments "*args" to __call => function-body
        #Arguments "*args" to wake => LHS of hibernate
        co = thisProcess.co
        if co != None and not co.dead:
            thisProcess.main = greenlet.getcurrent()
            thisProcess.suspended = False
            return co.switch(*args)
        else:
            raise SimianError("Attempted to wake a process: " + thisProcess.name + " failed")

    def hibernate(thisProcess, *args):
        #Arguments "*args" to hibernate => LHS of wake
        #Processes to be woken explicitly by events may return values
        thisProcess.suspended = True
        return thisProcess.main.switch(*args)

    def sleep(thisProcess, x, *args):
        #Processes which are to implicitly wake at set timeouts
        #All return values are passed to __call/wake
        if (not isinstance(x, (int, long, float))) or (x < 0):
            raise SimianError("Sleep not given non-negative number argument!" + thisProcess.name)

        entity = thisProcess.entity
        #Schedule a local alarm event after x timesteps to wakeup
        entity.engine.schedService(entity.engine.now + x, "_wakeProcess",
                        thisProcess.name, entity.name, entity.num)
        thisProcess.suspended = True
        return thisProcess.main.switch(*args)

    def categorize(thisProcess, kind, name):
        entity = thisProcess.entity
        entity.categorize(kind, name) #Also categorize as @kind on entity

    def unCategorize(thisProcess, kind, name):
        entity = thisProcess.entity
        entity.unCategorize(kind, name) #Also categorize as @kind on entity

    def spawn(thisProcess, name, fun, kind=None):
        #Create a new named processes as child or @kind
        entity = thisProcess.entity
        if name in entity._procList:
            raise SimianError("spawn: Process by name '" + name + "' already exists in entity " + entity.name + "[" + str(entity.num) + "]")

        entity.createProcess(name, fun, kind) #Creates a named process of kind type
        #Make this a child of thisProcess
        #NOTE: This is the difference between process.spawn and entity.createProcess
        entity._procList[name].parent = thisProcess
        thisProcess._childList[name] = True

    def _killallChildren(thisProcess): #Hidden function to kill all children
        entity = thisProcess.entity
        for name,_ in thisProcess._childList.items(): #So we can delete stuff in _childList
            proc = entity._procList[name] #Get actual process
            proc.kill() #Kill child and all its subprocesses
        thisProcess._childList = {} #A new child table

    def kill(thisProcess, name=None): #Kills itself, or named child-process
        #name: One of None or process-name
        entity = thisProcess.entity
        parent = thisProcess.parent
        if name == None: #Kill self
            thisProcess._killallChildren() #Killall children recursively

            #Parent process is guaranteed to be alive
            if parent: #Remove from child-list of parent
                parent._childList.pop(thisProcess.name)

            #Remove references from entity category and process lists
            for k in thisProcess._kindSet:
                entity._category[k].pop(thisProcess.name)

            entity._procList.pop(thisProcess.name) #Remove all references to this process

            co = thisProcess.co
            thisProcess.co = None
            co.throw() #Raise greenlet.GreenletExit
        elif name == "*": #Kill every chid-process
            self._killallChildren()
        elif thisProcess._childList[name]: #Is this a child process?
            proc = entity._procList[name]
            proc.kill() #Kill it

    def is_a(thisProcess, kind):
        name = thisProcess.name
        entity = thisProcess.entity
        if (kind in entity._category) and (name in entity._category[kind]) and (name in entity._procList): #Is indeed a @kind?
            return True
        return False

    def getCategoryNames(thisProcess):
        kindSet = {}
        n = 1
        for k in thisProcess._kindSet:
            kindSet[n] = k
            n = n + 1
        return kindSet

    def getChildNames(thisProcess):
        nameSet = {}
        n = 1
        for k in thisProcess._childList:
            nameSet[n] = k
            n = n + 1
        return nameSet

    def status(thisProcess):
        if thisProcess.started:
            try:
                co = thisProcess.co
                if co.dead:
                    return "dead"
                elif thisProcess.suspended:
                    return "suspended"
                else:
                    return "running"
            except:
                return "NonExistent"
        else:
            return "NotStarted"
