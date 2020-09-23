##############################################################################
# (c) Copyright 2015-. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.
#
# All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
#
# This is open source software; you can redistribute it and/or modify it under the terms of the BSD 3-clause License. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL. Full text of the BSD 3-clause License can be found in the License file in the main development branch of the repository.
#
##############################################################################
# BSD 3-clause license:
# Copyright 2015- Triad National Security, LLC
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
##############################################################################
# Author: Nandakishore Santhi
# Date: 23 November, 2014
# Copyright: Open source, must acknowledge original author
# Purpose: PDES Engine in CPython and PyPy, mirroring most of the original LuaJIT version of Simian JIT-PDES
# NOTE: If speed rivaling C/C++ PDES engines is desired, consider adopting the LuaJIT version of Simian JIT-PDES
# NOTE: SimianPie External Dependencies: Other than standard Python3/Python2 modules, we will optionally need:
#   greenlet module: For Simian process functionality only
#   libmpich.[so/dylib/dll] shared library: For Simian parallel DES functionality only
#
##############################################################################
# Changelog:
#
# NOTE: 11/232014: Author: Nandakishore Santhi
# Original version of Simian for Lua, Python-2.7, and Javascript
#
# NOTE: 4/21/2020: Changes: Stephan Eidenbenz
# Simian for Python 3.7
#
# NOTE: 5/9/2020: Changes: Nandakishore Santhi
# Combined all Simian modules into a single standalone module file
# Updated LICENSE and COPYRIGHT notices
# Version bumped to 1.55
#
# NOTE: 5/12/2020: Changes: Nandakishore Santhi
# Added ability to switch context to a historical list of older processes
#  when waking up from sleep. This is done using a stack by append/pop
# Made SimianError messages less cryptic at many places. More improvements are possible for later.
# Version bumped to 1.65 to better match prior version history
# Added a simulation related header/footer for all output log files
#
##############################################################################

from __future__ import print_function
import os, sys
import hashlib, heapq
import time as timeLib
import types #Used to bind Service at runtime to specific instances
import ctypes as C #For FFI of MPICH

__SimianVersion__ = "1.65"

# If mpich library is not explicitly provided when Simian is invoked with useMPI=True, then check for libmpich in parent directory of this file
defaultMpichLibName = os.path.join(os.path.dirname(__file__), "..", "libmpich.dylib")

#===========================================================================================
# Select a good performance timer clock based on Python version
#===========================================================================================
if sys.version_info >= (3, 7): perfTime = timeLib.perf_counter
elif sys.version_info >= (2, 7): perfTime = timeLib.clock
else: SimianError("Python version is not >= 2.7 or 3.7")

#===========================================================================================
# utils.py
#===========================================================================================
class SimianError(Exception):
    def __init__(self, value): self.value = str(value)

    def __str__(self): return self.value
#===========================================================================================

#===========================================================================================
# process.py
#===========================================================================================
greenlet = None
# This is a base class that all derived Processes will inherit from
class Process(object):
    #ent.createProcess/proc.hibernate <=> proc.wake/ent.wakeProcess
    #proc.sleep/proc.compute <=> ent.wakeProcess
    def __init__(self, name, fun, thisEntity, thisParent):
        global greenlet #Check for greenlet only if creating a process!
        if not greenlet:
            try:
                from greenlet import greenlet
            except:
                raise SimianError("process.__init__(): you have initialized a Simian process - please install greenlet before using SimianPie to run simulations")
        self.name = name
        self.co = greenlet(run=fun) ###
        self.started = False
        self.suspended = False
        self.main = [greenlet.getcurrent()] #To hold the main process for to/from context-switching within sleep/wake/hibernate

        self.entity = thisEntity
        self.parent = thisParent #Parent is None if created by entity

        self._kindSet = {} #Set of kinds that it belongs to on its entity
        self._childList = {}

    def wake(thisProcess, *args):
        #Arguments "*args" to __call => function-body
        #Arguments "*args" to wake => LHS of hibernate
        co = thisProcess.co
        if co != None and not co.dead:
            thisProcess.main.append(greenlet.getcurrent())
            thisProcess.suspended = False
            return co.switch(*args)
        else:
            raise SimianError("process.wake(): attempt to wake a process: " + thisProcess.name + " failed")

    def hibernate(thisProcess, *args):
        #Arguments "*args" to hibernate => LHS of wake
        #Processes to be woken explicitly by events may return values
        thisProcess.suspended = True
        if len(thisProcess.main) == 0:
            raise SimianError("process.hibernate(): attempt to context switch out of process: " + thisProcess.name + " failed")
        return thisProcess.main.pop().switch(*args)

    def sleep(thisProcess, x, *args):
        #Processes which are to implicitly wake at set timeouts
        #All return values are passed to __call/wake
        if (not isinstance(x, (int, float))) or (x < 0):
            raise SimianError("process.sleep(): not given non-negative number argument!" + thisProcess.name)

        entity = thisProcess.entity
        #Schedule a local alarm event after x timesteps to wakeup
        entity.engine.schedService(entity.engine.now + x, "_wakeProcess",
                        thisProcess.name, entity.name, entity.num)
        thisProcess.suspended = True
        if len(thisProcess.main) == 0:
            raise SimianError("process.sleep(): attempt to context switch out of process: " + thisProcess.name + " failed")
        return thisProcess.main.pop().switch(*args)

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
            raise SimianError("process.spawn(): process by name '" + name + "' already exists in entity " + entity.name + "[" + str(entity.num) + "]")

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
            thisProcess._killallChildren()
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
#===========================================================================================

#===========================================================================================
# entity.py
#===========================================================================================
# This is a base class that all derived Entity classes will inherit from
class Entity(object):
    def __init__(self, initInfo):
        #Constructor of derived entity to be called as <entityName>(name, out, engine, num, reqServiceProxy, <args>)
        #Here <args> are any additional arguments needed in the derived entity-class's __init__() method
        self.name = initInfo["name"]
        self.out = initInfo["out"] #Log file for this instance
        self.engine = initInfo["engine"] #Engine ... this will be the loop in asyncio
        self.num = initInfo["num"] #Serial Number
        self._procList = {} #A separate process table for each instance
        self._category = {} #A map of sets for each kind of process

    def __str__(self):
        return self.name + "(" + str(self.num) + ")"

    def reqService(self, offset, eventName, data, rx=None, rxId=None):
        #Purpose: Send an event if Simian is running.
        engine = self.engine #Get the engine for this entity

        if rx != None and offset < engine.minDelay:
            if not engine.running: raise SimianError("entity.reqService(): sending event when Simian is idle!")
            #If sending to self, then do not check against min-delay
            raise SimianError("entity.reqService(): " + self.name + "[" + str(self.num) + "]" + " attempted to send with too little delay")

        time = engine.now + offset
        if time > engine.endTime: #No need to send this event
            return

        if rx == None: rx = self.name
        if rxId == None: rxId = self.num
        e = {
                "tx": self.name, #String
                "txId": self.num, #Number
                "rx": rx, #String
                "rxId": rxId, #Number
                "name": eventName, #String
                "data": data, #Object
                "time": time, #Number
            }

        # this is a particular mechanism added by Jason Liu for
        # allowing different mappings from LPs to ranks
        recvRank = engine.getOffsetRank(rx, rxId)

        if recvRank == engine.rank: #Send to self
            self.engine.ec += 1
            heapq.heappush(engine.eventQueue, (time, self.engine.ec, e))
        else:
            engine.MPI.sendAndCount(e, recvRank)

    def attachService(self, name, fun):
        #Attaches a service at runtime to instance
        setattr(self, name, types.MethodType(fun, self))

    #Following code is to support coroutine processes on entities:
    #Entity methods to interact with processes
    def createProcess(self, name, fun, kind=None): #Creates a named process
        if name == "*":
            raise SimianError("entity.createProcess(): reserved name to represent all child processes: " + name)
        proc = Process(name, fun, self, None) #No parent means, entity is parent
        if not proc:
            raise SimianError("entity.createProcess(): could not create a valid process named: " + name)
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
               raise SimianError("entity.startProcess(): starting an already started process: " + proc.name)

    def _wakeProcess(self, name, tx=None, txId=None): #Hidden: implicit wake a named process without arguments
        if name in self._procList:
            proc = self._procList[name]
            return proc.wake()

    def wakeProcess(self, name, *args): #Wake a named process with arguments
        if not (name in self._procList):
            raise SimianError("entity.wakeProcess(): attempted to wake a non existant process: " + name)
        else: #If existing and not been killed asynchronously
            proc = self._procList[name]
            return proc.wake(*args)

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
            raise SimianError("entity.killProcessKind(): no category of processes on this entity called " + str(kind))
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
        else: raise SimianError("entity.categorizeProcess(): expects a proper child to categorize")

    def unCategorizeProcess(self, kind, name):
        #Check for existing process and then unCategorize
        if name in self._procList:
            proc = self._procList[name]
            #unCategorize both ways for easy lookup
            proc._kindSet.pop(kind) #Indicate to proc that it is not of this kind to its entity
            #Indicate to entity that proc is not of this kind
            if kind in self._category:
                self._category[kind].pop(name) #Existing kind deleted
        else: raise SimianError("entity.unCategorizeProcess(): expects a proper child to un-categorize")

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
#===========================================================================================

#===========================================================================================
# umsgPack.py
# umsgpack-python-pure can be substituted with msgpack-pure or msgpack-python
# NOTE: This module is under a different licence till this message appears again
#===========================================================================================
# u-msgpack-python v2.1 - vsergeev at gmail
# https://github.com/vsergeev/u-msgpack-python
#
# u-msgpack-python is a lightweight MessagePack serializer and deserializer
# module, compatible with both Python 2 and 3, as well CPython and PyPy
# implementations of Python. u-msgpack-python is fully compliant with the
# latest MessagePack specification.com/msgpack/msgpack/blob/master/spec.md). In
# particular, it supports the new binary, UTF-8 string, and application ext
# types.
#
# MIT License
#
# Copyright (c) 2013-2014 Ivan A. Sergeev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
"""
u-msgpack-python v2.1 - vsergeev at gmail
https://github.com/vsergeev/u-msgpack-python

u-msgpack-python is a lightweight MessagePack serializer and deserializer
module, compatible with both Python 2 and 3, as well CPython and PyPy
implementations of Python. u-msgpack-python is fully compliant with the
latest MessagePack specification.com/msgpack/msgpack/blob/master/spec.md). In
particular, it supports the new binary, UTF-8 string, and application ext
types.

License: MIT
"""
__version__ = "2.1"
"Module version string"

version = (2,1)
"Module version tuple"

import struct
import collections
import sys
import io

################################################################################
### Ext Class
################################################################################

# Extension type for application-defined types and data
class Ext:
    """
    The Ext class facilitates creating a serializable extension object to store
    an application-defined type and data byte array.
    """

    def __init__(self, type, data):
        """
        Construct a new Ext object.

        Args:
            type: application-defined type integer from 0 to 127
            data: application-defined data byte array

        Raises:
            TypeError:
                Specified ext type is outside of 0 to 127 range.

        Example:
        >>> foo = umsgpack.Ext(0x05, b"\x01\x02\x03")
        >>> umsgpack.packb({u"special stuff": foo, u"awesome": True})
        '\x82\xa7awesome\xc3\xadspecial stuff\xc7\x03\x05\x01\x02\x03'
        >>> bar = umsgpack.unpackb(_)
        >>> print(bar["special stuff"])
        Ext Object (Type: 0x05, Data: 01 02 03)
        >>>
        """
        # Application ext type should be 0 <= type <= 127
        if not isinstance(type, int) or not (type >= 0 and type <= 127):
            raise TypeError("ext type out of range")
        # Check data is type bytes
        elif sys.version_info[0] == 3 and not isinstance(data, bytes):
            raise TypeError("ext data is not type \'bytes\'")
        elif sys.version_info[0] == 2 and not isinstance(data, str):
            raise TypeError("ext data is not type \'str\'")
        self.type = type
        self.data = data

    def __eq__(self, other):
        """
        Compare this Ext object with another for equality.
        """
        return (isinstance(other, self.__class__) and
                self.type == other.type and
                self.data == other.data)

    def __ne__(self, other):
        """
        Compare this Ext object with another for inequality.
        """
        return not self.__eq__(other)

    def __str__(self):
        """
        String representation of this Ext object.
        """
        s = "Ext Object (Type: 0x%02x, Data: " % self.type
        for i in range(min(len(self.data), 8)):
            if i > 0:
                s += " "
            if isinstance(self.data[i], int):
                s += "%02x" % (self.data[i])
            else:
                s += "%02x" % ord(self.data[i])
        if len(self.data) > 8:
            s += " ..."
        s += ")"
        return s

################################################################################
### Exceptions
################################################################################

# Base Exception classes
class PackException(Exception):
    "Base class for exceptions encountered during packing."
    pass
class UnpackException(Exception):
    "Base class for exceptions encountered during unpacking."
    pass

# Packing error
class UnsupportedTypeException(PackException):
    "Object type not supported for packing."
    pass

# Unpacking error
class InsufficientDataException(UnpackException):
    "Insufficient data to unpack the encoded object."
    pass
class InvalidStringException(UnpackException):
    "Invalid UTF-8 string encountered during unpacking."
    pass
class ReservedCodeException(UnpackException):
    "Reserved code encountered during unpacking."
    pass
class UnhashableKeyException(UnpackException):
    """
    Unhashable key encountered during map unpacking.
    The serialized map cannot be deserialized into a Python dictionary.
    """
    pass
class DuplicateKeyException(UnpackException):
    "Duplicate key encountered during map unpacking."
    pass

# Backwards compatibility
KeyNotPrimitiveException = UnhashableKeyException
KeyDuplicateException = DuplicateKeyException

################################################################################
### Exported Functions and Globals
################################################################################

# Exported functions and variables, set up in __init()
pack = None
packb = None
unpack = None
unpackb = None
dump = None
dumps = None
load = None
loads = None

compatibility = False
"""
Compatibility mode boolean.

When compatibility mode is enabled, u-msgpack-python will serialize both
unicode strings and bytes into the old "raw" msgpack type, and deserialize the
"raw" msgpack type into bytes. This provides backwards compatibility with the
old MessagePack specification.

Example:
>>> umsgpack.compatibility = True
>>>
>>> umsgpack.packb([u"some string", b"some bytes"])
b'\x92\xabsome string\xaasome bytes'
>>> umsgpack.unpackb(_)
[b'some string', b'some bytes']
>>>
"""

################################################################################
### Packing
################################################################################

# You may notice struct.pack("B", obj) instead of the simpler chr(obj) in the
# code below. This is to allow for seamless Python 2 and 3 compatibility, as
# chr(obj) has a str return type instead of bytes in Python 3, and
# struct.pack(...) has the right return type in both versions.

def _pack_integer(obj, fp):
    if obj < 0:
        if obj >= -32:
            fp.write(struct.pack("b", obj))
        elif obj >= -2**(8-1):
            fp.write(b"\xd0" + struct.pack("b", obj))
        elif obj >= -2**(16-1):
            fp.write(b"\xd1" + struct.pack(">h", obj))
        elif obj >= -2**(32-1):
            fp.write(b"\xd2" + struct.pack(">i", obj))
        elif obj >= -2**(64-1):
            fp.write(b"\xd3" + struct.pack(">q", obj))
        else:
            raise UnsupportedTypeException("huge signed int")
    else:
        if obj <= 127:
            fp.write(struct.pack("B", obj))
        elif obj <= 2**8-1:
            fp.write(b"\xcc" + struct.pack("B", obj))
        elif obj <= 2**16-1:
            fp.write(b"\xcd" + struct.pack(">H", obj))
        elif obj <= 2**32-1:
            fp.write(b"\xce" + struct.pack(">I", obj))
        elif obj <= 2**64-1:
            fp.write(b"\xcf" + struct.pack(">Q", obj))
        else:
            raise UnsupportedTypeException("huge unsigned int")

def _pack_nil(obj, fp):
    fp.write(b"\xc0")

def _pack_boolean(obj, fp):
    fp.write(b"\xc3" if obj else b"\xc2")

def _pack_float(obj, fp):
    if _float_size == 64:
        fp.write(b"\xcb" + struct.pack(">d", obj))
    else:
        fp.write(b"\xca" + struct.pack(">f", obj))

def _pack_string(obj, fp):
    obj = obj.encode('utf-8')
    if len(obj) <= 31:
        fp.write(struct.pack("B", 0xa0 | len(obj)) + obj)
    elif len(obj) <= 2**8-1:
        fp.write(b"\xd9" + struct.pack("B", len(obj)) + obj)
    elif len(obj) <= 2**16-1:
        fp.write(b"\xda" + struct.pack(">H", len(obj)) + obj)
    elif len(obj) <= 2**32-1:
        fp.write(b"\xdb" + struct.pack(">I", len(obj)) + obj)
    else:
        raise UnsupportedTypeException("huge string")

def _pack_binary(obj, fp):
    if len(obj) <= 2**8-1:
        fp.write(b"\xc4" + struct.pack("B", len(obj)) + obj)
    elif len(obj) <= 2**16-1:
        fp.write(b"\xc5" + struct.pack(">H", len(obj)) + obj)
    elif len(obj) <= 2**32-1:
        fp.write(b"\xc6" + struct.pack(">I", len(obj)) + obj)
    else:
        raise UnsupportedTypeException("huge binary string")

def _pack_oldspec_raw(obj, fp):
    if len(obj) <= 31:
        fp.write(struct.pack("B", 0xa0 | len(obj)) + obj)
    elif len(obj) <= 2**16-1:
        fp.write(b"\xda" + struct.pack(">H", len(obj)) + obj)
    elif len(obj) <= 2**32-1:
        fp.write(b"\xdb" + struct.pack(">I", len(obj)) + obj)
    else:
        raise UnsupportedTypeException("huge raw string")

def _pack_ext(obj, fp):
    if len(obj.data) == 1:
        fp.write(b"\xd4" + struct.pack("B", obj.type & 0xff) + obj.data)
    elif len(obj.data) == 2:
        fp.write(b"\xd5" + struct.pack("B", obj.type & 0xff) + obj.data)
    elif len(obj.data) == 4:
        fp.write(b"\xd6" + struct.pack("B", obj.type & 0xff) + obj.data)
    elif len(obj.data) == 8:
        fp.write(b"\xd7" + struct.pack("B", obj.type & 0xff) + obj.data)
    elif len(obj.data) == 16:
        fp.write(b"\xd8" + struct.pack("B", obj.type & 0xff) + obj.data)
    elif len(obj.data) <= 2**8-1:
        fp.write(b"\xc7" + struct.pack("BB", len(obj.data), obj.type & 0xff) + obj.data)
    elif len(obj.data) <= 2**16-1:
        fp.write(b"\xc8" + struct.pack(">HB", len(obj.data), obj.type & 0xff) + obj.data)
    elif len(obj.data) <= 2**32-1:
        fp.write(b"\xc9" + struct.pack(">IB", len(obj.data), obj.type & 0xff) + obj.data)
    else:
        raise UnsupportedTypeException("huge ext data")

def _pack_array(obj, fp):
    if len(obj) <= 15:
        fp.write(struct.pack("B", 0x90 | len(obj)))
    elif len(obj) <= 2**16-1:
        fp.write(b"\xdc" + struct.pack(">H", len(obj)))
    elif len(obj) <= 2**32-1:
        fp.write(b"\xdd" + struct.pack(">I", len(obj)))
    else:
        raise UnsupportedTypeException("huge array")

    for e in obj:
        pack(e, fp)

def _pack_map(obj, fp):
    if len(obj) <= 15:
        fp.write(struct.pack("B", 0x80 | len(obj)))
    elif len(obj) <= 2**16-1:
        fp.write(b"\xde" + struct.pack(">H", len(obj)))
    elif len(obj) <= 2**32-1:
        fp.write(b"\xdf" + struct.pack(">I", len(obj)))
    else:
        raise UnsupportedTypeException("huge array")

    for k,v in obj.items():
        pack(k, fp)
        pack(v, fp)

########################################

# Pack for Python 2, with 'unicode' type, 'str' type, and 'long' type
def _pack2(obj, fp):
    """
    Serialize a Python object into MessagePack bytes.

    Args:
        obj: a Python object
        fp: a .write()-supporting file-like object

    Returns:
        None.

    Raises:
        UnsupportedType(PackException):
            Object type not supported for packing.

    Example:
    >>> f = open('test.bin', 'w')
    >>> umsgpack.pack({u"compact": True, u"schema": 0}, f)
    >>>
    """

    global compatibility

    if obj is None:
        _pack_nil(obj, fp)
    elif isinstance(obj, bool):
        _pack_boolean(obj, fp)
    elif isinstance(obj, int) or isinstance(obj, long):
        _pack_integer(obj, fp)
    elif isinstance(obj, float):
        _pack_float(obj, fp)
    elif compatibility and isinstance(obj, unicode):
        _pack_oldspec_raw(bytes(obj), fp)
    elif compatibility and isinstance(obj, bytes):
        _pack_oldspec_raw(obj, fp)
    elif isinstance(obj, unicode):
        _pack_string(obj, fp)
    elif isinstance(obj, str):
        _pack_binary(obj, fp)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        _pack_array(obj, fp)
    elif isinstance(obj, dict):
        _pack_map(obj, fp)
    elif isinstance(obj, Ext):
        _pack_ext(obj, fp)
    else:
        raise UnsupportedTypeException("unsupported type: %s" % str(type(obj)))

# Pack for Python 3, with unicode 'str' type, 'bytes' type, and no 'long' type
def _pack3(obj, fp):
    """
    Serialize a Python object into MessagePack bytes.

    Args:
        obj: a Python object
        fp: a .write()-supporting file-like object

    Returns:
        None.

    Raises:
        UnsupportedType(PackException):
            Object type not supported for packing.

    Example:
    >>> f = open('test.bin', 'w')
    >>> umsgpack.pack({u"compact": True, u"schema": 0}, fp)
    >>>
    """
    global compatibility

    if obj is None:
        _pack_nil(obj, fp)
    elif isinstance(obj, bool):
        _pack_boolean(obj, fp)
    elif isinstance(obj, int):
        _pack_integer(obj, fp)
    elif isinstance(obj, float):
        _pack_float(obj, fp)
    elif compatibility and isinstance(obj, str):
        _pack_oldspec_raw(obj.encode('utf-8'), fp)
    elif compatibility and isinstance(obj, bytes):
        _pack_oldspec_raw(obj, fp)
    elif isinstance(obj, str):
        _pack_string(obj, fp)
    elif isinstance(obj, bytes):
        _pack_binary(obj, fp)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        _pack_array(obj, fp)
    elif isinstance(obj, dict):
        _pack_map(obj, fp)
    elif isinstance(obj, Ext):
        _pack_ext(obj, fp)
    else:
        raise UnsupportedTypeException("unsupported type: %s" % str(type(obj)))

def _packb2(obj):
    """
    Serialize a Python object into MessagePack bytes.

    Args:
        obj: a Python object

    Returns:
        A 'str' containing serialized MessagePack bytes.

    Raises:
        UnsupportedType(PackException):
            Object type not supported for packing.

    Example:
    >>> umsgpack.packb({u"compact": True, u"schema": 0})
    '\x82\xa7compact\xc3\xa6schema\x00'
    >>>
    """
    fp = io.BytesIO()
    _pack2(obj, fp)
    return fp.getvalue()

def _packb3(obj):
    """
    Serialize a Python object into MessagePack bytes.

    Args:
        obj: a Python object

    Returns:
        A 'bytes' containing serialized MessagePack bytes.

    Raises:
        UnsupportedType(PackException):
            Object type not supported for packing.

    Example:
    >>> umsgpack.packb({u"compact": True, u"schema": 0})
    b'\x82\xa7compact\xc3\xa6schema\x00'
    >>>
    """
    fp = io.BytesIO()
    _pack3(obj, fp)
    return fp.getvalue()

################################################################################
### Unpacking
################################################################################

def _read_except(fp, n):
    data = fp.read(n)
    if len(data) < n:
        raise InsufficientDataException()
    return data

def _unpack_integer(code, fp):
    if (ord(code) & 0xe0) == 0xe0:
        return struct.unpack("b", code)[0]
    elif code == b'\xd0':
        return struct.unpack("b", _read_except(fp, 1))[0]
    elif code == b'\xd1':
        return struct.unpack(">h", _read_except(fp, 2))[0]
    elif code == b'\xd2':
        return struct.unpack(">i", _read_except(fp, 4))[0]
    elif code == b'\xd3':
        return struct.unpack(">q", _read_except(fp, 8))[0]
    elif (ord(code) & 0x80) == 0x00:
        return struct.unpack("B", code)[0]
    elif code == b'\xcc':
        return struct.unpack("B", _read_except(fp, 1))[0]
    elif code == b'\xcd':
        return struct.unpack(">H", _read_except(fp, 2))[0]
    elif code == b'\xce':
        return struct.unpack(">I", _read_except(fp, 4))[0]
    elif code == b'\xcf':
        return struct.unpack(">Q", _read_except(fp, 8))[0]
    raise Exception("logic error, not int: 0x%02x" % ord(code))

def _unpack_reserved(code, fp):
    if code == b'\xc1':
        raise ReservedCodeException("encountered reserved code: 0x%02x" % ord(code))
    raise Exception("logic error, not reserved code: 0x%02x" % ord(code))

def _unpack_nil(code, fp):
    if code == b'\xc0':
        return None
    raise Exception("logic error, not nil: 0x%02x" % ord(code))

def _unpack_boolean(code, fp):
    if code == b'\xc2':
        return False
    elif code == b'\xc3':
        return True
    raise Exception("logic error, not boolean: 0x%02x" % ord(code))

def _unpack_float(code, fp):
    if code == b'\xca':
        return struct.unpack(">f", _read_except(fp, 4))[0]
    elif code == b'\xcb':
        return struct.unpack(">d", _read_except(fp, 8))[0]
    raise Exception("logic error, not float: 0x%02x" % ord(code))

def _unpack_string(code, fp):
    if (ord(code) & 0xe0) == 0xa0:
        length = ord(code) & ~0xe0
    elif code == b'\xd9':
        length = struct.unpack("B", _read_except(fp, 1))[0]
    elif code == b'\xda':
        length = struct.unpack(">H", _read_except(fp, 2))[0]
    elif code == b'\xdb':
        length = struct.unpack(">I", _read_except(fp, 4))[0]
    else:
        raise Exception("logic error, not string: 0x%02x" % ord(code))

    # Always return raw bytes in compatibility mode
    global compatibility
    if compatibility:
        return _read_except(fp, length)

    try:
        return bytes.decode(_read_except(fp, length), 'utf-8')
    except UnicodeDecodeError:
        raise InvalidStringException("unpacked string is not utf-8")

def _unpack_binary(code, fp):
    if code == b'\xc4':
        length = struct.unpack("B", _read_except(fp, 1))[0]
    elif code == b'\xc5':
        length = struct.unpack(">H", _read_except(fp, 2))[0]
    elif code == b'\xc6':
        length = struct.unpack(">I", _read_except(fp, 4))[0]
    else:
        raise Exception("logic error, not binary: 0x%02x" % ord(code))

    return _read_except(fp, length)

def _unpack_ext(code, fp):
    if code == b'\xd4':
        length = 1
    elif code == b'\xd5':
        length = 2
    elif code == b'\xd6':
        length = 4
    elif code == b'\xd7':
        length = 8
    elif code == b'\xd8':
        length = 16
    elif code == b'\xc7':
        length = struct.unpack("B", _read_except(fp, 1))[0]
    elif code == b'\xc8':
        length = struct.unpack(">H", _read_except(fp, 2))[0]
    elif code == b'\xc9':
        length = struct.unpack(">I", _read_except(fp, 4))[0]
    else:
        raise Exception("logic error, not ext: 0x%02x" % ord(code))

    return Ext(ord(_read_except(fp, 1)), _read_except(fp, length))

def _unpack_array(code, fp):
    if (ord(code) & 0xf0) == 0x90:
        length = (ord(code) & ~0xf0)
    elif code == b'\xdc':
        length = struct.unpack(">H", _read_except(fp, 2))[0]
    elif code == b'\xdd':
        length = struct.unpack(">I", _read_except(fp, 4))[0]
    else:
        raise Exception("logic error, not array: 0x%02x" % ord(code))

    return [_unpack(fp) for i in xrange(length)]

def _deep_list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple([_deep_list_to_tuple(e) for e in obj])
    return obj

def _unpack_map(code, fp):
    if (ord(code) & 0xf0) == 0x80:
        length = (ord(code) & ~0xf0)
    elif code == b'\xde':
        length = struct.unpack(">H", _read_except(fp, 2))[0]
    elif code == b'\xdf':
        length = struct.unpack(">I", _read_except(fp, 4))[0]
    else:
        raise Exception("logic error, not map: 0x%02x" % ord(code))

    d = {}
    for i in xrange(length):
        # Unpack key
        k = _unpack(fp)

        if isinstance(k, list):
            # Attempt to convert list into a hashable tuple
            k = _deep_list_to_tuple(k)
        elif not isinstance(k, collections.Hashable):
            raise UnhashableKeyException("encountered unhashable key: %s, %s" % (str(k), str(type(k))))
        elif k in d:
            raise DuplicateKeyException("encountered duplicate key: %s, %s" % (str(k), str(type(k))))

        # Unpack value
        v = _unpack(fp)

        try:
            d[k] = v
        except TypeError:
            raise UnhashableKeyException("encountered unhashable key: %s" % str(k))
    return d

def _unpack(fp):
    code = _read_except(fp, 1)
    return _unpack_dispatch_table[code](code, fp)

########################################

def _unpack2(fp):
    """
    Deserialize MessagePack bytes into a Python object.

    Args:
        fp: a .read()-supporting file-like object

    Returns:
        A Python object.

    Raises:
        InsufficientDataException(UnpackException):
            Insufficient data to unpack the encoded object.
        InvalidStringException(UnpackException):
            Invalid UTF-8 string encountered during unpacking.
        ReservedCodeException(UnpackException):
            Reserved code encountered during unpacking.
        UnhashableKeyException(UnpackException):
            Unhashable key encountered during map unpacking.
            The serialized map cannot be deserialized into a Python dictionary.
        DuplicateKeyException(UnpackException):
            Duplicate key encountered during map unpacking.

    Example:
    >>> f = open("test.bin")
    >>> umsgpack.unpackb(f)
    {u'compact': True, u'schema': 0}
    >>>
    """
    return _unpack(fp)

def _unpack3(fp):
    """
    Deserialize MessagePack bytes into a Python object.

    Args:
        fp: a .read()-supporting file-like object

    Returns:
        A Python object.

    Raises:
        InsufficientDataException(UnpackException):
            Insufficient data to unpack the encoded object.
        InvalidStringException(UnpackException):
            Invalid UTF-8 string encountered during unpacking.
        ReservedCodeException(UnpackException):
            Reserved code encountered during unpacking.
        UnhashableKeyException(UnpackException):
            Unhashable key encountered during map unpacking.
            The serialized map cannot be deserialized into a Python dictionary.
        DuplicateKeyException(UnpackException):
            Duplicate key encountered during map unpacking.

    Example:
    >>> f = open("test.bin")
    >>> umsgpack.unpackb(f)
    {'compact': True, 'schema': 0}
    >>>
    """
    return _unpack(fp)

# For Python 2, expects a str object
def _unpackb2(s):
    """
    Deserialize MessagePack bytes into a Python object.

    Args:
        s: a 'str' containing serialized MessagePack bytes

    Returns:
        A Python object.

    Raises:
        TypeError:
            Packed data is not type 'str'.
        InsufficientDataException(UnpackException):
            Insufficient data to unpack the encoded object.
        InvalidStringException(UnpackException):
            Invalid UTF-8 string encountered during unpacking.
        ReservedCodeException(UnpackException):
            Reserved code encountered during unpacking.
        UnhashableKeyException(UnpackException):
            Unhashable key encountered during map unpacking.
            The serialized map cannot be deserialized into a Python dictionary.
        DuplicateKeyException(UnpackException):
            Duplicate key encountered during map unpacking.

    Example:
    >>> umsgpack.unpackb(b'\x82\xa7compact\xc3\xa6schema\x00')
    {u'compact': True, u'schema': 0}
    >>>
    """
    if not isinstance(s, str):
        raise TypeError("packed data is not type 'str'")
    return _unpack(io.BytesIO(s))

# For Python 3, expects a bytes object
def _unpackb3(s):
    """
    Deserialize MessagePack bytes into a Python object.

    Args:
        s: a 'bytes' containing serialized MessagePack bytes

    Returns:
        A Python object.

    Raises:
        TypeError:
            Packed data is not type 'bytes'.
        InsufficientDataException(UnpackException):
            Insufficient data to unpack the encoded object.
        InvalidStringException(UnpackException):
            Invalid UTF-8 string encountered during unpacking.
        ReservedCodeException(UnpackException):
            Reserved code encountered during unpacking.
        UnhashableKeyException(UnpackException):
            Unhashable key encountered during map unpacking.
            The serialized map cannot be deserialized into a Python dictionary.
        DuplicateKeyException(UnpackException):
            Duplicate key encountered during map unpacking.

    Example:
    >>> umsgpack.unpackb(b'\x82\xa7compact\xc3\xa6schema\x00')
    {'compact': True, 'schema': 0}
    >>>
    """
    if not isinstance(s, bytes):
        raise TypeError("packed data is not type 'bytes'")
    return _unpack(io.BytesIO(s))

################################################################################
### Module Initialization
################################################################################

# NOTE: Change: Nandakishore Santhi, Date: 5/9/2020.
#       Changelog: Modified to a proper class for initializing umsgPack from within this file itself
class umsgPack(object):
  def __init__(self):
    global pack
    global packb
    global unpack
    global unpackb
    global dump
    global dumps
    global load
    global loads
    global compatibility
    global _float_size
    global _unpack_dispatch_table
    global xrange

    # Compatibility mode for handling strings/bytes with the old specification
    compatibility = False

    # Auto-detect system float precision
    if sys.float_info.mant_dig == 53:
        _float_size = 64
    else:
        _float_size = 32

    # Map packb and unpackb to the appropriate version
    if sys.version_info[0] == 3:
        pack = _pack3
        packb = _packb3
        dump = _pack3
        dumps = _packb3
        unpack = _unpack3
        unpackb = _unpackb3
        load = _unpack3
        loads = _unpackb3
        xrange = range
    else:
        pack = _pack2
        packb = _packb2
        dump = _pack2
        dumps = _packb2
        unpack = _unpack2
        unpackb = _unpackb2
        load = _unpack2
        loads = _unpackb2

    # Build a dispatch table for fast lookup of unpacking function

    _unpack_dispatch_table = {}
    # Fix uint
    for code in range(0, 0x7f+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_integer
    # Fix map
    for code in range(0x80, 0x8f+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_map
    # Fix array
    for code in range(0x90, 0x9f+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_array
    # Fix str
    for code in range(0xa0, 0xbf+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_string
    # Nil
    _unpack_dispatch_table[b'\xc0'] = _unpack_nil
    # Reserved
    _unpack_dispatch_table[b'\xc1'] = _unpack_reserved
    # Boolean
    _unpack_dispatch_table[b'\xc2'] = _unpack_boolean
    _unpack_dispatch_table[b'\xc3'] = _unpack_boolean
    # Bin
    for code in range(0xc4, 0xc6+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_binary
    # Ext
    for code in range(0xc7, 0xc9+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_ext
    # Float
    _unpack_dispatch_table[b'\xca'] = _unpack_float
    _unpack_dispatch_table[b'\xcb'] = _unpack_float
    # Uint
    for code in range(0xcc, 0xcf+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_integer
    # Int
    for code in range(0xd0, 0xd3+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_integer
    # Fixext
    for code in range(0xd4, 0xd8+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_ext
    # String
    for code in range(0xd9, 0xdb+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_string
    # Array
    _unpack_dispatch_table[b'\xdc'] = _unpack_array
    _unpack_dispatch_table[b'\xdd'] = _unpack_array
    # Map
    _unpack_dispatch_table[b'\xde'] = _unpack_map
    _unpack_dispatch_table[b'\xdf'] = _unpack_map
    # Negative fixint
    for code in range(0xe0, 0xff+1):
        _unpack_dispatch_table[struct.pack("B", code)] = _unpack_integer
#===========================================================================================
# NOTE: This module used a different licence from when this message appeared at a previous line
#===========================================================================================

#===========================================================================================
# MPICH.py
#===========================================================================================
# Dynamically link the MPICH library into the global namespace
def loadMPICH(mpichLib):
    class MPI_Status(C.Structure):
        _fields_ = [
            ("count_lo", C.c_int),
            ("count_hi_and_cancelled", C.c_int),
            ("MPI_SOURC", C.c_int),
            ("MPI_TA", C.c_int),
            ("MPI_ERRO", C.c_int)
            ]

    mpi = C.CDLL(mpichLib) #Looks in current directory
    if not mpi:
        raise SimianError("loadMPICH(): could not load mpich dynamic library: " + mpichLib + ". Please check library path ...")

    mpi.MPI_Status = MPI_Status

    mpi.MPI_SUCCESS = 0
    mpi.MPI_ANY_TAG = -1
    mpi.MPI_ANY_SOURCE = -2

    mpi.MPI_Comm = C.c_int
    mpi.MPI_COMM_WORLD = mpi.MPI_Comm(0x44000000)

    mpi.MPI_Datatype = C.c_int
    mpi.MPI_BYTE = mpi.MPI_Datatype(0x4c00010d)
    mpi.MPI_DOUBLE = mpi.MPI_Datatype(0x4c00080b)
    mpi.MPI_LONG = mpi.MPI_Datatype(0x4c000807)

    mpi.MPI_Op = C.c_int
    mpi.MPI_MIN = mpi.MPI_Op(0x58000002)
    mpi.MPI_SUM = mpi.MPI_Op(0x58000003)

    mpi.MPI_Request = C.c_int
    mpi.MPI_Init.restype = C.c_int
    mpi.MPI_Finalize.restype = C.c_int
    mpi.MPI_Comm_size.restype = C.c_int
    mpi.MPI_Comm_rank.restype = C.c_int
    mpi.MPI_Iprobe.restype = C.c_int
    mpi.MPI_Probe.restype = C.c_int
    mpi.MPI_Send.restype = C.c_int
    mpi.MPI_Isend.restype = C.c_int
    mpi.MPI_Recv.restype = C.c_int
    mpi.MPI_Get_count.restype = C.c_int
    mpi.MPI_Get_elements.restype = C.c_int
    mpi.MPI_Allreduce.restype = C.c_int
    mpi.MPI_Barrier.restype = C.c_int
    mpi.MPI_Alltoall.restype = C.c_int

    mpi.MPI_Init.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_char_p)]
    mpi.MPI_Finalize.argtypes = []
    mpi.MPI_Comm_size.argtypes = [mpi.MPI_Comm, C.POINTER(C.c_int)]
    mpi.MPI_Comm_rank.argtypes = [mpi.MPI_Comm, C.POINTER(C.c_int)]
    mpi.MPI_Iprobe.argtypes = [C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(C.c_int), C.POINTER(mpi.MPI_Status)]
    mpi.MPI_Probe.argtypes = [C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(mpi.MPI_Status)]
    mpi.MPI_Send.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_int, C.c_int, mpi.MPI_Comm]
    mpi.MPI_Isend.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(mpi.MPI_Request)]
    mpi.MPI_Recv.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_int, C.c_int, mpi.MPI_Comm, C.POINTER(mpi.MPI_Status)]
    mpi.MPI_Get_count.argtypes = [C.POINTER(mpi.MPI_Status), mpi.MPI_Datatype, C.POINTER(C.c_int)]
    mpi.MPI_Get_elements.argtypes = [C.POINTER(mpi.MPI_Status), mpi.MPI_Datatype, C.POINTER(C.c_int)]
    mpi.MPI_Allreduce.argtypes = [C.c_void_p, C.c_void_p, C.c_int, mpi.MPI_Datatype, mpi.MPI_Op, mpi.MPI_Comm]
    mpi.MPI_Barrier.argtypes = [mpi.MPI_Comm]
    mpi.MPI_Alltoall.argtypes = [C.c_void_p, C.c_int, mpi.MPI_Datatype, C.c_void_p, C.c_int, mpi.MPI_Datatype, mpi.MPI_Comm]

    return mpi
#===========================================================================================

#===========================================================================================
# MPILib.py
#===========================================================================================
# Author: Nandakishore Santhi
# Date: 15 April, 2015
#  Common MPI wrapper for MPICH3 and Open-MPI
# NOTE: Currently, MPICH-v3.3.2 works well
# NOTE: There are some severe bugs in Open-MPI-v1.8.3 (Open-MPI-v1.6.5 seems to work), hence it has been blacklisted at present

# Following symbols are from umsgPack module
msg, Pack, Unpack = None, None, None

mpi = None
class MPI(object):
    def __init__(self, libName):
        global mpi
        if not mpi: mpi = loadMPICH(libName)

        global msg, Pack, Unpack
        try:
            msg = umsgPack()
            Pack, Unpack = packb, unpackb
        except:
            raise SimianError("MPI(): could not initialize umsgPack")

        if mpi.MPI_Init(None, None) != mpi.MPI_SUCCESS:
            raise SimianError("MPI(): could not initialize MPI")

        self.CBUF_LEN = 32*1024 #32kB

        self.comm = mpi.MPI_COMM_WORLD
        self.BYTE = mpi.MPI_BYTE
        self.DOUBLE = mpi.MPI_DOUBLE
        self.LONG = mpi.MPI_LONG
        self.MIN = mpi.MPI_MIN
        self.SUM = mpi.MPI_SUM

        self.request = mpi.MPI_Request()
        self.status = mpi.MPI_Status()
        self.itemp = C.c_int()
        self.dtemp0 = C.c_double()
        self.dtemp1 = C.c_double()
        self.ctemp = C.create_string_buffer(self.CBUF_LEN) #Preallocate

        self.numRanks = self.size()
        self.sndCounts = (C.c_long * self.numRanks)()
        for i in range(len(self.sndCounts)): self.sndCounts[i] = 0
        self.rcvCounts = (C.c_long * self.numRanks)()

    def finalize(self):
        if mpi.MPI_Finalize() == mpi.MPI_SUCCESS:
            return False
        raise SimianError("MPI.finalize(): error!")

    def rank(self):
        if mpi.MPI_Comm_rank(self.comm, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp.value
        raise SimianError("MPI.rank(): error!")

    def size(self):
        #size = (C.c_int * 1)()
        if mpi.MPI_Comm_size(self.comm, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp.value
        raise SimianError("MPI.size(): error!")

    def iprobe(self, src=None, tag=None): #Non-blocking asynch
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        if mpi.MPI_Iprobe(src, tag, self.comm, C.byref(self.itemp), C.byref(self.status)) == mpi.MPI_SUCCESS:
            return (self.itemp.value != 0)
        raise SimianError("MPI.iprobe(): error!")

    def probe(self, src=None, tag=None): #Blocking synch
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        return (mpi.MPI_Probe(src, tag, self.comm, C.byref(self.status)) == mpi.MPI_SUCCESS)

    def send(self, x, dst, tag=None): #Blocking
        m = Pack(x)
        tag = tag or len(m) #Set to message length if None
        if mpi.MPI_Send(m, len(m), self.BYTE, dst, tag, self.comm) != mpi.MPI_SUCCESS:
            raise SimianError("MPI.send(): error!")

    def isend(self, x, dst, tag=None): #Non-Blocking
        m = Pack(x)
        tag = tag or len(m) #Set to message length if None
        if mpi.MPI_Isend(m, len(m), self.BYTE, dst, tag, self.comm, C.byref(self.request)) != mpi.MPI_SUCCESS:
            raise SimianError("MPI.isend(): error!")

    def recv(self, maxSize, src=None, tag=None): #Blocking
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        m = self.ctemp
        if maxSize > self.CBUF_LEN: #Temporary buffer is too small
            m = C.create_string_buffer(maxSize)
        if mpi.MPI_Recv(m, maxSize, self.BYTE, src, tag, self.comm, C.byref(self.status)) == mpi.MPI_SUCCESS:
            #return Unpack(m.raw)
            return Unpack(m[:maxSize])
        raise SimianError("MPI.recv(): error!")

    def getCount(self): #Non-blocking
        if mpi.MPI_Get_count(C.byref(self.status), self.BYTE, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp.value
        raise SimianError("MPI.getCount(): error!")

    def getElements(self): #Non-blocking
        if mpi.MPI_Get_elements(C.byref(self.status), self.BYTE, C.byref(self.itemp)) == mpi.MPI_SUCCESS:
            return self.itemp
        raise SimianError("MPI.getElements(): error!")

    def recvAnySize(self, src=None, tag=None):
        src = src or mpi.MPI_ANY_SOURCE
        tag = tag or mpi.MPI_ANY_TAG
        return self.recv(self.getCount(), src, tag)

    def allreduce(self, partial, op):
        self.dtemp0 = C.c_double(partial)
        if (mpi.MPI_Allreduce(C.byref(self.dtemp0), C.byref(self.dtemp1),
                    1, self.DOUBLE, #Single double operand
                    op, self.comm) != mpi.MPI_SUCCESS):
            raise SimianError("MPI.allreduce(): error!")
        return self.dtemp1.value

    def barrier(self):
        if (mpi.MPI_Barrier(self.comm) != mpi.MPI_SUCCESS):
            raise SimianError("MPI.barrier(): error!")

    def alltoallSum(self):
        if (mpi.MPI_Alltoall(self.sndCounts, 1, self.LONG,
                             self.rcvCounts, 1, self.LONG, self.comm) != mpi.MPI_SUCCESS):
            raise SimianError("MPI.alltoallSum(): error!")
        toRcv = 0
        for i in range(self.numRanks):
            toRcv = toRcv + self.rcvCounts[i]
            self.sndCounts[i] = 0
        return toRcv

    def sendAndCount(self, x, dst, tag=None): #Blocking
        m = Pack(x)
        tag = tag or len(m) #Set to message length if None
        if mpi.MPI_Send(m, len(m), self.BYTE, dst, tag, self.comm) != mpi.MPI_SUCCESS:
            raise SimianError("MPI.sendAndCount(): error!")
        self.sndCounts[dst] += 1
#===========================================================================================

#===========================================================================================
# simian.py
# Main simumation engine class. Derive the PDES engine instance from this class.
#===========================================================================================
class Simian(object):
    # Note: changed interface here to add silent option and default values for start and end times
    def __init__(self, simName='simian_run', startTime=0.0, endTime=10e10, minDelay=1, useMPI=False, mpiLibName=defaultMpichLibName, silent=False):
        self.Entity = Entity #Include in the top Simian namespace

        self.name = simName
        self.silent = silent
        self.startTime = startTime
        self.endTime = endTime
        self.minDelay = minDelay
        self.now = startTime

        #If simulation is running
        self.running = False

        #Stores the entities available on this LP
        self.entities = {}

        #Events are stored in a priority-queue or heap, in increasing
        #order of time field. Heap top can be accessed using self.eventQueue[0]
        #event = {time, name, data, tx, txId, rx, rxId}.
        self.eventQueue = []
        self.ec = 0 # events created, used to work with the silly heap in Python 3 that can't compare dictionaries

        #Stores the minimum time of any event sent by this process,
        #which is used in the global reduce to ensure global time is set to
        #the correct minimum.
        self.infTime = endTime + 2*minDelay
        self.minSent = self.infTime

        #[[Base rank is an integer hash of entity's name]]
        self.baseRanks = {}

        #Make things work correctly with and without MPI
        if useMPI: #Initialize MPI
            try:
                self.useMPI = True
                self.MPI = MPI(mpiLibName)
                self.rank = self.MPI.rank()
                self.size = self.MPI.size()
            except:
                raise SimianError("Simian.__init__(): you have asserted useMPI - please ensure libmpich is available to ctypes before using Simian for MPI based simulations.\nTry passing absolute path to libmpich.[dylib/so/dll] to Simian.\nI tried to locate it at:\n\t" + mpiLibName + "\nand failed!")
        else:
            self.useMPI = False
            self.MPI = None
            self.rank = 0
            self.size = 1

        #One output file per rank
        self.out = open(self.name + "." + str(self.rank) + ".out", "w")

        # Write simulation related information in header for each output log file
        self.out.write("Simian JIT PDES Engine (" + __SimianVersion__ + ")\n")
        self.out.write("(c) Copyright 2015-. Triad National Security, LLC. All rights reserved. Released under BSD 3-clause license.\n")
        self.out.write("Author: Nandakishore Santhi (with other contributors as acknowledged in source code)\n\n")
        self.out.write("===================================================\n")
        self.out.write("------------SIMIAN-PIE JIT PDES ENGINE-------------\n")
        self.out.write("------------       VERSION: " + __SimianVersion__ + "      -------------\n")
        self.out.write("===================================================\n")
        if self.useMPI: self.out.write("MPI: ON\n")
        else: self.out.write("MPI: OFF\n")
        self.out.write("===================================================\n\n")

    def exit(self):
        sys.stdout.flush()
        self.out.close()
        del self.out

    def run(self): #Run the simulation
        startTime = perfTime()
        if self.rank == 0 and not self.silent:
            print("Simian JIT PDES Engine (" + __SimianVersion__ + ")")
            print("(c) Copyright 2015-. Triad National Security, LLC. All rights reserved. Released under BSD 3-clause license.")
            print("Author: Nandakishore Santhi (with other contributors as acknowledged in source code)\n")
            print("===================================================")
            print("------------SIMIAN-PIE JIT PDES ENGINE-------------")
            print("------------       VERSION: " + __SimianVersion__ + "      -------------")
            print("===================================================")
            if self.useMPI: print("MPI: ON")
            else: print("MPI: OFF")
            print("===================================================\n")
        numEvents = 0

        self.running = True
        globalMinLeft = self.startTime
        while globalMinLeft <= self.endTime:
            epoch = globalMinLeft + self.minDelay

            self.minSent = self.infTime
            while len(self.eventQueue) > 0 and self.eventQueue[0][0] < epoch:
                (time ,_, event) = heapq.heappop(self.eventQueue) #Next event
                if self.now > time:
                    raise SimianError("Out of order event: now=%f, evt=%f" % self.now, time)
                self.now = time #Advance time

                #Simulate event
                entity = self.entities[event["rx"]][event["rxId"]]
                service = getattr(entity, event["name"])
                service(event["data"], event["tx"], event["txId"]) #Receive TO BE CGECJED

                numEvents = numEvents + 1

            if self.size > 1:
                toRcvCount = self.MPI.alltoallSum()
                while toRcvCount > 0:
                    self.MPI.probe()
                    remoteEvent = self.MPI.recvAnySize()
                    self.ec += 1
                    heapq.heappush(self.eventQueue, (remoteEvent["time"], self.ec, remoteEvent))
                    toRcvCount -= 1

                minLeft = self.infTime
                if len(self.eventQueue) > 0: minLeft = self.eventQueue[0][0]
                globalMinLeft = self.MPI.allreduce(minLeft, self.MPI.MIN) #Synchronize minLeft
            else:
                globalMinLeft = self.infTime
                if len(self.eventQueue) > 0: globalMinLeft = self.eventQueue[0][0]

        self.running = False

        if self.size > 1:
            self.MPI.barrier()
            totalEvents = self.MPI.allreduce(numEvents, self.MPI.SUM)
        else:
            totalEvents = numEvents

        elapsedTime = perfTime() - startTime
        if self.rank == 0 and not self.silent:
            print("SIMULATION COMPLETED IN: " + str(elapsedTime) + " SECONDS")
            print("SIMULATED EVENTS: " + str(totalEvents))
            print("EVENTS PER SECOND: " + str(totalEvents/elapsedTime))
            print("===================================================")
        sys.stdout.flush()

        # Write simulation related information in footer for each output log file
        self.out.write("\n===================================================\n")
        self.out.write("SIMULATION COMPLETED IN: " + str(elapsedTime) + " SECONDS\n")
        self.out.write("SIMULATED EVENTS: " + str(totalEvents) + "\n")
        self.out.write("EVENTS PER SECOND: " + str(totalEvents/elapsedTime) + "\n")
        self.out.write("===================================================\n")

    def schedService(self, time, eventName, data, rx, rxId):
        #Purpose: Add an event to the event-queue.
        #For kicking off simulation and waking processes after a timeout
        if time > self.endTime: return #No need to push this event

        recvRank = self.getOffsetRank(rx, rxId)

        #print("Simian schedService: recvRank, self.rank", recvRank, self.rank)
        if recvRank == self.rank:
            e = {
                    "tx": None, #String (Implictly self.name)
                    "txId": None, #Number (Implictly self.num)
                    "rx": rx, #String
                    "rxId": rxId, #Number
                    "name": eventName, #String
                    "data": data, #Object
                    "time": time, #Number
                }
            #print("Simian schedService: time, e", time, e, self.ec)
            self.ec += 1
            heapq.heappush(self.eventQueue, (time, self.ec, e))

    def getBaseRank(self, name):
        #Can be overridden for more complex Entity placement on ranks
        return int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16) % self.size

    def getOffsetRank(self, name, num):
        #Can be overridden for more complex Entity placement on ranks
        val = (self.baseRanks[name] + num) % self.size
        return (self.baseRanks[name] + num) % self.size

    def getEntity(self, name, num):
        #Returns a reference to a named entity of given serial number
        if name in self.entities:
            entity = self.entities[name]
            if num in entity:
                return entity[num]

    def attachService(self, klass, name, fun):
        #Attaches a service at runtime to an entity klass type
        setattr(klass, name, fun)

    def addEntity(self, name, entityClass, num, *args, **kargs):
        #Purpose: Add an entity to the entity-list if Simian is idle
        #This function takes a pointer to a class from which the entities can
        #be constructed, a name, and a number for the instance.
        if self.running: raise SimianError("Simian.addEntity(): adding entity when Simian is running!")

        if not (name in self.entities):
            self.entities[name] = {} #To hold entities of this "name"
        entity = self.entities[name]

        self.baseRanks[name] = self.getBaseRank(name) #Register base-ranks

        computedRank = self.getOffsetRank(name, num)
        if computedRank == self.rank: #This entity resides on this engine
            #Output log file for this Entity
            self.out.write(name + "[" + str(num) + "]: Running on rank " + str(computedRank) + "\n")

            entity[num] = entityClass({
                "name": name,
                "out": self.out,
                "engine": self,
                "num": num,
                }, *args) #Entity is instantiated
