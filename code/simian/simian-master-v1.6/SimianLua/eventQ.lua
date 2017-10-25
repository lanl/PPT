--[[
Copyright (c) 2015, Los Alamos National Security, LLC
All rights reserved.

Copyright 2015. Los Alamos National Security, LLC. This software was produced under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National Laboratory (LANL), which is operated by Los Alamos National Security, LLC for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. 
	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution. 
	Neither the name of Los Alamos National Security, LLC, Los Alamos National Laboratory, LANL, the U.S. Government, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission. 
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
]]

--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Priority Queue or heap, with head being least time
--]]
local eventQ = {}

local floor, remove, insert = math.floor, table.remove, table.insert

function eventQ.push(list, item)
    insert(list, item)
    local parent, curPos = #list, nil
    while parent > 1 do
        curPos, parent = parent, floor(parent/2)
        if list[curPos].time < list[parent].time then
            list[curPos], list[parent] = list[parent], list[curPos]
        else
            break
        end
    end
    --print(#list)
end

function eventQ.pop(list)
    local len = #list
    list[1], list[len] = list[len], list[1]
    local pop = remove(list)
    local curPos, left, right, min = 1, 2, 3, 1
    while true do
        local state
        if len > left then
            if list[min].time > list[left].time then state, min = 1, left end
        end
        if len > right then
            if list[min].time > list[right].time then state, min = 2, right end
        end
        if not state then break end
        list[curPos], list[min] = list[min], list[curPos]
        curPos, left = min, min*2
        right = left+1
    end
    return pop
end

--[[TestCase
myList = {}
input = {7878, 78217, 90, 1, 425, 10}
for k, v in ipairs(input) do
    eventQ.push(myList, {time = v})
end
local pop
while #myList > 0 do
    pop = eventQ.pop(myList)
    print("pop", pop.time)
end
print(pop.time)
]]

return eventQ
