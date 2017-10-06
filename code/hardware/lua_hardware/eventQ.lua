--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Priority Queue or heap, with head being least time
--]]
local eventQ = {}

function eventQ.push(list, item)
    table.insert(list, item)
    local curPos
    local parent = #list
    while parent > 1 do
        curPos, parent = parent, math.floor(parent/2)
        if list[curPos].time < list[parent].time then
            list[curPos], list[parent] = list[parent], list[curPos]
        else
            break
        end
    end
end

function eventQ.pop(list)
    list[1], list[#list] = list[#list], list[1]
    local pop = table.remove(list)
    local curPos, left, right = 1, 2, 3
    while true do
        local state = 0
        local min = curPos
        if left <= #list then
            if list[min].time > list[left].time then
                state = 1
                min = left
            end
        end
        if right <= #list then
            if list[min].time > list[right].time then
                state = 2
                min = right
            end
        end
        if state == 0 then
            break
        elseif state == 1 then
            list[curPos], list[left] = list[left], list[curPos]
            curPos, left, right = left, left*2, left*2+1
        elseif state == 2 then
            list[curPos], list[right] = list[right], list[curPos]
            curPos, left, right = right, right*2, right*2+1
        end
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
