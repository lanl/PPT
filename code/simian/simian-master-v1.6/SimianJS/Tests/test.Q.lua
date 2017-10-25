#!/usr/bin/env luajit
local Q = require("eventQ");
local COUNT = 10000000;

local myList = {};

print("Pushing:");
for i=1,COUNT do
    Q.push(myList, {time = math.random()});
end

print("Checking Queue Pop Order:");
local prvMin = -1;
local topItem;
while (#myList > 1) do
    topItem = Q.pop(myList);
    if (prvMin > topItem.time) then print("Out of order"); end
    prvMin = topItem.time;
end

print("Pushing Time Check:");
local start = os.clock();
for i=1,COUNT do
    Q.push(myList, {time = math.random()});
end
 finish = os.clock() - start;
print("One Push took: " .. (1000*finish/COUNT) .. " ms")

print("Poping Time Check:");
local topItem;
local start = os.clock();
while (#myList > 1) do
    topItem = Q.pop(myList);
end
local finish = os.clock() - start;
print("One Pop took: " .. (1000*finish/COUNT) .. " ms")
