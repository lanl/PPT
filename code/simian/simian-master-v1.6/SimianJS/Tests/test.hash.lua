#!/usr/bin/env luajit
local hash = require ("hash")

local COUNT = 10000000

local start = os.clock()
local h = 0
for i=0,COUNT-1 do h = hash(tostring(i)) end
local finish = os.clock() - start
print(h);
print("Took: " .. finish*1000 .. " ms");
