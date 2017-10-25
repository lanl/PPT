#!/usr/bin/env luajit
local count = 10000000
local active = count/2
local clique = 20
local random = math.random

local start = os.clock()
local A = {}
for i=1,count do A[i] = random() end

local value = 0.0
local opCount = 0
for i=1,active do
    for j=1,clique do
        value = value + A[i]*random()
        opCount = opCount + 1
    end
end

local finish = os.clock() - start

print(value, opCount)
print("Took: " .. finish .. "(s)")
