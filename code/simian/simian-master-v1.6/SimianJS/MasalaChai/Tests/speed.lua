#!/usr/bin/env luajit
local ffi  = require("ffi")

--local libcFile = "libc.so.6"
local libcFile = "libSystem.dylib"
local libc = ffi.load(libcFile)

ffi.cdef[[
double sin(double);
int rand(void);
]]
local sin, rand = libc.sin, libc.rand

local INT_MAX = (2^31) - 1;
local COUNT = 10000000;

local function valFun()
    local val = 0.0
    for i=0,COUNT-1 do
        val = val + sin(i*rand()/INT_MAX)
    end
    return val
end

--local val = valFun() --JIT Warmup

local start = os.clock()
val = valFun()
local finish = os.clock() - start;
print(val);
print("Took:", finish*1000, "ms")
