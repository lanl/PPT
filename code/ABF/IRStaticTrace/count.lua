#!/usr/bin/env luajit
local op = {
    ['load'] = 0,
    ['store'] = 0,
    ['fmul'] = 0,
    ['fadd'] = 0,
    ['br'] = 0,
    ['call'] = 0,
    ['fsub'] = 0,
    ['icmp'] = 0,
    ['add'] = 0,
    ['fdiv'] = 0,
    ['ret'] = 0,
    ['mul'] = 0,
    ['fcmp'] = 0,
    ['sdiv'] = 0,
    ['and'] = 0,
}

for key,_ in pairs(op) do
    for line in io.lines(arg[1]) do
        if line.match(line, "%W" .. key .. "%W") then op[key] = op[key] + 1 end
    end
    print("Operand: " .. key .. ", Count: " .. op[key])
end
