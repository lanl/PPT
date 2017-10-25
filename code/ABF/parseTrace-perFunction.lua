#!/usr/bin/env luajit
--[[
Author: Nandakishore Santhi
Date: Nov 2016
--]]
local inspect = require "inspect"

function string:split(pat)
    local t = {}
    local fpat = "(.-)" .. pat
    local last_end = 1
    local s, e, cap = self:find(fpat, 1)
    while s do
        if s ~= 1 or cap ~= "" then
            table.insert(t,cap)
        end
        last_end = e+1
        s, e, cap = self:find(fpat, last_end)
    end
    if last_end <= #self then
        cap = self:sub(last_end)
        table.insert(t, cap)
    end
    return t
end

function string:strip()
    local from = self:match"^%s*()"
    return from > #self and "" or self:match(".*%S", from)
end

--[[
trace = {
Modules = {
    [1] = {
        name = ...
        TargetTriple = ...
        Functions = {
            [1] = {
                name = ...
                definition = true/false
                intrinsic = true/false
                ArgList = {
                    [1] = ..
                    ...
                }
                BasicBlocks = {
                    [1] = {
                        name = ...
                        Instructions = {
                            [1] = {
                                name = ...
                                pos = {
                                    line = ...
                                    col = ...
                                }
                                Operands = {
                                    [1] = ...
                                    ...
                                }
                            }
                        }
                        Succ = {
                            pos = {
                                line = ...
                                col = ...
                            }
                            unreachable = true/nil
                            return = "Fun_name"/nil
                            uncond = BB#/nil
                            true = BB#/nil
                            false = BB#/nil
                        }
                        Pred = {
                            [1] = BB#
                            [2] = BB#
                            ...
                        }
                    }
                    ...
                }
            }
            ...
        }
    }
    ...
}
}
]]
local function parseTrace(traceFile)
    local trace = {}

    local module, fun, bb, inst, ops --Holds current module, function, basic-block, instruction
    local current = ""
    for line in io.lines(traceFile) do
        line = line:strip()
        if line ~= "" then -- Ignore blank lines
            if line:match("Module : ") then
                current = "Module"
                if not trace.Modules then trace.Modules = {}; trace.ModuleTable = {} end
                module = {
                    name = line:split(":")[2]:strip(),
                }
                trace.Modules[#trace.Modules + 1] = module
                trace.ModuleTable[module.name] = #trace.Modules
            elseif line:match("Function ") then
                current = "Function"
                if not module.Functions then module.Functions = {}; module.FunctionTable = {} end
                local fields = line:split(":")
                fun = {
                    name = fields[2]:strip(),
                    definition = (fields[3]:strip() == "Definition"),
                    intrinsic = (fields[4]:strip() == "Intrinsic"),
                }
                module.Functions[#module.Functions + 1] = fun
                module.FunctionTable[fun.name] = #module.Functions
            elseif line:match("BasicBlock ") then
                current = "BasicBlock"
                if not fun.BasicBlocks then fun.BasicBlocks = {}; fun.BasicBlockTable = {} end
                bb = {
                    name = line:split(":")[2]:strip(),
                }
                fun.BasicBlocks[#fun.BasicBlocks + 1] = bb
                fun.BasicBlockTable[bb.name] = #fun.BasicBlocks
            elseif line:match("Instruction ") then
                current = "Instruction"
                if not bb.Instructions then bb.Instructions = {} end
                local fields = line:split(":")
                local pos = fields[2]:strip()
                inst = {
                    name = fields[3]:strip(),
                }
                local line, col = pos:match("(%d+), (%d+)")
                inst.pos = {
                    line = tonumber(line),
                    col = tonumber(col)
                }
                bb.Instructions[#bb.Instructions + 1] = inst
            elseif line:match("TargetTriple :") then
                current = "TargetTriple"
            elseif line:match("ArgList :") then
                current = "ArgList"
            else
                if current == "TargetTriple" then
                    module.TargetTriple = line
                elseif current == "ArgList" then
                    if not fun.ArgList then fun.ArgList = {} end
                    fun.ArgList[#fun.ArgList + 1] = line
                elseif current == "Instruction" then
                    if not inst.Operands then inst.Operands = {} end
                    inst.Operands[#inst.Operands + 1] = line
                else
                    error("Unknown parse condition: (" .. current .. ") Line: " .. line)
                end
            end
        end
    end

    return trace
end

local function idSuccPred(trace)
    if not trace.Modules then return end
    for _,mod in ipairs(trace.Modules) do
        for _,fun in ipairs(mod.Functions) do
            if fun.definition then
                for _,bb in ipairs(fun.BasicBlocks) do --Successors
                    local terminator = bb.Instructions[#bb.Instructions]
                    local sucs = { pos = terminator.pos }
                    if terminator.name == "unreachable" then
                        sucs["unreachable"] = true
                    elseif terminator.name == "ret" then
                        sucs["return"] = fun.name
                    elseif terminator.name == "br" then
                        local ops = terminator.Operands
                        if #ops == 1 then --Unconditional branch
                            sucs["uncond"] = fun.BasicBlockTable[ops[1]:split("<@")[1]]
                        elseif #ops == 3 then -- Conditional branch
                            sucs["true"] = fun.BasicBlockTable[ops[2]:split("<@")[1]]
                            sucs["false"] = fun.BasicBlockTable[ops[3]:split("<@")[1]]
                        else --Unknown
                            error("Unknown branch instruction in the basic-block " .. bb.name .. " in function " .. fun.name .. " in module " .. mod.name)
                        end
                    elseif terminator.name == "indirectbr" then
                        error("Terminator instruction " .. terminator.name .. " not yet implemented")
                    elseif terminator.name == "switch" then
                        error("Terminator instruction " .. terminator.name .. " not yet implemented")
                    elseif terminator.name == "invoke" then
                        error("Terminator instruction " .. terminator.name .. " not yet implemented")
                    elseif terminator.name == "resume" then
                        error("Terminator instruction " .. terminator.name .. " not yet implemented")
                    else
                        error("Unknown terminator instruction " .. terminator.name .. " for the basic-block " .. bb.name .. " in function " .. fun.name .. " in module " .. mod.name)
                    end

                    bb.Succ = sucs
                    bb.Pred = {}
                end
                for bbNum,bb in ipairs(fun.BasicBlocks) do --Predecessors
                    if bbNum == 1 then
                        local pred = bb.Pred
                        pred[#pred + 1] = { --Function entry basic-block
                            kind = "entry",
                            name = fun.name,
                            id = 0, --The canonical (and fictional) entry BB in any function
                            pos = {
                                line = 0,
                                col = 0,
                            },
                        }
                    end
                    if bb.Succ.uncond then
                        local pred = fun.BasicBlocks[bb.Succ.uncond].Pred
                        pred[#pred + 1] = {
                            kind = "uncond",
                            name = bb.name,
                            id = bbNum,
                            pos = bb.Succ.pos,
                        }
                    elseif bb.Succ["true"] then
                        local pred = fun.BasicBlocks[bb.Succ["true"]].Pred
                        pred[#pred + 1] = {
                            kind = "true",
                            name = bb.name,
                            id = bbNum,
                            pos = bb.Succ.pos,
                        }
                        pred = fun.BasicBlocks[bb.Succ["false"]].Pred
                        pred[#pred + 1] = {
                            kind = "false",
                            name = bb.name,
                            id = bbNum,
                            pos = bb.Succ.pos,
                        }
                    end
                end
            end
        end
    end
end

local function getLinearConstraints(trace)
    local C = {}

    if not trace.Modules then return C end

    for _,mod in ipairs(trace.Modules) do
        local modC = {}
        for _,fun in ipairs(mod.Functions) do
            if fun.definition then
                local funC = {}
                --Form the constraints for fun
                for bbNum,bb in ipairs(fun.BasicBlocks) do --Examine all basic-blocks; 1 equation for each BB
                    local E = {
                        [0] = {
                            id = bbNum,
                        }
                    }
                    for _,pred in ipairs(bb.Pred) do --Examine all predecessors
                        E[#E + 1] = {
                            id = pred.id,
                            kind = pred.kind,
                            pos = pred.pos.line .. "_" .. pred.pos.col,
                        }
                    end
                    funC[#funC + 1] = E
                end
                modC[fun.name] = funC
            end
        end
        C[mod.name] = modC
    end
    return C
end

local function printConstraints(C)
    io.write("\n\nLinear constraints:\n(# times a BB_i is executed in a given function definition is denoted as N_i)\n")
    for modName,mod in pairs(C) do --Examine all modules
        io.write("\nModule: " .. modName .. "\n")
        for funName,fun in pairs(mod) do --Examine all functions
            io.write("\n\tFunction: " .. funName .. "\n")
            for _,eqn in ipairs(fun) do --Examine all equations
                io.write("\t\tN_" .. eqn[0].id .. " = ") --LHS term
                for termId=1,#eqn do --Examine all RHS terms
                    local term = eqn[termId]
                    if termId > 1 then io.write(" + ") end
                    if (term.kind == "entry") then
                        io.write("1")
                    elseif (term.kind == "uncond") then
                        io.write("N_" .. term.id)
                    else
                        io.write("P_" .. term.pos .. "_" .. term.kind .. " * N_" .. term.id)
                    end
                end
                io.write("\n")
            end
        end
    end
    io.write("\n\n")
end

local function printTaskLists(trace)
    io.write("\n\nTask lists for each basic-block in every function:\n")
    if not trace.Modules then return end
    for _,mod in ipairs(trace.Modules) do
        io.write("\nModule: " .. mod.name .. "\n")
        for _,fun in ipairs(mod.Functions) do
            if fun.definition then
                io.write("\n\tFunction: " .. fun.name .. "\n")
                for bbNum,bb in ipairs(fun.BasicBlocks) do --BasicBlocks
                    io.write("\t\tBB_" .. bbNum .. " [" .. bb.name .. "]: { ")
                    for numInst,inst in ipairs(bb.Instructions) do --Instructions
                        io.write(inst.name .. " ")
                    end
                    io.write("}\n")
                end
            end
        end
    end
    io.write("\n\n")
end

local function outputProblemLNA(C, outputDir)
    local modNum = 0
    for modName,mod in pairs(C) do --Examine all modules
        modNum = modNum + 1
        for funName,fun in pairs(mod) do --Examine all functions
            local A, b, P = {}, {}, {} --A, b are matrices; P is the set of probability-variables
            for eqnNum,eqn in ipairs(fun) do --Examine all equations
                b[eqnNum] = 0
                A[eqnNum] = {}
                A[eqnNum][eqn[0].id] = 1
                for termId=1,#eqn do --Examine all RHS terms
                    local term = eqn[termId]
                    if (term.kind == "entry") then
                        b[eqnNum] = 1
                    elseif (term.kind == "uncond") then
                        A[eqnNum][term.id] = -1
                    else
                        local PrVar
                        if term.kind == "true" then
                            PrVar = "P_" .. term.pos .. "_true"
                            P[PrVar] = true
                        else
                            PrVar = "(1-P_" .. term.pos .. "_true)"
                        end
                        A[eqnNum][term.id] = "-" .. PrVar
                    end
                end
            end

            --Output the problem
            local fileName = outputDir .. "/" .. modNum .. "_" .. funName .. ".lua"
            print("Writing to file: " .. fileName)

            local fp = io.open(fileName, "w")
            fp:write("local matrix = require 'lna.matrix'\n\n")

            fp:write("--BEGIN TODO: Fill\n")
            for key,_ in pairs(P) do
                fp:write("local " .. key .. " = \n")
            end
            fp:write("--END TODO: Fill\n")
            fp:write("\n")

            fp:write("local A = matrix.fromtable{\n")
            for i=1,#fun do
                fp:write("  {")
                for j=1,#fun do
                    a_ij = A[i][j] or 0
                    fp:write(" " .. a_ij .. ",")
                end
                fp:write(" },\n")
            end
            fp:write("}\n\n")

            fp:write("local b = matrix.fromtable{\n")
            for _,bVal in pairs(b) do
                fp:write("  {" .. bVal .. "},\n")
            end
            fp:write("}\n\n")

            fp:write("local N =  A:solve(b)\n")
            fp:write("print('Function: " .. funName .. "')\n")
            fp:write("print('Solution:')\n")
            fp:write("for i=0,N.m-1 do\n")
            fp:write("\tprint('N_' .. i+1 .. ' = ' .. tonumber(N[i][0]))\n")
            fp:write("end\n")

            fp:close()
        end
    end
end

local function outputProblemPure(C, outputDir)
    local modNum = 0
    for modName,mod in pairs(C) do --Examine all modules
        modNum = modNum + 1
        for funName,fun in pairs(mod) do --Examine all functions
            local A, b, P = {}, {}, {} --A, b are matrices; P is the set of probability-variables
            for eqnNum,eqn in ipairs(fun) do --Examine all equations
                b[eqnNum] = 0
                A[eqnNum] = {}
                A[eqnNum][eqn[0].id] = 1
                for termId=1,#eqn do --Examine all RHS terms
                    local term = eqn[termId]
                    if (term.kind == "entry") then
                        b[eqnNum] = 1
                    elseif (term.kind == "uncond") then
                        A[eqnNum][term.id] = -1
                    else
                        local PrVar
                        if term.kind == "true" then
                            PrVar = "P_" .. term.pos .. "_true"
                            P[PrVar] = true
                        else
                            PrVar = "(1-P_" .. term.pos .. "_true)"
                        end
                        A[eqnNum][term.id] = "-" .. PrVar
                    end
                end
            end

            --Output the problem
            local fileName = outputDir .. "/" .. modNum .. "_" .. funName .. ".lua"
            print("Writing to file: " .. fileName)

            local fp = io.open(fileName, "w")
            fp:write("local matrix = require 'matrix.matrix'\n\n")

            fp:write("--BEGIN TODO: Fill\n")
            for key,_ in pairs(P) do
                fp:write("local " .. key .. " = \n")
            end
            fp:write("--END TODO: Fill\n")
            fp:write("\n")

            fp:write("local A = matrix{\n")
            for i=1,#fun do
                fp:write("  {")
                for j=1,#fun do
                    a_ij = A[i][j] or 0
                    fp:write(" " .. a_ij .. ",")
                end
                fp:write(" },\n")
            end
            fp:write("}\n\n")

            fp:write("local b = matrix{\n")
            for _,bVal in pairs(b) do
                fp:write("  {" .. bVal .. "},\n")
            end
            fp:write("}\n\n")

            fp:write("local N =  A^-1 * b\n")
            fp:write("print('Function: " .. funName .. "')\n")
            fp:write("print('Solution:')\n")
            fp:write("for i=1,#N do\n")
            fp:write("\tprint('N_' .. i .. ' = ' .. tonumber(N[i][1]))\n")
            fp:write("end\n")

            fp:close()
        end
    end
end

local trace = parseTrace(arg[1])
idSuccPred(trace)
--print(inspect(trace, false, 10))
local constraints = getLinearConstraints(trace)
--print(inspect(constraints, false, 10))
printConstraints(constraints)
printTaskLists(trace)
--outputProblemLNA(constraints, arg[2]) --Use OpenBLAS routines to solve the matrix equations
outputProblemPure(constraints, arg[2]) --Use pure Lua routines to solve the matrix equations
