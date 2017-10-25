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
                if not trace.Modules then trace.Modules = {}; trace["-ModuleTable"] = {} end
                module = {
                    name = line:split(":")[2]:strip(),
                }
                trace.Modules[#trace.Modules + 1] = module
                trace["-ModuleTable"][module.name] = #trace.Modules
            elseif line:match("Function ") then
                current = "Function"
                if not module.Functions then module.Functions = {}; module["-FunctionTable"] = {} end
                local fields = line:split(":")
                fun = {
                    name = fields[2]:strip(),
                    definition = (fields[3]:strip() == "Definition"),
                    intrinsic = (fields[4]:strip() == "Intrinsic"),
                }
                module.Functions[#module.Functions + 1] = fun
                module["-FunctionTable"][fun.name] = #module.Functions
            elseif line:match("BasicBlock ") then
                current = "BasicBlock"
                if not fun.BasicBlocks then fun.BasicBlocks = {}; fun.BasicBlockTable = {} end
                bb = {
                    name = line:split(":")[2]:strip(),
                }
                --print("DBG:", line, bb.name)
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

    --print(inspect(trace))
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
                            sucs["uncond"] = fun.BasicBlockTable[ops[1]:split("<@")[1]:split(":")[2]:strip()]
                            --print("DBG:", sucs["uncond"])
                        elseif #ops == 3 then -- Conditional branch
                            sucs["true"] = fun.BasicBlockTable[ops[2]:split("<@")[1]:split(":")[2]:strip()]
                            sucs["false"] = fun.BasicBlockTable[ops[3]:split("<@")[1]:split(":")[2]:strip()]
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
        local modNum = trace["-ModuleTable"][mod.name]
        local modC = {["-id"] = modNum}
        for _,fun in ipairs(mod.Functions) do
            if fun.definition then
                local funNum = mod["-FunctionTable"][fun.name]
                local funC = {["-id"] = funNum}
                --Form the constraints for fun
                for bbNum,bb in ipairs(fun.BasicBlocks) do --Examine all basic-blocks; 1 equation for each BB
                    local E = {
                        name = bb.name,
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

local function getFunctionEntries(trace, C)
    --NOTE: It is assumed that there is just a single module present
    if not trace.Modules then return C end

    for modNum,mod in ipairs(trace.Modules) do
        local modC = C[mod.name]
        for funNum,fun in ipairs(mod.Functions) do
            if fun.definition then
                for bbNum,bb in ipairs(fun.BasicBlocks) do --Examine all basic-blocks; 1 equation for each BB
                    for instNum,inst in ipairs(bb.Instructions) do --Examine all instructions for 'call's
                        if inst.name == "call" then
                            local operands = inst.Operands
                            local targetFunString = operands[#operands]
                            local targetFunName = targetFunString:split("<@")[1]:split(":")[2]:strip()
                            local targetFunNum = mod["-FunctionTable"][targetFunName]
                            local target = mod.Functions[targetFunNum]

                            --print("Calling: " .. targetFunName .. " (" .. (target.definition and "Defined" or "Declared") .. ", " .. (target.intrinsic and "Intrinsic" or "Extrinsic") .. ")")
                            if (target.definition and not target.intrinsic) then
                                --print(modNum .. "_" .. funNum .. "_" .. bbNum .. ": Call[" .. targetFunName .. "]")
                                if not modC[targetFunName]["-Entry"] then modC[targetFunName]["-Entry"] = {} end
                                local E = modC[targetFunName]["-Entry"]
                                local predBlockId = funNum .. "_" .. bbNum
                                if not E[predBlockId] then E[predBlockId] = { from = { funNum, bbNum }, count = 0} end
                                E[predBlockId].count = E[predBlockId].count + 1
                            end
                        end
                    end
                end
            end
        end
    end
    return C
end

local function printTaskLists(trace)
    io.write("\n\nTask lists for each basic-block in every function:\n")
    if not trace.Modules then return end
    for _,mod in ipairs(trace.Modules) do
        io.write("\nModule: " .. mod.name .. "\n")
        for funNum,fun in ipairs(mod.Functions) do
            if fun.definition then
                io.write("\n\tFunction: " .. fun.name .. "\n")
                for bbNum,bb in ipairs(fun.BasicBlocks) do --BasicBlocks
                    io.write("\t\tBB_" .. funNum .. "_" .. bbNum .. " [" .. bb.name .. "]: { ")
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

local function outputModule(pure, A, b, P, entryEqId, modOps, outputDir, modName, modNum)
    --Output the problem for this module
    local fileName = outputDir .. "/module_" .. modNum .. ".lua"
    print("Writing to file: " .. fileName)

    local fp = io.open(fileName, "w")
    fp:write("local matrix = " .. (pure and "require 'matrix.matrix'\n\n" or "require 'lna.matrix'\n\n"))
    fp:write("local Prob = require 'probabilities' --TODO: Make sure to provide the file 'probabilities.lua' externally\n\n")

    fp:write(pure and "local A = matrix{\n" or "local A = matrix.fromtable{\n")
    for i=0,#A do
        fp:write("  {")
        for j=0,#A do
            a_ij = A[i][j] or 0
            fp:write(" " .. a_ij .. ",")
        end
        fp:write(" }, --" .. A[i].name .. "\n")
    end
    fp:write("}\n\n")

    fp:write(pure and "local b = matrix{\n" or "local b = matrix.fromtable{\n")
    for i,bVal in pairs(b) do
        fp:write("  {" .. bVal .. "}, --" .. A[i].name .. "\n")
    end
    fp:write("}\n\n")

    fp:write("local blockName = {\n")
    for i,val in pairs(A) do
        fp:write("  '" .. A[i].name:gsub("%%.","%%%1"):format("%q") .. "',\n")
    end
    fp:write("}\n\n")

    fp:write("local instCounts = {\n")
    for i,val in pairs(A) do
        --print(i, A[i].name)
        if (not A[i].name:match('/ENTRY/')) then
            fp:write("  " .. A[i].TOTAL .. ",\n")
        else
            fp:write("  0,\n")
        end
    end
    fp:write("}\n\n")

    fp:write("local N =  " .. (pure and "A^-1 * b\n" or "A:solve(b)\n"))

    fp:write("\nlocal N_tot = 0\n")
    fp:write(pure and "for i=1,#N do\n" or "for i=0,N.m-1 do\n")
    fp:write("\tif (not blockName[i]:match('/ENTRY/')) then N_tot = N_tot + tonumber(N[i][" .. (pure and 1 or 0) .. "]) end\n")
    fp:write("end\n")
    --fp:write("print(N_tot)\n")

    fp:write("print('Module: " .. modName .. "')\n")
    fp:write("print('Solution:')\nlocal PBB = {}\n")
    fp:write(pure and "for i=1,#N do\n" or "for i=0,N.m-1 do\n")
    fp:write("\tif (not blockName[i]:match('/ENTRY/')) then PBB[i] = tonumber(N[i][" .. (pure and 1 or 0) .. "])/N_tot; print('N_' .. blockName[i" .. (pure and "" or "+1") .. "] .. ' = ' .. tonumber(N[i][" .. (pure and 1 or 0) .. "]) .. ' (' .. PBB[i] .. ')'); else PBB[i] = 0; end\n")
    fp:write("end\n")

    --Opcount part
    local ops = {}
    for funNum,funOps in pairs(modOps) do
        --print("\nFUN:", funNum)
        for bbNum,bbOps in pairs(funOps) do
            --print("\nBB:", bbNum)
            local I = entryEqId[funNum] + bbNum + 1
            for op,count in pairs(bbOps) do
                --print(op, count)
                ops[op] = (ops[op] and (ops[op] .. " + ") or "") .. count .. " * N[" .. I .. "][1]"
            end
        end
    end

    local opsList = {}
    for op,val in pairs(ops) do
        opsList[#opsList+1] = {op, val}
    end
    table.sort(opsList, function(x, y) return (x[1] < y[1]) end)

    fp:write("\nlocal ops = {}\n")
    for i,val in pairs(opsList) do
        local opName = val[1]:gsub("%%.","%%%1"):format("%q")
        fp:write("ops[" .. i .. "] = {'" .. opName .. "', " .. val[2] .. "}\n")
    end
    fp:write("\nprint('\\n\\nDYNAMIC OP COUNTS FOR THIS INSTANCE:')\n")
    fp:write("\nfor _,val in pairs(ops) do print(val[1] .. ' : ' .. val[2]) end\n")
    fp:write([[

P = {}
for i=1,#A do P[i] = {} end
for i=1,#A do
    for j=1,#A[i] do
        if (A[i][j] < 0) then
            P[j][i] = -A[i][j]
        else
            P[j][i] = 0
        end
    end
end
io.write("\n\nP =\n")
for i=1,#P do
    io.write(blockName[i], " : ")
    for j=1,#P[i] do
        io.write(P[i][j], " ")
    end
    io.write("\n")
end

print("\n\nInstruction Counts:")
for i=1,#instCounts do
    io.write(blockName[i], " : ", instCounts[i], "\n")
end

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

--Generate CFG
local fp = io.open("cfg.dot", "w")
fp:write("digraph PDG {\n\n\trankdir=TB;\n\tnode [style=filled];\n\tsize=\"10x10\";\n\n")
local funName=""
local funCount=1
for i=1,#blockName do --Nodes
    local thisFun = blockName[i]:split(":")[1]
    if thisFun ~= funName then
        if funName ~= "" then
            fp:write("\t}\n\n")
        end
        funName = thisFun
        fp:write("\tsubgraph cluster_", funCount, " {\n")
        fp:write("\t\tlabel=\"", funName, "\";\n")
        funCount = funCount + 1
    end

    local sumProb=0.0
    for j=1,#P[i] do
        sumProb = sumProb + P[i][j]
    end
    if sumProb <= 0.0 then
        fp:write("\t\tBB", i, "\t[fillcolor=\"#FF0000\", label=\"", blockName[i], "\\n", N[i][1], ", ", PBB[i], "\\n", instCounts[i], "\\nRET\"];\n")
    else
        if blockName[i]:match("/ENTRY/") then
            fp:write("\t\tBB", i, "\t[fillcolor=\"#0000FF\", label=\"", blockName[i], "\"];\n")
        elseif PBB[i] <= 0.0 then
            fp:write("\t\tBB", i, "\t[fillcolor=\"#FF0000\", label=\"", blockName[i], "\\n", N[i][1], ", ", PBB[i], "\\n", instCounts[i], "\"];\n")
        else
            fp:write("\t\tBB", i, "\t[fillcolor=\"#FFEFD5\", label=\"", blockName[i], "\\n", N[i][1], ", ", PBB[i], "\\n", instCounts[i], "\"];\n")
        end
    end
end
fp:write("\t}\n\n")
for i=1,#P do
    for j=1,#P[i] do --Edges
        if (P[i][j] > 0) then
            fp:write("\tBB", i, " -> ", "BB", j, "\t[color=\"#00FF00\", label=\"", P[i][j], "\"];\n")
        end
    end
end
fp:write("\n}\n")
]])

    fp:close()

    local fileName = outputDir .. "/probabilities_module_" .. modNum .. ".lua"
    local fp = io.open(fileName, "w")
    fp:write("local BranchingProbs = { --TODO: Fill in static values for the various branching probbabilities for your instance\n")
    local prv = ""
    for probVar,val in pairs(P) do
        local prVar, whichBB, funName, delimit = val[1], val[2], val[3], ""
        if prv ~= funName then
            if prv ~= "" then delimit = "\n" end
            prv = funName
        end
        fp:write(delimit .. "    " .. prVar .. " = , -- " .. whichBB .. "\n")
    end
    fp:write("}\n\n")
    fp:write("return BranchingProbs\n")
    fp:close()
end

local function outputProblem(pure, C, opCount, outputDir)
    for modName,mod in pairs(C) do --Examine all modules
        if modName ~= "-id" then
            local modNum = mod["-id"]

            local entryEqId = {}
            local I = 0
            for funName,fun in pairs(mod) do --Examine all functions
                if funName ~= "-id" then
                    local funNum = fun["-id"]
                    entryEqId[funNum] = I
                    I = I + #fun + 1
                end
            end

            local A, b, P = {}, {}, {} --A, b are matrices; P is the set of probability-variables
            for funName,fun in pairs(mod) do --Examine all functions
                if funName ~= "-id" then
                    local funNum = fun["-id"]

                    I = entryEqId[funNum]
                    b[I] = (funName == "main") and 1 or 0 --Main function is entered just once, through the OS shell/kernel

                    --Entry to other functions must be from other basic-blocks in other functions
                    local E = fun["-Entry"] or {}
                    local thisEqn = {name = funName .. ":0:[/ENTRY/]"}
                    thisEqn[I] = 1
                    for key,val in pairs(E) do
                        local from = entryEqId[val.from[1]] + val.from[2]
                        thisEqn[from] = (thisEqn[from] or 0) - val.count
                    end
                    A[I] = thisEqn

                    for eqnNum,eqn in ipairs(fun) do --Examine all local equations (ie., each basic-block) from the function
                        local whichBB = funName .. ":" .. eqnNum .. ":[" .. eqn.name .. "]" --Identity of the current BB
                        b[I + eqnNum] = 0
                        local thisEqn = {name = whichBB}
                        thisEqn[I + eqn[0].id] = 1
                        for termId=1,#eqn do --Examine all RHS terms
                            local term = eqn[termId]
                            if ((term.kind == "entry") or (term.kind == "uncond")) then
                                thisEqn[I + term.id] = (thisEqn[I + term.id] or 0) - 1
                            else
                                local PrVar
                                if term.kind == "true" then
                                    PrVar = "Prob.T_" .. term.pos
                                    P[#P+1] = { "T_" .. term.pos, whichBB, funName }
                                else
                                    PrVar = "(1-Prob.T_" .. term.pos .. ")"
                                end
                                thisEqn[I + term.id] = (thisEqn[I + term.id] or "") .. "-" .. PrVar
                            end
                        end
                        --print(modNum, funNum, funName, eqnNum, eqn.name, opCount[modNum][funNum][eqnNum].TOTAL)
                        thisEqn.TOTAL = opCount[modNum][funNum][eqnNum].TOTAL
                        A[I + eqnNum] = thisEqn
                    end
                end
            end
            table.sort(P, function(x, y) return (x[1] < y[1]) end)

            outputModule(pure, A, b, P, entryEqId, opCount[modNum], outputDir, modName, modNum)
        end
    end
end

local function getTypeVectorWidth(typeName)
    if (typeName:sub(1, 1) == "<") then
        local vectorWidth = typeName:sub(2):split("x")[1]:strip()
        return vectorWidth
    else
        return 1
    end
end

local function countBasicBlockOps(trace)
    --NOTE: It is assumed that there is just a single module present
    if not trace.Modules then return C end

    local opCount = {}
    for modNum,mod in ipairs(trace.Modules) do
        opCount[modNum] = {}
        local modOpCount = opCount[modNum]
        for funNum,fun in ipairs(mod.Functions) do
            if fun.definition then
                modOpCount[funNum] = {}
                local funOpCount = modOpCount[funNum]
                for bbNum,bb in ipairs(fun.BasicBlocks) do --Examine all basic-blocks; 1 equation for each BB
                    funOpCount[bbNum] = {}
                    local bbOpCount = funOpCount[bbNum]
                    bbOpCount["TOTAL"] = 0
                    for instNum,inst in ipairs(bb.Instructions) do --Examine all instructions for 'call's
                        local operands = inst.Operands
                        local name, vectorWidth = inst.name, 1
                        if operands and #operands == 1 then
                            local typeName = operands[1]:split("<@")[1]:split(":")[1]:strip()
                            vectorWidth = getTypeVectorWidth(typeName)
                            name = inst.name .. ":" .. vectorWidth .. ":" .. typeName
                        end

                        if inst.name == "call" then
                            local operands = inst.Operands
                            local targetFunString = operands[#operands]
                            --print(inspect(operands), targetFunString)
                            local targetFunName = targetFunString:split("<@")[1]:split(":")[2]:strip()
                            local targetFunNum = mod["-FunctionTable"][targetFunName]
                            local target = mod.Functions[targetFunNum]

                            --print("Calling: " .. targetFunName .. " (" .. (target.definition and "Defined" or "Declared") .. ", " .. (target.intrinsic and "Intrinsic" or "Extrinsic") .. ")")
                            if (target.definition and not target.intrinsic) then
                                bbOpCount["call"] = (bbOpCount["call"] or 0) + 1
                                bbOpCount["TOTAL"] = bbOpCount["TOTAL"] + 1
                            end
                        elseif inst.name == "br" then
                            local ops = inst.Operands
                            if #ops == 1 then --Unconditional branch
                                bbOpCount["br:uncond"] = (bbOpCount["br:uncond"] or 0) + 1
                                bbOpCount["TOTAL"] = bbOpCount["TOTAL"] + 1
                            elseif #ops == 3 then -- Conditional branch
                                bbOpCount["br:cond"] = (bbOpCount["br:cond"] or 0) + 1
                                bbOpCount["TOTAL"] = bbOpCount["TOTAL"] + 1
                            else --Unknown
                                error("Unknown branch instruction in the basic-block " .. bb.name .. " in function " .. fun.name .. " in module " .. mod.name)
                                bbOpCount["TOTAL"] = bbOpCount["TOTAL"] + 1
                            end
                        else
                            bbOpCount[name] = (bbOpCount[name] or 0) + vectorWidth
                            bbOpCount["TOTAL"] = bbOpCount["TOTAL"] + vectorWidth
                        end
                    end
                end
            end
        end
    end

    return opCount
end

local pureMode = true
local trace = parseTrace(arg[1])

local opCount = countBasicBlockOps(trace)
print(inspect(opCount, false, 10))

idSuccPred(trace)
--print(inspect(trace, false, 10))
--os.exit(0)

local constraints = getLinearConstraints(trace)
--print(inspect(constraints, false, 10))
--os.exit(0)
printTaskLists(trace)
getFunctionEntries(trace, constraints)
outputProblem(pureMode, constraints, opCount, arg[2]) --Use either pure Lua or LNA routines to solve the linear system

--TODO:
--Form a tasklet data dependency graph for each basic-block
