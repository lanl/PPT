--[[
--Author: Nandakishore Santhi
--Date: 23 November, 2014
--Copyright: Open source, must acknowledge original author
--Purpose: JITed PDES Engine in LuaJIT
--  Simple string to integer hashing
--]]
-- djb2 @ http://www.cse.yorku.ca/~oz/hash.html
local function hash(str)
    local res = 5381;
    for i=1,#str do --hash(i) = 33*hash(i-1) + c
        res = 33*res + str:byte(i)
    end
    return res
end

return hash
