require "defines"
local ffi = require("ffi")
local math = require "math"

ffi.cdef[[
typedef uint64_t key_t;
void print_bitset(uint64_t bitset[2]);
void hash(
    uint64_t res[2],
    double x, 
    double y, 
    double z
    );
void rshift(uint64_t res[2], uint64_t set[2], int);
void lshift(uint64_t res[2], uint64_t set[2], int);
void get_daughter(uint64_t res[2], uint64_t key[2], uint64_t daughter);
bool equals(uint64_t set[2], uint64_t set2[2]);
bool lt(uint64_t set[2], uint64_t set2[2]);
bool gt(uint64_t set[2], uint64_t set2[2]);
bool is_one(uint64_t set[2]);
uint64_t band(uint64_t val1, uint64_t val2);
uint64_t bor(uint64_t val1, uint64_t val2);
void create_root(uint64_t res[2]);
int get_short_hash(uint64_t val[2]);
]]
local bit = ffi.load("./BitOp/libOp.so")

local Key = {}
Key.__index = Key

function Key.reconstruct(first, second, level)
  local key = {
    value = ffi.new("uint64_t[2]",{}),
    parent= nil,
    dnum=nil,
    level = level 
  }
  key.value[0] = ffi.typeof'uint64_t'(ffi.cast(ffi.typeof'uint64_t *', first)[0])
  key.value[1] = ffi.typeof'uint64_t'(ffi.cast(ffi.typeof'uint64_t *', second)[0])
  setmetatable(key,Key)
  return key
end

function Key.create(point, minVal)
  local key = {
    value = ffi.new("uint64_t[2]",{}),
    parent= nil,
    dnum=nil,
    level = 0
  }
  setmetatable(key,Key)  
  bit.hash(key.value, point.x-minVal.x, 
           point.y-minVal.y, point.z-minVal.z)    
  return key
end
function Key:copy()
  local copy = {
    value = ffi.new("uint64_t[2]",{}),
    parent= nil,
    dnum=self.dnum,
    level = self.level
  }
  copy.value[0] = self.value[0]
  copy.value[1] = self.value[1]
  setmetatable(copy,Key)    
  return copy
end
function Key.create_root()
  local key = {}           
  setmetatable(key,Key)  
  key.value = ffi.new("uint64_t[2]",{}) 
  bit.create_root(key.value)
  key.level = 31
  return key 
end
function Key.new(bitset)
  local key = {}           
  setmetatable(key,Key)  
  key.value = bitset    
  return key
end

function Key:get_parent()
  if not self.parent then
    local res = ffi.new("uint64_t[2]",{})
    bit.rshift(res, self.value, 3)
    self.parent = Key.new(res)
    self.parent.level = self.level+1
  end
  return self.parent
end

function Key:update(point, minVal)
  self.value[0] = 0
  self.value[1] = 0
  bit.hash(self.value, point.x-minVal.x, point.y-minVal.y, point.z-minVal.z)
end
---
--Return lower 15 bits of the key
--
function Key:hash()
  return bit.get_short_hash(self.value)
end

function Key:get_daughter(daughter)
  local res = ffi.new("uint64_t[2]",{})
  bit.get_daughter(res, self.value, daughter)
  return Key.new(res)
end
function Key:get_ancestor(level)
  local res = ffi.new("uint64_t[2]",{})
  bit.rshift(res, self.value, 3*level)
  local key = Key.new(res)
  key.level = self.level+level
  return  key
end

function Key:get_first()
  local res = ffi.string(ffi.typeof'uint64_t[1]'(self.value[0]), 8)
  return res
end

function Key:get_second()
  local res = ffi.string(ffi.typeof'uint64_t[1]'(self.value[1]), 8)
  return res
end
---
-- Return lower three bits of the key
-- 
function Key:daughter_number()
  if not self.dnum then
    self.dnum = tonumber(bit.band(self.value[1], 7))
  end
  return self.dnum
end

function Key:equals(key)
  return bit.equals(self.value,key.value)
end

function Key:lt(key)
  return bit.lt(self.value,key.value)
end

function Key:gt(key)
  return bit.gt(self.value,key.value)
end

function Key:print_bitset()
  bit.print_bitset(self.value)
end

function Key:is_one()
  return bit.is_one(self.value)
end

function Key:is_ancestor(key)
  local res = false
  local diff = self.level - key.level
  if diff>0 then
    res = self:equals(key:get_ancestor(diff))
  end
  return res 
end
return Key