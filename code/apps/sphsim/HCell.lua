local HCell = {}
HCell.__index = HCell
local math = require "math"
local BBox = require "BoundingBox"
local ffi = require("ffi")

ffi.cdef[[
  int bor(int val1, int val2);
  int band(int val1, int val2);
]]
local bit = ffi.load("./BitOp/libOp.so")
local Key = require "Key"

function HCell.create(key, point, owner)
  local hcell = {
    bbox = BBox.create(point),
    --daughters = 0,
    daughters_array = {},
    key = key,
    owner = owner,
    parent = nil
  }           
  setmetatable(hcell,HCell)
  return hcell
end
function HCell.create_with_bbox(key, bbox, owner)
  local hcell = {
    bbox = bbox,
    --daughters = 0,
    daughters_array = {},
    key = key,
    owner = owner,
    parent = nil
  }           
  setmetatable(hcell,HCell)
  return hcell
end
function HCell:create_copy_for_transfer()
  local copy = {
    bbox = self.bbox,
    daughters_array = {},
    key_first = self.key:get_first(),
    key_last = self.key:get_second(),
    key_level = self.key.level,
    owner = self.owner,
    parent = nil,
    class = "HCell"
  }
  for i=0,7 do
    if self:get_daughter(i) then
      copy.daughters_array[i+1] = true
    else
      copy.daughters_array[i+1] = nil
    end
  end
  setmetatable(copy,HCell)
  return copy
end
function HCell:reconstruct()
  self.key = Key.reconstruct(self.key_first, self.key_last, self.key_level)
  self.status = "remote"
  self.key_first = nil
  self.key_last = nil
  self.key_level = nil
  self.reconstructed = true
  setmetatable(self.bbox, BBox)
end

function HCell:copy()
  local hcell = {
    bbox = self.bbox:copy(),
    daughters_array = {},
    key = self.key:copy(),
    owner = self.owner,
    status = "remote",
    parent = nil
  }
  for i=0,7 do
    hcell.daughters_array[i+1] = self.daughters_array[i+1]
  end
  setmetatable(hcell,HCell)
  return hcell
end
function HCell:copy_branch()
  local hcell = {
    bbox = self.bbox:copy(),
    daughters_array = {},
    key_first = self.key_first,
    key_last = self.key_last,
    key_level = self.key_level,
    owner = self.owner,
    parent = nil,
    class = "HCell"
  }
  for i=0,7 do
    hcell.daughters_array[i+1] = self.daughters_array[i+1]
  end
  setmetatable(hcell,HCell)
  return hcell
end
function HCell.is_leaf()
  return false
end

function HCell:set_daughter(dnum, cell)
  --self.daughters = bit.bor(self.daughters, math.pow(2,dnum))
  self.daughters_array[dnum+1] = cell
end

function HCell:get_daughter(dnum)
  return self.daughters_array[dnum+1]
end

function HCell:print()
  io.write ("{ key = ")
  self.key:print_bitset()
  io.write (", bbox = ")
  self.bbox:print()
  io.write(", daughters = {")
  for i = 0, 7 do
    io.write(tostring(not (self:get_daughter(i) == nil)).." ")
  end
  print ("} }")
end

return HCell