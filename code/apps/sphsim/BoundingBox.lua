local math = require "math"

local BBox = {}
BBox.__index = BBox

function BBox.create(point)
  local bbox = {
    minx = point.x,
    maxx = point.x,
    miny = point.y,
    maxy = point.y,
    minz = point.z,
    maxz = point.z}           
  setmetatable(bbox,BBox)
  return bbox
end
function BBox.create_radius(point, radius)
  local bbox = {}           
  setmetatable(bbox,BBox)
  bbox.minx = point.x-radius
  bbox.maxx = point.x+radius
  bbox.miny = point.y-radius
  bbox.maxy = point.y+radius
  bbox.minz = point.z-radius
  bbox.maxz = point.z+radius
  return bbox
end
function BBox:insert(point)
  local min, max = math.min, math.max
  self.minx = min(self.minx, point.x)
  self.maxx = max(self.maxx, point.x)
  self.miny = min(self.miny, point.y)
  self.maxy = max(self.maxy, point.y)
  self.minz = min(self.minz, point.z)
  self.maxz = max(self.maxz, point.z)
end

function BBox:merge(bbox)
  local min, max = math.min, math.max
  self.minx = min(self.minx, bbox.minx)
  self.maxx = max(self.maxx, bbox.maxx)
  self.miny = min(self.miny, bbox.miny)
  self.maxy = max(self.maxy, bbox.maxy)
  self.minz = min(self.minz, bbox.minz)
  self.maxz = max(self.maxz, bbox.maxz)
end
function BBox:copy()
  local copy = {
    minx = self.minx,
    maxx = self.maxx,
    miny = self.miny,
    maxy = self.maxy,
    minz = self.minz,
    maxz = self.maxz}           
  setmetatable(copy,BBox)
  return copy
end
-- May not intersect in reality as it considers the englobing cube defined by
-- the radius
function BBox:intersects(bbox)
  return bbox.minx < self.maxx and
         bbox.maxx > self.minx and
         bbox.miny < self.maxy and
         bbox.maxy > self.miny and
         bbox.minz < self.maxz and
         bbox.maxz > self.minz
end

function BBox:englobes(bbox)
  return self.minx < bbox.minx and 
         self.maxx > bbox.maxx and
         self.miny < bbox.miny and 
         self.maxy > bbox.maxy and
         self.minz < bbox.minz and 
         self.maxz > bbox.maxz
end

function BBox:get_width()
  return self.maxx-self.minx
end

function BBox:get_depth()
  return self.maxy-self.miny
end

function BBox:get_height()
  return self.maxz-self.minz
end

function BBox:print()
  io.write ("{x = ["..tostring(self.minx).." - "..tostring(self.maxx).."]; ")
  io.write ("y = ["..tostring(self.miny).." - "..tostring(self.maxy).."]; ")
  io.write ("z = ["..tostring(self.minz).." - "..tostring(self.maxz).."] }")
end

return BBox