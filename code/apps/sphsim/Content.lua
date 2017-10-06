local Content = {}
Content.__index = Content
local Key = require "Key"

function Content:create_copy_for_transfer()
  local copy = {
    point = self.point,
    key_first = self.key:get_first(),
    key_last = self.key:get_second(),
    mass=self.mass,
    class = "Content"
  }       
  setmetatable(copy,Content)
  return copy
end

function Content:reconstruct()
  self.key = Key.reconstruct(self.key_first,self.key_last,0)
  --self.key_first = nil
  --self.key_last = nil
end
function Content:copy()
  local copy = {
    point = {x=self.point.x,y=self.point.y,z=self.point.z},
    key = self.key:copy(),
    mass=self.mass
  }       
  setmetatable(copy,Content)
  return copy
end
function Content:copy_branch()
  local copy = {
    point = {x=self.point.x,y=self.point.y,z=self.point.z},
    key_first = self.key_first,
    key_last = self.key_last,
    mass=self.mass,
    class = "Content"
  }       
  setmetatable(copy,Content)
  return copy
end
function Content.create(point, minVal)
  local content = {
    point = point,
    key = Key.create(point,minVal),
    mass=0
  }       
  setmetatable(content,Content)
  return content
end

function Content.cmp(ct1, ct2)
  return ct1.key:lt(ct2.key)
end

function Content:update(minVal)
  self.key:update(self.point, minVal)
end

function Content:move(force)
  self.velocity.x = self.velocity.x+force.x
  self.velocity.y = self.velocity.y+force.y
  self.velocity.z = self.velocity.z+force.z
  self.point.x = self.point.x + self.velocity.x 
  self.point.y = self.point.y + self.velocity.y
  self.point.z = self.point.z + self.velocity.z
end

return Content