local Cell = {}
Cell.__index = Cell
local BBox = require "BoundingBox"
local Key = require "Key"
local Content = require "Content"


function Cell:create_copy_for_transfer()
  local copy = {
    content = nil,
    bbox = self.bbox,
    last = nil,
    key_first = self.key:get_first(),
    key_last = self.key:get_second(),
    parent = nil,
    class = "Cell"
  }
  local content = self.content  
  local content_copy = content.value:create_copy_for_transfer()
  copy.content = {value=content_copy, next=nil}
  local last_copy = copy.content
  content = content.next
  while content do
    last_copy.next = {value=content.value:create_copy_for_transfer(), next=nil}
    last_copy=last_copy.next
    content = content.next
  end
  copy.last = last_copy
  setmetatable(copy,Cell)
  return copy
end

function Cell:reconstruct()
  self.key = Key.reconstruct(self.key_first,self.key_last,0)
  self.key_first = nil
  self.key_last = nil
  self.reconstructed = true
  setmetatable(self.bbox, BBox)
  
  local content = self.content
  while content do
    setmetatable(content.value, Content)
    content.value:reconstruct()
    content = content.next
  end  
end
function Cell:copy()
  local copy = {
    content = nil,
    bbox    = self.bbox:copy(),
    last    = nil,
    key     = self.key:copy(),
    parent  = nil
  }
  local content = self.content
  local content_copy = content.value:copy()
  copy.content = {value=content_copy, next=nil}
  local last_copy = copy.content
  content = content.next
  while content do
    last_copy.next = {value=content.value:copy(), next=nil}
    last_copy = last_copy.next
    content = content.next
  end
  copy.last = last_copy
  setmetatable(copy,Cell)
  return copy
end

function Cell:copy_branch()
  local copy = {
    content = nil,
    bbox    = self.bbox:copy(),
    last    = nil,
    key_first = self.key_first,
    key_last = self.key_last,
    parent  = nil,
    class = "Cell"
  }
  local content = self.content
  local content_copy = content.value:copy_branch()
  copy.content = {value=content_copy, next=nil}
  local last_copy = copy.content
  content = content.next
  while content do
    last_copy.next = {value=content.value:copy_branch(), next=nil}
    last_copy = last_copy.next
    content = content.next
  end
  copy.last = last_copy
  setmetatable(copy,Cell)
  return copy
end

function Cell.create(content, key)
  local cell = {
    content = {value = content, next = nil},
    bbox    = BBox.create(content.point),
    last    = nil,
    key     = key,
    parent  = nil
  }           
  setmetatable(cell,Cell)
  cell.last = cell.content
  return cell
end

function Cell:insert_content(content)
  local new_list = {value=content,next=nil}
  self.bbox:insert(content.point)
  self.last.next = new_list
  self.last = new_list
end

function Cell.is_leaf()
  return true
end

function Cell:print()
  io.write("{ key = ")
  self.key:print_bitset()
  io.write(", bbox = ")
  self.bbox:print()
  io.write(", points = {")
  local content = self.content
  while content do
    io.write("["..tostring(content.value.point.x)..", "
      ..tostring(content.value.point.y)..", "
      ..tostring(content.value.point.z).."] ")
    content = content.next
  end
  print("} }")
end

return Cell