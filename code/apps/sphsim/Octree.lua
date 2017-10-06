local Octree = {}
Octree.__index = Octree
local Key = require "Key"
local HCell = require "HCell"
local Cell = require "Cell"
local Content = require "Content"
local ffi = require("ffi")

function Octree.reconstruct(cell)
  if cell.class == "HCell" then
    setmetatable(cell, HCell)
  elseif cell.class == "Cell" then
    setmetatable(cell, Cell)
  end
  if cell.reconstructed then
    cell = cell:copy()
  else
    cell:reconstruct()
  end
  return cell
end

ffi.cdef[[
  void set_sizes(double, double, double);
]]
local bit = ffi.load("./BitOp/libOp.so")

function Octree.create(sizes, init_cell, update_cell)
  bit.set_sizes(sizes.x, sizes.y, sizes.z)
  local point = {x=sizes.x/2,y=sizes.y/2,z=sizes.z/2}
  local content = Content.create(point,{x=0,y=0,z=0})
  local octree = {
    cells = {},
    init_cell = init_cell,
    update_cell = update_cell,
    root = HCell.create(Key.create_root(),point)
  }
  setmetatable(octree,Octree)
  init_cell(octree.root, content)
  octree:insert_rec(octree.root)
  return octree
end
function Octree.set_sizes(sizes)
  bit.set_sizes(sizes.x, sizes.y, sizes.z)
end

function Octree:insert(content)
  local key = content.key
  local cell = self:find_cell(key)
  if cell then
    cell:insert_content(content)
    self.update_cell(cell, content)
    self:update_parents(cell, content)
  else  
    cell = Cell.create(content, key)
    self.init_cell(cell, content)
    self:insert_rec(cell)
    self:insert_after_first_ancestor(cell, content)
  end
end

function Octree:insert_foreign_cell(cell)

  local my_cell = self:find_cell(cell.key)
  if my_cell then
    print ("Error while inserting foreign cell with key:")
    cell.key:print_bitset()
    print("\n Already have a cell with key:")
    my_cell.key:print_bitset()
    print("")
  end
  cell.status = "remote"



  self:insert_rec(cell)
  local parent_key = cell.key:get_parent()
  local parent = self:find_cell(parent_key)
  local dnum = cell.key:daughter_number()
  while not parent do
    local new_cell = HCell.create_with_bbox(parent_key, cell.bbox)
    self:insert_rec(new_cell)
    cell.parent = new_cell
    new_cell:set_daughter(dnum, cell)
    
    cell = new_cell
    parent_key = cell.key:get_parent()
    parent = self:find_cell(parent_key)
    dnum = cell.key:daughter_number()
  end
  self:update_parents_with_bbox(parent, cell.bbox)
  cell.parent = parent
  parent:set_daughter(dnum, cell)
  
end

function Octree:insert_after_first_ancestor(cell, content)
  local parent_key = cell.key:get_parent()
  local parent = self:find_cell(parent_key)
  local dnum = cell.key:daughter_number()
  while not parent do
    local new_cell = HCell.create(parent_key, content.point)
    self.init_cell(new_cell,content)
    self:insert_rec(new_cell)
    cell.parent = new_cell
    new_cell:set_daughter(dnum, cell)
    
    cell = new_cell
    parent_key = cell.key:get_parent()
    parent = self:find_cell(parent_key)
    dnum = cell.key:daughter_number()
  end
  self:update_parents(parent, content)
  cell.parent = parent
  parent:set_daughter(dnum, cell)
end
-- TODO refactor
--function Octree:insert_after_first_ancestor(cell, content)
--  local point = content.point
--  local key = cell.key
--  if not key:is_one() then
--    local parent_key = key:get_parent()
--    local parent_cell = self:find_cell(parent_key)
--    local level = 0
--    local dnums = {}
--    dnums[0] = key:daughter_number()
--    while not parent_cell do
--      level = level+1
--      dnums[level] = parent_key:daughter_number()
--      parent_key = parent_key:get_parent()
--      parent_cell = self:find_cell(parent_key)
--    end
--    parent_cell.bbox:insert(content.point)
--    self.update_cell(parent_cell, content)
--    self:update_parents(parent_cell, content)
--    local daughter = parent_cell:get_daughter(dnums[level])
--    local c_ancestor_key = key:get_ancestor(level)
--    if daughter then
--      local d_ancestor_key = daughter.key:get_ancestor(level)
--      while (d_ancestor_key:equals(c_ancestor_key)) do
--        local new_cell = HCell.create(d_ancestor_key,point)
--        self.init_cell(new_cell, content)
--        parent_cell:set_daughter(dnums[level], new_cell)
--        self:insert_rec(new_cell)
--        new_cell.parent = parent_cell
--        local ct = daughter.content
--        while ct do
--          new_cell.bbox:insert(ct.value.point)
--          self.update_cell(new_cell, ct.value)
--          ct = ct.next
--        end
--        level = level-1
--        d_ancestor_key = daughter.key:get_ancestor(level)
--        c_ancestor_key = key:get_ancestor(level)
--        parent_cell = new_cell
--      end
--      parent_cell:set_daughter(d_ancestor_key:daughter_number(), daughter)
--      daughter.parent = parent_cell
--    end
--    parent_cell:set_daughter(c_ancestor_key:daughter_number(), cell)
--    cell.parent = parent_cell
--  end   
--end

function Octree:update_parents(cell, content)
  local key = cell.key
  if not key:is_one() then
    local parent_cell = cell.parent
    local daughter_number = key:daughter_number()
    parent_cell:set_daughter(daughter_number, cell)
    parent_cell.bbox:insert(content.point)
    self.update_cell(parent_cell, content)
    self:update_parents(parent_cell, content)
  end 
end

function Octree:update_parents_with_bbox(cell, bbox)
  local key = cell.key
  cell.bbox:merge(bbox)
  if not key:is_one() then
    local parent_cell = cell.parent
    self:update_parents_with_bbox(parent_cell, bbox)
  end 
end

---
--Insert an absent key in the hash table
--
function Octree:insert_rec(cell)
  local hash = cell.key:hash()
  local entry = self.cells[hash]
  if entry then
    local new_entry = {value=cell, next=entry}
    self.cells[hash] = new_entry
  else
    self.cells[hash] = {value=cell, next=nil}
  end
end

--- 
-- Return the cell with requested key if present in the hash table or nil
-- 
function Octree:find_cell(key)
  local res = nil
  local hash = key:hash()
  local first = self.cells[hash]
  local entry = first
  local prev = nil
  local accesses = 1
  while entry do
    if entry.value.key:equals(key) then
      if prev then
        prev.next = entry.next
        entry.next = first
        self.cells[hash] = entry
      end
      res = entry.value
      break
    end
    prev  = entry
    entry = entry.next
    accesses = accesses + 1
  end
  return res, accesses
end

function Octree:print()
  local root = self.root
  if root then
    local prefix = ""
    self:print_rec(root, prefix)
  else
    print "Empty tree"
  end
end

function Octree:print_rec(cell, prefix)
  io.write(prefix)
  cell:print()
  if not cell.is_leaf() then
    for i = 0, 7 do
      local daughter = cell:get_daughter(i)
      if daughter then
        self:print_rec(daughter, prefix.." ")
      end
    end
  end
end

local function check_daughters(cell, res)
  local kg = false
  local count = 0
  res.out.accesses = res.out.accesses+1
  if not cell:is_leaf() then
    if (res.prev and cell.key:is_ancestor(res.prev)) or
          (res.next and cell.key:is_ancestor(res.next)) then
      kg = true
    else
      table.insert(res.out.branches, cell:create_copy_for_transfer())
    end
  else
    table.insert(res.out.branches, cell:create_copy_for_transfer())
  end
  return kg, res
end

function Octree:traverse(func, args)
  local root = self.root  
  local result = args
  if root then
    result = self:traverse_rec(root, func, args)
  end
  return result
end

function Octree:traverse_rec(cell, func, args)
  local kg, result = func(cell, args)
  if kg and not cell:is_leaf() then
    for i = 0, 7 do
      local daughter = cell:get_daughter(i)
      if daughter then
        result = self:traverse_rec(daughter, func, result)
      end
    end
  end
  return result
end

function Octree:find_englobing_branches(prev, next)
  return self:traverse(check_daughters, {out={accesses=0,branches={}}, prev=prev,next=next}).out
end

return Octree