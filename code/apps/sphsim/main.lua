local Octree = require "Octree"
local Content = require "Content"
local BBox = require "BoundingBox"
local math = require "math"
---
--Define how to handle custom data at tree creation
--

-- If passed when creating the tree, this function is called whenever a new 
-- cell is added
function init_cell(cell, content)
  cell.center = {}
  cell.center.x = content.point.x
  cell.center.y = content.point.y
  cell.center.z = content.point.z
  cell.mass = content.mass
end

-- If passed when creating the tree, this function is called whenever a new 
-- point is added to a cell or one of its daughters
function update_cell(cell, content)
  local total_mass = cell.mass+content.mass
  cell.center.x = (cell.center.x*cell.mass+content.point.x*content.mass)
                  /total_mass
  cell.center.y = (cell.center.y*cell.mass+content.point.y*content.mass)
                  /total_mass
  cell.center.z = (cell.center.z*cell.mass+content.point.z*content.mass)
                  /total_mass
  cell.mass = total_mass
end

function count_fat_boxes(cell, count)
  if cell.bbox:get_width() > 0 or
    cell.bbox:get_height() > 0 or
    cell.bbox:get_depth() > 0 then
    count = count+1
  end
  return true, count
end

function count_fat_leaves(cell, count)
  if cell:is_leaf() and (cell.bbox:get_width() > 0 or
    cell.bbox:get_height() > 0 or
    cell.bbox:get_depth() > 0) then
    count = count+1
  end
  return true, count
end

function find_heaviest_leaf(cell, res)
  if cell.is_leaf() and (not res or cell.mass > res.mass) then
    res = cell
  end
  return true, res
end

function find_most_populated_leaf(cell, res)
  if cell.is_leaf() and (not res or cell.n_points > res.n_points) then
    res = cell
  end
  return true, res
end

---
--Define physics functions
--
function compute_gravity(cell, data)
  local kg=false
  if data.bbox:intersects(cell.bbox) then
    if cell.is_leaf() then
      local entry = cell.content
      while entry do
        local content = entry.value
        local force = 
          gravity(data.point, data.mass, content.point, content.mass)
        data.force.x = data.force.x+force.x
        data.force.y = data.force.y+force.y
        data.force.z = data.force.z+force.z
        entry = entry.next
        data.nb_interactions = data.nb_interactions+1
      end
    else
      kg=true
    end
  else
    local force = gravity(data.point, data.mass, cell.center, cell.mass)
    data.force.x = data.force.x+force.x
    data.force.y = data.force.y+force.y
    data.force.z = data.force.z+force.z
    data.nb_interactions = data.nb_interactions+1
  end
  return kg, data
end

function gravity(p1, m1, p2, m2)
  local result = {x=0,y=0,z=0}
  local x, y, z = p2.x-p1.x, p2.y-p1.y, p2.z-p1.z
  local pow = math.pow
  local sqdist = pow(x,2)+
                 pow(y,2)+
                 pow(z,2)
  if sqdist > 100 then
    local grav = 6.6*10^(-11)*(m1*m2)/sqdist
    local dist = math.sqrt(sqdist)
    result.x = grav*x/dist
    result.y = grav*y/dist
    result.z = grav*z/dist
  end
  return result
end


local function main()
  local minVal = {x=math.huge,y=math.huge,z=math.huge}
  local sizes = {x=0,y=0,z=0}
  
  local nsteps  = 600
  local radius  = 10000
  local nbodies = 10000
  
  math.randomseed(0)--os.time())
  
  ---
  --Initialize data points
  --
  local bodies = {}
  local min, max, random = math.min, math.max, math.random
  for i=1,nbodies  do
    local point = 
    {
      x = random()*2097152,
      y = random()*2097152,
      z = random()*2097152
    }
    minVal.x = min(minVal.x, point.x)
    minVal.y = min(minVal.y, point.y)
    minVal.z = min(minVal.z, point.z)
    sizes.x = max(sizes.x,point.x)
    sizes.y = max(sizes.y,point.y)
    sizes.z = max(sizes.z,point.z)
    local velocity = 
    {
      x = (random()-random())*1000,
      y = (random()-random())*1000,
      z = (random()-random())*1000
    }
    local content = Content.create(point,minVal)
    content.mass=random()*50+10. 
    content.velocity=velocity
    bodies[i] = content
  end
  sizes.x = sizes.x - minVal.x
  sizes.y = sizes.y - minVal.y
  sizes.z = sizes.z - minVal.z
  ---
  --This will update the keys for each point with minVal
  Octree.set_sizes(sizes)
  for i=1, nbodies do
    local content = bodies[i]
    content:update(minVal)
    --print(tostring(0)..
      --" "..tostring((content.point.x/2097152)*200)..
      --" "..tostring((content.point.y/2097152)*200)..
      --" "..tostring((content.point.z/2097152)*200)..
      --" "..tostring(content.mass/80))
  end
  table.sort(bodies, Content.cmp)
  ---
  --Run simulation
  --
  local start = os.time() 
  for i=1, nsteps do
    print("Computing step "..tostring(i).." after "..tostring(os.time()-start))
    local oct = Octree.create(sizes, init_cell, update_cell)
    for j = 1, #bodies do
      oct:insert(bodies[j])
    end
    local forces = {}
    local grav_func = compute_gravity
    local total_interactions = 0
    for j = 1, #bodies do
      local content = bodies[j]
      local data = {
        force = {x=0,y=0,z=0}, 
        point = content.point, 
        mass = content.mass,
        bbox = BBox.create_radius(content.point, radius),
        nb_interactions = 0
      }
      data = oct:traverse(compute_gravity, data)
      total_interactions = total_interactions+data.nb_interactions
      local force = data.force
      force.x = force.x*5*10^16
      force.y = force.y*5*10^16
      force.z = force.z*5*10^16
      forces[j] = force
      if j%1000 == 0 then
        --print(tostring(j/100).."%")
      end
    end
    sizes.x = 0
    sizes.y = 0
    sizes.z = 0
    for j = 1, #bodies do
      local content = bodies[j]
      content:move(forces[j], minVal)
      local point = content.point
      minVal.x = min(minVal.x,point.x)
      minVal.y = min(minVal.y,point.y)
      minVal.z = min(minVal.z,point.z)
      sizes.x = max(sizes.x,point.x)
      sizes.y = max(sizes.y,point.y)
      sizes.z = max(sizes.z,point.z)
    end
    sizes.x = sizes.x - minVal.x
    sizes.y = sizes.y - minVal.y
    sizes.z = sizes.z - minVal.z
    for j = 1, #bodies do
      local content = bodies[j]
      content:update(minVal)
      --print(tostring(i)..
        --" "..tostring((content.point.x/2097152)*200)..
        --" "..tostring((content.point.y/2097152)*200)..
        --" "..tostring((content.point.z/2097152)*200)..
        --" "..tostring(content.mass/80))
    end
    print("Computed "..tostring(total_interactions)..
      " interactions at this step")
    table.sort(bodies, Content.cmp)
  end
end
main()
