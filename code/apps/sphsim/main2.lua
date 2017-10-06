local Octree = require "Octree"
local Content = require "Content"
local BBox = require "BoundingBox"
local math = require "math"
local Key = require "Key"
package.path = "../../LuaHM/src/?.lua;" .. package.path
local Simian = require "simian"
local clusters = require "clusters"
local nodes = require "nodes"



---
-- Parameters
---
local simName, startTime, endTime, minDelay, useMPI = "SPH",0,10000000,0.000000001,true
local nsteps = 600
local radius = 280
local nbodies = 15057 
local n_procs = tonumber(arg[1])  
local cell_size = 20 -- size of a tree cell in bytes
local transfer = 0.000000003 -- 0.0000000006 -- time to transfer a single byte over the network
                   -- To be replaced by network simulator
                   
local time_local = 0.000000044 -- 0.000000045 -- Time to compute a local interaction
                          -- To be replaced by tasklist and evaluation
local time_distant = 0.0000000005 -- Time to compute a distant interaction
                             -- To be replaced by tasklist and evaluation 
local cache_access_time = 0.0000000001 -- 0.0000000001
local access_time = 0.000000000205 -- 0.00000000020


local function SPH_compute_handler(this)
  local function count_interactions(cell, res)
    local kg=false
    local acc
    _, acc = this.oct:find_cell(cell.key)
    res.out.accesses = res.out.accesses+acc-1
    res.out.cache_accesses = res.out.cache_accesses+1
    if res.bbox:intersects(cell.bbox) then
      --print(cell.status)
      if cell.is_leaf() then
        local entry = cell.content
        acc = 0
        while entry do
          local content = entry.value
          res.out.local_interactions = res.out.local_interactions+1
          entry = entry.next
          acc = acc+1 
        end
        res.out.accesses = res.out.accesses+acc-1
        res.out.cache_accesses = res.out.cache_accesses+1
      elseif cell.status == "remote" then
        cell.status = "pending"
        table.insert(res.out.requests, 
          {
            owner = cell.owner, 
            key_first = cell.key:get_first(), 
            key_last = cell.key:get_second(),
            level = cell.key.level
          })
        this.resume_list[cell] = {{body=res.body, bbox = res.bbox}}
        res.out.stalls = res.out.stalls+1
      elseif cell.status == "pending" then
        table.insert(this.resume_list[cell],{body=res.body, bbox = res.bbox})
        res.out.stalls = res.out.stalls+1
      else
        kg=true
      end
    else
      res.out.distant_interactions = res.out.distant_interactions+1
    end
    return kg, res
  end
  ---
  -- Send cells requested by others
  ---
  local time = 0
  local accesses = 0
  local cache_accesses = 0
  while next(this.foreign_requests) do
    local request = table.remove(this.foreign_requests,1)
    local data = {}
    local cell, acc = this.oct:find_cell(request.key)
    accesses = accesses+acc-1
    cache_accesses = cache_accesses+1
    accesses = accesses + acc
    local nb_cells = 0
    for i=0,7 do
      local daughter = cell:get_daughter(i)
      if daughter then
        _, acc = this.oct:find_cell(daughter.key)
        accesses = accesses+acc-1
        cache_accesses = cache_accesses+1
        table.insert(data, daughter:create_copy_for_transfer())
        nb_cells = nb_cells+1
      end
    end
    time = time + cache_accesses*cache_access_time + accesses*access_time
    this.nb_outgoing_transfers = this.nb_outgoing_transfers+1
    local transfer_time = time + nb_cells * cell_size * transfer
    --print ("Sending cells with delay", transfer_time)
    this:reqService( 
      math.max(transfer_time,minDelay),
      "SPH_cells_receive_handler",data,"Node",request.rank)
  end
  ---
  -- Create data to traverse tree
  ---
  local res = {
    body = nil,
    bbox = nil,
    out = {
      local_interactions = 0,
      distant_interactions =0,
      requests = {},
      accesses = 0,
      cache_accesses = 0,
      stalls = 0      
    }
  } 
  ---
  --  Resume previous traversals
  ---
  while next(this.resumable) do
    local resume = table.remove(this.resumable, 1)
    res.body = resume.body
    res.bbox = resume.bbox
    res = this.oct:traverse_rec(resume.cell, count_interactions, res)
  end
  ---
  -- Process current body if any
  ---
  if this.current_body<=this.nbodies then
    res.body = this.current_body
    res.bbox = BBox.create_radius(this.bodies[this.current_body].point,radius)
    res = this.oct:traverse(count_interactions, res)
    this.current_body = this.current_body+1    
  end
  
  --TODO: Add number of tree accesses to the time taken
  this.local_int = this.local_int+res.out.local_interactions
  this.distant_int = this.distant_int+res.out.distant_interactions
  time = time + res.out.local_interactions*time_local + 
    res.out.distant_interactions*time_distant + 
    res.out.accesses*access_time+ res.out.cache_accesses*cache_access_time
    
  for _,request in pairs(res.out.requests) do
    local out = {
      rank = this.rank,
      key_first = request.key_first,
      key_last = request.key_last,
      level = request.level    
    }
    --print ("Sending request with delay", time)
    this:reqService(
      math.max(time,minDelay),
      "SPH_request_receive_handler",
      out,
      "Node",
      request.owner)
    time = time+transfer
    this.pending = this.pending+1
  end
  if this.current_body > this.nbodies then
    this.finished = true
    local next = next
    if next(this.resume_list) == nil and this.pending == 0 then
      print("Rank "..this.rank.." has finished computations at time "..this.engine.now+time)
      print("Local interactions: "..this.local_int)
      print("Distant interactions: "..this.distant_int)
      print("Outgoing transfers: "..this.nb_outgoing_transfers)
    end
  else
    --print ("Scheduling next body with delay", time)
    this:reqService(
      math.max(time,minDelay), 
      "SPH_compute_handler", nil, "Node", this.rank)
  end
end

---
--Handle to be executed by each participating node for tree creation
---
local function SPH_init_handler(this, data)
  this.rank = data.rank
  this.world = data.world
  this.received = 1
  this.nbodies = 0
  this.finished = false
  this.local_int = 0
  this.distant_int = 0
  this.nb_outgoing_transfers = 0
  this.pending = 0
  ---
  --Define how to handle custom data at tree creation
  --
  
  -- If passed when creating the tree, this function is called whenever a new 
  -- cell is added
  local function init_cell(cell, content)
    cell.owner = content.owner
  end
  
  -- If passed when creating the tree, this function is called whenever a new 
  -- point is added to a cell or one of its daughters
  local function update_cell(cell, content)
    
  end
  this.oct = Octree.create(data.sizes, init_cell, update_cell)
  local oct = this.oct
  for i, content in ipairs(data.bodies) do
    oct:insert(content)
    this.nbodies = this.nbodies+1
  end
  local result = oct:find_englobing_branches(data.prev_key, data.next_key)
  --TODO
  -- Compute time taken so far (parallel sorting of keys, local tree creation, 
  -- branches identification)
  local init_time = result.accesses*access_time
  local out = {branches=result.branches, sender = this.rank}
  for rank=1,this.world do
    if rank ~= this.rank then   
      this:reqService(
        math.max(init_time,minDelay),
        "SPH_branch_receive_handler",out,"Node",rank)
      local new_out = {branches={}, sender = this.rank}
      for k,cell in pairs(out.branches) do
        local copy = cell:copy_branch()
        table.insert(new_out.branches, copy)
      end
      out = new_out
    end
  end
  this.current_body = 1
  this.bodies = data.bodies
  this.resumable = {}
  this.resume_list = {}
  this.foreign_requests = {}
  
  if n_procs == 1 then
    SPH_compute_handler(this)
  end
end
local function SPH_request_receive_handler(this, data)
  local key = Key.reconstruct(data.key_first, data.key_last, data.level)
  --key:print_bitset()
  --print("")
  if this.finished then
    local out = {}
    local cell = this.oct:find_cell(key)
    local nb_cells = 0
    for i=0,7 do
      local daughter = cell:get_daughter(i)
      if daughter then
        table.insert(out, daughter:create_copy_for_transfer())
      end
    end
    local transfer_time = nb_cells * cell_size * transfer
    this:reqService( 
      math.max(transfer_time,minDelay),
      "SPH_cells_receive_handler",out,"Node",data.rank)
  else
    table.insert(this.foreign_requests, 
          {
            rank = data.rank,
            key = key  
          })
  end
end

local function SPH_branch_receive_handler(this, data)
  for _, cell in pairs(data.branches) do
    Octree.reconstruct(cell)
    this.oct:insert_foreign_cell(cell)
  end
  this.received = this.received+1
  --print ("Rank", this.rank, "has received", this.received,"groups of branches out of", this.world)
  if this.received == this.world then
    --print("Rank "..this.rank.." has received all branches")
    SPH_compute_handler(this)
  end
end


local function SPH_cells_receive_handler(this, data)
  local first = true
  local parent = nil
  for _, cell in pairs(data) do
    Octree.reconstruct(cell)
    this.oct:insert_foreign_cell(cell)
    if first then
      parent = cell.parent
      first = false
    end
  end
  parent.status = "local"
  for _, resume in pairs(this.resume_list[parent]) do
    resume.cell = parent
    table.insert(this.resumable,resume)
  end
  this.resume_list[parent] = nil
  this.pending = this.pending-1
  if this.finished then
    SPH_compute_handler(this)
  end
end

local function main()
  --- 
  -- Simian stuff
  ---
  print("Running simulation on ", n_procs,"simulated nodes")
  Simian:init(simName, startTime, endTime, minDelay, useMPI)
  local trin = clusters.HalfTrinity(Simian,n_procs)
  Simian:attachService(
    nodes.KNLNode, "SPH_init_handler", SPH_init_handler)
  Simian:attachService(
    nodes.KNLNode, "SPH_branch_receive_handler", SPH_branch_receive_handler)
  Simian:attachService(
    nodes.KNLNode, "SPH_compute_handler", SPH_compute_handler)
  Simian:attachService(
    nodes.KNLNode, "SPH_request_receive_handler", SPH_request_receive_handler)
  Simian:attachService(
    nodes.KNLNode, "SPH_cells_receive_handler", SPH_cells_receive_handler)

  local minVal = {}
  minVal.x = 0
  minVal.y = 0
  minVal.z = 0
  
  ---
  --Define how to handle custom data at tree creation
  ---
  
  -- If passed when creating the tree, this function is called whenever a new 
  -- cell is added
  local function init_cell(cell, content)
    cell.owner = content.owner
  end
  
  -- If passed when creating the tree, this function is called whenever a new 
  -- point is added to a cell or one of its daughters
  local function update_cell(cell, content)
    
  end
  
  local function count_interactions(cell, data)
    local kg=false
    if data.bbox:intersects(cell.bbox) then
      kg=true
      if cell.is_leaf() then
        local entry = cell.content
        while entry do
          local content = entry.value
          data.local_interactions = data.local_interactions+1
          entry = entry.next
        end
      end
    else
      data.distant_interactions = data.distant_interactions+1
    end
    return kg, data
  end
  
  math.randomseed(0)
  
  ---
  --Initialize data points
  ---
  local f = assert(io.open("./input.dat","r"))
  local bodies = {}
  local min, max, random = math.min, math.max, math.random
  local minVal = {x=math.huge,y=math.huge,z=math.huge}
  local sizes = {x=0,y=0,z=0}
  for i=1,nbodies  do
    local point = {}
    point.x, point.y, point.z = f:read("*number", "*number", "*number")
    minVal.x = min(minVal.x, point.x)
    minVal.y = min(minVal.y, point.y)
    minVal.z = min(minVal.z, point.z)
    sizes.x = max(sizes.x,point.x)
    sizes.y = max(sizes.y,point.y)
    sizes.z = max(sizes.z,point.z)
    
    table.insert(bodies,i,Content.create(point, minVal))
  end
  f:close()
  sizes.x = sizes.x - minVal.x
  sizes.y = sizes.y - minVal.y
  sizes.z = sizes.z - minVal.z
  local oct = Octree.create(sizes, init_cell, update_cell)
  for i=1,nbodies do
    bodies[i]:update(minVal)
  end
  table.sort(bodies, Content.cmp)
  --print ("Values have been sorted")
  ---
  --This will update the keys for each point with minVal
  ---
  for i, content in ipairs(bodies) do
    oct:insert(content)
  end
  local total_weight, local_weight, distant_weight = 0,0,0
  for j, content in ipairs(bodies) do
    local data = {}
    data.point=content.point
    data.bbox = BBox.create_radius(data.point, radius)
    data.local_interactions=0
    data.distant_interactions=0
    data = oct:traverse(count_interactions, data)
    content.weight = data.local_interactions+data.distant_interactions
    local_weight = local_weight+data.local_interactions
    distant_weight = distant_weight+data.distant_interactions
    total_weight = total_weight+content.weight
  end
  --print ("Weights have been computed")
  oct = nil
  local workload = total_weight/n_procs
  local load = 0
  local prev = 0
  local rank =1
  local data = 
  {
    bodies={}, 
    rank=rank,
    world=n_procs, 
    sizes = sizes, 
    prev_key=nil, 
    next_key=nil, 
    foreign_bodies = nil
  }
  for i =1,nbodies do
    load = load + bodies[i].weight
    data.bodies[i-prev] = bodies[i]
    data.bodies[i-prev].owner = rank
    if load > workload then
      if i <nbodies then
        data.next_key = bodies[i+1].key
      end
      --print ("Rank ", rank, "is in charge of points", prev, "to", i)
      Simian:schedService(0, "SPH_init_handler", data, "Node", rank)
      prev = i
      rank = rank+1
      data = 
      {
        bodies={}, 
        rank = rank, 
        world=n_procs, 
        sizes = sizes, 
        prev_key = bodies[i].key, 
        next_key = nil, 
        foreign_bodies = nil
      }
      load = 0
    end
  end
  if prev > 0 then
    data.prev_key = bodies[prev].key
  end
  --print ("Rank ", data.rank, "is in charge of points", prev, "to", nbodies)
  Simian:schedService(0, "SPH_init_handler", data, "Node", rank)
  data = nil
  bodies = nil
  ---
  -- Run simulation
  ---
  Simian:run()
  print("Simulation finished at time ", Simian.now)
  Simian:exit()
end
main()
