//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mesh.h
 * \author Alex Long
 * \date   July 18 2014
 * \brief  Object that holds mesh and manages decomposition and communication
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef mesh_h_
#define mesh_h_

#include <algorithm>
#include <iterator>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "buffer.h"
#include "cell.h"
#include "constants.h"
#include "mpi_types.h"
#include "info.h"
#include "input.h"
#include "imc_state.h"


//==============================================================================
/*!
 * \class Mesh
 * \brief Manages data access, decomposition and parallel communication for mesh
 *
 * Using an Input class, make the mesh with the correct material properties
 * for each region. The mesh numbering and mapping between global IDs and local
 * indices are all determined with the aid of Parmetis in the decompose_mesh
 * function. The mesh class also manages two-sided messaging in the mesh-
 * passing method.
 *
 */
//==============================================================================
class Mesh {

  public:

  //! constructor
  Mesh(Input* input, MPI_Types *mpi_types, const Info& mpi_info)
  : ngx(input->get_global_n_x_cells()),
    ngy(input->get_global_n_y_cells()),
    ngz(input->get_global_n_z_cells()),
    rank(mpi_info.get_rank()),
    n_rank(mpi_info.get_n_rank()),
    n_off_rank(n_rank-1),
    silo_x(input->get_silo_x_ptr()),
    silo_y(input->get_silo_y_ptr()),
    silo_z(input->get_silo_z_ptr())
  {
    using std::vector;
    using Constants::bc_type;
    using Constants::X_POS;  using Constants::Y_POS; using Constants::Z_POS;
    using Constants::X_NEG;  using Constants::Y_NEG; using Constants::Z_NEG;
    using Constants::ELEMENT;

    max_map_size = input->get_map_size();
    double dx,dy,dz;

    // make off processor map
    for (uint32_t i=0; i<n_off_rank; i++) {
      int r_index = i + int(i>=rank);
      proc_map[i] = r_index;
    }

    regions = input->get_regions();
    // map region IDs to index in the region
    for (uint32_t i =0; i<regions.size(); i++) {
      region_ID_to_index[regions[i].get_ID()] = i;
    }

    vector<bc_type> bc(6);
    bc[X_POS] = input->get_bc(X_POS); bc[X_NEG] = input->get_bc(X_NEG);
    bc[Y_POS] = input->get_bc(Y_POS); bc[Y_NEG] = input->get_bc(Y_NEG);
    bc[Z_POS] = input->get_bc(Z_POS); bc[Z_NEG] = input->get_bc(Z_NEG);

    //initialize number of RMA requests as zero
    off_rank_reads=0;

    uint32_t global_count =0; //global cell count

    //this rank's cells
    n_global = ngx*ngy*ngz;
    uint32_t cell_id_begin = floor(rank*n_global/double(n_rank));
    uint32_t cell_id_end = floor((rank+1)*n_global/double(n_rank));

    uint32_t on_rank_count =0;

    uint32_t n_x_div = input->get_n_x_divisions();
    uint32_t n_y_div = input->get_n_y_divisions();
    uint32_t n_z_div = input->get_n_z_divisions();

    Region region;
    uint32_t region_index, nx, ny, nz;
    double x_start, y_start, z_start;
    double x_cell_end;
    double y_cell_end;
    double z_cell_end;

    uint32_t g_i=0; //! Global x index
    uint32_t g_j=0; //! Global y index
    uint32_t g_k=0; //! Global z index

    for (uint32_t iz_div=0; iz_div<n_z_div; iz_div++) {
      dz = input->get_dz(iz_div);
      nz = input->get_z_division_cells(iz_div);
      z_start = input->get_z_start(iz_div);
      for (uint32_t k=0; k<nz; k++) {
        g_j = 0;
        for (uint32_t iy_div=0; iy_div<n_y_div; iy_div++) {
          dy = input->get_dy(iy_div);
          ny = input->get_y_division_cells(iy_div);
          y_start = input->get_y_start(iy_div);
          for (uint32_t j=0; j<ny; j++) {
            g_i = 0;
            for (uint32_t ix_div=0; ix_div<n_x_div; ix_div++) {
              dx = input->get_dx(ix_div);
              nx = input->get_x_division_cells(ix_div);
              x_start = input->get_x_start(ix_div);
              for (uint32_t i=0; i<nx; i++) {
                if (global_count >= cell_id_begin && global_count < cell_id_end) {
                  //find the region for this cell
                  region_index = input->get_region_index(ix_div, iy_div, iz_div);
                  region = regions[region_index];
                  Cell e;

                  // set ending coordinates explicity to match the start of
                  // the next division to avoid weird roundoff errors
                  if (i == nx-1 && ix_div != n_x_div-1)
                    x_cell_end = input->get_x_start(ix_div+1);
                  else x_cell_end = x_start+(i+1)*dx;

                  if (j == ny-1 && iy_div != n_y_div-1)
                    y_cell_end = input->get_y_start(iy_div+1);
                  else y_cell_end = y_start+(j+1)*dy;

                  if (k == nz-1 && iz_div != n_z_div-1)
                    z_cell_end = input->get_z_start(iz_div+1);
                  else z_cell_end = z_start+(k+1)*dz;

                  e.set_coor(x_start+i*dx, x_cell_end, y_start+j*dy, y_cell_end,
                    z_start+k*dz, z_cell_end);
                  e.set_ID(global_count);
                  e.set_region_ID(region.get_ID());

                  // set cell physical properties using region
                  e.set_cV(region.get_cV());
                  e.set_T_e(region.get_T_e());
                  e.set_T_r(region.get_T_r());
                  e.set_T_s(region.get_T_s());
                  e.set_rho(region.get_rho());

                  // set the global index for SILO plotting--this will always
                  // be the current global count (g_i +g_j*ngx + g_k*(ngy_*ngz))
                  e.set_silo_index(global_count);

                  // set neighbors in x direction
                  if (g_i<(ngx-1)) {
                    e.set_neighbor( X_POS, global_count+1);
                    e.set_bc(X_POS, ELEMENT);
                  }
                  else {
                    e.set_neighbor( X_POS, global_count);
                    e.set_bc(X_POS, bc[X_POS]);
                  }
                  if (g_i>0) {
                    e.set_neighbor( X_NEG, global_count-1);
                    e.set_bc(X_NEG, ELEMENT);
                  }
                  else {
                    e.set_neighbor( X_NEG, global_count);
                    e.set_bc(X_NEG, bc[X_NEG]);
                  }

                  // set neighbors in y direction
                  if (g_j<(ngy-1)) {
                    e.set_neighbor(Y_POS, global_count+ngx);
                    e.set_bc(Y_POS, ELEMENT);
                  }
                  else {
                    e.set_neighbor(Y_POS, global_count);
                    e.set_bc(Y_POS, bc[Y_POS]);
                  }
                  if (g_j>0) {
                    e.set_neighbor(Y_NEG, global_count-ngx);
                    e.set_bc(Y_NEG, ELEMENT);
                  }
                  else {
                    e.set_neighbor(Y_NEG, global_count);
                    e.set_bc(Y_NEG, bc[Y_NEG]);
                  }

                  // set neighbors in z direction
                  if (g_k<(ngz-1)) {
                    e.set_neighbor(Z_POS, global_count+ngx*ngy);
                    e.set_bc(Z_POS, ELEMENT);
                  }
                  else {
                    e.set_neighbor(Z_POS, global_count);
                    e.set_bc(Z_POS, bc[Z_POS]);
                  }
                  if (g_k>0) {
                    e.set_neighbor(Z_NEG, global_count-ngx*ngy);
                    e.set_bc(Z_NEG, ELEMENT);
                  }
                  else {
                    e.set_neighbor(Z_NEG, global_count);
                    e.set_bc(Z_NEG, bc[Z_NEG]);
                  }

                  // add cell to mesh
                  cell_list.push_back(e);
                  //increment on rank count
                  on_rank_count++;
                } // end if on processor check
                global_count++;
                g_i++;
              } // end i loop
            } // end x division loop
            g_j++;
          } // end j loop
        } // end y division loop
        g_k++;
      } // end k loop
    } // end z division loop
    n_cell = on_rank_count;

    total_photon_E = 0.0;

    mpi_cell_size = mpi_types->get_cell_size();

    mpi_window_set = false;
  }

  // destructor, free buffers and delete MPI allocated cell
  ~Mesh() {
    // free MPI window (also frees associated memory)
    if (mpi_window_set) MPI_Win_free(&mesh_window);
  }

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//
  uint32_t get_max_grip_size(void) const {return max_grip_size;}
  uint32_t get_n_local_cells(void) const {return n_cell;}
  uint32_t get_my_rank(void) const {return  rank;}
  uint32_t get_offset(void) const {return on_rank_start;}
  uint32_t get_global_num_cells(void) const {return n_global;}
  std::unordered_map<uint32_t, uint32_t> get_proc_adjacency_list(void) const {
    return adjacent_procs;
  }
  double get_total_photon_E(void) const {return total_photon_E;}

  std::vector<uint32_t> get_off_rank_bounds(void) {return off_rank_bounds;}

  void pre_decomp_print(void) const {
    for (uint32_t i= 0; i<n_cell; i++)
      cell_list[i].print();
  }

  uint32_t get_grip_ID_from_cell_ID(uint32_t cell_ID) const {
    uint32_t local_ID = cell_ID - on_rank_start;
    return cells[local_ID].get_grip_ID();
  }

  void post_decomp_print(void) const {
    for (uint32_t i= 0; i<n_cell; i++)
      cells[i].print();
  }

  //! returns a mapping of old cell indices to new simple global indices
  std::unordered_map<uint32_t, uint32_t> get_new_global_index_map(void) const {
    std::unordered_map<uint32_t, uint32_t> local_map;
    uint32_t g_ID;
    for (uint32_t i=0; i<n_cell; i++) {
      g_ID = cell_list[i].get_ID();
      local_map[g_ID] = i+on_rank_start;
    }
    return local_map;
  }

  //! returns a mapping of new global cell indices to the global cell index of
  // a grip, this must be used after grips with cell IDs have been set
  std::unordered_map<uint32_t, uint32_t> get_grip_map(void) const {
    std::unordered_map<uint32_t, uint32_t> local_grip_map;
    uint32_t new_g_ID;
    for (uint32_t i=0; i<n_cell; i++) {
      new_g_ID = i+on_rank_start;
      local_grip_map[new_g_ID] = cell_list[i].get_grip_ID();
    }
    return local_grip_map;
  }

  //! Gets cell from vector list of cells before it's deleted
  Cell get_pre_window_allocation_cell(const uint32_t& local_ID) const
  {
    return cell_list[local_ID];
  }

  Cell get_cell(const uint32_t& local_ID) const {
    return cells[local_ID];
  }

  const Cell* get_cell_ptr(const uint32_t& local_ID) const {
    return &cells[local_ID];
  }

  const Cell * get_const_cells_ptr(void) const {
    return cells;
  }

  uint32_t get_off_rank_id(const uint32_t& index) const {
    //find rank of index
    bool found = false;
    uint32_t min_i = 0;
    uint32_t max_i = off_rank_bounds.size()-1;
    uint32_t s_i; //search index
    while(!found) {
      s_i =(max_i + min_i)/2;
      if (s_i == max_i || s_i == min_i) found = true;
      else if (index >= off_rank_bounds[s_i]) min_i = s_i;
      else max_i = s_i;
    }
    return s_i;
  }

  uint32_t get_rank(const uint32_t& index) const {
    uint32_t r_rank;
    if (on_processor(index)) r_rank = rank;
    else  r_rank = get_off_rank_id(index);
    return r_rank;
  }

  uint32_t get_local_ID(const uint32_t& index) const {
    return index-on_rank_start;
  }

  uint32_t get_global_ID(const uint32_t& local_index) const {
    return on_rank_start+local_index;
  }

  Cell get_on_rank_cell(const uint32_t& index) {
    // this can only be called with valid on rank indexes
    if (on_processor(index))
      return cells[index-on_rank_start];
    else
      return stored_cells[index];
  }

  bool on_processor(const uint32_t& index) const {
    return  (index>=on_rank_start) && (index<=on_rank_end) ;
  }

  void print_map(void) {
    for ( std::unordered_map<uint32_t,Cell>::iterator map_i =
      stored_cells.begin();
      map_i!=stored_cells.end(); map_i++)
      (map_i->second).print();
  }

  bool mesh_available(const uint32_t& index) const {
    if (on_processor(index)) return true;
    else if (stored_cells.find(index) != stored_cells.end())
      return true;
    else
      return false;
  }

  std::vector<double> get_census_E(void) const {return m_census_E;}
  std::vector<double> get_emission_E(void) const {return m_emission_E;}
  std::vector<double> get_source_E(void) const {return m_source_E;}

  uint32_t get_global_n_x_faces(void) const {return ngx+1;}
  uint32_t get_global_n_y_faces(void) const {return ngy+1;}
  uint32_t get_global_n_z_faces(void) const {return ngz+1;}

  float * get_silo_x(void) const {return silo_x;}
  float * get_silo_y(void) const {return silo_y;}
  float * get_silo_z(void) const {return silo_z;}

  //--------------------------------------------------------------------------//
  // non-const functions                                                      //
  //--------------------------------------------------------------------------//

  //! set the grip ID to be the global index of the cell at the center of the
  // grip
  void set_grip_ID_using_cell_index(void)
  {
    using std::max;
    using std::unordered_map;

    uint32_t new_grip_ID, grip_end_index;
    // start by looking at the first grip
    uint32_t current_grip_ID = cell_list.front().get_grip_ID();
    uint32_t grip_start_index = 0;
    uint32_t grip_count = 0;
    // start with max_grip_size at zero
    max_grip_size = 0;

    unordered_map<uint32_t, uint32_t> start_index_to_count;

    // map the starting index of cells with the same grip to the number in
    // that grip
    for (uint32_t i=0; i<n_cell; i++) {
      Cell & cell = cell_list[i];
      if (cell.get_grip_ID() != current_grip_ID) {
        grip_start_index = i;
        current_grip_ID = cell.get_grip_ID();
      }

      // if in grip, incerement count...
      if (start_index_to_count.find(grip_start_index) !=
        start_index_to_count.end())
      {
        start_index_to_count[grip_start_index]++;
      }
      // otherwise initialize count to 1
      else {
        start_index_to_count[grip_start_index] =1;
      }
    }

    // set grip ID using the map of start indices and number of cells
    for (auto map_i = start_index_to_count.begin();
      map_i!=start_index_to_count.end(); map_i++)
    {
      grip_start_index = map_i->first;
      grip_count = map_i->second;
      // set the new grip ID to be the index of the cell at the center of
      // this grip for odd grip sizes and one above center for even grip
      // sizes (for convenience in parallel comm)
      new_grip_ID = on_rank_start + grip_start_index + grip_count/2;
      // update max grip size
      max_grip_size = max(max_grip_size, grip_count);
      // loop over cells in grip and set new ID
      grip_end_index = grip_start_index+grip_count;
      for (uint32_t j=grip_start_index; j<grip_end_index;j++)
        cell_list[j].set_grip_ID(new_grip_ID);
    }
  }

  //! set the global ID of the start and end cell on this rank
  void set_global_bound(uint32_t _on_rank_start, uint32_t _on_rank_end) {
    on_rank_start = _on_rank_start;
    on_rank_end = _on_rank_end;
  }

  //! set the global ID starting indices for all ranks
  void set_off_rank_bounds(std::vector<uint32_t> _off_rank_bounds) {
    off_rank_bounds=_off_rank_bounds;
  }

  //! Gets cell reference from vector list of cells before it's deleted
  Cell& get_pre_window_allocation_cell_ref(const uint32_t& local_ID)
  {
    return cell_list[local_ID];
  }

  //! Calculate new physical properties and emission energy for each cell on
  // the mesh
  void calculate_photon_energy(IMC_State* imc_s) {
    using Constants::c;
    using Constants::a;
    total_photon_E = 0.0;
    double dt = imc_s->get_dt();
    double op_a, op_s, f, cV, rho;
    double vol;
    double T, Tr, Ts;
    uint32_t step = imc_s->get_step();
    double tot_census_E = 0.0;
    double tot_emission_E = 0.0;
    double tot_source_E = 0.0;
    double pre_mat_E = 0.0;

    uint32_t region_ID;
    Region region;
    for (uint32_t i=0; i<n_cell;++i) {
      Cell& e = cells[i];
      vol = e.get_volume();
      cV = e.get_cV();
      T = e.get_T_e();
      Tr = e.get_T_r();
      Ts = e.get_T_s();
      rho = e.get_rho();

      region_ID = e.get_region_ID();
      region =  regions[region_ID_to_index[region_ID]];

      op_a = region.get_absorption_opacity(T);
      op_s = region.get_scattering_opacity();
      f =1.0/(1.0 + dt*op_a*c*(4.0*a*pow(T,3)/(cV*rho)));
      e.set_op_a(op_a);
      e.set_op_s(op_s);
      e.set_f(f);

      m_emission_E[i] = dt*vol*f*op_a*a*c*pow(T,4);
      if (step > 1) m_census_E[i] = 0.0;
      else m_census_E[i] =vol*a*pow(Tr,4);
      m_source_E[i] = dt*op_a*a*c*pow(Ts,4);

      pre_mat_E+=T*cV*vol*rho;
      tot_emission_E+=m_emission_E[i];
      tot_census_E  +=m_census_E[i];
      tot_source_E  +=m_source_E[i];
      total_photon_E += m_emission_E[i] + m_census_E[i] + m_source_E[i];
    }

    // set energy for conservation checks
    imc_s->set_pre_mat_E(pre_mat_E);
    imc_s->set_emission_E(tot_emission_E);
    imc_s->set_source_E(tot_source_E);
    if(imc_s->get_step() == 1) imc_s->set_pre_census_E(tot_census_E);
  }


  //! Correctly set the connectivity of cells given a new mesh numbering
  // after mesh decomposition. Also, determine adjacent ranks.
  void update_off_rank_connectivity(
    std::unordered_map<uint32_t, uint32_t> off_map,
    std::unordered_map<uint32_t, uint32_t> off_grip_map,
    std::vector< std::vector<bool> >& remap_flag) {

    using Constants::PROCESSOR;
    using Constants::dir_type;
    using std::unordered_map;
    using std::set;

    uint32_t next_index;
    unordered_map<uint32_t, uint32_t>::iterator end = off_map.end();
    uint32_t new_index, new_grip_index;
    // check to see if neighbors are on or off processor
    for (uint32_t i=0; i<n_cell; i++) {
      Cell& cell = cell_list[i];
      for (uint32_t d=0; d<6; d++) {
        next_index = cell.get_next_cell(d);
        unordered_map<uint32_t, uint32_t>::iterator map_i =
          off_map.find(next_index);
        if (map_i != end && remap_flag[i][d] ==false ) {
          // update index and bc type, this will always be an off processor cell
          // so if an index is updated it will always be at a processor bound
          remap_flag[i][d] = true;
          new_index = map_i->second;
          // new_grip_map maps new global indices to new grip IDs
          new_grip_index = off_grip_map[new_index];
          cell.set_neighbor( dir_type(d) , new_index );
          cell.set_grip_neighbor(dir_type(d), new_grip_index);
          cell.set_bc(dir_type(d), PROCESSOR);

          // determine adjacent ranks for minimizing communication
          uint32_t off_rank = get_off_rank_id(new_index);
          if (adjacent_procs.find(off_rank) == adjacent_procs.end()) {
            uint32_t rank_count = adjacent_procs.size();
            adjacent_procs[off_rank] = rank_count;
          } // if adjacent_proc.find(off_rank)
        }
      }
    }
  }

  //! Renumber the local cell IDs and connectivity of local cells after
  // decomposition using simple global  numbering
  void renumber_local_cell_indices(
    std::unordered_map<uint32_t, uint32_t> local_map,
    std::unordered_map<uint32_t, uint32_t> local_grip_map)
  {

    using Constants::PROCESSOR;
    using Constants::bc_type;
    using Constants::dir_type;
    using std::unordered_map;

    uint32_t next_index;
    unordered_map<uint32_t, uint32_t>::iterator end = local_map.end();
    uint32_t new_index, new_grip_index;
    bc_type current_bc;
    // renumber global cell and check to see if neighbors are on or off
    // processor
    for (uint32_t i=0; i<n_cell; i++) {
      Cell& cell = cell_list[i];
      cell.set_ID(i+on_rank_start);
      for (uint32_t d=0; d<6; d++) {
        current_bc = cell.get_bc(bc_type(d));
        next_index = cell.get_next_cell(d);
        unordered_map<uint32_t, uint32_t>::iterator map_i =
          local_map.find(next_index);
        //if this index is not a processor boundary, update it
        if (local_map.find(next_index) != end && current_bc != PROCESSOR) {
          new_index = map_i->second;
          // new_grip_map maps new global indices to new grip IDs
          new_grip_index = local_grip_map[new_index];
          cell.set_neighbor( dir_type(d) , new_index );
          cell.set_grip_neighbor( dir_type(d) , new_grip_index);
        }
      } // end direction
    } // end cell
  }

  // Remove old mesh cells after decomposition and communication of new cells
  void set_post_decomposition_mesh_cells(void) {
    using std::vector;
    vector<Cell> new_mesh;
    for (uint32_t i =0; i< cell_list.size(); i++) {
      bool delete_flag = false;
      for (vector<uint32_t>::iterator rmv_itr= remove_cell_list.begin();
        rmv_itr != remove_cell_list.end();
        rmv_itr++)
      {
        if (*rmv_itr == i)  delete_flag = true;
      }
      if (delete_flag == false) new_mesh.push_back(cell_list[i]);
    }

    for (uint32_t i =0; i< new_cell_list.size(); i++)
      new_mesh.push_back(new_cell_list[i]);
    cell_list = new_mesh;
    n_cell = cell_list.size();
    new_cell_list.clear();
    remove_cell_list.clear();

    // sort based on global cell ID
    sort(cell_list.begin(), cell_list.end());

    //use the final number of cells to size vectors
    m_census_E = vector<double>(n_cell, 0.0);
    m_emission_E = vector<double>(n_cell, 0.0);
    m_source_E = vector<double>(n_cell, 0.0);
  }

  //! sort pre-winodw allocation cell vector based on the grip ID of each cell
  void sort_cells_by_grip_ID(void) {
    using std::sort;
    // sort based on global cell ID
    sort(cell_list.begin(), cell_list.end(), Cell::sort_grip_ID);
  }

  //! Use MPI allocation routines, copy in cell data and make the MPI window
  // object
  void make_MPI_window(void) {
    //make the MPI window with the sorted cell list
    MPI_Aint n_bytes(n_cell*mpi_cell_size);
    //MPI_Alloc_mem(n_bytes, MPI_INFO_NULL, &cells);
    MPI_Win_allocate(n_bytes, mpi_cell_size, MPI_INFO_NULL,
      MPI_COMM_WORLD, &cells, &mesh_window);
    //copy the cells list data into the cells array
    memcpy(cells,&cell_list[0], n_bytes);

    mpi_window_set = true;
    cell_list.clear();
  }

  //! Use the absorbed energy and update the material temperature of each
  // cell on the mesh. Set diagnostic and conservation values.
  void update_temperature(std::vector<double>& abs_E, IMC_State* imc_s) {
    //abs E is a global vector
    double total_abs_E = 0.0;
    double total_post_mat_E = 0.0;
    double vol,cV,rho,T, T_new;
    uint32_t region_ID;
    Region region;
    for (uint32_t i=0; i<n_cell;++i) {
      region_ID = cells[i].get_region_ID();
      region = regions[region_ID_to_index[ region_ID]];
      cV = region.get_cV();
      rho = region.get_rho();
      Cell& e = cells[i];
      vol = e.get_volume();
      T = e.get_T_e();
      T_new = T + (abs_E[i+on_rank_start] - m_emission_E[i])/(cV*vol*rho);
      e.set_T_e(T_new);
      total_abs_E+=abs_E[i+on_rank_start];
      total_post_mat_E+= T_new*cV*vol*rho;
    }
    //zero out absorption tallies for all cells (global)
    for (uint32_t i=0; i<abs_E.size();++i) {
      abs_E[i] = 0.0;
    }
    imc_s->set_absorbed_E(total_abs_E);
    imc_s->set_post_mat_E(total_post_mat_E);
    imc_s->set_step_cells_requested(off_rank_reads);
    off_rank_reads = 0;
  }

  //! Return reference to MPI window (used by rma_mesh_manager class)
  MPI_Win& get_mesh_window_ref(void) {return mesh_window;}

  //! Add off-rank mesh data to the temporary mesh storage and manage the
  // temporary mesh
  void add_non_local_mesh_cells(std::vector<Cell> new_recv_cells) {
    using std::advance;
    using std::unordered_map;

    // if new_recv_cells is bigger than maximum map size truncate it
    if (new_recv_cells.size() > max_map_size) {
      new_recv_cells.erase(new_recv_cells.begin() + max_map_size,
        new_recv_cells.end());
    }

    // remove a chunk of working mesh data if the new cells won't fit
    uint32_t stored_cell_size = stored_cells.size();
    if (stored_cell_size + new_recv_cells.size() > max_map_size) {
      // remove enough cells so all new cells will fit
      unordered_map<uint32_t, Cell>::iterator i_start = stored_cells.begin();
      advance(i_start, max_map_size - new_recv_cells.size());
      stored_cells.erase(i_start, stored_cells.end());
    }

    // add received cells to the stored_cells map
    for (uint32_t i=0; i<new_recv_cells.size();i++) {
      uint32_t index = new_recv_cells[i].get_ID();
      stored_cells[index] = new_recv_cells[i];
    }
  }

  //! Remove the temporary off-rank mesh data after the end of a timestep
  // (the properties will be updated so it can't be reused)
  void purge_working_mesh(void) {
    stored_cells.clear();
  }

  //! Set maximum grip size
  void set_max_grip_size(const uint32_t& new_max_grip_size) {
    max_grip_size = new_max_grip_size;
  }

  //! Add mesh cell (used during decomposition, not parallel communication)
  void add_mesh_cell(Cell new_cell) {new_cell_list.push_back(new_cell);}

  //! Remove mesh cell (used during decomposition, not parallel communication)
  void remove_cell(uint32_t index) {remove_cell_list.push_back(index);}

  //! Get census energy vector needed to source particles
  std::vector<double>& get_census_E_ref(void) {return m_census_E;}

  //! Get emission energy vector needed to source particles
  std::vector<double>& get_emission_E_ref(void) {return m_emission_E;}

  //! Get external source energy vector needed to source particles
  std::vector<double>& get_source_E_ref(void) {return m_source_E;}

  //--------------------------------------------------------------------------//
  // member variables
  //--------------------------------------------------------------------------//
  private:

  uint32_t ngx; //! Number of global x sizes
  uint32_t ngy; //! Number of global y sizes
  uint32_t ngz; //! Number of global z sizes
  uint32_t rank; //! MPI rank of this mesh
  uint32_t n_rank; //! Number of global ranks
  uint32_t n_off_rank; //! Number of other ranks
  float *silo_x; //! Global array of x face locations for SILO
  float *silo_y; //! Global array of y face locations for SILO
  float *silo_z; //! Global array of z face locations for SILO

  uint32_t n_cell; //! Number of local cells
  uint32_t n_global; //! Nuber of global cells

  uint32_t on_rank_start; //! Start of global index on rank
  uint32_t on_rank_end; //! End of global index on rank

  std::vector<double> m_census_E; //! Census energy vector
  std::vector<double> m_emission_E; //! Emission energy vector
  std::vector<double> m_source_E; //! Source energy vector

  Cell *cells; //! Cell data allocated with MPI_Alloc
  std::vector<Cell> cell_list; //! On processor cells
  std::vector<Cell> new_cell_list; //! New received cells
  std::vector<uint32_t> remove_cell_list; //! Cells to be removed
  std::vector<uint32_t> off_rank_bounds; //! Ending value of global ID for each rank

  std::unordered_map<uint32_t, uint32_t> adjacent_procs; //! List of adjacent processors

  std::vector<Region> regions; //! Vector of regions in the problem
  std::unordered_map<uint32_t, uint32_t> region_ID_to_index; //! Maps region ID to index

  double total_photon_E; //! Total photon energy on the mesh

  uint32_t max_map_size; //! Maximum size of map object
  uint32_t off_rank_reads; //! Number of off rank reads
  int32_t mpi_cell_size; //! Size of custom MPI_Cell type

  MPI_Win mesh_window; //! Handle to shared memory window of cell data

  //! Cells that have been accessed off rank
  std::unordered_map<uint32_t, Cell> stored_cells;

  std::unordered_map<int,int> proc_map; //! Maps number of off-rank processor to global rank

  uint32_t max_grip_size; //! Size of largest grip on this rank

  bool mpi_window_set; //! Flag indicating if MPI_Window was created

};

#endif // mesh_h_
//---------------------------------------------------------------------------//
// end of mesh.h
//---------------------------------------------------------------------------//
