//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   decompose_mesh.h
 * \author Alex Long
 * \date   June 17 2015
 * \brief  Functions to decompose mesh with ParMetis
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef decompose_mesh_h_
#define decompose_mesh_h_

#include <mpi.h>
#include <parmetis.h>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <vector>

#include "mesh.h"
#include "mpi_types.h"
#include "buffer.h"


//! Print the mesh information for each rank, one at a time
void print_MPI_out(Mesh *mesh, uint32_t rank, uint32_t size) {
  using std::cout;
  cout.flush();
  MPI_Barrier(MPI_COMM_WORLD);

  for (uint32_t p_rank = 0; p_rank<size; p_rank++) {
    if (rank == p_rank) {
      mesh->post_decomp_print();
      cout.flush();
    }
    usleep(100);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100);
  }
}


//! Print the remapping information for each rank, one at a time
void print_MPI_maps(Mesh *mesh, uint32_t rank, uint32_t size) {
  using std::cout;
  cout.flush();
  MPI_Barrier(MPI_COMM_WORLD);

  for (uint32_t p_rank = 0; p_rank<size; p_rank++) {
    if (rank == p_rank) {
      mesh->print_map();
      cout.flush();
    }
    usleep(100);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100);
  }
}

//! Generate new partitioning with ParMetis, send and receive cells, renumber
// mesh and communicate renumbering
void decompose_mesh(Mesh* mesh, MPI_Types* mpi_types, const Info& mpi_info,
  const uint32_t& grip_size)
{

  using Constants::X_POS;  using Constants::Y_POS; using Constants::Z_POS;
  using Constants::X_NEG;  using Constants::Y_NEG; using Constants::Z_NEG;
  using std::vector;
  using std::unordered_map;
  using std::unordered_set;
  using std::partial_sum;

  int rank = mpi_info.get_rank();
  int nrank = mpi_info.get_n_rank();
  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  // make off processor map
  uint32_t n_off_rank = nrank -1; // implicit conversion from int to uint32_t
  unordered_map<int,int> proc_map;
  for (uint32_t i=0; i<n_off_rank; i++) {
    int r_index = i + int(int(i)>=rank);
    proc_map[i] = r_index;
  }

  MPI_Datatype MPI_Cell = mpi_types->get_cell_type();

  // begin PARMETIS routines

  // get the number of cells on each processor
  // vtxdist has number of vertices on each rank, same for all ranks
  vector<int> start_ncells(nrank, 0);
  vector<int> vtxdist(nrank, 0);

  uint32_t ncell_on_rank = mesh->get_n_local_cells();
  start_ncells[rank] = ncell_on_rank;

  MPI_Allreduce(MPI_IN_PLACE, &start_ncells[0], nrank, MPI_INT, MPI_SUM,
    MPI_COMM_WORLD);
  partial_sum(start_ncells.begin(), start_ncells.end(), vtxdist.begin());
  vtxdist.insert(vtxdist.begin(), 0);

  // build adjacency list needed for ParMetis call for each rank also get
  // coordinates of cell centers for geometry based partitioning
  float *xyz = new float[ncell_on_rank*3];
  vector<int> xadj;
  vector<int> adjncy;
  int adjncy_ctr = 0;
  Cell cell;
  uint32_t g_ID; //! Global ID
  for (uint32_t i=0; i<ncell_on_rank;++i) {
    cell = mesh->get_pre_window_allocation_cell(i);
    g_ID = cell.get_ID();
    cell.get_center(&xyz[i*3]);
    uint32_t xm_neighbor =cell.get_next_cell(X_NEG);
    uint32_t xp_neighbor =cell.get_next_cell(X_POS);
    uint32_t ym_neighbor =cell.get_next_cell(Y_NEG);
    uint32_t yp_neighbor =cell.get_next_cell(Y_POS);
    uint32_t zm_neighbor =cell.get_next_cell(Z_NEG);
    uint32_t zp_neighbor =cell.get_next_cell(Z_POS);

    xadj.push_back(adjncy_ctr); //starting index in xadj for this cell's nodes
    if (xm_neighbor != g_ID) {adjncy.push_back(xm_neighbor); adjncy_ctr++;}
    if (xp_neighbor != g_ID) {adjncy.push_back(xp_neighbor); adjncy_ctr++;}
    if (ym_neighbor != g_ID) {adjncy.push_back(ym_neighbor); adjncy_ctr++;}
    if (yp_neighbor != g_ID) {adjncy.push_back(yp_neighbor); adjncy_ctr++;}
    if (zm_neighbor != g_ID) {adjncy.push_back(zm_neighbor); adjncy_ctr++;}
    if (zp_neighbor != g_ID) {adjncy.push_back(zp_neighbor); adjncy_ctr++;}
  }
  xadj.push_back(adjncy_ctr);

  int ndim = 3;
  int wgtflag = 0; // no weights for cells
  int numflag = 0; // C-style numbering
  int ncon = 1;
  int nparts = nrank; // sub-domains = nrank

  float *tpwgts = new float[nparts];
  for (int i=0; i<nparts; i++) tpwgts[i]=1.0/float(nparts);

  float *ubvec = new float[ncon];
  for (int i=0; i<ncon; i++) ubvec[i]=1.05;

  int options[3];
  options[0] = 1; // 0--use default values, 1--use the values in 1 and 2
  options[1] = 3; //output level
  options[2] = 1242; //random number seed

  int edgecut =0;
  int *part = new int[ncell_on_rank];

  ParMETIS_V3_PartGeomKway( &vtxdist[0],   // array describing how cells are distributed
                        &xadj[0],   // how cells are stored locally
                        &adjncy[0], // how cells are stored loccaly
                        NULL,       // weight of vertices
                        NULL,       // weight of edges
                        &wgtflag,   // 0 means no weights for node or edges
                        &numflag,   // numbering style, 0 for C-style
                        &ndim,      // n dimensions
                        xyz,        // coorindates of vertices
                        &ncon,      // weights per vertex
                        &nparts,    // number of sub-domains
                        tpwgts,     // weight per sub-domain
                        ubvec,      // unbalance in vertex weight
                        options,    // options array
                        &edgecut,   // OUTPUT: Number of edgecuts
                        part,       // OUTPUT: partition of each vertex
                        &comm); // MPI communicator

  // if edgecuts are made (edgecut > 0) send cells to other processors
  // otherwise mesh is already partitioned

  vector<int> recv_from_rank(n_off_rank,0);
  vector<int> send_to_rank(n_off_rank, 0);

  MPI_Request *reqs = new MPI_Request[n_off_rank*2];
  vector<Buffer<Cell> > send_cell(n_off_rank);
  vector<Buffer<Cell> > recv_cell(n_off_rank);

  if (edgecut) {
    for (uint32_t ir=0; ir<n_off_rank; ir++) {
      //sends
      int off_rank = proc_map[ir];
      // make list of cells to send to off_rank
      vector<Cell> send_list;
      for (uint32_t i=0; i<ncell_on_rank; i++) {
        if(part[i] == off_rank)
          send_list.push_back(mesh->get_pre_window_allocation_cell(i));
      }
      send_to_rank[ir] = send_list.size();
      send_cell[ir].fill(send_list);

      MPI_Isend(&send_to_rank[ir], 1,  MPI_UNSIGNED, off_rank, 0,
        MPI_COMM_WORLD, &reqs[ir]);

      MPI_Irecv(&recv_from_rank[ir], 1, MPI_UNSIGNED, off_rank, 0,
        MPI_COMM_WORLD, &reqs[ir+n_off_rank]);

      // erase sent cells from the mesh
      for (uint32_t i=0; i<ncell_on_rank; i++) {
        if(part[i] == off_rank) mesh->remove_cell(i);
      }
    }

    MPI_Waitall(n_off_rank*2, reqs, MPI_STATUS_IGNORE);

    // now send the buffers and post receives
    for (uint32_t ir=0; ir<n_off_rank; ir++) {
      int off_rank = proc_map[ir];
      MPI_Isend(send_cell[ir].get_buffer(), send_to_rank[ir], MPI_Cell,
        off_rank, 0, MPI_COMM_WORLD, &reqs[ir]);

      recv_cell[ir].resize(recv_from_rank[ir]);

      MPI_Irecv(recv_cell[ir].get_buffer(), recv_from_rank[ir], MPI_Cell,
        off_rank, 0, MPI_COMM_WORLD, &reqs[ir+n_off_rank]);
    }

    MPI_Waitall(n_off_rank*2, reqs, MPI_STATUS_IGNORE);

    for (uint32_t ir=0; ir<n_off_rank; ir++) {
      vector<Cell> new_cells = recv_cell[ir].get_object();
      for (uint32_t i = 0; i< new_cells.size(); i++) {
        mesh->add_mesh_cell(new_cells[i]);
      }
    }
  }

  // update the cell list on each processor
  mesh->set_post_decomposition_mesh_cells();

  // if using grips of cell data, get additional decomposition
  {
    // get post-decomposition number of cells on this rank
    uint32_t n_cell_on_rank = mesh->get_n_local_cells();

    // make map of global IDs to local indices
    unordered_map<uint32_t, uint32_t> mesh_cell_ids;
    for (uint32_t i=0;i<n_cell_on_rank;i++) {
      mesh_cell_ids[mesh->get_pre_window_allocation_cell(i).get_ID()] = i;
    }
    unordered_map<uint32_t, uint32_t>::const_iterator end = mesh_cell_ids.end();

    vector<int> xadj;
    vector<int> adjncy;
    int adjncy_ctr = 0;
    Cell cell;
    uint32_t g_ID; // global ID
    for (uint32_t i=0; i<mesh->get_n_local_cells();i++) {
      cell = mesh->get_pre_window_allocation_cell(i);
      g_ID = cell.get_ID();
      uint32_t xm_neighbor =cell.get_next_cell(X_NEG);
      uint32_t xp_neighbor =cell.get_next_cell(X_POS);
      uint32_t ym_neighbor =cell.get_next_cell(Y_NEG);
      uint32_t yp_neighbor =cell.get_next_cell(Y_POS);
      uint32_t zm_neighbor =cell.get_next_cell(Z_NEG);
      uint32_t zp_neighbor =cell.get_next_cell(Z_POS);

       // starting index in xadj for this cell's nodes
      xadj.push_back(adjncy_ctr);
      // use local indices for the CSV to pass to METIS
      if (xm_neighbor != g_ID && mesh_cell_ids.find(xm_neighbor) != end ) {
        adjncy.push_back(mesh_cell_ids[xm_neighbor]);
        adjncy_ctr++;
      }
      if (xp_neighbor != g_ID && mesh_cell_ids.find(xp_neighbor) != end ) {
        adjncy.push_back(mesh_cell_ids[xp_neighbor]);
        adjncy_ctr++;
      }
      if (ym_neighbor != g_ID && mesh_cell_ids.find(ym_neighbor) != end ) {
        adjncy.push_back(mesh_cell_ids[ym_neighbor]);
        adjncy_ctr++;
      }
      if (yp_neighbor != g_ID && mesh_cell_ids.find(yp_neighbor) != end ) {
        adjncy.push_back(mesh_cell_ids[yp_neighbor]);
        adjncy_ctr++;
      }
      if (zm_neighbor != g_ID && mesh_cell_ids.find(zm_neighbor) != end ) {
        adjncy.push_back(mesh_cell_ids[zm_neighbor]);
        adjncy_ctr++;
      }
      if (zp_neighbor != g_ID  && mesh_cell_ids.find(zp_neighbor) != end ) {
        adjncy.push_back(mesh_cell_ids[zp_neighbor]);
        adjncy_ctr++;
      }
    } // end loop over cells

    xadj.push_back(adjncy_ctr);

    int ncon = 1;

    // get the desired number of grips on a mesh, round down
    int n_grips = n_cell_on_rank/grip_size;
    if (!n_grips) n_grips=1;

    int rank_options[METIS_NOPTIONS];

    METIS_SetDefaultOptions(rank_options);
    rank_options[METIS_OPTION_NUMBERING] = 0; // C-style numbering

    int objval;
    int *grip_index = new int[n_cell_on_rank];
    int metis_return;

    // make a signed version of this for METIS call
    int signed_n_cell_on_rank = int(n_cell_on_rank);

    // Metis does not seem to like partitioning with one part, so skip in
    // that case
    if (n_grips > 1 && grip_size > 1) {

      metis_return =
        METIS_PartGraphKway( &signed_n_cell_on_rank, // number of on-rank vertices
                              &ncon,  // weights per vertex
                              &xadj[0], // how cells are stored locally
                              &adjncy[0], // how cells are stored loccaly
                              NULL, // weight of vertices
                              NULL, // size of vertices for comm volume
                              NULL, // weight of the edges
                              &n_grips,  // number of mesh grips on this mesh
                              NULL, // tpwgts (NULL = equal weight domains)
                              NULL, // unbalance in v-weight (NULL=1.001)
                              rank_options, // options array
                              &objval,  // OUTPUT: Number of edgecuts
                              grip_index);  // OUTPUT: grip ID of each vertex

      if (metis_return==METIS_ERROR_INPUT)
        std::cout<<"METIS: Input error"<<std::endl;
      else if (metis_return==METIS_ERROR_MEMORY)
        std::cout<<"METIS: Memory error"<<std::endl;
      else if (metis_return==METIS_ERROR)
        std::cout<<"METIS: Input error"<<std::endl;
      //else if (metis_return==METIS_OK)
        //std::cout<<"METIS: Metis OK"<<std::endl;
      //else
      //  std::cout<<"METIS: Output not recognized"<<std::endl;

      // set the grip index for each cell
      for (uint32_t i =0; i<uint32_t(signed_n_cell_on_rank);i++) {
        Cell& cell = mesh->get_pre_window_allocation_cell_ref(i);
        cell.set_grip_ID(grip_index[i]);
      }
      // sort cells by grip ID
      mesh->sort_cells_by_grip_ID();

      // delete dynamically allocated data
      delete[] grip_index;
    } // end if n_grips > 1

    else if (n_grips == 1) {
      // one grip on this rank, all cells have the same grip index
      // set the grip index for each cell
      for (uint32_t i =0; i<uint32_t(signed_n_cell_on_rank);i++) {
        Cell& cell = mesh->get_pre_window_allocation_cell_ref(i);
        cell.set_grip_ID(0);
      }
    }

    else if (grip_size ==1) {
      for (uint32_t i =0; i<uint32_t(signed_n_cell_on_rank);i++) {
        Cell& cell = mesh->get_pre_window_allocation_cell_ref(i);
        cell.set_grip_ID(i);
      }
      mesh->sort_cells_by_grip_ID();
    }

  } // end within rank partitioning scope


  // gather the number of cells on each processor
  uint32_t n_cell_post_decomp = mesh->get_n_local_cells();
  vector<uint32_t> out_cells_proc(nrank, 0);
  out_cells_proc[rank] = mesh->get_n_local_cells();
  MPI_Allreduce(MPI_IN_PLACE,
                &out_cells_proc[0],
                nrank,
                MPI_UNSIGNED,
                MPI_SUM,
                MPI_COMM_WORLD);

  // prefix sum on out_cells to get global numbering
  vector<uint32_t> prefix_cells_proc(nrank, 0);
  partial_sum(out_cells_proc.begin(), out_cells_proc.end(), prefix_cells_proc.begin());

  // set global numbering
  uint32_t g_start = prefix_cells_proc[rank]-n_cell_post_decomp;
  uint32_t g_end = prefix_cells_proc[rank]-1;
  mesh->set_global_bound(g_start, g_end);

  // set the grip ID to be the cells at the center of grips using global cell
  // indices
  mesh->set_grip_ID_using_cell_index();

  // get the maximum grip size for correct parallel operations
  uint32_t max_grip_size = mesh->get_max_grip_size();
  uint32_t global_max_grip_size=0;
  uint32_t global_min_grip_size=10000000;
  MPI_Allreduce(&max_grip_size, &global_max_grip_size, 1, MPI_UNSIGNED,
    MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&max_grip_size, &global_min_grip_size, 1, MPI_UNSIGNED,
    MPI_MIN, MPI_COMM_WORLD);
  mesh->set_max_grip_size(global_max_grip_size);

  if (rank==0) {
    std::cout<<"Minimum/Maximum grip size: "<<global_min_grip_size;
    std::cout<<" / "<<global_max_grip_size<<std::endl;
  }

  // prepend zero to the prefix array to make it a standard bounds array
  prefix_cells_proc.insert(prefix_cells_proc.begin(), 0);
  mesh->set_off_rank_bounds(prefix_cells_proc);

  //make sure each off rank cell index and grip ID is remapped ONLY ONCE!
  vector< vector<bool> > remap_flag;
  for (uint32_t i=0; i<n_cell_post_decomp; i++)
    remap_flag.push_back(vector<bool> (6,false));

  // change global indices to match a simple number system for easy sorting,
  // this involves sending maps to each processor to get new indicies
  unordered_map<uint32_t, uint32_t> local_map = mesh->get_new_global_index_map();
  unordered_map<uint32_t, uint32_t> local_grip_map = mesh->get_grip_map();
  vector<uint32_t> packed_map(n_cell_post_decomp*2);
  vector<uint32_t> packed_grip_map(n_cell_post_decomp*2);
  vector<Buffer<uint32_t> > recv_packed_maps(n_off_rank);
  vector<Buffer<uint32_t> > recv_packed_grip_maps(n_off_rank);

  MPI_Request *grip_reqs = new MPI_Request[n_off_rank*2];

  // pack up the index map
  uint32_t i_packed = 0;
  for(unordered_map<uint32_t, uint32_t>::iterator map_i=local_map.begin();
    map_i!=local_map.end(); map_i++) {
    packed_map[i_packed] = map_i->first;
    i_packed++;
    packed_map[i_packed] = map_i->second;
    i_packed++;
  }

  // pack up the grip map
  uint32_t i_grip_packed = 0;
  for(unordered_map<uint32_t, uint32_t>::iterator map_i=local_grip_map.begin();
    map_i!=local_grip_map.end(); map_i++) {
    packed_grip_map[i_grip_packed] = map_i->first;
    i_grip_packed++;
    packed_grip_map[i_grip_packed] = map_i->second;
    i_grip_packed++;
  }

  // Send and receive packed maps for remapping boundaries
  for (uint32_t ir=0; ir<n_off_rank; ir++) {
    int off_rank = proc_map[ir];
    // Send your packed index map
    MPI_Isend(&packed_map[0],
              n_cell_post_decomp*2,
              MPI_UNSIGNED,
              off_rank,
              0,
              MPI_COMM_WORLD,
              &reqs[ir]);
    // Send your packed grip map
    MPI_Isend(&packed_grip_map[0],
              n_cell_post_decomp*2,
              MPI_UNSIGNED,
              off_rank,
              1,
              MPI_COMM_WORLD,
              &grip_reqs[ir]);

    // Receive other packed index maps
    recv_packed_maps[ir].resize(out_cells_proc[off_rank]*2);
    MPI_Irecv(recv_packed_maps[ir].get_buffer(),
              out_cells_proc[off_rank]*2 ,
              MPI_UNSIGNED,
              off_rank,
              0,
              MPI_COMM_WORLD,
              &reqs[ir+n_off_rank]);

    // Receive other packed grip maps
    recv_packed_grip_maps[ir].resize(out_cells_proc[off_rank]*2);
    MPI_Irecv(recv_packed_grip_maps[ir].get_buffer(),
              out_cells_proc[off_rank]*2,
              MPI_UNSIGNED,
              off_rank,
              1,
              MPI_COMM_WORLD,
              &grip_reqs[ir+n_off_rank]);
  }

  // wait for completion of all sends/receives
  MPI_Waitall(n_off_rank*2, reqs, MPI_STATUS_IGNORE);
  MPI_Waitall(n_off_rank*2, grip_reqs, MPI_STATUS_IGNORE);

  // remake the map objects from the packed maps
  for (uint32_t ir=0; ir<n_off_rank; ir++) {
    vector<uint32_t> off_packed_map = recv_packed_maps[ir].get_object();
    vector<uint32_t> off_packed_grip_map =
      recv_packed_grip_maps[ir].get_object();
    unordered_map<uint32_t, uint32_t> off_map;
    unordered_map<uint32_t, uint32_t> off_grip_map;

    for (uint32_t m=0; m<off_packed_map.size(); m++) {
      off_map[off_packed_map[m]] =off_packed_map[m+1];
      off_grip_map[off_packed_grip_map[m]] =off_packed_grip_map[m+1];
      m++;
    }
    mesh->update_off_rank_connectivity(off_map, off_grip_map, remap_flag);
  }

  // now update the indices of local IDs
  mesh->renumber_local_cell_indices(local_map, local_grip_map);

  // reallocate mesh data in new MPI window and delete the old vector object
  mesh->make_MPI_window();

  // clean up dynamically allocated memory
  delete[] grip_reqs;
  delete[] reqs;
  delete[] part;
  delete[] ubvec;
  delete[] tpwgts;
  delete[] xyz;
}

#endif // decompose_mesh_h
//---------------------------------------------------------------------------//
// end of decompose_mesh.h
//---------------------------------------------------------------------------//
