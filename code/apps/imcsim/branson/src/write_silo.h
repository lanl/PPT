//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   write_silo.h
 * \author Alex Long
 * \date   April 11 2016
 * \brief  Writes SILO output file for data visualization
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef write_silo_h_
#define write_silo_h_

#include <array>
#include <string>
#include <sstream>
#include <vector>

#ifdef VIZ_LIBRARIES_FOUND
  #include <silo.h>
#endif

#include "constants.h"
#include "config.h"
#include "imc_state.h"

//! All ranks perform reductions to produce global arrays and rank zero
// writes the SILO file for visualization
void write_silo(Mesh *mesh, const double& arg_time, const uint32_t& step,
  const double& r_transport_time, const double& r_mpi_time, const int& rank,
  const int& n_rank, std::vector<uint32_t>& rank_requests) 
{

#ifdef VIZ_LIBRARIES_FOUND
  using std::array;
  using std::stringstream;
  using std::string;
  using std::vector;
  using Constants::X_NEG;
  using Constants::X_POS;
  using Constants::Y_NEG;
  using Constants::Y_POS;
  using Constants::ELEMENT;

  // need a non-const double to pass to SILO
  double time = arg_time;

  //generate name for this silo file
  stringstream ss;
  ss.setf(std::ios::showpoint);
  ss<<std::setprecision(3); 
  ss << "output_"<<step<< ".silo";
  string file = ss.str();

  int nx = mesh->get_global_n_x_faces();
  int ny = mesh->get_global_n_y_faces();
  int nz = mesh->get_global_n_z_faces();

  // set number of dimensions
  int ndims;
  // Use a 2D mesh for one z cell (2 faces)
  if (nz == 2) ndims = 2;
  // Otherwise use 3D mesh for 3 or more z faces
  else ndims=3; 

  // generate title of plot
  stringstream tt;
  tt.setf(std::ios::showpoint);
  tt<<std::setprecision(3); 
  if (ndims==2) tt << "2D rectangular mesh, t = " << time << " (sh)";
  else tt << "3D rectangular mesh, t = " << time << " (sh)";
  string title = tt.str();

  // get total cells for MPI all_reduce calls
  uint32_t n_xyz_cells;
  if (ndims == 2) n_xyz_cells = (nx-1)*(ny-1);
  else n_xyz_cells = (nx-1)*(ny-1)*(nz-1);

  // reduce rank requests based on global ID
  MPI_Allreduce(MPI_IN_PLACE, &rank_requests[0], n_xyz_cells, MPI_INT, MPI_SUM,
    MPI_COMM_WORLD);

  // make vectors of data for plotting
  vector<int> rank_data(n_xyz_cells,0);
  vector<int> n_requests(n_xyz_cells,0);
  vector<double> T_e(n_xyz_cells,0.0);
  vector<double> transport_time(n_xyz_cells,0.0);
  vector<double> mpi_time(n_xyz_cells,0.0);
  vector<int> grip_ID(n_xyz_cells,0);

  // get rank data, map values from from global ID to SILO ID
  uint32_t n_local = mesh->get_n_local_cells();
  Cell cell;
  uint32_t g_index, silo_index;
  for (uint32_t i=0;i<n_local;i++) {
    cell = mesh->get_cell(i);
    g_index = cell.get_ID();
    silo_index = cell.get_silo_index();
    rank_data[silo_index] = rank;
    // set silo plot variables
    n_requests[silo_index] = rank_requests[g_index];
    T_e[silo_index] = cell.get_T_e();
    transport_time[silo_index] = r_transport_time;
    mpi_time[silo_index] = r_mpi_time;
    grip_ID[silo_index] = cell.get_grip_ID();
  }

  // reduce to get rank of each cell across all ranks
  MPI_Allreduce(MPI_IN_PLACE, &rank_data[0], n_xyz_cells, MPI_INT, MPI_SUM,
    MPI_COMM_WORLD);

  // reduce to get n_requests across all ranks
  MPI_Allreduce(MPI_IN_PLACE, &n_requests[0], n_xyz_cells, MPI_INT, MPI_SUM,
    MPI_COMM_WORLD);

  // reduce to get T_e across all ranks
  MPI_Allreduce(MPI_IN_PLACE, &T_e[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM,
    MPI_COMM_WORLD);

  // reduce to get transport runtime from all ranks
  MPI_Allreduce(MPI_IN_PLACE, &transport_time[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM,
    MPI_COMM_WORLD);

  // reduce to get mpi time from all ranks
  MPI_Allreduce(MPI_IN_PLACE, &mpi_time[0], n_xyz_cells, MPI_DOUBLE, MPI_SUM,
    MPI_COMM_WORLD);

  // reduce to get T_e across all ranks
  MPI_Allreduce(MPI_IN_PLACE, &grip_ID[0], n_xyz_cells, MPI_INT, MPI_SUM,
    MPI_COMM_WORLD);

  // First rank writes the SILO file
  if (rank ==0) {

    // write the global mesh
    float *x = mesh->get_silo_x();
    float *y = mesh->get_silo_y();
    float *z = mesh->get_silo_z();

    int *dims;
    float **coords;
    int *cell_dims;
    uint32_t n_xyz_cells;

    // do 2D write
    if (ndims == 2) {
      dims = new int[2];
      dims[0] = nx; dims[1] = ny;
      coords = new float*[2];  
      coords[0] = x; coords[1] = y;
      cell_dims = new int[2]; 
      cell_dims[0] = nx-1; cell_dims[1] = ny-1;
    }
    // do 3D write
    else {
      dims = new int[3];
      dims[0] = nx; dims[1] = ny; dims[2] =nz;
      coords = new float*[3];  
      coords[0]=x; coords[1] = y; coords[2]=z;
      cell_dims = new int[3]; 
      cell_dims[0]=nx-1; cell_dims[1]=ny-1; cell_dims[2]=nz-1;
    } 

    //create SILO file for this mesh
    DBfile *dbfile = NULL;
    dbfile = DBCreate(file.c_str() , 0, DB_LOCAL, NULL, DB_PDB);

    // make the correct potion list for 2D and 3D meshes
    DBoptlist *optlist;
    if (ndims ==2) {
      optlist = DBMakeOptlist(4);
      DBAddOption(optlist, DBOPT_XLABEL, (void *)"x");
      DBAddOption(optlist, DBOPT_XUNITS, (void *)"cm");
      DBAddOption(optlist, DBOPT_YLABEL, (void *)"y");
      DBAddOption(optlist, DBOPT_YUNITS, (void *)"cm");
    }
    if (ndims ==3) {
      optlist = DBMakeOptlist(6);
      DBAddOption(optlist, DBOPT_XLABEL, (void *)"x");
      DBAddOption(optlist, DBOPT_XUNITS, (void *)"cm");
      DBAddOption(optlist, DBOPT_YLABEL, (void *)"y");
      DBAddOption(optlist, DBOPT_YUNITS, (void *)"cm");
      DBAddOption(optlist, DBOPT_ZLABEL, (void *)"z");
      DBAddOption(optlist, DBOPT_ZUNITS, (void *)"cm");
    }

    DBPutQuadmesh(dbfile, "quadmesh", NULL, coords, dims, ndims,
                  DB_FLOAT, DB_COLLINEAR, optlist);
   
    // write rank IDs xy to 1D array
    int *rank_ids = new int[n_rank];
    for (uint32_t i=0;i<n_rank;i++) rank_ids[i] = i;

    DBPutMaterial(dbfile, "Rank_ID", "quadmesh", n_rank, rank_ids, 
      &rank_data[0], cell_dims, ndims, 0, 0, 0, 0, 0, DB_INT, NULL);   

    // write the n_requests scalar field
    DBoptlist *req_optlist = DBMakeOptlist(2);
    DBAddOption(req_optlist, DBOPT_UNITS, (void *)"requests");
    DBAddOption(req_optlist, DBOPT_DTIME, &time);
    DBPutQuadvar1(dbfile, "mesh_cell_requests", "quadmesh", &n_requests[0], 
      cell_dims, ndims, NULL, 0, DB_INT, DB_ZONECENT, req_optlist);

    // write the material temperature scalar field
    DBoptlist *Te_optlist = DBMakeOptlist(2);
    DBAddOption(Te_optlist, DBOPT_UNITS, (void *)"keV");
    DBAddOption(Te_optlist, DBOPT_DTIME, &time);
    DBPutQuadvar1(dbfile, "T_e", "quadmesh", &T_e[0], 
      cell_dims, ndims, NULL, 0, DB_DOUBLE, DB_ZONECENT, Te_optlist);

    // write the transport time scalar field
    DBoptlist *t_time_optlist = DBMakeOptlist(2);
    DBAddOption(t_time_optlist, DBOPT_UNITS, (void *)"seconds");
    DBAddOption(t_time_optlist, DBOPT_DTIME, &time);
    DBPutQuadvar1(dbfile, "transport_time", "quadmesh", &transport_time[0], 
      cell_dims, ndims, NULL, 0, DB_DOUBLE, DB_ZONECENT, t_time_optlist);

    // write the mpi time scalar field
    DBoptlist *mpi_time_optlist = DBMakeOptlist(2);
    DBAddOption(mpi_time_optlist, DBOPT_UNITS, (void *)"seconds");
    DBAddOption(mpi_time_optlist, DBOPT_DTIME, &time);
    DBPutQuadvar1(dbfile, "mpi_time", "quadmesh", &mpi_time[0], 
      cell_dims, ndims, NULL, 0, DB_DOUBLE, DB_ZONECENT, mpi_time_optlist);

    // write the grip_ID scalar field
    DBoptlist *grip_id_optlist = DBMakeOptlist(2);
    DBAddOption(grip_id_optlist, DBOPT_UNITS, (void *)"grip_ID");
    DBAddOption(grip_id_optlist, DBOPT_DTIME, &time);
    DBPutQuadvar1(dbfile, "grip_ID", "quadmesh", &grip_ID[0], 
      cell_dims, ndims, NULL, 0, DB_INT, DB_ZONECENT, grip_id_optlist);

    // free option lists
    DBFreeOptlist(optlist);
    DBFreeOptlist(req_optlist);
    DBFreeOptlist(Te_optlist);
    DBFreeOptlist(t_time_optlist);
    DBFreeOptlist(mpi_time_optlist);
    DBFreeOptlist(grip_id_optlist);

    // free data
    delete[] rank_ids;
    delete[] cell_dims;
    delete[] coords;
    delete[] dims;

    // close file
    DBClose(dbfile);
  } // end rank==0
#endif
}

  // rank based silo write with polygonal unstructured mesh
  /*
    uint32_t n_cell = mesh->get_n_local_cells();

    // array of nodes indices (4) for each cell
    vector<array<int,4> > node_list(n_cell);
    for (uint32_t i =0;i<n_cell; i++) {
      node_list[i][0] = -1;
      node_list[i][1] = -1;
      node_list[i][2] = -1;
      node_list[i][3] = -1;
    }


    vector<int> directions(4);
    directions[0] = X_NEG;
    directions[1] = Y_POS;
    directions[2] = X_POS;
    directions[3] = Y_NEG;
    vector<array<int,2> > dir_indices(4);
    dir_indices[0] = array<int,2>{{0,1}};
    dir_indices[1] = array<int,2>{{1,2}};
    dir_indices[2] = array<int,2>{{2,3}};
    dir_indices[3] = array<int,2>{{3,0}};
    vector<array<int,2> > opp_dir_indices(4);
    opp_dir_indices[0] = array<int,2>{{3,2}};
    opp_dir_indices[1] = array<int,2>{{0,3}};
    opp_dir_indices[2] = array<int,2>{{1,0}};
    opp_dir_indices[3] = array<int,2>{{2,1}};

    Cell cell;
    int node_index = 0;
    int nbr_index, n1, n2, o_n1, o_n2;
    int direction;
    for (uint32_t i =0;i<n_cell; i++) {
      cell = mesh->get_cell(i);
      for (uint32_t i_dir=0; i_dir<directions.size();i_dir++) {
        n1 = dir_indices[i_dir][0];
        n2 = dir_indices[i_dir][1];
        o_n1 = opp_dir_indices[i_dir][0];
        o_n2 = opp_dir_indices[i_dir][1];
        direction = directions[i_dir];
        if (cell.get_bc(direction) == ELEMENT) {
          nbr_index = cell.get_next_cell(direction);
          if (node_list[i][n1] == -1 && node_list[i][n2] ==-1) {
            node_list[i][n1] = node_index; 
            node_list[nbr_index][o_n1] = node_index;
            node_index++;
            node_list[i][n2] = node_index; 
            node_list[nbr_index][o_n2] = node_index;
            node_index++;
          }
          else if (node_list[i][n1] == -1 && node_list[i][n2] != -1)  {
            node_list[i][n1] = node_index;
            node_list[nbr_index][o_n1] = node_index;
            node_index++;
            // node 2 already set, match neighbor
            node_list[nbr_index][o_n2] = node_list[i][n2];
          }
          else if (node_list[i][n1] != -1 && node_list[i][n2] == -1)  {
            node_list[i][n2] = node_index;
            node_list[nbr_index][o_n2] = node_index;
            node_index++;
            // node 1 already set, match neighbor
            node_list[nbr_index][o_n1] = node_list[i][n1];
          }
          // both nodes set, set neighbor to match
          else if (node_list[i][n1] != -1 && node_list[i][n2] != -1)  {
            node_list[nbr_index][o_n1] = node_list[i][n1];
            node_list[nbr_index][o_n2] = node_list[i][n2];
          }
        } // end neighbor boundary setting

        // no element across this face (processor/domain bound), set independently
        else {
          // neither node set, set new indices
          if (node_list[i][n1] == -1 && node_list[i][n2] == -1) {
            node_list[i][n1] = node_index; 
            node_index++;
            node_list[i][n2] = node_index; 
            node_index++;
          }
          // second node set, set first
          else if (node_list[i][n1] == -1 && node_list[i][n2] != -1)  {
            node_list[i][n1] = node_index;
            node_index++; 
          }
          // first node set, set second
          else if (node_list[i][n1] != -1 && node_list[i][n2] == -1)  {
            node_list[i][n2] = node_index;
            node_index++; 
          }
        } // end processor/domain bound boundary setting
      } // end loop over directions
    } // end loop over cells

    // number of nodes is equal to the final node index (it was incremented
    // after setting the last node value, making it equal to the node count)

    uint32_t n_node = node_index;

    // set x and y arrays (position of each node)
    float *x = new float[n_node];
    float *y = new float[n_node];

    float xl,xh,yl,yh;
    int n3, n4;
    for (uint32_t i=0;i<n_cell;i++) {
      cell = mesh->get_cell(i);
      xl = cell.get_x_low();
      xh = cell.get_x_high();
      yl = cell.get_y_low();
      yh = cell.get_y_high();
      n1 = node_list[i][0];
      n2 = node_list[i][1];
      n3 = node_list[i][2];
      n4 = node_list[i][3];
      x[n1] = xl;
      x[n2] = xl;
      x[n3] = xh;
      x[n4] = xh;

      y[n1] = yl;
      y[n2] = yh;
      y[n3] = yh;
      y[n4] = yl;
    }

    float *coords[2] = {x, y};

    // options
    DBoptlist *optlist = DBMakeOptlist(4);
    DBAddOption(optlist, DBOPT_XLABEL, (void *)"x");
    DBAddOption(optlist, DBOPT_XUNITS, (void *)"cm");
    DBAddOption(optlist, DBOPT_YLABEL, (void *)"y");
    DBAddOption(optlist, DBOPT_YUNITS, (void *)"cm");

    int ndims = 2;
    DBPutUcdmesh( dbfile, //Database file pointer.
      "mesh", //Name of the mesh.
      ndims, //Number of spatial dimensions represented by this UCD mesh.
      NULL, //Array of length ndims containing pointers to the names to be provided when writing out the coordinate arrays. This parameter is currently ignored and can be set as NULL.
      coords, // Array of length ndims containing pointers to the coordinate arrays.
      n_node, //Number of nodes in this UCD mesh.
      n_cell, //Number of zones in this UCD mesh.
      NULL, //Name of the zonelist structure associated with this variable [written with DBPutZonelist]. If no association is to be made or if the mesh is composed solely of arbitrary, polyhedral elements, this value should be NULL. If a polyhedral-zonelist is to be associated with the mesh, DO NOT pass the name of the polyhedral-zonelist here. Instead, use the DBOPT_PHZONELIST option described below. For more information on arbitrary, polyhedral zonelists, see below and also see the documentation for DBPutPHZonelist.
      NULL, //Name of the facelist structure associated with this variable [written with DBPutFacelist]. If no association is to be made, this value should be NULL.
      DB_FLOAT, //Datatype of the coordinate arrays
      optlist); //Pointer to an option list structure containing additional information 

    // free option list
    DBFreeOptlist(optlist);
  */

#endif // write_silo_h_
//---------------------------------------------------------------------------//
// end of write_silo.h
//---------------------------------------------------------------------------//
