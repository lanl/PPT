//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_mesh.h
 * \author Alex Long
 * \date   April 7 2016
 * \brief  Test region assignment after mesh construction
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <string>
#include <vector>

#include "../constants.h"
#include "../input.h"
#include "../mesh.h"
#include "../mpi_types.h"
#include "testing_functions.h"

int main (int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  
  const Info mpi_info;

  MPI_Types *mpi_types = new MPI_Types();

  using std::cout;
  using std::endl;
  using std::string;

  int nfail = 0;

  // Test that a simple mesh (one division in each dimension) is constructed
  // correctly from the input file (simple_input.xml) and that each cell 
  // is assigned the correct region
  {
    string filename("simple_input.xml");
    Input *input = new Input(filename);

    Mesh mesh(input, mpi_types, mpi_info);

    bool simple_mesh_pass = true;

    uint32_t n_cell = mesh.get_n_local_cells();
    if (n_cell != 10*20*30) simple_mesh_pass =false;

    Cell cell;
    for (uint32_t i = 0; i<n_cell; i++) {
      cell = mesh.get_pre_window_allocation_cell(i);
      if ( cell.get_region_ID() != 6) simple_mesh_pass =false;
    }

    if (simple_mesh_pass) cout<<"TEST PASSED: simple mesh construction"<<endl;
    else { 
      cout<<"TEST FAILED: simple mesh construction"<<endl; 
      nfail++;
    }
    delete input;
  }
  
  // Test that a multi-region mesh is constructed correctly from the input file
  // (three_region_input_mesh.xml) and that each cell is assigned the correct 
  // region
  {
    bool three_region_mesh_pass = true;
    // first test large particle input file
    string three_reg_filename("three_region_mesh_input.xml");
    Input *three_reg_input = new Input(three_reg_filename);

    Mesh mesh(three_reg_input, mpi_types, mpi_info);

    uint32_t n_cell = mesh.get_n_local_cells();
    if (n_cell != 21*10) three_region_mesh_pass =false;

    Cell cell;
    const double *coor;
    double x_low;
    // check the lower x position of the cell to see if the region matches
    // the divisions set in the input file
    for (uint32_t i = 0; i<n_cell; i++) {
      cell = mesh.get_pre_window_allocation_cell(i);
      coor = cell.get_node_array();
      x_low = coor[0];
      //cells in the first region
      if (x_low < 4.0) {
        if ( cell.get_region_ID() != 230) three_region_mesh_pass =false;
      }
      else if (x_low >= 4.0 && x_low < 8.0) {
        if ( cell.get_region_ID() != 177) three_region_mesh_pass =false;
      }
      else if (x_low >= 8.0) {
        if ( cell.get_region_ID() != 11) three_region_mesh_pass =false;
      }
      else {
        // this should not occur, test fails
        three_region_mesh_pass = false;
      }
    }

    if (three_region_mesh_pass) cout<<"TEST PASSED: three region mesh construction"<<endl;
    else { 
      cout<<"TEST FAILED: three region mesh construction"<<endl; 
      nfail++;
    }
    delete three_reg_input;
  }

  delete mpi_types;

  MPI_Finalize();

  return nfail;
}
//----------------------------------------------------------------------------//
// end of test_mesh.cc
//----------------------------------------------------------------------------//
