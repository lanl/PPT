//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_cell.cc
 * \author Alex Long
 * \date   January 11 2016
 * \brief  Test cell check_in_cell and distance_to_boundary functions
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include "../cell.h"
#include "testing_functions.h"

int main (void) {

  using std::cout;
  using std::endl;

  int nfail = 0;

  //setup cell
  Cell cell;
  // simple cube of size 1.0
  double x_low = 0.0;
  double x_high = 1.0;
  double y_low = 0.0;
  double y_high = 1.0;
  double z_low = 0.0;
  double z_high = 1.0;
  
  cell.set_coor(x_low, x_high, y_low, y_high, z_low, z_high);

  // test the check_in_cell function
  {
    double true_pos_1[3] = { 0.5, 0.5, 0.5};
    double true_pos_2[3] = { 0.9, 0.9, 0.9};
    double true_pos_3[3] = { 0.1, 0.1, 0.1};
    double true_pos_4[3] = { 0.1, 0.9, 0.1};
    double false_pos_1[3] = {-1.0, -1.0, -1.0};
    double false_pos_2[3] = {1.0, -1.0, -1.0};
    double false_pos_3[3] = {1.0, 1.0, -1.0};
    double false_pos_4[3] = {-1.0, 1.0, -1.0};

    bool check_in_cell_pass = true;
    //positions in cell
    if (!cell.check_in_cell(true_pos_1)) check_in_cell_pass =false;
    if (!cell.check_in_cell(true_pos_2)) check_in_cell_pass =false;
    if (!cell.check_in_cell(true_pos_3)) check_in_cell_pass =false;
    if (!cell.check_in_cell(true_pos_4)) check_in_cell_pass =false;
    //positions out of cell
    if (cell.check_in_cell(false_pos_1)) check_in_cell_pass =false;
    if (cell.check_in_cell(false_pos_2)) check_in_cell_pass =false;
    if (cell.check_in_cell(false_pos_3)) check_in_cell_pass =false;
    if (cell.check_in_cell(false_pos_4)) check_in_cell_pass =false;

    if (check_in_cell_pass) cout<<"TEST PASSED: check_in_cell function"<<endl;
    else { 
      cout<<"TEST FAILED: check_in_cell function"<<endl; 
      nfail++;
    }
  }

  
  // test distance to boundary function
  {
    double tolerance = 1.0e-8;

    double pos[3] = { 0.5, 0.5, 0.5};

    double angle_1[3] = {0.999, 0.031614, 0.031614};
    double angle_2[3] = {0.031614, 0.999, 0.031614};
    double angle_3[3] = {0.031614, 0.31614, 0.999};

    double angle_4[3] = {-0.999, 0.031614, 0.031614};
    double angle_5[3] = {0.031614, -0.999, 0.031614};
    double angle_6[3] = {0.031614, 0.31614, -0.999};

    unsigned int surface_cross;
    bool distance_to_bound_pass = true;

    // tests for true
    double distance_1 = 
      cell.get_distance_to_boundary(pos, angle_1, surface_cross);
    if ( !soft_equiv(distance_1, 0.5/0.999, tolerance) ) 
      distance_to_bound_pass=false;

    double distance_2 = 
      cell.get_distance_to_boundary(pos, angle_2, surface_cross);
    if ( !soft_equiv(distance_2, 0.5/0.999, tolerance) ) 
      distance_to_bound_pass=false;

    double distance_3 = 
      cell.get_distance_to_boundary(pos, angle_3, surface_cross);
    if ( !soft_equiv(distance_3, 0.5/0.999, tolerance) ) 
      distance_to_bound_pass=false;

    double distance_4 = 
      cell.get_distance_to_boundary(pos, angle_4, surface_cross);
    if ( !soft_equiv(distance_4, 0.5/0.999, tolerance) ) 
      distance_to_bound_pass=false;

    double distance_5 = 
      cell.get_distance_to_boundary(pos, angle_5, surface_cross);
    if ( !soft_equiv(distance_5, 0.5/0.999, tolerance) ) 
      distance_to_bound_pass=false;

    double distance_6 = 
      cell.get_distance_to_boundary(pos, angle_6, surface_cross);
    if ( !soft_equiv(distance_6, 0.5/0.999, tolerance) ) 
      distance_to_bound_pass=false;

    if (distance_to_bound_pass) cout<<"TEST PASSED: distance_to_boundary function"<<endl;
    else { 
      cout<<"TEST FAILED: distance_to_boundary function"<<endl; 
      nfail++;
    }
  }


  // test get_volume function
  {
    bool get_volume_pass = true;

    double tolerance = 1.0e-8;

    if (!soft_equiv(cell.get_volume(), 1.0, tolerance)) get_volume_pass = false;

    //make an oblong cell
    Cell oblong_cell;
    oblong_cell.set_coor(0.01, 0.02, 0.0, 10.0, -0.1, 0.1);

    if (!soft_equiv(oblong_cell.get_volume(), 0.02, tolerance)) get_volume_pass = false;

    if (get_volume_pass) cout<<"TEST PASSED: get_volume function"<<endl;
    else { 
      cout<<"TEST FAILED: get_volume function"<<endl; 
      nfail++;
    }
  }

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_cell.cc
//---------------------------------------------------------------------------//
