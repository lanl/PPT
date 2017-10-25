//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_photon.cc
 * \author Alex Long
 * \date   January 11 2016
 * \brief  Test photon construction and move functionality
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>

#include "../photon.h"
#include "testing_functions.h"

int main (void) {

  using std::cout;
  using std::endl;

  int nfail = 0;

  // test construction, get and set functions
  {
    bool test_photon = true;
    Photon photon;

    //set position matches get position
    double pos[3] = {0.1, 0.2, 0.3}; 
    photon.set_position(pos);
    const double *pos_from_get = photon.get_position();
    
    if (pos[0] != pos_from_get[0]) test_photon= false;
    if (pos[1] != pos_from_get[1]) test_photon= false;
    if (pos[2] != pos_from_get[2]) test_photon= false;

    //set angle matches get angle
    double angle[3] = {0.57735, 0.37735, 0.52427};
    photon.set_angle(angle);
    const double *angle_from_get = photon.get_angle();
    
    if (angle[0] != angle_from_get[0]) test_photon= false;
    if (angle[1] != angle_from_get[1]) test_photon= false;
    if (angle[2] != angle_from_get[2]) test_photon= false;
   
    // set cell and grip matches get cell and grip
    uint32_t cell = 129120;
    uint32_t grip = 213191;
    photon.set_cell(cell);
    photon.set_grip(grip);
    
    if (photon.get_cell() != cell) test_photon= false;
    if (photon.get_grip() != grip) test_photon= false;
 
    if (test_photon) cout<<"TEST PASSED: Photon construction, get and set functions"<<endl;
    else { 
      cout<<"TEST FAILED: Photon construction, get and set functions function"<<endl; 
      nfail++;
    }
  }


  //test move function
  {
    bool test_photon_move = true;

    double tolerance = 1.0e-8;
    Photon photon;
    
    // first move    
    {
      double pos[3] =   {0.0, 0.0, 0.0}; 
      double angle[3] = {1.0, 0.0, 0.0};
      photon.set_position(pos);
      photon.set_angle(angle);
      photon.set_distance_to_census(10.0);
      
      photon.move(7.5);
      const double *moved_position_1 = photon.get_position();

      if ( !soft_equiv(moved_position_1[0], 7.5, tolerance) ) 
        test_photon_move= false;
      if ( !soft_equiv(moved_position_1[1], 0.0, tolerance) ) 
        test_photon_move= false;
      if ( !soft_equiv(moved_position_1[2], 0.0, tolerance) ) 
        test_photon_move= false;
      //distance remaining is distance to census - move distance
      if ( !soft_equiv(photon.get_distance_remaining(), 2.5, tolerance) ) 
        test_photon_move= false;
    }

    // second move
    {
      double pos[3] =   {1.64, 6.40, -5.64}; 
      double angle[3] = {0.57735, -0.57735, 0.57735};
      photon.set_position(pos);
      photon.set_angle(angle);
      photon.set_distance_to_census(10.0);
      
      photon.move(1.001648);
      const double *moved_position_2 = photon.get_position();

      if ( !soft_equiv(moved_position_2[0], 2.2183014728, tolerance) ) 
        test_photon_move= false;
      if ( !soft_equiv(moved_position_2[1], 5.8216985272, tolerance) ) 
        test_photon_move= false;
      if ( !soft_equiv(moved_position_2[2],-5.0616985272, tolerance) ) 
        test_photon_move= false;
      //distance remaining is distance to census - move distance
      if ( !soft_equiv(photon.get_distance_remaining(), 8.998352, tolerance) )
        test_photon_move= false;
    }
    
    if (test_photon_move) cout<<"TEST PASSED: Photon move function"<<endl;
    else { 
      cout<<"TEST FAILED: Photon move function"<<endl; 
      nfail++;
    }
  } 

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_photon.cc
//---------------------------------------------------------------------------//
