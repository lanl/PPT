//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_work_packet.cc
 * \author Alex Long
 * \date   May 5 2016
 * \brief  Test work packet construction and splitting functionality
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>

#include "../constants.h"
#include "../work_packet.h"

int main (void) {

  using std::cout;
  using std::endl;

  int nfail = 0;

  // test construction, get and set functions
  {
    bool test_work_packet = true;
    Work_Packet work_packet;

    // after construction, photons, cell ID and creation energy are zero
    if (work_packet.get_global_cell_ID() != 0) test_work_packet = false;
    if (work_packet.get_global_grip_ID() != 0) test_work_packet = false;
    if (work_packet.get_n_particles() != 0) test_work_packet = false;
    if (work_packet.get_create_E() != 0) test_work_packet = false;

    // default source type is emission, set and check initial census type
    if (work_packet.get_source_type() != Constants::EMISSION) 
      test_work_packet = false;
    work_packet.set_source_type(Constants::INITIAL_CENSUS);
    if (work_packet.get_source_type() != Constants::INITIAL_CENSUS) 
      test_work_packet = false;

    // set cell ID, emission energy and number of particles
    uint32_t cell_ID = 165470;
    uint32_t grip_ID = 520468;
    uint32_t n_particles = 994908;
    double emission_E = 1904.304;
    double cell_coor[6] = {1.0,2.0, 0.0,0.5, 3.3,3.6};

    work_packet.set_global_cell_ID(cell_ID);
    work_packet.set_global_grip_ID(grip_ID);
    work_packet.attach_creation_work(emission_E, n_particles);
    work_packet.set_coor(cell_coor);

    const double *work_packet_coor = work_packet.get_node_array();

    if (work_packet.get_global_cell_ID() != cell_ID) test_work_packet = false;
    if (work_packet.get_global_grip_ID() != grip_ID) test_work_packet = false;
    if (work_packet.get_n_particles() != n_particles) test_work_packet = false;
    if (work_packet.get_create_E() != emission_E) test_work_packet = false;
    for (uint32_t i=0;i<6;i++) {
      if (work_packet_coor[i] != cell_coor[i]) test_work_packet = false;
    }

    if (test_work_packet) {
      cout<<"TEST PASSED: Work_Packet construction, get ";
      cout<<"and set functions"<<endl;
    }
    else { 
      cout<<"TEST FAILED: Work_Packet construction, get and set functions ";
      cout<<endl;
      nfail++;
    }
  }

  //test split function
  {
    bool test_split_work_packet = true;

    Work_Packet big_work_packet;

    // set cell ID, emission energy and number of particles
    uint32_t cell_ID = 10;
    uint32_t grip_ID = 670;
    uint32_t n_particles = 106547;
    double emission_E = 100.0;
    double cell_coor[6] = {1.0,2.0, 0.0,0.5, 3.3,3.6};

    big_work_packet.set_global_cell_ID(cell_ID);
    big_work_packet.set_global_grip_ID(grip_ID);
    big_work_packet.attach_creation_work(emission_E, n_particles);
    big_work_packet.set_coor(cell_coor);

    // split work packet
    uint32_t n_remain = 100000;
    Work_Packet leftover_work = big_work_packet.split(n_remain);

    double e_leftover = double(n_particles-n_remain)/double(n_particles)*emission_E;
    double e_remain = emission_E - e_leftover;
    const double * big_work_coor = big_work_packet.get_node_array();

    // test big work packet
    if (big_work_packet.get_global_cell_ID() != cell_ID) 
      test_split_work_packet = false;
    if (big_work_packet.get_global_grip_ID() != grip_ID) 
      test_split_work_packet = false;
    if (big_work_packet.get_n_particles() != n_remain) 
      test_split_work_packet = false;
    if (big_work_packet.get_create_E() != e_remain) 
      test_split_work_packet = false;
    for (uint32_t i=0;i<6;i++) {
      if (big_work_coor[i] != cell_coor[i]) test_split_work_packet = false;
    }

    const double * leftover_work_coor = leftover_work.get_node_array();
    // test leftover packet
    if (leftover_work.get_global_cell_ID() != cell_ID) 
      test_split_work_packet = false;
    if (leftover_work.get_n_particles() != n_particles - n_remain) 
      test_split_work_packet = false;
    if (leftover_work.get_create_E() != e_leftover) 
      test_split_work_packet = false;
    for (uint32_t i=0;i<6;i++) {
      if (leftover_work_coor[i] != cell_coor[i]) test_split_work_packet = false;
    }
    
    // test combination of both packets
    if (leftover_work.get_n_particles()+big_work_packet.get_n_particles()
      != n_particles) 
      test_split_work_packet = false;
    if (leftover_work.get_create_E()+big_work_packet.get_create_E() 
      != emission_E) 
      test_split_work_packet = false;
  

    if (test_split_work_packet) cout<<"TEST PASSED: Split Work_Packet"<<endl;
    else { 
      cout<<"TEST FAILED: Split Work_Packet"<<endl;
      nfail++;
    }
  }
  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_work_packet.cc
//---------------------------------------------------------------------------//
