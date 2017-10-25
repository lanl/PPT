//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_remap_census.cc
 * \author Alex Long
 * \date   August 31, 2016
 * \brief  Test census remap functions
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>

#include "../photon.h"
#include "../remap_census.h"
#include "../RNG.h"
#include "../info.h"
#include "testing_functions.h"

int main (int argc, char *argv[]) {

  using std::cout;
  using std::endl;
  using std::vector;

  MPI_Init(&argc, &argv);
  
  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  MPI_Types *mpi_types = new MPI_Types();
  const Info mpi_info;

  int nfail = 0;

  // test helper functions used by remap census
  {
    bool test_helper_functions = true;

    // test get communication partner function against expected outputs
    uint32_t t_rank = 2;
    uint32_t t_n_rank = 32;
    uint32_t k =0;
    uint32_t r_partner = get_pairing(t_rank, t_n_rank, k);

    if (r_partner != 3) test_helper_functions= false;
    if (get_pairing(r_partner, t_n_rank, k) != t_rank) 
      test_helper_functions= false;

    k=1;
    r_partner = get_pairing(t_rank, t_n_rank, k);

    if (r_partner != 0) test_helper_functions= false;
    if (get_pairing(r_partner, t_n_rank, k) != t_rank) 
      test_helper_functions= false;

    k=2;
    r_partner = get_pairing(t_rank, t_n_rank, k);

    if (r_partner != 6) test_helper_functions= false;
    if (get_pairing(r_partner, t_n_rank, k) != t_rank) 
      test_helper_functions= false;

    k=3;
    r_partner = get_pairing(t_rank, t_n_rank, k);

    if (r_partner != 10) test_helper_functions= false;
    if (get_pairing(r_partner, t_n_rank, k) != t_rank) 
      test_helper_functions= false;

    if (test_helper_functions) 
      cout<<"TEST PASSED: remap_census helper functions"<<endl;
    else { 
      cout<<"TEST FAILED: remap_census helper functions"<<endl;
      nfail++;
    }
  }


  // setup the rank boundaries
  uint32_t n_cell = 100000;

  // make up cell bounds for this rank
  vector<uint32_t> rank_bounds(n_rank+1);
  uint32_t r_bound = 0;
  for (uint32_t i=0; i<n_rank+1;++i) {
    if (i!=n_rank) {
      rank_bounds[i] = r_bound;
      r_bound+=n_cell/n_rank;
    }
    else {
      rank_bounds[i] = n_cell;
    }
  }
  uint32_t rank_start = rank_bounds[rank];
  uint32_t rank_end = rank_bounds[rank+1];

  //test remap_census function with standard off-rank photons
  {

    bool test_remap_to_rank = true;

    // need RNG to randomize particle ranks
    RNG *rng = new RNG();
    rng->set_seed(rank*4106);

    // make up off-rank census data that ended up on this rank
    // test to make sure all post rebalance census particles are on this rank
    // also check to make sure data has not been corrupted
    
    vector<Photon> off_rank_census;
    uint64_t rank_photons = 0;
    uint32_t new_cell;
    double c_dt = 10.0;
    uint32_t n_off_rank_photons = 1000;
    for (uint32_t i=0;i<n_off_rank_photons;++i) {
      Photon temp_photon;
      double pos[3] =   {0.0, 0.0, 0.0}; 
      double angle[3] = {1.0, 0.0, 0.0};
      temp_photon.set_position(pos);
      temp_photon.set_angle(angle);
      temp_photon.set_distance_to_census(c_dt);
      new_cell = uint32_t(rng->generate_random_number()*n_cell);
      while (new_cell > rank_start && new_cell < rank_end)
        new_cell = uint32_t(rng->generate_random_number()*n_cell); 
      temp_photon.set_cell(new_cell);
      off_rank_census.push_back(temp_photon);
    }

    vector<Photon> post_rebalance_census = rebalance_census(off_rank_census, 
      rank_photons, rank_bounds, mpi_types, mpi_info);

    uint32_t phtn_cell;
    for (vector<Photon>::const_iterator iphtn =post_rebalance_census.cbegin();
      iphtn!=post_rebalance_census.cend(); ++iphtn)
    {
      phtn_cell =  iphtn->get_cell();
      if(phtn_cell < rank_start || phtn_cell >= rank_end) 
        test_remap_to_rank=false;
      if(iphtn->get_distance_remaining() != c_dt)
        test_remap_to_rank=false;
    }

    if (test_remap_to_rank) cout<<"TEST PASSED: remap_census to rank"<<endl;
    else { 
      cout<<"TEST FAILED: remap_census to rank"<<endl; 
      nfail++;
    }
    delete rng;
  }

  // call rebalance census with an empty vector of off rank census, make
  // sure it finishes and returns an empty vector
  {
    bool test_empty_census = true;

    // need RNG to randomize particle ranks
    RNG *rng = new RNG();
    rng->set_seed(rank*4106);

    vector<Photon> empty_census;
    uint64_t rank_photons = 0;
    vector<Photon> post_rebalance_empty = rebalance_census(empty_census,
      rank_photons, rank_bounds, mpi_types, mpi_info);

    if(post_rebalance_empty.size() != 0) test_empty_census=false;

    if (test_empty_census) cout<<"TEST PASSED: remap_census with empty census"<<endl;
    else { 
      cout<<"TEST FAILED: remap_census with empty census"<<endl; 
      nfail++;
    }
    delete rng;
  }

  // call rebalance when the on rank photon count is more than the allowed
  // limit for this node, no census photons should be sent
  {
    bool test_full_node = true;

    // need RNG to randomize particle ranks
    RNG *rng = new RNG();

    rng->set_seed(rank*4106);
    vector<Photon> off_rank_census;
    uint32_t new_cell;
    double c_dt = 10.0;
    uint64_t rank_photons = mpi_info.get_node_mem();
    uint32_t n_off_rank_photons = 1000;
    for (uint32_t i=0;i<n_off_rank_photons;++i) {
      Photon temp_photon;
      double pos[3] =   {0.0, 0.0, 0.0}; 
      double angle[3] = {1.0, 0.0, 0.0};
      temp_photon.set_position(pos);
      temp_photon.set_angle(angle);
      temp_photon.set_distance_to_census(c_dt);
      new_cell = uint32_t(rng->generate_random_number()*n_cell);
      while (new_cell > rank_start && new_cell < rank_end)
        new_cell = uint32_t(rng->generate_random_number()*n_cell); 
      temp_photon.set_cell(new_cell);
      off_rank_census.push_back(temp_photon);
    }

    vector<Photon> post_rebalance_full = rebalance_census(off_rank_census,
      rank_photons, rank_bounds, mpi_types, mpi_info);

    uint32_t phtn_cell;
    for (vector<Photon>::const_iterator iphtn =post_rebalance_full.cbegin();
      iphtn!=post_rebalance_full.cend(); ++iphtn)
    {
      phtn_cell =  iphtn->get_cell();
      if(phtn_cell >= rank_start && phtn_cell < rank_end) 
        test_full_node=false;
      if(iphtn->get_distance_remaining() != c_dt)
        test_full_node=false;
    }

    if (test_full_node) cout<<"TEST PASSED: remap_census with full node"<<endl;
    else { 
      cout<<"TEST FAILED: remap_census with full node"<<endl; 
      nfail++;
    }
    delete rng;
  }

  // call rebalance when all off-rank particles belong to one rank and would
  // overflow the limit for a node
  {
    bool test_large_imbalance = true;

    // need RNG to randomize particle ranks
    RNG *rng = new RNG();

    rng->set_seed(rank*4106);
    vector<Photon> off_rank_census;
    uint32_t new_cell;
    double c_dt = 10.0;
    uint64_t rank_photons = 0;
    uint64_t n_off_rank_photons = uint64_t(
      0.1*(mpi_info.get_node_mem()/sizeof(Photon))/16.0) ;
    if (rank != 0) {
      for (uint64_t i=0;i<n_off_rank_photons;++i) {
        Photon temp_photon;
        double pos[3] =   {0.0, 0.0, 0.0}; 
        double angle[3] = {1.0, 0.0, 0.0};
        temp_photon.set_position(pos);
        temp_photon.set_angle(angle);
        temp_photon.set_distance_to_census(c_dt);
        // set cell to be rank on 0
        temp_photon.set_cell(0);
        off_rank_census.push_back(temp_photon);
      }
    }

    vector<Photon> post_rebalance_full = rebalance_census(off_rank_census,
      rank_photons, rank_bounds, mpi_types, mpi_info);

    uint32_t phtn_cell;
    for (vector<Photon>::const_iterator iphtn =post_rebalance_full.cbegin();
      iphtn!=post_rebalance_full.cend(); ++iphtn)
    {
      phtn_cell =  iphtn->get_cell();
      if(phtn_cell >= rank_start && phtn_cell < rank_end) 
        test_large_imbalance=false;
      if(iphtn->get_distance_remaining() != c_dt)
        test_large_imbalance=false;
    }

    if (test_large_imbalance) cout<<"TEST PASSED: remap_census with full large imbalance"<<endl;
    else { 
      cout<<"TEST FAILED: remap_census with large imbalance"<<endl; 
      nfail++;
    }
    delete rng;
  }
  

  delete mpi_types;

  MPI_Finalize();

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_remap_census.cc
//---------------------------------------------------------------------------//
