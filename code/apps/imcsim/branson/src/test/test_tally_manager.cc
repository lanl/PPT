//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_tally_manager.cc
 * \author Alex Long
 * \date   August 31, 2016
 * \brief  Test census remap functions
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <unordered_map>
#include <vector>

#include "../message_counter.h"
#include "../RNG.h"
#include "../tally_manager_rma.h"
#include "testing_functions.h"

int main (int argc, char *argv[]) {

  using std::cout;
  using std::endl;
  using std::vector;

  MPI_Init(&argc, &argv);

  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

  int nfail = 0;

  // setup the rank boundaries
  uint32_t n_cell = 1000;

  // make up cell bounds for this rank
  vector<uint32_t> rank_bounds(n_rank+1);
  uint32_t r_bound = 0;
  for (uint32_t i=0; i<uint32_t(n_rank+1);++i) {
    if (i!=uint32_t(n_rank)) {
      rank_bounds[i] = r_bound;
      r_bound+=n_cell/n_rank;
    }
    else {
      rank_bounds[i] = n_cell;
    }
  }
  uint32_t rank_start = rank_bounds[rank];
  uint32_t rank_end = rank_bounds[rank+1];
  uint32_t n_cell_on_rank = rank_end - rank_start;

  // test Tally_Manager with random tally writes
  {
    bool test_tally_manager = true;

    Tally_Manager t_manager(rank, rank_bounds, n_cell_on_rank);
    Message_Counter mctr;

    double const * const abs_E_from_other_ranks = t_manager.get_tally_ptr();

    // test to make sure all tallies are initially zero
    for (uint32_t i=0; i<n_cell_on_rank;++i) {

      // test to make sure the energy in each cell is zero
      if (abs_E_from_other_ranks[i] != 0.0) test_tally_manager = false;
    }

    // need RNG to randomize particle ranks
    RNG *rng = new RNG();
    rng->set_seed(rank*4106);

    // setup tally test parameters and variables
    uint32_t n_tally = n_cell;
    double E_event = 0.01;
    std::unordered_map<uint32_t, double> off_rank_abs_E;
    std::vector<double> E_to_rank(n_rank, 0.0);

    // write tallies
    uint32_t cell_id, write_rank;
    for (uint32_t i=0; i<n_tally;++i) {
      //write_rank = rank;
      //while(write_rank == rank) {
      //  cell_id = uint32_t(rng->generate_random_number()*n_cell);
      //  write_rank = t_manager.get_off_rank_id(cell_id);
      //}
      cell_id = i;
      write_rank = t_manager.get_off_rank_id(cell_id);
      E_to_rank[write_rank] += E_event;
      if (off_rank_abs_E.find(cell_id) == off_rank_abs_E.end())
        off_rank_abs_E[cell_id] = E_event;
      else
        off_rank_abs_E[cell_id]+=E_event;
    }

    // set the expected amount of energy
    double expected_total_abs_E = n_rank*n_tally*E_event;

    // Write the absorbed energy
    bool force_send = false;

    cout<<"Rank "<<rank<<" about to remote write"<<endl;
    t_manager.process_off_rank_tallies(mctr, off_rank_abs_E, force_send);
    t_manager.finish_remote_writes(mctr, off_rank_abs_E);


    cout<<"Rank "<<rank<<" finished"<<endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // sum the total energy tallied on this rank
    double actual_rank_abs_E = 0.0;
    for (uint32_t i=0; i<n_cell_on_rank;++i) {
      actual_rank_abs_E += abs_E_from_other_ranks[i];

      // test to make sure the energy in each cell is valid (greater than zero
      // and not a NaN)
      if (abs_E_from_other_ranks[i] < 0.0) test_tally_manager = false;
      if (abs_E_from_other_ranks[i] != abs_E_from_other_ranks[i])
        test_tally_manager = false;
    }

    // reduce to get the actual energy absorbed across all ranks
    double actual_total_abs_E = 0.0;
    MPI_Allreduce(&actual_rank_abs_E, &actual_total_abs_E, 1, MPI_DOUBLE,
      MPI_SUM, MPI_COMM_WORLD);

    // reduce the E_to_rank array to get the expected energy absorbed on each
    // rank
    vector<double> expected_rank_abs_E(n_rank, 0.0);
    MPI_Allreduce(&E_to_rank[0], &expected_rank_abs_E[0], n_rank, MPI_DOUBLE,
      MPI_SUM, MPI_COMM_WORLD);

    double tol = 1.0e-8;

    // check to make sure the actual absorbed energy matches the expected
    // absorbed energy for this rank
    if (!soft_equiv(actual_total_abs_E, expected_total_abs_E, tol))
      test_tally_manager = false;

    // check to make sure the total absorbed energy accross all ranks matches
    // the expected value
    if (!soft_equiv(actual_rank_abs_E, expected_rank_abs_E[rank], tol))
      test_tally_manager = false;

    if (test_tally_manager) {
      cout<<"TEST PASSED: Tally manager correctly moves energy back to origin ";
      cout<<"rank: "<<rank<<endl;
    }
    else {
      cout<<"TEST FAILED: Tally manager failed"<<endl;
      nfail++;
    }

    cout.flush();
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) {
      cout.flush();
      cout<<"Expected total abs E: "<<expected_total_abs_E<<" actual total";
      cout<<" abs E: "<<actual_total_abs_E<<endl;
    }

  }

  MPI_Finalize();

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_tally_manager.cc
//---------------------------------------------------------------------------//
