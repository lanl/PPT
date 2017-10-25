//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_completion_manager.cc
 * \author Alex Long
 * \date   March 18 2016
 * \brief  Test both completion manager's construction and completion routines
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <mpi.h>

#include "../constants.h"
#include "../completion_manager_milagro.h"
#include "../completion_manager_rma.h"
#include "../message_counter.h"
#include "testing_functions.h"

using std::cout;
using std::endl;

int main (int argc, char *argv[]) {

  MPI_Init(&argc, &argv);

  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

  int nfail = 0;


  // test construction and MPI window type
  {
    bool construction_pass = true;

    Completion_Manager_RMA comp(rank, n_rank);

    if (comp.get_mpi_window_memory_type() != MPI_WIN_UNIFIED) construction_pass = false;

    if (construction_pass) cout<<"TEST PASSED: Construction and MPI window type "
      <<n_rank<<" ranks"<<endl;
    else {
      cout<<"TEST FAILED: Construction and MPI window type with"<<n_rank<<" ranks"<<endl;
      nfail++;
    }
  }

  // test completion routine for RMA manager
  {
    bool completion_routine_pass = true;

    uint64_t rank_particles = 10000;
    uint64_t global_count = n_rank*rank_particles;
    Completion_Manager_RMA comp(rank, n_rank);

    //set global particle count
    comp.set_timestep_global_particles(global_count);

    uint64_t rank_complete = 0;
    double work = 0.0;
    bool finished = false;

    Message_Counter mctr;

    while(!finished ) {
      if (rank_complete <rank_particles) rank_complete++;
      work+=exp(-rank_complete%8);
      comp.process_completion(true, rank_complete, mctr);
      finished=comp.is_finished();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // test to make sure all particles were tallied over all ranks
    if (comp.get_n_complete_tree() !=  global_count) completion_routine_pass = false;

    // all ranks should be finished
    if (!comp.is_finished()) completion_routine_pass = false;

    // reset for next timestep and test value again
    comp.end_timestep(mctr);

    // after reset, all ranks should not be finished
    if (comp.is_finished()) completion_routine_pass = false;

    // at beginning of timestep count should be zero
    comp.start_timestep(mctr);

    if (comp.get_n_complete_tree() !=  0) completion_routine_pass = false;

    // should post/complete some receives
    if (mctr.n_receives_posted == 0) completion_routine_pass = false;
    if (mctr.n_receives_completed == 0) completion_routine_pass = false;
    if (mctr.n_receives_posted != mctr.n_receives_completed)
      completion_routine_pass = false;

    // shold post/complete no sends
    if (mctr.n_sends_posted != 0) completion_routine_pass = false;
    if (mctr.n_sends_completed != 0) completion_routine_pass = false;
    if (mctr.n_sends_posted != mctr.n_sends_completed) completion_routine_pass = false;

    if (completion_routine_pass) {
      cout<<"TEST PASSED: Completion_Manager_RMA ";
      cout<<"with "<<n_rank<<" ranks"<<endl;
    }
    else {
      cout<<"TEST FAILED: Completion_Manager_RMA with "<<n_rank;
      cout<<" ranks"<<endl;
      nfail++;
    }
    cout<<"RMA Number of requests: "<<mctr.n_receives_completed<<endl;
  }


  // test completion routine for Milagro
  {
    bool milagro_completion_pass = true;

    uint64_t rank_particles = 10000;
    uint64_t global_count = n_rank*rank_particles;
    Completion_Manager_Milagro comp(rank, n_rank);

    //set global particle count
    comp.set_timestep_global_particles(global_count);

    uint64_t rank_complete = 0;
    uint64_t rank_sourced = 0;
    double work = 0.0;
    bool finished = false;

    Message_Counter mctr;

    comp.start_timestep(mctr);

    while(!finished ) {
      // add completed work if there is work to do
      if (rank_sourced <rank_particles) {
        rank_complete++;
        rank_sourced++;
      }
      work+=exp(-rank_complete%8);
      comp.process_completion(true, rank_complete, mctr);
      finished=comp.is_finished();
    }

    // all ranks should be finished
    if (!comp.is_finished()) milagro_completion_pass = false;

    // send completed count down the tree
    comp.end_timestep(mctr);

    // after reset, all ranks should not be finished
    if (comp.is_finished()) milagro_completion_pass = false;

    MPI_Barrier(MPI_COMM_WORLD);

    // should post/complete some receives and post should equal complete
    if (mctr.n_receives_posted == 0) milagro_completion_pass = false;
    if (mctr.n_receives_completed == 0) milagro_completion_pass = false;
    if (mctr.n_receives_posted != mctr.n_receives_completed)
      milagro_completion_pass = false;

    // shold post/complete some sends and post should equal complete
    if (mctr.n_sends_posted == 0) milagro_completion_pass = false;
    if (mctr.n_sends_completed == 0) milagro_completion_pass = false;
    if (mctr.n_sends_posted != mctr.n_sends_completed) milagro_completion_pass = false;

    if (milagro_completion_pass) {
      cout<<"TEST PASSED: ";
      cout<<"Milagro Completion_Manager with "<<n_rank<<" ranks"<<endl;
    }
    else {
      cout<<"TEST FAILED: ";
      cout<<"Milagro Completion_Manager with "<<n_rank<<" ranks"<<endl;
      nfail++;
    }

    cout<<"Milagro number of sends posted: "<<mctr.n_sends_posted<<endl;
    cout<<"Milagro number of sends completed: "<<mctr.n_sends_completed<<endl;
    cout<<"Milagro number of receives posted: "<<mctr.n_receives_posted<<endl;
    cout<<"Milagro number of receives completed: "<<mctr.n_receives_completed<<endl;
  }

  MPI_Finalize();

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_completion_manager.cc
//---------------------------------------------------------------------------//
