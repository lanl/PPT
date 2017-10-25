//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   completion_manager.h
 * \author Alex Long
 * \date   May 18 2016
 * \brief  Virtual base class for completion routines
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef completion_manager_h_
#define completion_manager_h_

#include <iostream>
#include <mpi.h>
#include <vector>

#include "constants.h"
#include "message_counter.h"

//==============================================================================
/*!
 * \class Completion_Manager
 * \brief Virtual base class for completion routines
 *
 * \example no test yet
 */
//==============================================================================
class Completion_Manager
{

  public:
  Completion_Manager(const int& rank,
                     const int& n_rank)
    : n_complete_c1(0),
      n_complete_c2(0),
      n_complete_p(0),
      n_particle_global(0),
      finished(false)
  {
    //set up binary tree rank structure
    parent = (rank + 1) / 2 - 1;
    child1 = rank * 2 + 1;
    child2 = child1 + 1;
    // set missing nodes to MPI_PROC_NULL
    if (!rank) parent = MPI_PROC_NULL;

    // maximum valid node id
    const int last_node = n_rank - 1;

    if (child1 > last_node) {
      child1 = MPI_PROC_NULL;
      child2 = MPI_PROC_NULL;
    }
    else if (child1 == last_node) child2 = MPI_PROC_NULL;

  }
  virtual ~Completion_Manager() {}

  //non-const functions
  void set_timestep_global_particles(uint64_t _n_particle_global) {
    n_particle_global = _n_particle_global;
  }

  bool is_finished(void) {return finished;}

  virtual void start_timestep(Message_Counter& mctr) =0;

  virtual void end_timestep(Message_Counter& mctr) = 0;

  virtual void process_completion(bool waiting_for_work,
                                  uint64_t& n_complete_tree,
                                  Message_Counter& mctr) = 0;

  protected:
  uint64_t n_complete_c1; //! Completed particles in first child's tree
  uint64_t n_complete_c2; //! Completed particles in second child's tree
  uint64_t n_complete_p; //! Completed particles in parent's tree
  uint64_t n_particle_global; //! Total particles across all ranks
  bool finished; //! Finished with transport flag
  int child1; //! Rank ID of first child
  int child2; //! Rank ID of second child
  int parent; //! Rank ID of parent
  int flag_c1; //! Return flag for MPI test of first child
  int flag_c2; //! Return flag for MPI test of second child
  int flag_p; //! Return flag for MPI test of parent
};

#endif // completion_manager_h_
//---------------------------------------------------------------------------//
// end of completion_manager.h
//---------------------------------------------------------------------------//
