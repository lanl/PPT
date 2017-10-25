//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   completion_manager_rma.h
 * \author Alex Long
 * \date   March 3 2016
 * \brief  One-sided transport completion manager
 *
 * Manages completion of domain-decomposed IMC transport with one-sided
 * messaging and MPI windows. Uses a binary tree communication pattern
 *
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef completion_manager_rma_h_
#define completion_manager_rma_h_

#include <iostream>
#include <mpi.h>

#include "completion_manager.h"
#include "constants.h"


//==============================================================================
/*!
 * \class Completion_Manager_RMA
 * \brief Manages completion of particle transport with one-sided messages
 *
 * \example no test yet
 */
//==============================================================================

class Completion_Manager_RMA : public Completion_Manager
{

  public:

  //! constructor
  Completion_Manager_RMA(const int& rank, const int& n_rank)
    : Completion_Manager(rank, n_rank),
      buffer_c1(0),
      buffer_c2(0),
      buffer_p(0),
      c1_req_flag(false),
      c2_req_flag(false),
      p_req_flag(false)
  {
    // Get the size of MPI_UNSIGNED_LONG
    int size_mpi_uint64;
    MPI_Type_size(MPI_UNSIGNED_LONG, &size_mpi_uint64);

    //MPI_Win_set_attr(completion_window, MPI_WIN_MODEL, MPI_WIN_UNIFIED);

    // Make the MPI window for the number of particles completed on this
    // sub-tree, which includes this ranks completed particles

    MPI_Win_allocate(size_mpi_uint64, size_mpi_uint64, MPI_INFO_NULL,
      MPI_COMM_WORLD, &n_complete_tree_data, &completion_window);

    int flag;
    MPI_Win_get_attr(completion_window, MPI_WIN_MODEL, &memory_model, &flag);

    // open MPI window for one-sided messaging
    int assert =0;
    MPI_Win_lock_all(assert,completion_window);

  }

  //! destructor
  virtual ~Completion_Manager_RMA() {
    // closes MPI window for one-sided messaging
    MPI_Win_unlock_all(completion_window);
  }

  // const functions

  //! Get the type of memory model used by the MPI implementation on this system
  int get_mpi_window_memory_type(void) const {
    return *memory_model;
  }

  //! Get the number of completed particles
  uint64_t get_n_complete_tree(void) const {
    return *n_complete_tree_data;
  }

  //! No posting of receives is necessary so don't do anything
  virtual void start_timestep(Message_Counter& mctr) {
    *n_complete_tree_data=0;
  }

  // non-const functions

  //! Resets all particle counts and finishes open requests (send counts are
  // not used)
  virtual void end_timestep( Message_Counter& mctr)
  {
    //reset tree counts
    //*n_complete_tree_data = 0;
    n_complete_c1 = 0;
    n_complete_c2 = 0;
    n_complete_p = 0;
    buffer_c1 = 0;
    buffer_c2 = 0;
    buffer_p = 0;

    // reset finished flag
    finished = false;

    //wait on outstanding requests, parent has already completed
    if (c1_req_flag) {
      MPI_Wait(&req_c1, MPI_STATUS_IGNORE);
      c1_req_flag = false;
      mctr.n_receives_completed++;
    }
    if (c2_req_flag) {
      MPI_Wait(&req_c2, MPI_STATUS_IGNORE);
      c2_req_flag = false;
      mctr.n_receives_completed++;
    }
  }

  //! Add number of completed particles to this tree count and get the number
  // of completed particles by your children (n_sends_posted is not used
  virtual void process_completion(bool waiting_for_work,
                                  uint64_t& n_complete_tree,
                                  Message_Counter& mctr)
  {

    // only do this if rank has no particles to transport and received
    // buffers are empty
    if (waiting_for_work) {
      // Test for completion of non-blocking RMA requests
      // If child requests were completed, add to tree complete count and
      // make a new request

      // child 1
      if (child1!= MPI_PROC_NULL) {
        if (c1_req_flag) {
          MPI_Test(&req_c1, &flag_c1, MPI_STATUS_IGNORE);
          if (flag_c1) {
            n_complete_c1 = buffer_c1;
            c1_req_flag = false;
            mctr.n_receives_completed++;
          }
        }
        if (!c1_req_flag) {
          MPI_Rget(&buffer_c1, 1, MPI_UNSIGNED_LONG, child1, 0,
            1, MPI_UNSIGNED_LONG, completion_window, &req_c1);
          c1_req_flag=true;
          mctr.n_receives_posted++;
        }
      }

      // child 2
      if (child2!= MPI_PROC_NULL) {
        if (c2_req_flag) {
          MPI_Test(&req_c2, &flag_c2, MPI_STATUS_IGNORE);
          if (flag_c2) {
            n_complete_c2 = buffer_c2;
            c2_req_flag = false;
            mctr.n_receives_completed++;
          }
        }
        if (!c2_req_flag) {
          MPI_Rget(&buffer_c2, 1, MPI_UNSIGNED_LONG, child2, 0,
            1, MPI_UNSIGNED_LONG, completion_window, &req_c2);
          c2_req_flag=true;
          mctr.n_receives_posted++;
        }
      }

      // update total count for this tree
      *n_complete_tree_data =n_complete_tree + n_complete_c1 + n_complete_c2;

      // If parent is complete, test parent count for overall completion
      if (parent != MPI_PROC_NULL) {
        if (p_req_flag) {
          MPI_Test(&req_p, &flag_p, MPI_STATUS_IGNORE);
          if (flag_p) {
            n_complete_p = buffer_p;
            // If complete, set finished to true
            if (n_complete_p == n_particle_global) {
              finished=true;
              *n_complete_tree_data = n_complete_p;
            }
            p_req_flag =false;
            mctr.n_receives_completed++;
          }
        }
        if (!p_req_flag && !finished) {
          MPI_Rget(&buffer_p, 1, MPI_UNSIGNED_LONG, parent, 0,
            1, MPI_UNSIGNED_LONG, completion_window, &req_p);
          p_req_flag=true;
          mctr.n_receives_posted++;
        }
      }
      // If root, test tree count for completion
      else {
        if (*n_complete_tree_data == n_particle_global) finished=true;
      }

    } //end if waiting_for_work
  }

  private:
  //! Pointer to complete tree, allocated by MPI_Mem_Alloc with a size of 1
  uint64_t *n_complete_tree_data;
  uint64_t buffer_c1; //! Buffer to receive first child's tree count
  uint64_t buffer_c2; //! Buffer to receive second child's tree count
  uint64_t buffer_p; //! Buffer to receive parent's tree count
  bool c1_req_flag; //! Active request flag for first child
  bool c2_req_flag; //! Active request flag for second child
  bool p_req_flag; //! Active request flag for parent
  MPI_Win completion_window; //! MPI window to complete count
  MPI_Request req_c1; //! MPI request for first child
  MPI_Request req_c2; //! MPI request for second child
  MPI_Request req_p; //!  MPI request for parent
  int *memory_model; //! Memory model of MPI window (implementation dependent)
};

#endif // def completion_manager_rma_h_
//---------------------------------------------------------------------------//
// end of completion_manager_rma.h
//---------------------------------------------------------------------------//
