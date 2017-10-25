//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   completion_manager_milagro.h
 * \author Alex Long
 * \date   March 31 2016
 * \brief  Two-sided transport completion manager
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef completion_manager_milagro_h_
#define completion_manager_milagro_h_

#include <iostream>
#include <mpi.h>
#include <vector>

#include "buffer.h"
#include "completion_manager.h"
#include "constants.h"


//==============================================================================
/*!
 * \class Completion_Manager_Milagro
 * \brief The number of completed particles is sent up the chain and then reset.
 * This allows us to send the completed count up the tree without trying to
 * synchronize completion from both children. The root never resets the tree
 * count
 *
 * \example no test yet
 */
//==============================================================================
class Completion_Manager_Milagro : public Completion_Manager
{
  public:
  Completion_Manager_Milagro(const int& rank, const int& n_rank)
    : Completion_Manager(rank,n_rank)
  {
    // buffers will always receive only one 64 bit integer
    c1_recv_buffer.resize(1);
    c2_recv_buffer.resize(1);
    p_recv_buffer.resize(1);
  }
  virtual ~Completion_Manager_Milagro() {}

  //non-const functions
  virtual void start_timestep(Message_Counter& mctr) {
    using Constants::count_tag;
    // Messages are sent up the tree whenever a rank has completed its local work
    // or received an updated particle complete count from its child
    // Messages are sent down the tree only after completion and starting at the
    // root node.
    // Post receives for photon counts from children and parent now
    if (child1!=MPI_PROC_NULL) {
      MPI_Irecv(c1_recv_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, child1,
        count_tag, MPI_COMM_WORLD, &c1_recv_req);
      mctr.n_receives_posted++;
      c1_recv_buffer.set_awaiting();
    }
    if (child2!=MPI_PROC_NULL) {
      MPI_Irecv(c2_recv_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, child2,
        count_tag, MPI_COMM_WORLD, &c2_recv_req);
      mctr.n_receives_posted++;
      c2_recv_buffer.set_awaiting();
    }
    if (parent != MPI_PROC_NULL) {
      MPI_Irecv(p_recv_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, parent,
        count_tag, MPI_COMM_WORLD, &p_recv_req);
      mctr.n_receives_posted++;
      p_recv_buffer.set_awaiting();
    }
  }

  //! Check for completed particle counts from children and parent.
  // Add children to current tree count. If parent count is received,
  // it will be the global problem particle count, indicating completion
  void check_messages(uint64_t& n_complete_tree, Message_Counter& mctr) {
    using Constants::count_tag;

    //test receives from children and add work to tree count
    if (c1_recv_buffer.awaiting()) {
      MPI_Test(&c1_recv_req, &flag_c1, MPI_STATUS_IGNORE);
      if (flag_c1) {
        mctr.n_receives_completed++;
        c1_recv_buffer.set_received();
        n_complete_c1 = c1_recv_buffer.get_object()[0];
        //update tree count
        n_complete_tree+=n_complete_c1;
        //post receive again
        c1_recv_buffer.reset();
        MPI_Irecv(c1_recv_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, child1,
          count_tag, MPI_COMM_WORLD, &c1_recv_req);
        mctr.n_receives_posted++;
        c1_recv_buffer.set_awaiting();
      }
    }
    if (c2_recv_buffer.awaiting()) {
      MPI_Test(&c2_recv_req, &flag_c2, MPI_STATUS_IGNORE);
      if (flag_c2) {
        mctr.n_receives_completed++;
        c2_recv_buffer.set_received();
        n_complete_c2 = c2_recv_buffer.get_object()[0];
        //update tree count
        n_complete_tree+=n_complete_c2;
        //post receive again
        c2_recv_buffer.reset();
        MPI_Irecv(c2_recv_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, child2,
          count_tag, MPI_COMM_WORLD, &c2_recv_req);
        mctr.n_receives_posted++;
        c2_recv_buffer.set_awaiting();
      }
    }

    // test receive from parent (indicates completion of time step)
    if (p_recv_buffer.awaiting()) {
      MPI_Test(&p_recv_req, &flag_p, MPI_STATUS_IGNORE);
      if (flag_p) {
        mctr.n_receives_completed++;
        p_recv_buffer.set_received();
        n_complete_p = p_recv_buffer.get_object()[0];
        finished = true;
      }
    }

    // test sends to parent
    if (p_send_buffer.sent() ) {
      MPI_Test(&p_send_req, &flag_p, MPI_STATUS_IGNORE);
      if (flag_p) {
        mctr.n_sends_completed++;
        p_send_buffer.reset();
      }
    }
  }

  //! For the Milagro method, send the completed count up the tree
  virtual void process_completion(bool waiting_for_work,
                                  uint64_t& n_complete_tree,
                                  Message_Counter& mctr)
  {
    using std::vector;
    using Constants::count_tag;

    check_messages(n_complete_tree, mctr);

    // non-root ranks send complete counts up the tree
    if ((n_complete_tree && waiting_for_work) && (parent!=MPI_PROC_NULL &&
      !p_send_buffer.sent()) ) {
      p_send_buffer.fill(vector<uint64_t> (1,n_complete_tree));
      MPI_Isend(p_send_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, parent,
        count_tag, MPI_COMM_WORLD, &p_send_req);
      mctr.n_sends_posted++;
      //n_complete_messages++;
      p_send_buffer.set_sent();
      //reset tree count so work is not double counted
      n_complete_tree =0;
    }
    // root checks for completion
    else {
      if (n_complete_tree == n_particle_global) finished=true;
    }
  }

  //! Send finish message down the tree and finish off messages
  virtual void end_timestep(Message_Counter& mctr)
  {
    using std::vector;
    using Constants::count_tag;

    // finish off sends and send empty messages to complete awaiting receives
    //send finished count down tree to children and wait for completion
    if (child1 != MPI_PROC_NULL) {
      if (c1_send_buffer.sent()) {
        MPI_Wait(&c1_send_req, MPI_STATUS_IGNORE);
        mctr.n_sends_completed++;
      }
      c1_send_buffer.fill(vector<uint64_t> (1,n_particle_global));
      MPI_Isend(c1_send_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, child1,
        count_tag, MPI_COMM_WORLD, &c1_send_req);
      mctr.n_sends_posted++;
      MPI_Wait(&c1_send_req, MPI_STATUS_IGNORE);
      mctr.n_sends_completed++;
    }
    if (child2 != MPI_PROC_NULL)  {
      if (c2_send_buffer.sent()) {
        MPI_Wait(&c2_send_req, MPI_STATUS_IGNORE);
        mctr.n_sends_completed++;
      }
      c2_send_buffer.fill(vector<uint64_t> (1,n_particle_global));
      MPI_Isend(c2_send_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, child2,
        count_tag, MPI_COMM_WORLD, &c2_send_req);
      mctr.n_sends_posted++;
      MPI_Wait(&c2_send_req, MPI_STATUS_IGNORE);
      mctr.n_sends_completed++;
    }

    // wait for parent send to complete, if sent then finish off
    // parent's receive calls
    if (parent!=MPI_PROC_NULL) {
      if (p_send_buffer.sent()) {
        MPI_Wait(&p_send_req, MPI_STATUS_IGNORE);
        mctr.n_sends_completed++;
      }
      p_send_buffer.fill(vector<uint64_t> (1,1));
      MPI_Isend(p_send_buffer.get_buffer(), 1, MPI_UNSIGNED_LONG, parent,
        count_tag, MPI_COMM_WORLD, &p_send_req);
      mctr.n_sends_posted++;
      MPI_Wait(&p_send_req, MPI_STATUS_IGNORE);
      mctr.n_sends_completed++;
    }

    if (child1 != MPI_PROC_NULL) {
      MPI_Wait(&c1_recv_req, MPI_STATUS_IGNORE);
      mctr.n_receives_completed++;
    }
    if (child2 != MPI_PROC_NULL) {
      MPI_Wait(&c2_recv_req, MPI_STATUS_IGNORE);
      mctr.n_receives_completed++;
    }

    //reset tree counts
    n_particle_global = 0;
    n_complete_c1 = 0;
    n_complete_c2 = 0;
    n_complete_p = 0;

    // reset finished flag
    finished = false;
  }

  private:
  Buffer<uint64_t> c1_recv_buffer;
  Buffer<uint64_t> c2_recv_buffer;
  Buffer<uint64_t> p_recv_buffer;
  Buffer<uint64_t> c1_send_buffer;
  Buffer<uint64_t> c2_send_buffer;
  Buffer<uint64_t> p_send_buffer;
  MPI_Request p_recv_req;
  MPI_Request c1_recv_req;
  MPI_Request c2_recv_req;
  MPI_Request p_send_req;
  MPI_Request c1_send_req;
  MPI_Request c2_send_req;
};

#endif // def completion_manager_milagro_h_
//---------------------------------------------------------------------------//
// end of completion_manager_milagro.h
//---------------------------------------------------------------------------//
