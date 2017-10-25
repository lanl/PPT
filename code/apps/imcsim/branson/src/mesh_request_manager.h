//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   mesh_request_manager.h
 * \author Alex Long
 * \date   August 10, 2016
 * \brief  Manages two-sided requests for remote mesh data
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//----------------------------------------------------------------------------//

#ifndef mesh_request_manager_h_
#define mesh_request_manager_h_

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "buffer.h"
#include "cell.h"
#include "constants.h"
#include "mpi_types.h"


//==============================================================================
/*!
 * \class RMA_Manager
 * \brief Makes and processes requests for remote mesh data
 *
 * \example no test yet
 */
//==============================================================================

class Mesh_Request_Manager
{

  public:
  //! constructor
  Mesh_Request_Manager(const int& _rank,
    const std::vector<uint32_t>& _rank_bounds, const uint32_t n_global_cell,
    const uint32_t _grip_size, MPI_Types * mpi_types, const Cell * const _cells)
  : rank(_rank),
    rank_bounds(_rank_bounds),
    MPI_Cell(mpi_types->get_cell_type()),
    cells(_cells),
    grip_size(_grip_size),
    rank_start(rank_bounds[rank]),
    rank_end(rank_bounds[rank+1]),
    max_reqs(100),
    max_ids(100)
  {
    using std::vector;

    // send ID variables
    s_id_reqs = vector<MPI_Request> (max_reqs);
    s_id_buffers = vector<Buffer<uint32_t> >(max_reqs);
    s_id_max_index = 0;
    s_id_count = 0;

    // send cell variables
    s_cell_reqs = vector<MPI_Request> (max_reqs);
    s_cell_buffers = vector<Buffer<Cell> > (max_reqs);
    s_cell_max_index = 0;
    s_cell_count = 0;

    // receive ID variables
    r_id_reqs = vector<MPI_Request> (max_reqs);
    r_id_buffers = vector<Buffer<uint32_t> >(max_reqs);
    r_id_status = vector<MPI_Status> (max_reqs);

    // receive cell variables
    r_cell_reqs = vector<MPI_Request> (max_reqs);
    r_cell_buffers = vector<Buffer<Cell> > (max_reqs);
    r_cell_status = vector<MPI_Status> (max_reqs);
    r_cell_max_index = 0;
    r_cell_count = 0;

    // resize buffers for receiving IDs to max ID size
    for (uint32_t i=0;i<max_reqs;++i)
      r_id_buffers[i].resize(max_ids);

    complete_indices = vector<int> (max_reqs);
    n_requests_vec= vector<uint32_t> (n_global_cell,0);
  }

  //! destructor
  ~Mesh_Request_Manager() {}

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//

  //! Check to see if mesh has already been requestsed
  bool mesh_is_requested(uint32_t g_index) const {
    return ids_requested.find(g_index) != ids_requested.end();
  }

  //! Search rank bounds to get remote mesh owner's rank
  uint32_t get_off_rank_id(const uint32_t& g_index) const {
    //find rank of index
    bool found = false;
    uint32_t min_i = 0;
    uint32_t max_i = rank_bounds.size()-1;
    uint32_t s_i; //search index
    while(!found) {
      s_i =(max_i + min_i)/2;
      if (s_i == max_i || s_i == min_i) found = true;
      else if (g_index >= rank_bounds[s_i]) min_i = s_i;
      else max_i = s_i;
    }
    return s_i;
  }

  //--------------------------------------------------------------------------//
  // non-const functions                                                      //
  //--------------------------------------------------------------------------//

  private:

  //! Returns the index of the next available send ID request and buffer
  uint32_t get_next_send_id_request_and_buffer_index(void)
  {
    // check to see if request at count is in use
    while(s_id_in_use.find(s_id_count) != s_id_in_use.end() ) {
      s_id_count++;
      if (s_id_count==max_reqs) s_id_count=0;
    }
    s_id_max_index = std::max(s_id_count,s_id_max_index);

    // record this index as in use
    s_id_in_use.insert(s_id_count);

    return s_id_count;
  }

  //! Returns the index of the next available send cell request and buffer
  uint32_t get_next_send_cell_request_and_buffer_index(void)
  {
    // check to see if request at count is in use
    while(s_cell_in_use.find(s_cell_count) != s_cell_in_use.end() ) {
      s_cell_count++;
      if (s_cell_count==max_reqs) s_cell_count=0;
    }
    s_cell_max_index = std::max(s_cell_count,s_cell_max_index);

    // record this index as in use
    s_cell_in_use.insert(s_cell_count);

    return s_cell_count;
  }

  //! Return the index of the next available receive cell request and buffer
  uint32_t get_next_receive_cell_request_and_buffer_index(void)
  {
    // check to see if request at count is in use
    while(r_cell_in_use.find(r_cell_count) != r_cell_in_use.end() ) {
      r_cell_count++;
      if (r_cell_count==max_reqs) r_cell_count=0;
    }
    r_cell_max_index = std::max(r_cell_count,r_cell_max_index);

    // record this index as in use
    r_cell_in_use.insert(r_cell_count);

    return r_cell_count;
  }

  //! Test active send and receives request objects for completion (sent IDs and sent cells)
  void test_sends_and_receives(Message_Counter& mctr) {
    using Constants::cell_id_tag;

    // test sends of cell IDs, don't test if no active requests
    if (!s_id_in_use.empty()) {
      MPI_Testsome(s_id_max_index+1, &s_id_reqs[0], &n_req_complete,
        &complete_indices[0], MPI_STATUSES_IGNORE);

      for (uint32_t i=0; i<uint32_t(n_req_complete);++i)
        s_id_in_use.erase(complete_indices[i]);
      mctr.n_sends_completed+=n_req_complete;
    }

    // test sends of cells, don't test if no active requests
    if (!s_cell_in_use.empty()) {
      MPI_Testsome(s_cell_max_index+1, &s_cell_reqs[0], &n_req_complete,
        &complete_indices[0], MPI_STATUSES_IGNORE);
      for (uint32_t i=0; i<uint32_t(n_req_complete);++i)
        s_cell_in_use.erase(complete_indices[i]);
      mctr.n_sends_completed+=n_req_complete;
    }

    // test receives of local IDs needed by other ranks
    MPI_Testsome(max_reqs, &r_id_reqs[0], &n_req_complete,
      &complete_indices[0], &r_id_status[0]);

    int comp_index, n_ids;
    mctr.n_receives_completed+=n_req_complete;

    // for each complete request, add cells to send lists
    for (int i = 0;i<n_req_complete;++i) {
      // get actual number of ids received
      comp_index = complete_indices[i];
      MPI_Get_count(&r_id_status[i], MPI_UNSIGNED, &n_ids);
      r_id_buffers[comp_index].set_rank(r_id_status[i].MPI_SOURCE);
      r_id_buffers[comp_index].set_receive_size(n_ids);
      r_id_stack.push(r_id_buffers[comp_index]);

      // repost the receive at this index
      MPI_Irecv(r_id_buffers[comp_index].get_buffer(), max_reqs, MPI_UNSIGNED,
        MPI_ANY_SOURCE, cell_id_tag, MPI_COMM_WORLD, &r_id_reqs[comp_index]);
      mctr.n_receives_posted++;
    }

    // test receives for remote mesh
    new_cells.clear();

    // test all receives for needed cells
    if (!r_cell_in_use.empty()) {

      MPI_Testsome(r_cell_max_index+1, &r_cell_reqs[0], &n_req_complete,
        &complete_indices[0], &r_cell_status[0]);

      int n_cell_recv;

      mctr.n_receives_completed+=n_req_complete;
      // for each complete request, send the cells
      for (int i = 0;i<n_req_complete;++i) {
        comp_index = complete_indices[i];

        MPI_Get_count(&r_cell_status[i], MPI_Cell, &n_cell_recv);

        // remove request index from index_in_use set
        r_cell_in_use.erase(comp_index);

        // copy actual number of received cells from the buffer to the
        // new_cells vector
        std::vector<Cell>& complete_cells =
          r_cell_buffers[comp_index].get_object();
        new_cells.insert(new_cells.begin(), complete_cells.begin(),
          complete_cells.begin() + n_cell_recv);

        // erase the receives IDs from the request list
        const std::vector<uint32_t>& receive_ids =
          r_cell_buffers[comp_index].get_grip_IDs();
        for (uint32_t j=0;j<receive_ids.size();++j) {
          ids_requested.erase(receive_ids[j]);
        }
      } // end loop over received requests
    } // end if !r_cell_in_use.empty()
  }

  //! Post sends and receives for needed mesh data
  void process_requests_for_remote_mesh(Message_Counter& mctr) {
    using Constants::cell_id_tag;
    using Constants::cell_tag;
    uint32_t off_rank;

    // can't erase from a set while iterating over it, use this
    std::unordered_set<uint32_t> ranks_to_erase;

    for (auto irank=ranks_to_recv.begin();
      irank!=ranks_to_recv.end();++irank)
    {
      if (s_id_in_use.size() < max_reqs && r_cell_in_use.size() < max_reqs) {
        off_rank = *irank;
        auto rank_range = s_rank_to_ids.equal_range(off_rank);
        std::vector<uint32_t> send_ids;
        for (auto it=rank_range.first; it!=rank_range.second; ++it)
          send_ids.push_back(it->second);

        // store request and buffer for testing, post needed ID sends
        uint32_t s_id_index = get_next_send_id_request_and_buffer_index();
        s_id_buffers[s_id_index].fill(send_ids);

        MPI_Isend(s_id_buffers[s_id_index].get_buffer(), send_ids.size(),
          MPI_UNSIGNED, off_rank, cell_id_tag, MPI_COMM_WORLD,
          &s_id_reqs[s_id_index]);
        mctr.n_sends_posted++;

        // size the receive buffer to maximum possible size
        uint32_t r_cell_index =
          get_next_receive_cell_request_and_buffer_index();
        uint32_t recv_size = grip_size*send_ids.size();
        r_cell_buffers[r_cell_index].resize(recv_size);

        // post receive with number of grips in the tag and store request for
        // testing
        int custom_tag = cell_tag + send_ids.size();
        r_cell_buffers[r_cell_index].set_grip_IDs(send_ids);

        MPI_Irecv(r_cell_buffers[r_cell_index].get_buffer(), recv_size, MPI_Cell,
          off_rank, custom_tag, MPI_COMM_WORLD, &r_cell_reqs[r_cell_index]);
        mctr.n_receives_posted++;

        ranks_to_erase.insert(off_rank);
        s_rank_to_ids.erase(off_rank);
      } // end if requests available
    } // end for off_rank in ranks_to_recv

    // delete the ranks that were requested from the set
    for (auto irank=ranks_to_erase.begin();
      irank!=ranks_to_erase.end();++irank)
    {
      ranks_to_recv.erase(*irank);
    }
  }

  //! If requests for local mesh have been made, send the mesh to the
  // requesting rank
  void process_requests_for_local_mesh(Message_Counter& mctr) {
    using Constants::cell_tag;

    // can't erase from a set while iterating over it, use this
    std::unordered_set<uint32_t> ranks_to_erase;

    uint32_t off_rank, n_ids, g_index;
    uint32_t r_id_stack_size = r_id_stack.size();
    Buffer<uint32_t> id_buffer;

    for (uint32_t i=0; i<r_id_stack_size; ++i) {
      if (s_cell_in_use.size() < max_reqs) {

        // pop a received ID buffer off stack
        id_buffer = r_id_stack.front();
        r_id_stack.pop();

        off_rank = id_buffer.get_rank();
        n_ids = id_buffer.get_receive_size();
        std::vector<uint32_t> send_ids = id_buffer.get_object();

        // pop off elements to get to actual receive size
        while (send_ids.size() > n_ids) send_ids.pop_back();

        // fill vector with all cells requested by off rank
        std::vector<Cell> send_cells;
        send_cells.resize(n_ids*grip_size);
        uint32_t copy_index = 0;
        uint32_t n_cells_to_send = 0;
        uint32_t start_index;
        for (uint32_t j=0; j<n_ids; ++j) {
          g_index = send_ids[j];
          start_index = g_index;
          if (int32_t(start_index) - int32_t(grip_size)/2 < int32_t(rank_start))
            start_index = rank_start;
          else
            start_index -= grip_size/2;

          uint32_t n_cells_to_copy;
          if (start_index + grip_size > rank_end)
            n_cells_to_copy = rank_end - start_index;
          else
            n_cells_to_copy = grip_size;

          // transform start index to local cell index
          start_index-=rank_start;

          if (g_index > rank_bounds.back())
            std::cout<<"this is bad: g index > bounds"<<std::endl;
          uint32_t n_bytes = sizeof(Cell)*n_cells_to_copy;
          n_cells_to_send+=n_cells_to_copy;
          memcpy(&send_cells[copy_index],&cells[start_index], n_bytes);
          copy_index+=n_cells_to_copy;
        } // end grip cell copy

        // truncate send vector to actual size
        while(send_cells.size() > n_cells_to_send) send_cells.pop_back();

        // get next available request and buffer, fill buffer, post send
        uint32_t s_cell_index = get_next_send_cell_request_and_buffer_index();
        s_cell_buffers[s_cell_index].fill(send_cells);

        // send with a custom tag with the number of grips in message
        int custom_tag = cell_tag + n_ids;

        MPI_Isend(s_cell_buffers[s_cell_index].get_buffer(), n_cells_to_send,
          MPI_Cell, off_rank, custom_tag, MPI_COMM_WORLD,
          &s_cell_reqs[s_cell_index]);
        mctr.n_sends_posted++;
        mctr.n_cell_messages++;
        mctr.n_cells_sent+=n_cells_to_send;

      } // end if send_cell requests available
    } // end loop over stack of received ID buffers
  }

  public:
  std::vector<Cell>& process_mesh_requests(Message_Counter& mctr) {
    // first, test sends
    test_sends_and_receives(mctr);

    // test requests for local (on-rank) mesh data and send it out
    process_requests_for_local_mesh(mctr);

    // send needed IDs to other ranks and post receive for data
    process_requests_for_remote_mesh(mctr);

    return new_cells;
  }

  //! Add requested cell to maps and sets for later request
  void request_cell(const uint32_t& g_index, Message_Counter& mctr) {
    typedef std::pair<uint32_t, uint32_t> rpair;
    // store cell for requesting if not already stored
    if (g_index > rank_bounds.back())
      std::cout<<"this is bad: g index > bounds"<<std::endl;
    if (!mesh_is_requested(g_index)) {
      // get the rank of the global index, add to set if there are fewer than
      // max_ids for that rank
      uint32_t off_rank_id = get_off_rank_id(g_index);
      if (s_rank_to_ids.count(off_rank_id) < max_ids) {
        s_rank_to_ids.insert(rpair(off_rank_id, g_index));
        ids_requested.insert(g_index);
        ranks_to_recv.insert(off_rank_id);
      }
    }
  }

  //! Begin timestep by getting updated cell data and posting receives for
  // local IDs needed by other ranks
  void start_simulation(Message_Counter& mctr) {
    using Constants::cell_id_tag;
    for (uint32_t i=0; i<max_reqs;++i) {
      MPI_Irecv(r_id_buffers[i].get_buffer(), max_ids, MPI_UNSIGNED, MPI_ANY_SOURCE,
        cell_id_tag, MPI_COMM_WORLD, &r_id_reqs[i]);
      mctr.n_receives_posted++;
    }
  }

  //! End timestep by resetting active indices and request counts
  void end_timestep(void) {
    s_id_max_index =0;
    s_id_count=0;
    s_cell_max_index=0;
    s_cell_count=0;
    r_cell_max_index=0;
    r_cell_count=0;
    for(uint32_t i=0; i<n_requests_vec.size();i++) n_requests_vec[i]=0;
  }

  //! End simulation by cancelling pending receives requests
  void end_simulation(Message_Counter& mctr)
  {
    for (uint32_t i=0; i<max_reqs;++i)
      MPI_Cancel(&r_id_reqs[i]);

    MPI_Waitall(max_reqs, &r_id_reqs[0], MPI_STATUSES_IGNORE);

    mctr.n_receives_completed+=max_reqs;
  }

  //! Return vector of requets counts (used only for plotting SILO file)
  std::vector<uint32_t> get_n_request_vec(void) {return n_requests_vec;}

  //! If IDs requested is empty that means all outstanding requests for remote
  // mesh have completed
  bool no_active_requests(void) {return ids_requested.empty();}

  private:
  int rank; //! MPI rank
  std::vector<uint32_t> rank_bounds; //! Global cell ID bounds on each rank
  MPI_Datatype MPI_Cell; //! Custom MPI datatype for cells
  const Cell * const cells; //! Cell data to sent to other ranks

  //! Global maximum grip size (number of cells in a request)
  uint32_t grip_size;

  uint32_t rank_start; //! Inex of first cell
  uint32_t rank_end; //! Index of one after last cell

  const uint32_t max_reqs; //! Maximum number of concurrent requests
  const uint32_t max_ids; //! Maximum number of IDs in a request

  // send id variables
  std::vector<MPI_Request> s_id_reqs;
  std::vector<Buffer<uint32_t> > s_id_buffers;
  std::unordered_set<uint32_t> s_id_in_use;
  uint32_t s_id_max_index;
  uint32_t s_id_count;

  // send cell variables
  std::vector<MPI_Request> s_cell_reqs;
  std::vector<Buffer<Cell> > s_cell_buffers;
  std::unordered_set<uint32_t> s_cell_in_use;
  uint32_t s_cell_max_index;
  uint32_t s_cell_count;

  //receive id variables
  std::vector<MPI_Request> r_id_reqs;
  std::vector<Buffer<uint32_t> > r_id_buffers;
  std::vector<MPI_Status> r_id_status;

  // receive cell variables
  std::vector<MPI_Request> r_cell_reqs;
  std::vector<Buffer<Cell> > r_cell_buffers;
  std::vector<MPI_Status> r_cell_status;
  std::unordered_set<uint32_t> r_cell_in_use;
  uint32_t r_cell_max_index;
  uint32_t r_cell_count;

  //! Stack of buffers to process IDs
  std::queue<Buffer<uint32_t> > r_id_stack;

  //! Number of times a cell was requested
  std::vector<uint32_t> n_requests_vec;

  //! Returned from MPI_Testsome, indicates completed requests at index
  std::vector<int> complete_indices;

  int n_req_complete; //! Number of completed requests after MPI_Testsome

  std::vector<Cell> new_cells; //! New cells after MPI_Testsome

  //! Stores global IDs of requested cells
  std::unordered_set<uint32_t> ids_requested;

  //! Ranks to receive mesh data from
  std::unordered_set<uint32_t> ranks_to_recv;

  std::unordered_set<uint32_t> ranks_to_send; //! Ranks to send mesh data to

  //! Key is remote rank, value is non-local ID needed by this rank
  std::unordered_multimap<uint32_t, uint32_t> s_rank_to_ids;
};

#endif // def mesh_request_manager_h_

//----------------------------------------------------------------------------//
// end of mesh_request_manager.h
//----------------------------------------------------------------------------//
