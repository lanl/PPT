//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   tally_manager.h
 * \author Alex Long
 * \date   September, 6 2016
 * \brief  Buffers and communicates tally data on other ranks 
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//----------------------------------------------------------------------------//

#ifndef tally_manager_h_
#define tally_manager_h_

#include <iostream>
#include <mpi.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "buffer.h"
#include "constants.h"
#include "mpi_types.h"
#include "message_counter.h"


struct Tally
{
  Tally(void)
  : cell(0),
    abs_E(0)
  {}
  ~Tally(void) {}

  uint32_t cell;
  double abs_E;
};


//==============================================================================
/*!
 * \class Tally_Manager
 * \brief Processes absorbed energy by remote particles on local mesh and sends
 * absorbed energy from local particles corresponding to non-local mesh
 * \example no test yet
 */
//==============================================================================


class Tally_Manager
{

  public:
  //! constructor
  Tally_Manager(const int _rank, const std::vector<uint32_t>& _rank_bounds, 
    MPI_Types * mpi_types)
  : rank(_rank),
    rank_bounds(_rank_bounds),
    rank_start(rank_bounds[rank]),
    rank_end(rank_bounds[rank+1]), 
    max_reqs(100),
    max_tally_size(10000),
    MPI_Tally(mpi_types->get_tally_type())
  {
    using std::vector;

    // send ID variables
    s_id_reqs = vector<MPI_Request> (max_reqs);
    s_id_buffers = vector<Buffer<uint32_t> >(max_reqs);
    s_id_max_index = 0;
    s_id_count = 0;
    
    // send abs_E variables
    s_tally_reqs = vector<MPI_Request> (max_reqs);
    s_tally_buffers = vector<Buffer<Tally> > (max_reqs);
    s_tally_max_index = 0;
    s_tally_count = 0;

    // receive ID variables
    r_id_reqs = vector<MPI_Request> (max_reqs);
    r_id_buffers = vector<Buffer<uint32_t> >(max_reqs);
    r_id_status = vector<MPI_Status> (max_reqs);

    // receive tally variables
    r_tally_reqs = vector<MPI_Request> (max_reqs);
    r_tally_buffers = vector<Buffer<Tally> > (max_reqs);
    r_tally_max_index = 0;
    r_tally_count = 0;

    complete_indices = vector<int> (max_reqs);
  }

  //! destructor
  ~Tally_Manager() {}

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//

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

  //! Returns the index of the next available send tally request and buffer
  uint32_t get_next_send_tally_request_and_buffer_index(void) 
  {
    // check to see if request at count is in use
    while(s_tally_in_use.find(s_tally_count) != s_tally_in_use.end() ) {
      s_tally_count++;
      if (s_tally_count==max_reqs) s_tally_count=0;
    }
    s_tally_max_index = std::max(s_tally_count,s_tally_max_index);

    // record this index as in use
    s_tally_in_use.insert(s_tally_count);

    return s_tally_count;
  }

  //! Return the index of the next available receive tally request and buffer
  uint32_t get_next_receive_tally_request_and_buffer_index(void)
  {
    // check to see if request at count is in use
    while(r_tally_in_use.find(r_tally_count) != r_tally_in_use.end() ) {
      r_tally_count++;
      if (r_tally_count==max_reqs) r_tally_count=0;
    }
    r_tally_max_index = std::max(r_tally_count,r_tally_max_index);

    // record this index as in use
    r_tally_in_use.insert(r_tally_count);

    return r_tally_count;
  }

  //! Test active send and receives request objects for completion (sent IDs 
  // and sent tallies)
  void test_sends_and_receives(Message_Counter& mctr,
    std::vector<double>& rank_abs_E)
  {
    using Constants::n_tally_tag;
    using Constants::tally_tag;
    using std::vector;

    // test sends of tally IDs, don't test if no active requests
    if (!s_id_in_use.empty()) {
      MPI_Testsome(s_id_max_index+1, &s_id_reqs[0], &n_req_complete,
        &complete_indices[0], MPI_STATUSES_IGNORE);

      for (uint32_t i=0; i<n_req_complete;++i)
        s_id_in_use.erase(complete_indices[i]);
      mctr.n_sends_completed+=n_req_complete;
    }

    // test sends of tallies, don't test if no active requests
    if (!s_tally_in_use.empty()) {
      MPI_Testsome(s_tally_max_index+1, &s_tally_reqs[0], &n_req_complete,
        &complete_indices[0], MPI_STATUSES_IGNORE);
      for (uint32_t i=0; i<n_req_complete;++i)
        s_tally_in_use.erase(complete_indices[i]);
      mctr.n_sends_completed+=n_req_complete;
    }

    // test receives for number of IDs being sent by other ranks
    MPI_Testsome(max_reqs, &r_id_reqs[0], &n_req_complete,
      &complete_indices[0], &r_id_status[0]);
    
    int comp_index, n_ids;
    uint32_t g_index, off_rank; 
    mctr.n_receives_completed+=n_req_complete;

    // for each complete request, post receive for all tallies
    for (int i = 0;i<n_req_complete;++i) {
      comp_index = complete_indices[i];
      // get number of ids to be received
      off_rank = r_id_status[i].MPI_SOURCE;
      n_ids = r_id_buffers[comp_index].get_object()[0];

      // post receive for tallies from this rank
      uint32_t r_tally_index = 
        get_next_receive_tally_request_and_buffer_index();
      r_tally_buffers[r_tally_index].set_receive_size(n_ids);
      int custom_tag = tally_tag + n_ids;

      MPI_Irecv(r_tally_buffers[r_tally_index].get_buffer(), n_ids, MPI_Tally,
        off_rank, tally_tag, MPI_COMM_WORLD, &r_tally_reqs[r_tally_index]);

      // repost the receive at this index
      MPI_Irecv(r_id_buffers[comp_index].get_buffer(), 1, MPI_UNSIGNED,
        MPI_ANY_SOURCE, n_tally_tag, MPI_COMM_WORLD, &r_id_reqs[comp_index]);
      mctr.n_receives_posted++;
    }

    // test tally data receives
    if (!r_tally_in_use.empty()) {

      MPI_Testsome(r_tally_max_index+1, &r_tally_reqs[0], &n_req_complete,
        &complete_indices[0], MPI_STATUSES_IGNORE);
 
      int n_tally_recv;
      int tallys_in_req;

      mctr.n_receives_completed+=n_req_complete;
      // for each complete request, add the tallies to local absorbed E
      for (int i = 0;i<n_req_complete;++i) {
        comp_index = complete_indices[i];

        vector<Tally>& recv_tally = r_tally_buffers[comp_index].get_object();
        n_tally_recv = recv_tally.size();

        // remove request index from index_in_use set
        r_tally_in_use.erase(comp_index);

        uint32_t l_index;
        for (uint32_t i=0; i<n_tally_recv; ++i) {
          l_index = recv_tally[i].cell - rank_start;
          rank_abs_E[l_index] += recv_tally[i].abs_E;
        }

      } // end loop over received requests
    } // end if !r_tally_in_use.empty()
  }


  void send_tally_data(Message_Counter& mctr,
    std::unordered_map<uint32_t, double>& off_rank_abs_E)
  {
    using std::vector;
    using Constants::tally_tag;
    using Constants::n_tally_tag;

    std::unordered_map<uint32_t, vector<Tally> > rank_tally;
    uint32_t off_rank;
    vector<Tally> temp_tally_vec(1);
    Tally tally;
    for( auto map_i=off_rank_abs_E.begin(); map_i!=off_rank_abs_E.end();++map_i)
    {
      off_rank = get_off_rank_id(map_i->first);
      tally.cell = map_i->first;
      tally.abs_E = map_i->second;
      if (rank_tally.find(off_rank) == rank_tally.end()) {
        temp_tally_vec[0] = tally;
        rank_tally[off_rank] = temp_tally_vec;
      }
      else {
        rank_tally[off_rank].push_back(tally);
      }
    }

    
    for (auto map_i=rank_tally.begin(); map_i != rank_tally.end();++map_i) {
      vector<Tally>& send_tallies = map_i->second;
      uint32_t n_ids = send_tallies.size();

      // get next available ID request and buffer, fill buffer, post send
      uint32_t s_id_index = get_next_send_id_request_and_buffer_index();
      s_id_buffers[s_id_index].fill(vector<uint32_t> (1, n_ids));

      MPI_Isend(s_id_buffers[s_id_index].get_buffer(), 1, MPI_UNSIGNED, 
        off_rank, n_tally_tag, MPI_COMM_WORLD, &s_id_reqs[s_id_index]);
      
      // get next available tally request and buffer, fill buffer, post send
      uint32_t s_tally_index = get_next_send_tally_request_and_buffer_index();
      s_tally_buffers[s_tally_index].fill(send_tallies);

      MPI_Isend(s_tally_buffers[s_tally_index].get_buffer(), n_ids, MPI_Tally, 
        off_rank, tally_tag, MPI_COMM_WORLD, &s_tally_reqs[s_tally_index]);
    }
    off_rank_abs_E.clear();
  }

  public:
  void process_off_rank_tallies(Message_Counter& mctr,
    std::vector<double>& rank_abs_E, 
    std::unordered_map<uint32_t, double>& off_rank_abs_E, 
    const bool force_send)
  {
    // first, test sends and receives of tally data
    test_sends_and_receives(mctr, rank_abs_E);

    // then send off-rank tally data if map is full
    if (off_rank_abs_E.size() > max_tally_size || force_send) 
      send_tally_data(mctr, off_rank_abs_E);
  }

  //! Begin simulation by posting receives for the number of IDs that will be sent
  // by other ranks
  void start_simulation(Message_Counter& mctr) {
    using Constants::n_tally_tag;
    for (uint32_t i=0; i<max_reqs;++i) {
      MPI_Irecv(r_id_buffers[i].get_buffer(), 1, MPI_UNSIGNED, MPI_ANY_SOURCE,
        n_tally_tag, MPI_COMM_WORLD, &r_id_reqs[i]);
      mctr.n_receives_posted++;
    }
  }

  //! End timestep by resetting active indices and request counts
  void end_timestep(void) {
    s_id_max_index =0;
    s_id_count=0;
    s_tally_max_index=0;
    s_tally_count=0;
    r_tally_max_index=0;
    r_tally_count=0;
  }

  //! End simulation by canceling pending receives requests
  void end_simulation(Message_Counter& mctr)
  {
    for (uint32_t i=0; i<max_reqs;++i)
      MPI_Cancel(&r_id_reqs[i]);

    MPI_Waitall(max_reqs, &r_id_reqs[0], MPI_STATUSES_IGNORE);

    mctr.n_receives_completed+=max_reqs;
  }

  private:
  int rank; //! MPI rank
  std::vector<uint32_t> rank_bounds; //! Global tally ID bounds on each rank

  uint32_t rank_start; //! Index of first tally
  uint32_t rank_end; //! Index of one after last tally 

  const uint32_t max_reqs; //! Maximum number of concurrent requests
  const uint32_t max_tally_size; //! Maximum number of tallies to write 

  MPI_Datatype MPI_Tally; //! Custom MPI datatype for tally

  // send id variables
  std::vector<MPI_Request> s_id_reqs;
  std::vector<Buffer<uint32_t> > s_id_buffers;
  std::unordered_set<uint32_t> s_id_in_use;
  uint32_t s_id_max_index;
  uint32_t s_id_count;

  // send tally variables
  std::vector<MPI_Request> s_tally_reqs;
  std::vector<Buffer<Tally> > s_tally_buffers;
  std::unordered_set<uint32_t> s_tally_in_use;
  uint32_t s_tally_max_index;
  uint32_t s_tally_count;

  //receive id variables
  std::vector<MPI_Request> r_id_reqs;
  std::vector<Buffer<uint32_t> > r_id_buffers;
  std::vector<MPI_Status> r_id_status;

  // receive tally variables  
  std::vector<MPI_Request> r_tally_reqs;
  std::vector<Buffer<Tally> > r_tally_buffers;
  std::unordered_set<uint32_t> r_tally_in_use;
  uint32_t r_tally_max_index;
  uint32_t r_tally_count;

  //! Returned from MPI_Testsome, indicates completed requests at index
  std::vector<int> complete_indices;

  int n_req_complete; //! Number of completed requests after MPI_Testsome
};

#endif // def tally_manager_h_

//----------------------------------------------------------------------------//
// end of tally_manager.h
//----------------------------------------------------------------------------//
