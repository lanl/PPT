//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   tally_manager_rma.h
 * \author Alex Long
 * \date   September, 6 2016
 * \brief  Buffers and communicates tally data on other ranks
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//----------------------------------------------------------------------------//

#ifndef tally_manager_rma_h_
#define tally_manager_rma_h_

#include <iostream>
#include <mpi.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "buffer.h"
#include "constants.h"
#include "message_counter.h"


struct Tally
{
  Tally(void)
  : cell(0),
    abs_E(0)
  {}
  ~Tally(void) {}

  static bool sort_cell_ID(const Tally& compare_1, const Tally& compare_2) {
    return compare_1.cell < compare_2.cell;
  }

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
    const uint32_t n_cell)
  : rank(_rank),
    rank_bounds(_rank_bounds),
    rank_start(rank_bounds[rank]),
    rank_end(rank_bounds[rank+1]),
    max_reqs(100),
    max_tally_size(10000),
    write_size(20)
  {
    // make the MPI window of size n_cell*double and MPI_INFO with 
    // accumulate_ops set to "same_op"
    MPI_Info tally_info;
    MPI_Info_create(&tally_info);
    MPI_Info_set(tally_info, "accumulate_ops", "same_op");
    int mpi_double_size;
    MPI_Type_size(MPI_DOUBLE, &mpi_double_size);
    MPI_Aint n_bytes(n_cell*mpi_double_size);
    MPI_Win_allocate(n_bytes, mpi_double_size, tally_info,
      MPI_COMM_WORLD, &tally, &tally_window);
    MPI_Info_free(&tally_info);

    // initialize tally to zero
    for (uint32_t i=0; i<n_cell;++i)
      tally[i] = 0.0;

    // barrier to make sure the memory location has been updated everywhere
    MPI_Barrier(MPI_COMM_WORLD);

    // send tally variables
    s_tally_reqs = std::vector<MPI_Request> (max_reqs);
    s_tally_buffers = std::vector<Buffer<double> > (max_reqs);
    s_tally_max_index = 0;
    s_tally_count = 0;

    complete_indices = std::vector<int> (max_reqs);

    int assert =0;
    MPI_Win_lock_all(assert,tally_window);
  }

  //! destructor
  ~Tally_Manager() {
    MPI_Win_unlock_all(tally_window);
    MPI_Win_free(&tally_window);
  }

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

  //! Return a pointer to the constant tally data
  double const * get_tally_ptr(void) {return tally;}

  //--------------------------------------------------------------------------//
  // non-const functions                                                      //
  //--------------------------------------------------------------------------//

  private:

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

  //! Test active send and receives request objects for completion (sent IDs
  // and sent tallies)
  void test_remote_writes(Message_Counter& mctr)
  {
    // test sends of tallies, don't test if no active requests
    if (!s_tally_in_use.empty()) {
      MPI_Testsome(s_tally_max_index+1, &s_tally_reqs[0], &n_req_complete,
        &complete_indices[0], MPI_STATUSES_IGNORE);
      for (uint32_t i=0; i<uint32_t(n_req_complete);++i)
        s_tally_in_use.erase(complete_indices[i]);
      mctr.n_sends_completed+=n_req_complete;
    }
  }

  void remote_tally_accumulate(Message_Counter& mctr,
    std::unordered_map<uint32_t, double>& off_rank_abs_E)
  {
    using std::vector;

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

    // clear the map, we'll write it back if we are at maximum number of
    // remote writes
    off_rank_abs_E.clear();

    for (auto map_i=rank_tally.begin(); map_i != rank_tally.end();++map_i) {
      vector<Tally>& send_tallies = map_i->second;

      // sort based on global cell ID
      sort(send_tallies.begin(), send_tallies.end(), Tally::sort_cell_ID);

      off_rank = map_i->first;

      vector<vector<Tally> > grouped_tallies;

      uint32_t start_buffer = send_tallies.front().cell;
      vector<Tally> tallies;
      for (auto t_i=send_tallies.begin(); t_i!=send_tallies.end();++t_i) {
        if (t_i->cell < start_buffer+write_size) {
          tallies.push_back(*t_i);
        }
        else {
          // push this group onto grouped_tallies and reset
          grouped_tallies.push_back(tallies);
          tallies.clear();

          // make this tally the start of a new group
          start_buffer = t_i->cell;
          tallies.push_back(*t_i);
        }
      }
      // add the last block of tallies
      grouped_tallies.push_back(tallies);

      for (uint32_t i=0; i<grouped_tallies.size();++i) {
        vector<double> write_buffer(write_size, 0.0);
        vector<Tally>& tallies = grouped_tallies[i];
        uint32_t remote_start_write = tallies.front().cell;

        // fill buffer
        for (auto t_i=tallies.begin(); t_i != tallies.end(); ++t_i)
          write_buffer[t_i->cell - remote_start_write] = t_i->abs_E;

        // write this buffer to remote window, if there are available buffers
        if (s_tally_in_use.size() < max_reqs) {

          // get next available ID request and buffer, fill buffer
          uint32_t s_tally_index = get_next_send_tally_request_and_buffer_index();
          s_tally_buffers[s_tally_index].fill(write_buffer);

          int32_t buffer_size = write_buffer.size();
          if (remote_start_write + buffer_size >= rank_bounds[off_rank+1])
            buffer_size = rank_bounds[off_rank+1] - remote_start_write;

          // transform the remote_start_write to a local index for that rank
          remote_start_write -= rank_bounds[off_rank];

          MPI_Raccumulate(s_tally_buffers[s_tally_index].get_buffer(),
            buffer_size, MPI_DOUBLE, off_rank, remote_start_write, buffer_size,
            MPI_DOUBLE, MPI_SUM, tally_window, &s_tally_reqs[s_tally_index]);
        }

        // no available buffer, put this data back in off_rank_abs_E
        else {
          uint32_t cell_ID = remote_start_write;
          for (uint32_t j=0; j<write_buffer.size(); ++j) {
            if (write_buffer[j] != 0.0)
              off_rank_abs_E[cell_ID] = write_buffer[j];
            cell_ID++;
          }
        }

      } // end loop over grouped tallies

    } // end loop over ranks
  }

  public:

  void process_off_rank_tallies(Message_Counter& mctr,
    std::unordered_map<uint32_t, double>& off_rank_abs_E,
    const bool force_send)
  {
    // first, test sends and receives of tally data
    test_remote_writes(mctr);

    // then send off-rank tally data if map is full
    if (off_rank_abs_E.size() > max_tally_size || force_send)
      remote_tally_accumulate(mctr, off_rank_abs_E);
  }


  void finish_remote_writes(Message_Counter& mctr,
    std::unordered_map<uint32_t, double>& off_rank_abs_E)
  {
    while(!off_rank_abs_E.empty() || !s_tally_in_use.empty()  ) {
      // first, test sends and receives of tally data
      test_remote_writes(mctr);

      // always send in finish mode
      remote_tally_accumulate(mctr, off_rank_abs_E);
    }
  }

  //! End timestep by resetting active indices and request counts
  void end_timestep(void) {
    s_tally_max_index=0;
    s_tally_count=0;
  }

  private:
  int rank; //! MPI rank
  std::vector<uint32_t> rank_bounds; //! Global tally ID bounds on each rank

  uint32_t rank_start; //! Index of first tally
  uint32_t rank_end; //! Index of one after last tally

  const uint32_t max_reqs; //! Maximum number of concurrent requests
  const uint32_t max_tally_size; //! Maximum number of tallies to write
  const uint32_t write_size; //! Number of indices to accumulate in remote write

  double *tally; //! Pointer to be allocated by MPI_Win_Allocate
  MPI_Win tally_window; //! Window for tally

  // send tally variables
  std::vector<MPI_Request> s_tally_reqs;
  std::vector<Buffer<double> > s_tally_buffers;
  std::unordered_set<uint32_t> s_tally_in_use;
  uint32_t s_tally_max_index;
  uint32_t s_tally_count;

  //! Returned from MPI_Testsome, indicates completed requests at index
  std::vector<int> complete_indices;

  int n_req_complete; //! Number of completed requests after MPI_Testsome
};

#endif // def tally_manager_rma_h_

//----------------------------------------------------------------------------//
// end of tally_manager_rma.h
//----------------------------------------------------------------------------//
