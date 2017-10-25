//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   remap_census.h
 * \author Alex Long
 * \date   August 30 2016
 * \brief  Send census particles back to the ranks that own the mesh they need
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef remap_census_h_
#define remap_census_h_

#include <algorithm>
#include <bitset>
#include <mpi.h>
#include <unordered_map>
#include <vector>

#include "info.h"
#include "photon.h"
#include "mpi_types.h"

//! Get the rank ID of a given cell index
uint32_t get_rank(const std::vector<uint32_t>& rank_bounds, 
  const uint32_t index)
{
  //find rank of index
  bool found = false;
  uint32_t min_i = 0;
  uint32_t max_i = rank_bounds.size()-1;
  uint32_t s_i; //search index
  while(!found) {
    s_i =(max_i + min_i)/2;
    if (s_i == max_i || s_i == min_i) found = true;
    else if (index >= rank_bounds[s_i]) min_i = s_i;
    else max_i = s_i;
  }
  return s_i;
}

//! At iteration k, return the communication partner for a rank in binary 
// tree communication pattern
uint32_t get_pairing(const int32_t rank, const int32_t n_rank, const int32_t k) {
  uint32_t r_partner;
  std::bitset<32> b_rank(rank);
  // if kth bit is 0, add 2**k otherwise subtract 2**k
  r_partner = rank + (-2*int(b_rank[k]) + 1)*pow(2,k);

  // check the bounds
  if (r_partner < 0 || r_partner > n_rank-1 )
    r_partner = MPI_PROC_NULL;

  return r_partner;
}

//! Determine the ``forwarding address'' for rank from off_rank (in binary
// tree algorithms processors do not communicate directly)
uint32_t get_send_rank(const int32_t rank, const int32_t off_rank, const int32_t n_rank) {

  // take exclusive or of the two numbers, the position of the highest 
  // significant bit is the level of the binary tree where these two ranks 
  // effectively interact
  uint32_t x = uint32_t(rank)^uint32_t(off_rank);

  // find the position of the highest significant bit 
  int k = 32;
  if (!x)
    return 0;
  if (!(x & 0xffff0000u)) {
    x <<= 16;
    k -= 16;
  }
  if (!(x & 0xff000000u)) {
    x <<= 8;
    k -= 8;
  }
  if (!(x & 0xf0000000u)) {
    x <<= 4;
    k -= 4;
  }
  if (!(x & 0xc0000000u)) {
    x <<= 2;
    k -= 2;
  }
  if (!(x & 0x80000000u)) {
    x <<= 1;
    k -= 1;
  }

  // bits differ at k, use k-1 to get zero based numbering

  // get the communication partner at this level for this rank
  return get_pairing(rank, n_rank, k-1);
}

//! Set the two map objects that aid in sending photons to their communication
// partners
void set_census_send_maps(const uint32_t rank, const uint32_t n_rank,
  const std::vector<uint32_t>& rank_bounds, const std::vector<Photon>& census,
  std::unordered_map<uint32_t, uint32_t>& census_on_rank, 
  std::unordered_map<uint32_t, uint32_t>& census_start_index)
{
  using std::vector;

  uint32_t i=0;
  uint32_t send_rank, p_rank, cell_ID;
  for (vector<Photon>::const_iterator iphtn =census.cbegin();
    iphtn!=census.cend(); ++iphtn)
  {
    cell_ID = iphtn->get_cell();
    p_rank = get_rank(rank_bounds, cell_ID);
    if (p_rank == 3) {
      uint32_t x=0;
    }
    if (p_rank != rank) 
      send_rank = get_send_rank(rank, p_rank, n_rank);
    else
      send_rank = rank;

    // at first instance of send_rank set the starting index
    if (census_on_rank.find(send_rank) == census_on_rank.end()) {
      census_on_rank[send_rank] = 1;
      census_start_index[send_rank]=i;
    }
    else {
      census_on_rank[send_rank]++;
    }
    i++;
  }
}


//! Use a binary tree approach to send census particles back to their
// owners
std::vector<Photon> rebalance_census(std::vector<Photon>& off_rank_census, 
  uint64_t& rank_photons, const std::vector<uint32_t>& rank_bounds,
  MPI_Types* mpi_types, const Info& mpi_info)
{
  using std::vector;
  using std::unordered_map;

  const int n_tag(100);
  const int phtn_tag(200);

  int rank = mpi_info.get_rank();
  int n_rank = mpi_info.get_n_rank();

  MPI_Datatype MPI_Particle = mpi_types->get_particle_type();

  // split the communicator based on the node number
  MPI_Comm local_comm;
  MPI_Comm_split(MPI_COMM_WORLD, mpi_info.get_color(), rank, &local_comm);
  int n_rank_local;
  MPI_Comm_size(local_comm, &n_rank_local); 

  vector<Photon> new_on_rank_census;

  // add the off_rank census to the number of node photons
  rank_photons += off_rank_census.size();

  // sort the census vector by cell ID (global ID)
  sort(off_rank_census.begin(), off_rank_census.end());

  // do the binary tree pattern to the nearest log2(rank), rounding down
  int32_t n_levels = int32_t(log2(n_rank));
  uint32_t max_rank = pow(2, n_levels) - 1;

  // make n_levels of send buffers
  vector< vector<Photon> > send_buffers(n_levels);
  
  unordered_map<uint32_t, uint32_t> census_on_rank;
  unordered_map<uint32_t, uint32_t> census_start_index;

  // map the off-rank photons to their correct receiving rank
  set_census_send_maps(rank, n_rank, rank_bounds, 
    off_rank_census, census_on_rank, census_start_index);


  uint32_t send_rank, start_index, count;
  for (uint32_t i=0;i<n_levels;++i) {
    send_rank = get_pairing(rank, n_rank, i);
    // if send_rank is in the map, add photons to buffer
    if (census_on_rank.find(send_rank) != census_on_rank.end()) {
      start_index = census_start_index[send_rank];
      count = census_on_rank[send_rank];
      send_buffers[i].insert(send_buffers[i].begin(),
        off_rank_census.begin() + start_index,
        off_rank_census.begin() + start_index + count);
    }
  }
  off_rank_census.clear();
 

  // try to manage the memory at the node level, allow 75% of the node memory 
  // to be photons
  uint64_t node_photons = 0;
  uint64_t n_recv_node = 0;
  uint64_t max_node_photons = uint64_t(
    0.75*(mpi_info.get_node_mem()/sizeof(Photon)));

  uint32_t r_partner;
  uint64_t  n_max_recv, n_max_send, n_recv, n_node_recv, n_send;
  vector<Photon> recv_buffer;
  std::vector<MPI_Request> reqs(4);

  // get the current number of photons on this node
  MPI_Allreduce(&rank_photons, &node_photons, 1, MPI_UNSIGNED_LONG, MPI_SUM,
      local_comm);

  // run the binary tree comm pattern in reverse, going from largest
  // communication distance to smallest--this gets particles as close to their
  // owner as possible in a step-by-step procedure
  for (int32_t k=n_levels-1; k>=0;--k) {

    // get current communication partner
    r_partner = get_pairing(rank, n_rank, k);
    n_send = send_buffers[k].size();

    // send and receive number of photons that will be communicated
    MPI_Isend(&n_send, 1, MPI_UNSIGNED_LONG, r_partner, n_tag, MPI_COMM_WORLD, 
      &reqs[0]);
    MPI_Irecv(&n_recv, 1, MPI_UNSIGNED_LONG, r_partner, n_tag, MPI_COMM_WORLD, 
      &reqs[1]);
    MPI_Waitall(2, &reqs[0], MPI_STATUS_IGNORE);

    // check to see how many photons this node can receive, if it will
    // overrun the limits, only allow proc to receive 1/n_rank_local
    // of the remaining particles
    n_node_recv = 0;
    MPI_Allreduce(&n_recv, &n_node_recv, 1, MPI_UNSIGNED_LONG, MPI_SUM,
        local_comm);
    if (node_photons > max_node_photons) {
      n_max_recv = 0;
    }
    else if (node_photons + n_node_recv > max_node_photons) {
      n_max_recv = (max_node_photons - node_photons)/n_rank_local;
    }
    else n_max_recv = max_node_photons;

    // send and receive max number you can receive and send
    MPI_Isend(&n_max_recv, 1, MPI_UNSIGNED_LONG, r_partner, n_tag, MPI_COMM_WORLD,
      &reqs[0]);
    MPI_Irecv(&n_max_send, 1, MPI_UNSIGNED_LONG, r_partner, n_tag, MPI_COMM_WORLD,
      &reqs[1]);
    MPI_Waitall(2, &reqs[0], MPI_STATUS_IGNORE);

    // send and receive photons
    if (n_max_send < n_send) n_send = n_max_send;
    if (n_recv > n_max_recv) n_recv = n_max_recv;

    recv_buffer.resize(n_recv);
    MPI_Isend(&send_buffers[k][0], n_send, MPI_Particle, r_partner, phtn_tag,
      MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&recv_buffer[0], n_recv, MPI_Particle, r_partner, phtn_tag,
      MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, &reqs[0], MPI_STATUS_IGNORE);

    // update the number of photons on this rank and get new node photon count
    rank_photons = rank_photons - n_send + n_recv;
    node_photons =0;
    MPI_Allreduce(&rank_photons, &node_photons, 1, MPI_UNSIGNED_LONG, MPI_SUM,
        local_comm);

    // if not all photons were sent, add them to this rank's census
    new_on_rank_census.insert(new_on_rank_census.end(),
      send_buffers[k].begin() + n_send, send_buffers[k].end());
    send_buffers[k].clear();

    // sort the received census vector by cell ID (global ID)
    sort(recv_buffer.begin(), recv_buffer.end());

    // find the destination of these new photons
    unordered_map<uint32_t, uint32_t> recv_census_on_rank;
    unordered_map<uint32_t, uint32_t> recv_census_start_index;
    set_census_send_maps(rank, n_rank, rank_bounds, recv_buffer, 
      recv_census_on_rank, recv_census_start_index);

    // first copy out the photons on this rank
    if (recv_census_on_rank.find(rank) != recv_census_on_rank.end()) {
        start_index = recv_census_start_index[rank];
        count = recv_census_on_rank[rank];
        new_on_rank_census.insert(new_on_rank_census.end(),
          recv_buffer.begin() + start_index, 
          recv_buffer.begin() + start_index + count);
    }

    // now copy out photons going to other ranks to 
    for (uint32_t i=0;i<n_levels;++i) {
      send_rank = get_pairing(rank, n_rank, i);
      if (recv_census_on_rank.find(send_rank) != recv_census_on_rank.end()) {
        start_index = recv_census_start_index[send_rank];
        count = recv_census_on_rank[send_rank];
        send_buffers[i].insert(send_buffers[i].begin(),
          recv_buffer.begin() + start_index,
          recv_buffer.begin() + start_index + count);
      }
    }

  } // end for loop over levels
  return new_on_rank_census;
}

#endif // remap_census_h_
