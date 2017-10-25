//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   decompose_photons.h
 * \author Alex Long
 * \date   June 17 2015
 * \brief  Load balance census photons after a step in mesh passing DD mode
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef decompose_photons_h_
#define decompose_photons_h_

#include <iostream>
#include <mpi.h>
#include <unordered_map>
#include <algorithm>
#include <vector>

#include "mpi_types.h"
#include "photon.h"


void print_MPI_photons( const std::vector<Photon>& phtn_vec,
                        const uint32_t& rank,
                        const uint32_t& size) {

  using std::cout;

  cout.flush();
  MPI_Barrier(MPI_COMM_WORLD);

  for (uint32_t p_rank = 0; p_rank<size; p_rank++) {
    if (rank == p_rank) {
      for(uint32_t i=0; i<phtn_vec.size();i++)
        phtn_vec[i].print_info(rank);
      cout.flush();
    }
    usleep(100);
    MPI_Barrier(MPI_COMM_WORLD);
    usleep(100);
  }
  usleep(100);
  cout.flush();
  usleep(100);
}


std::vector<Photon> rebalance_census(std::vector<Photon>& off_rank_census,
                                     Mesh* mesh, MPI_Types* mpi_types)
{
  using std::unordered_map;
  using std::sort;
  using std::vector;

  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);
  uint32_t n_off_rank = n_rank-1;

  MPI_Datatype MPI_Particle = mpi_types->get_particle_type();

  // make off processor map
  unordered_map<int,int> proc_map;
  for (int i=0; i<int(n_off_rank); i++) {
    int r_index = i + int(i>=rank);
    proc_map[i] = r_index;
  }

  //sort the census vector by cell ID (global ID)
  sort(off_rank_census.begin(), off_rank_census.end());

  //size of census list
  uint32_t n_census = off_rank_census.size();

  //count the photons belonging to each rank and the start index of each
  //count the ranks that you will send to, add them to a vector
  vector<uint32_t> rank_count(n_rank, 0);
  vector<uint32_t> rank_start(n_rank+1, 0);
  vector<bool> rank_found(n_rank, false);
  uint32_t r;
  for (uint32_t i=0; i<n_census; i++) {
    r = mesh->get_rank(off_rank_census[i].get_cell());
    rank_count[r]++;
    if(rank_found[r]==false) {
      rank_found[r]=true;
      rank_start[r] =i;
    }
  }

  // end of rank count is the total number of census photons
  rank_start[n_rank] = n_census;

  // make requests for non-blocking communication
  MPI_Request* reqs = new MPI_Request[n_off_rank*2];

  // make n_off_rank receive buffers
  vector<vector<Photon> > recv_photons;
  for (uint32_t ir=0; ir<n_off_rank; ir++) {
    vector<Photon> empty_vec;
    recv_photons.push_back(empty_vec);
  }

  //get the number of photons received from each rank
  vector<int> recv_from_rank(n_off_rank, 0);

  for (uint32_t ir=0; ir<n_off_rank; ir++) {
    int off_rank = proc_map[ir];
    MPI_Isend(&rank_count[off_rank], 1,  MPI_UNSIGNED, off_rank, 0,
      MPI_COMM_WORLD, &reqs[ir]);
    MPI_Irecv(&recv_from_rank[ir], 1, MPI_UNSIGNED, off_rank, 0, MPI_COMM_WORLD,
      &reqs[ir+n_off_rank]);
  }

  MPI_Waitall(n_off_rank*2, reqs, MPI_STATUS_IGNORE);

  // now send the buffers and post receives
  // resize receive buffers with recv_from_rank
  for (uint32_t ir=0; ir<n_off_rank; ir++) {
    int off_rank = proc_map[ir];
    int start_copy = rank_start[off_rank];
    MPI_Isend(&off_rank_census[start_copy], rank_count[off_rank], MPI_Particle,
      off_rank, 0, MPI_COMM_WORLD, &reqs[ir]);
    recv_photons[ir].resize(recv_from_rank[ir]);
    MPI_Irecv(&recv_photons[ir][0], recv_from_rank[ir], MPI_Particle, off_rank,
      0, MPI_COMM_WORLD, &reqs[ir+n_off_rank]);
  }

  MPI_Waitall(n_off_rank*2, reqs, MPI_STATUS_IGNORE);

  //free memory from off rank census list
  off_rank_census.clear();

  //copy received census photons to a new census list
  vector<Photon> new_on_rank_census;
  for (uint32_t ir=0; ir<uint32_t(n_rank-1); ir++) {
    new_on_rank_census.insert(new_on_rank_census.end(),
      recv_photons[ir].begin(), recv_photons[ir].end());
  }

  // explicitly delete the MPI requests
  delete[] reqs;

  return new_on_rank_census;
}

#endif // decompose_photons_h
//---------------------------------------------------------------------------//
// end of decompose_photons.h
//---------------------------------------------------------------------------//
