//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   load_balance.h
 * \author Alex Long
 * \date   May 2 2016
 * \brief  Function for determining balance and communicating work
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//----------------------------------------------------------------------------//

#ifndef load_balance_h_
#define load_balance_h_

#include <iostream>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "buffer.h"
#include "constants.h"
#include "mpi_types.h"
#include "photon.h"
#include "work_packet.h"

//! Load balance the work on all ranks given an array of work packets and
// census particles
void load_balance(const int& rank, const int& n_rank,
  const uint64_t n_particle_on_rank, std::vector<Work_Packet>& work,
  std::vector<Photon>& census_list, MPI_Types *mpi_types)
{
  using std::unordered_map;
  using std::unordered_set;
  using std::vector;
  using Constants::photon_tag;
  using Constants::work_tag;

  // get MPI datatypes
  MPI_Datatype MPI_Particle = mpi_types->get_particle_type();

  MPI_Datatype MPI_WPacket = mpi_types->get_work_packet_type();

  //--------------------------------------------------------------------------//
  // Calculate load imbalance for each rank
  //--------------------------------------------------------------------------//

  vector<int64_t> domain_particles(n_rank, 0);
  vector<int64_t> domain_delta_p(n_rank, 0);

  domain_particles[rank] = n_particle_on_rank;

  MPI_Allreduce(MPI_IN_PLACE, &domain_particles[0], n_rank, MPI_UNSIGNED_LONG,
    MPI_SUM, MPI_COMM_WORLD);

  int64_t n_global_particles =
    std::accumulate(domain_particles.begin(), domain_particles.end(),0);

  int64_t n_balance_particles = n_global_particles/n_rank;

  for (uint32_t ir=0; ir<uint32_t(n_rank); ir++)
    domain_delta_p[ir] = domain_particles[ir] - n_balance_particles;

  //--------------------------------------------------------------------------//
  // Determine which ranks to send or reiceve work from
  //--------------------------------------------------------------------------//

  // first, ignore particle imbalance on the order of 5%
  for (uint32_t ir=0; ir<uint32_t(n_rank); ir++) {
    // ignore particle imbalance on the order of 5%
    if ( domain_delta_p[ir]<0 && domain_particles[ir] > 0.95*n_balance_particles)
      domain_delta_p[ir] = 0;
  }

  unordered_map<int32_t, int32_t> n_send_rank;
  vector<uint32_t> work_donor_ranks;

  int64_t max_particles = 0;
  uint32_t max_rank = 0;

  double overload_factor = 0.0; // 40%

  int send_to_nbr, rank_delta_p, left_node, right_node;
  for (uint32_t ir=0; ir<uint32_t(n_rank); ir++) {
    if (domain_particles[ir] > max_particles) {
      max_particles=domain_particles[ir];
      max_rank = ir;
    }
    rank_delta_p = domain_delta_p[ir];

    // allow overload on donor ranks by effectively reducing the number of
    // particles they can donate (signed integer math)
    if (rank_delta_p>0) {
      rank_delta_p = domain_particles[ir] -
        (1.0 + overload_factor)*n_balance_particles;
    }

    left_node = ir-1;
    right_node = ir+1;
    while (rank_delta_p > 0 && (left_node>=0 || right_node<n_rank)) {
      if (left_node >= 0) {
        if (domain_delta_p[left_node] < 0 && rank_delta_p > 0) {
          send_to_nbr = abs(domain_delta_p[left_node]);
          if (send_to_nbr > rank_delta_p) send_to_nbr = rank_delta_p;

          // if this is your rank, remember it in your map
          if (uint32_t(rank)==ir) n_send_rank[left_node] = send_to_nbr;

          // if you are going to receive work, remember donor rank
          if (rank==left_node) work_donor_ranks.push_back(ir);

          domain_delta_p[left_node]+=send_to_nbr;
          rank_delta_p -= send_to_nbr;
        }
        left_node--;
      } // if left_node >= 0
      if (right_node < n_rank) {
        if (domain_delta_p[right_node] < 0 && rank_delta_p > 0) {
          send_to_nbr = abs(domain_delta_p[right_node]);
          if (send_to_nbr > rank_delta_p) send_to_nbr = rank_delta_p;

          // if this is your rank, remember it in your map
          if (uint32_t(rank)==ir) n_send_rank[right_node] = send_to_nbr;
          // if you are going to receive work, remember donor rank

          if (rank==right_node) work_donor_ranks.push_back(ir);

          domain_delta_p[right_node]+=send_to_nbr;
          rank_delta_p -= send_to_nbr;
        }
        right_node++;
      } // if ir+dist < n_rank
    } //end while
  } // end for ir in n_rank

  if (rank==0) {
    std::cout<<"Load balancing "<<n_global_particles<<" particles"<<std::endl;
    std::cout<<"Max particles: "<<max_particles<<" on rank "<<max_rank<<", ";
    std::cout<<100.0*double(max_particles)/double(n_global_particles);
    std::cout<<" percent of total work"<<std::endl;
  }

  //--------------------------------------------------------------------------//
  // Receive work from donating ranks
  //--------------------------------------------------------------------------//

  // make MPI requests for the number of work sources
  uint32_t n_donors = work_donor_ranks.size();
  if (n_donors) {
    MPI_Request *work_recv_request = new MPI_Request[n_donors];
    MPI_Request *phtn_recv_request = new MPI_Request[n_donors];
    MPI_Status *work_recv_status = new MPI_Status[n_donors];
    MPI_Status *phtn_recv_status = new MPI_Status[n_donors];
    vector<int32_t> n_work_recv(n_donors);
    vector<int32_t> n_phtn_recv(n_donors);
    vector<Buffer<Photon> > photon_buffer(n_donors);
    vector<Buffer<Work_Packet> > work_buffer(n_donors);
    // get the total number of particles deficient on this rank to size buffers
    uint32_t n_deficient = n_balance_particles - n_particle_on_rank;

    // post receives from ranks that are sending you work
    for (uint32_t i=0; i<n_donors; i++) {
      // make buffer maximum possible size
      work_buffer[i].resize(n_deficient);

      // post work packet receives
      MPI_Irecv(work_buffer[i].get_buffer(), n_deficient, MPI_WPacket,
        work_donor_ranks[i], work_tag, MPI_COMM_WORLD, &work_recv_request[i]);

      // make buffer maximum possible size
      photon_buffer[i].resize(n_deficient);

      // post particle receives
      MPI_Irecv(photon_buffer[i].get_buffer(), n_deficient, MPI_Particle,
        work_donor_ranks[i], photon_tag, MPI_COMM_WORLD, &phtn_recv_request[i]);
    }

    // wait on all requests
    MPI_Waitall(n_donors, work_recv_request, work_recv_status);
    MPI_Waitall(n_donors, phtn_recv_request, phtn_recv_status);

    for (uint32_t i=0; i<n_donors; i++) {
      // get received count for work and photons from this rank
      MPI_Get_count(&work_recv_status[i], MPI_WPacket, &n_work_recv[i]);
      MPI_Get_count(&phtn_recv_status[i], MPI_Particle, &n_phtn_recv[i]);

      // add received work to your work
      vector<Work_Packet> temp_work = work_buffer[i].get_object();
      work.insert(work.begin(), temp_work.begin(),
        temp_work.begin() + n_work_recv[i]);

      // add received census photons
      vector<Photon> temp_photons = photon_buffer[i].get_object();
      census_list.insert(census_list.begin(), temp_photons.begin(),
        temp_photons.begin() + n_phtn_recv[i]);
    }
    delete[] work_recv_request;
    delete[] phtn_recv_request;
    delete[] work_recv_status;
    delete[] phtn_recv_status;

  } // end if(n_donors)

  //--------------------------------------------------------------------------//
  // Send work to accepting ranks
  //--------------------------------------------------------------------------//

  uint32_t n_acceptors =  n_send_rank.size();
  if (n_acceptors) {
    // sort work to put larger work packets near the end of the vector
    sort(work.begin(), work.end());

    MPI_Request *work_send_request = new MPI_Request[n_acceptors];
    MPI_Request *phtn_send_request = new MPI_Request[n_acceptors];
    uint32_t dest_rank;
    uint32_t temp_n_send;
    uint32_t ireq=0;  //! Request index
    uint32_t start_cut_index = 0; //! Begin slice of census list
    uint32_t n_census_remain = census_list.size(); //! Where to cut census list

    // iterate over unordered map and prepare work for other ranks
    for (unordered_map<int32_t,int32_t>::iterator map_itr=n_send_rank.begin();
      map_itr !=n_send_rank.end(); map_itr++)
    {

      dest_rank = map_itr->first;
      temp_n_send = map_itr->second;

      vector<Work_Packet> work_to_send;
      Work_Packet temp_packet, leftover_packet;
      // first, try to send work packets to the other ranks
      while (!work.empty() && temp_n_send > 0) {
        temp_packet = work.back();
        // split work if it's larger than the number needed
        if (temp_packet.get_n_particles() > temp_n_send) {
          leftover_packet = temp_packet.split(temp_n_send);
          work[work.size()-1] = leftover_packet;
        }
        // otherwise, pop the temp work packet off the stack
        else work.pop_back();

        // add packet to send list
        work_to_send.push_back(temp_packet);
        // subtract particle in packet from temp_n_send
        temp_n_send -= temp_packet.get_n_particles();
      }

      // send census particles instead (not preferred, these photons are likely
      // to travel farther and thus require more memory)
      uint32_t n_send_census =0;
      if (temp_n_send > 0 && n_census_remain > temp_n_send) {
        n_send_census = temp_n_send;
        start_cut_index = n_census_remain - n_send_census;
        n_census_remain -= n_send_census;
      }

      // send both work and photon vectors, even if they're empty
      MPI_Isend(&work_to_send[0], work_to_send.size(), MPI_WPacket,
        dest_rank, work_tag, MPI_COMM_WORLD, &work_send_request[ireq]);
      MPI_Isend(&census_list[start_cut_index], n_send_census, MPI_Particle,
        dest_rank, photon_tag, MPI_COMM_WORLD, &phtn_send_request[ireq]);

      // wait on this request so buffers are not invalidated when going
      // out of loop scope
      MPI_Wait(&work_send_request[ireq], MPI_STATUS_IGNORE);
      MPI_Wait(&phtn_send_request[ireq], MPI_STATUS_IGNORE);

      // increment the request index counter
      ireq++;
    }

    // remove census photons that were sent off
    census_list.erase(census_list.begin() + n_census_remain,
      census_list.end());

    // clean up
    delete[] work_send_request;
    delete[] phtn_send_request;

  } // end if(n_acceptors)

}

#endif // load_balance_h_

//----------------------------------------------------------------------------//
// end of load_balance.h
//----------------------------------------------------------------------------//
