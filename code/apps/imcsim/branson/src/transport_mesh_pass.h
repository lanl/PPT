//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   transport_mesh_pass.h
 * \author Alex Long
 * \date   June 6 2015
 * \brief  Transport routine using two sided messaging and mesh-passing DD
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef transport_mesh_pass_h_
#define transport_mesh_pass_h_

#include <algorithm>
#include <vector>
#include <numeric>
#include <queue>
#include <mpi.h>

#include "completion_manager_milagro.h"
#include "completion_manager_rma.h"
#include "mesh_request_manager.h"
#include "constants.h"
#include "decompose_photons.h"
#include "info.h"
#include "message_counter.h"
#include "mesh.h"
#include "mpi_types.h"
#include "RNG.h"
#include "sampling_functions.h"

//! Transport a single photon until it has a terminating event (kill, exit,
// wait for data, census)
Constants::event_type transport_photon_mesh_pass(Photon& phtn,
                              Mesh* mesh,
                              RNG* rng,
                              double& next_dt,
                              double& exit_E,
                              double& census_E,
                              std::vector<double>& rank_abs_E)

{
  using Constants::VACUUM; using Constants::REFLECT;
  using Constants::ELEMENT; using Constants::PROCESSOR;
  //events
  using Constants::WAIT; using Constants::CENSUS;
  using Constants::KILL; using Constants::EXIT;
  using Constants::bc_type;
  using Constants::event_type;
  using Constants::c;
  using std::min;

  uint32_t cell_id, next_cell, next_grip;
  bc_type boundary_event;
  event_type event;
  double dist_to_scatter, dist_to_boundary, dist_to_census, dist_to_event;
  double sigma_a, sigma_s, f, absorbed_E;
  double angle[3];
  Cell cell;

  uint32_t surface_cross = 0;
  const double cutoff_fraction = 0.01; //note: get this from IMC_state

  cell_id=phtn.get_cell();
  cell = mesh->get_on_rank_cell(cell_id);
  bool active = true;

  //transport this photon
  while(active) {
    sigma_a = cell.get_op_a();
    sigma_s = cell.get_op_s();
    f = cell.get_f();

    //get distance to event
    dist_to_scatter = -log(rng->generate_random_number())/
      ((1.0-f)*sigma_a + sigma_s);
    dist_to_boundary = cell.get_distance_to_boundary(phtn.get_position(),
                                                      phtn.get_angle(),
                                                      surface_cross);
    dist_to_census = phtn.get_distance_remaining();

    //select minimum distance event
    dist_to_event = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

    //Calculate energy absorbed by material, update photon and material energy
    absorbed_E = phtn.get_E()*(1.0 - exp(-sigma_a*f*dist_to_event));
    phtn.set_E(phtn.get_E() - absorbed_E);

    rank_abs_E[cell_id] += absorbed_E;

    //update position
    phtn.move(dist_to_event);

    //Apply variance/runtime reduction
    if (phtn.below_cutoff(cutoff_fraction)) {
      rank_abs_E[cell_id] += phtn.get_E();
      active=false;
      event=KILL;
    }
    // or apply event
    else {
      //Apply event
      //EVENT TYPE: SCATTER
      if(dist_to_event == dist_to_scatter) {
        get_uniform_angle(angle, rng);
        phtn.set_angle(angle);
      }
      //EVENT TYPE: BOUNDARY CROSS
      else if(dist_to_event == dist_to_boundary) {
        boundary_event = cell.get_bc(surface_cross);
        if(boundary_event == ELEMENT || boundary_event == PROCESSOR) {
          next_cell = cell.get_next_cell(surface_cross);
          next_grip = cell.get_next_grip(surface_cross);
          phtn.set_cell(next_cell);
          phtn.set_grip(next_grip);
          cell_id=next_cell;
          //look for this cell, if it's not there transport later
          if (mesh->mesh_available(cell_id))
            cell = mesh->get_on_rank_cell(cell_id);
          else {
            event= WAIT;
            active=false;
          }
        }
        else if(boundary_event == VACUUM) {
          active=false;
          exit_E+=phtn.get_E();
          event=EXIT;
        }
        else phtn.reflect(surface_cross);
      }
      //EVENT TYPE: REACH CENSUS
      else if(dist_to_event == dist_to_census) {
        phtn.set_distance_to_census(c*next_dt);
        active=false;
        census_E+=phtn.get_E();
        event=CENSUS;
      }
    } //end event loop
  } // end while alive
  return event;
}



//! Transport photons from a source object using the mesh-passing algorithm
// and two-sided messaging to fulfill requests for mesh data
std::vector<Photon> transport_mesh_pass(Source& source,
                                        Mesh* mesh,
                                        IMC_State* imc_state,
                                        IMC_Parameters* imc_parameters,
                                        Completion_Manager* comp,
                                        Mesh_Request_Manager* req_manager,
                                        Message_Counter& mctr,
                                        std::vector<double>& rank_abs_E,
                                        MPI_Types *mpi_types,
                                        const Info& mpi_info)
{
  using std::queue;
  using std::vector;
  using Constants::event_type;
  //events
  using Constants::WAIT; using Constants::CENSUS;
  using Constants::KILL; using Constants::EXIT;

  uint32_t n_local = source.get_n_photon();
  uint32_t n_local_sourced = 0;

  uint32_t cell_id;

  double census_E = 0.0;
  double exit_E = 0.0;
  double dt = imc_state->get_next_dt(); //! For making current photons
  double next_dt = imc_state->get_next_dt(); //! For census photons

  RNG *rng = imc_state->get_rng();
  Photon phtn;

  //timing
  Timer t_transport;
  Timer t_mpi;
  t_transport.start_timer("timestep transport");

  //set global particles to be n_rank, every rank sets it completed particles
  //to 1 after finished local work
  comp->set_timestep_global_particles(mpi_info.get_n_rank());

  // New data flag is initially false
  bool new_data = false;
  std::vector<Cell> new_cells; // New cells from completed RMA requests

  // Number of particles to run between MPI communication
  const uint32_t batch_size = imc_parameters->get_batch_size();

  event_type event;
  uint32_t wait_list_size;

  //--------------------------------------------------------------------------//
  // main loop over photons
  //--------------------------------------------------------------------------//
  vector<Photon> census_list; //! Local end of timestep census list
  vector<Photon> off_rank_census_list; //! Off rank end of timestep census list
  queue<Photon> wait_list; //! Photons waiting for mesh data
  while ( n_local_sourced < n_local) {

    uint32_t n = batch_size;

    while (n && n_local_sourced < n_local) {

      phtn =source.get_photon(rng, dt);
      n_local_sourced++;

      //get start cell, this only changea with cell crossing event
      cell_id=phtn.get_cell();

      // if mesh available, transport and process, otherwise put on the
      // waiting list
      if (mesh->mesh_available(cell_id)) {
        event = transport_photon_mesh_pass(phtn, mesh, rng, next_dt, exit_E,
          census_E, rank_abs_E);
        cell_id = phtn.get_cell();
      }
      else event = WAIT;

      if (event==CENSUS) {
        if (mesh->on_processor(cell_id)) census_list.push_back(phtn);
        else off_rank_census_list.push_back(phtn);
      }
      else if (event==WAIT) {
        t_mpi.start_timer("timestep mpi");
        req_manager->request_cell(phtn.get_grip(), mctr);
        t_mpi.stop_timer("timestep mpi");
        wait_list.push(phtn);
      }
      n--;
    } // end batch transport

    //process mesh requests
    t_mpi.start_timer("timestep mpi");
    new_cells = req_manager->process_mesh_requests(mctr);
    new_data = !new_cells.empty();
    if (new_data) mesh->add_non_local_mesh_cells(new_cells);
    t_mpi.stop_timer("timestep mpi");
    // if data was received, try to transport photons on waiting list
    if (new_data) {
      wait_list_size = wait_list.size();
      for (uint32_t wp =0; wp<wait_list_size; wp++) {
        phtn = wait_list.front();
        wait_list.pop();
        cell_id=phtn.get_cell();
        if (mesh->mesh_available(cell_id)) {
          event = transport_photon_mesh_pass(phtn, mesh, rng, next_dt, exit_E,
                                          census_E, rank_abs_E);
          cell_id = phtn.get_cell();
        }
        else event = WAIT;

        if (event==CENSUS) {
            if (mesh->on_processor(cell_id)) census_list.push_back(phtn);
            else off_rank_census_list.push_back(phtn);
        }
        else if (event==WAIT) {
          t_mpi.start_timer("timestep mpi");
          req_manager->request_cell(phtn.get_grip(), mctr);
          t_mpi.stop_timer("timestep mpi");
          wait_list.push(phtn);
        }
      } // end wp in wait_list
    }
  } //end while (n_local_source < n_local)

  //--------------------------------------------------------------------------//
  // Main transport loop finished, transport photons waiting for data
  //--------------------------------------------------------------------------//
  while (!wait_list.empty()) {
    //process mesh requests
    t_mpi.start_timer("timestep mpi");
    new_cells = req_manager->process_mesh_requests(mctr);
    new_data = !new_cells.empty();
    if (new_data) mesh->add_non_local_mesh_cells(new_cells);
    t_mpi.stop_timer("timestep mpi");

    // if new data received, transport waiting list
    if (new_data || req_manager->no_active_requests() ) {
      wait_list_size = wait_list.size();
      for (uint32_t wp =0; wp<wait_list_size; wp++) {
        phtn = wait_list.front();
        wait_list.pop();
        cell_id=phtn.get_cell();
        if (mesh->mesh_available(cell_id)) {
          event = transport_photon_mesh_pass(phtn, mesh, rng, next_dt, exit_E,
            census_E, rank_abs_E);
          cell_id = phtn.get_cell();
        }
        else event = WAIT;

        if (event==CENSUS) {
          if (mesh->on_processor(cell_id)) census_list.push_back(phtn);
          else off_rank_census_list.push_back(phtn);
        }
        else if (event==WAIT) {
          t_mpi.start_timer("timestep mpi");
          req_manager->request_cell(phtn.get_grip(), mctr);
          t_mpi.stop_timer("timestep mpi");
          wait_list.push(phtn);
        }
      }
    }
  } //end while wait_list not empty

  // record time of transport work for this rank
  t_transport.stop_timer("timestep transport");

  // set complete to be 1 (true) when all ranks set this in the tree,
  // the root will see n_complete == n_rank and finish
  uint64_t complete =1;

  //--------------------------------------------------------------------------//
  // While waiting for other ranks to finish, check for other messages
  //--------------------------------------------------------------------------//
  bool finished = false;
  bool waiting_for_work = true;
  while (!finished) {
    req_manager->process_mesh_requests(mctr);
    comp->process_completion(waiting_for_work, complete, mctr);
    finished = comp->is_finished();
  } // end while

  // Milagro version sends completed count down, RMA version just resets
  comp->end_timestep(mctr);

  // wait for all ranks to finish transport to finish off cell and cell id
  // requests and sends
  MPI_Barrier(MPI_COMM_WORLD);

  // all ranks have now finished transport, set diagnostic quantities
  imc_state->set_exit_E(exit_E);
  imc_state->set_post_census_E(census_E);
  imc_state->set_network_message_counts(mctr);
  imc_state->set_rank_transport_runtime(
    t_transport.get_time("timestep transport"));

  // send the off-rank census back to ranks that own the mesh its on and receive
  // census particles that are on your mesh
  vector<Photon> rebalanced_census = rebalance_census(off_rank_census_list,
                                                      mesh, mpi_types);
  census_list.insert(census_list.end(), rebalanced_census.begin(),
    rebalanced_census.end());

  imc_state->set_rank_mpi_time(t_mpi.get_time("timestep mpi"));

  // sort on census vectors by cell ID (global ID)
  sort(census_list.begin(), census_list.end());

  // set post census size after sorting and merging
  imc_state->set_census_size(census_list.size());

  return census_list;
}

#endif // def transport_mesh_pass_h_
//----------------------------------------------------------------------------//
// end of transport_mesh_pass.h
//----------------------------------------------------------------------------//
