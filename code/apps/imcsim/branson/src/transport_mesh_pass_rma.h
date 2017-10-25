//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   transport_mesh_pass_rma.h
 * \author Alex Long
 * \date   March 2 2016
 * \brief  Transport routine using one sided messaging and mesh-passing DD
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#ifndef transport_rma_mesh_pass_h_
#define transport_rma_mesh_pass_h_

#include <algorithm>
#include <vector>
#include <numeric>
#include <queue>
#include <mpi.h>

#include "constants.h"
#include "decompose_photons.h"
#include "info.h"
#include "mesh.h"
#include "mesh_rma_manager.h"
#include "message_counter.h"
#include "mpi_types.h"
#include "RNG.h"
#include "sampling_functions.h"
#include "timer.h"
#include "transport_mesh_pass.h"


//! Transport photons from a source object using the mesh-passing algorithm
// and one-sided messaging to fulfill requests for mesh data
std::vector<Photon> transport_mesh_pass_rma(Source& source,
                                            Mesh* mesh,
                                            IMC_State* imc_state,
                                            IMC_Parameters* imc_parameters,
                                            RMA_Manager* rma_manager,
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

  // timing
  Timer t_transport;
  Timer t_mpi;
  t_transport.start_timer("timestep transport");

  bool new_data = false; //! New data flag is initially false
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

      //get start cell, this only change with cell crossing event
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
        rma_manager->request_cell_rma(phtn.get_grip(), mctr);
        t_mpi.stop_timer("timestep mpi");
        wait_list.push(phtn);
      }
      n--;
    } // end batch transport

    //process mesh requests
    t_mpi.start_timer("timestep mpi");
    new_cells = rma_manager->process_rma_mesh_requests(mctr);
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
          rma_manager->request_cell_rma(phtn.get_grip(), mctr);
          wait_list.push(phtn);
          t_mpi.stop_timer("timestep mpi");
        }
      } // end wp in wait_list
    } // end if no data

  } //end while (n_local_source < n_local)

  //--------------------------------------------------------------------------//
  // Main transport loop finished, transport photons waiting for data
  //--------------------------------------------------------------------------//
  wait_list_size = wait_list.size();
  while (!wait_list.empty()) {
    t_mpi.start_timer("timestep mpi");
    new_cells = rma_manager->process_rma_mesh_requests(mctr);
    new_data = !new_cells.empty();
    if (new_data) mesh->add_non_local_mesh_cells(new_cells);
    t_mpi.stop_timer("timestep mpi");
    // if new data received or there are no active mesh requests, try to
    // transport waiting list (it could be that there are no active memory
    // requests because the request queue was full at the time)
    if (new_data || rma_manager->no_active_requests()) {
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
          rma_manager->request_cell_rma(phtn.get_grip(), mctr);
          wait_list.push(phtn);
          t_mpi.stop_timer("timestep mpi");
        }
      } // end wp in wait_list
    } // end if new_data
  } //end while wait_list not empty

  // record time of transport work for this rank
  t_transport.stop_timer("timestep transport");

  MPI_Barrier(MPI_COMM_WORLD);

  // all ranks have now finished transport set diagnostic quantities
  imc_state->set_exit_E(exit_E);
  imc_state->set_post_census_E(census_E);
  imc_state->set_network_message_counts(mctr);
  imc_state->set_rank_transport_runtime(
    t_transport.get_time("timestep transport"));

  // send the off-rank census back to ranks that own the mesh its on and receive
  // census particles that are on your mesh

  //t_mpi.start_timer("timestep mpi");
  vector<Photon> rebalanced_census =
    rebalance_census(off_rank_census_list, mesh, mpi_types);
  //t_mpi.stop_timer("timestep mpi");

  imc_state->set_rank_mpi_time(t_mpi.get_time("timestep mpi"));

  census_list.insert(census_list.end(), rebalanced_census.begin(),
    rebalanced_census.end());

  // sort on census vectors by cell ID (global ID)
  sort(census_list.begin(), census_list.end());

  // set post census size after sorting and merging
  imc_state->set_census_size(census_list.size());

  return census_list;
}

#endif // def transport_rma_mesh_pass_h_
//----------------------------------------------------------------------------//
// end of transport_mesh_pass_rma.h
//----------------------------------------------------------------------------//
