//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   imc_state.h
 * \author Alex Long
 * \date   July 24 2014
 * \brief  Holds and prints diagnostic state data
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef imc_state_h_
#define imc_state_h_

#include <functional>
#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

#include "constants.h"
#include "input.h"
#include "message_counter.h"
#include "photon.h"
#include "RNG.h"

//==============================================================================
/*!
 * \class IMC_State
 * \brief Holds high level information and diagnostic data
 * 
 * Keeps track of the simulation time, step, particle counts and
 * energy conservation qualities. Also holds the RNG pointer.
 * \example no test yet
 */
//==============================================================================
class IMC_State
{
  public:

  //! constructor
  IMC_State(Input *input, uint32_t _rank)
    : rank(_rank),
      m_dt(input->get_dt()),
      m_time(input->get_time_start()),
      m_time_stop(input->get_time_finish()),
      m_step(1),
      m_dt_mult(input->get_time_mult()),
      m_dt_max(input->get_dt_max())
  {
    pre_census_E = 0.0;
    post_census_E = 0.0;
    pre_mat_E = 0.0;
    post_mat_E = 0.0;
    emission_E = 0.0;
    exit_E = 0.0;
    absorbed_E = 0.0;
    source_E = 0.0;

    // 64 bit
    trans_particles = 0;
    census_size = 0;
    step_particles_sent = 0;
    total_particles_sent=0;
    // 32 bit
    total_cells_requested=0;
    total_cells_sent=0;
    total_cell_messages=0;
    total_particle_messages=0;

    step_particle_messages=0;
    step_cells_requested=0;
    step_cell_messages=0;
    step_cells_sent=0;

    step_sends_posted=0;
    step_sends_completed=0;
    step_receives_posted=0;
    step_receives_completed=0;

    m_RNG = new RNG();
    m_RNG->set_seed(input->get_rng_seed()+rank*4106);

    rank_transport_runtime = 0.0;
    rank_mpi_time = 0.0;
    rank_load_balance_time = 0.0;
  }

  //! Destructor
  ~IMC_State() { delete m_RNG;}

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//

  //! Get current simulation time
  double get_time(void) const {return m_time;}

  //! Get current simulation timestep
  double get_dt(void) const {return m_dt;}

  //! Get current step
  uint32_t get_step(void) const {return m_step;}

  //! Get transported particles for current timestep
  uint64_t get_transported_particles(void) const {return trans_particles;}

  //! Get number of particles in census
  uint64_t get_census_size(void) const {return census_size;}

  //! Get number of particles sent over MPI for current timestep
  uint64_t get_step_particles_sent(void) const {return step_particles_sent;}

  //! Get number of particles sent over MPI for entire simulation
  uint64_t get_total_particles_sent(void) const {return total_particles_sent;}

  //! Get census energy at the beginning of timestep
  double get_pre_census_E(void) {return pre_census_E;}

  //! Get emission energy for current timestep
  double get_emission_E(void) {return emission_E;}

  //! Get next timestep size
  double get_next_dt(void) const {
    double next_dt;
    //multiply by dt_mult if it's going to be less than dt_max
    //other set to dt_max
    if (m_dt*m_dt_mult < m_dt_max) next_dt = m_dt*m_dt_mult;
    else next_dt = m_dt_max;
    //don't overrun the end time
    if (m_time + next_dt > m_time_stop) next_dt = m_time_stop - m_time;

    return next_dt;
  }  

  //! Check to see if simulation has completed
  bool finished(void) const {
    using std::abs;
    if ( abs(m_time-m_time_stop) < 1.0e-8) return true;
    else return false;
  }

  //! Print beginning of timestep information
  void print_timestep_header(void) const  {
    using std::cout;
    using std::endl;
    cout<<"****************************************";
    cout<<"****************************************"<<endl;
    cout<<"Step: "<<m_step<<"  Start Time: "<<m_time<<"  End Time: ";
    cout<<m_time +m_dt<<"  dt: "<<m_dt<<endl;
  }

  //! Print end of timestep information
  void print_simulation_footer(uint32_t dd_type ) const {
    using std::cout;
    using std::endl;
    using Constants::PARTICLE_PASS;
    using Constants::CELL_PASS;
    if (dd_type == PARTICLE_PASS) {
      cout<<"Total particles sent: "<<total_particles_sent<<endl;
      cout<<"Total particle messages: "<<total_particle_messages<<endl;
    }
    else {
      cout<<"Total cells requested: "<<total_cells_requested<<endl;
      cout<<"Total cells sent: "<<total_cells_sent<<endl;
      cout<<"Total cell messages: "<<total_cell_messages<<endl;
    }
    cout<<"****************************************";
    cout<<"****************************************"<<endl;
  }

  //! Get transport time for this rank on current timestep
  double get_rank_transport_runtime(void) {return rank_transport_runtime;}

  //! Get MPI time for this rank on current timestep
  double get_rank_mpi_time(void) {return rank_mpi_time;}

  //--------------------------------------------------------------------------//
  // non-const functions                                                      //
  //--------------------------------------------------------------------------//

  //! Perform reduction on diagnostic and conservation quantities and print
  void print_conservation(uint32_t dd_type) {
    using std::cout;
    using std::endl;
    using std::plus;
    using Constants::PARTICLE_PASS;
    using Constants::CELL_PASS;

    //define global value
    double g_absorbed_E=0.0; 
    double g_emission_E=0.0;
    double g_pre_census_E=0.0;
    double g_pre_mat_E=0.0;
    double g_post_census_E=0.0;
    double g_post_mat_E=0.0;
    double g_exit_E = 0.0;
    // timing values
    double max_mpi_time =0.0;
    double min_mpi_time =0.0;
    double max_transport_time = 0.0;
    double min_transport_time = 0.0;
    // 64 bit global integers
    uint64_t g_census_size=0;
    uint64_t g_trans_particles=0;
    uint64_t g_step_particles_sent=0;
    // 32 bit global integers
    uint32_t g_step_particle_messages=0;
    uint32_t g_step_cells_requested=0;
    uint32_t g_step_cell_messages=0;
    uint32_t g_step_cells_sent=0;
    uint32_t g_step_sends_posted=0;
    uint32_t g_step_sends_completed=0;
    uint32_t g_step_receives_posted=0;
    uint32_t g_step_receives_completed=0;

    //reduce energy conservation values (double)
    MPI_Allreduce(&absorbed_E, &g_absorbed_E, 1, MPI_DOUBLE, MPI_SUM, 
      MPI_COMM_WORLD); 
    MPI_Allreduce(&emission_E, &g_emission_E, 1, MPI_DOUBLE, MPI_SUM, 
      MPI_COMM_WORLD); 
    MPI_Allreduce(&pre_census_E, &g_pre_census_E, 1, MPI_DOUBLE, MPI_SUM, 
      MPI_COMM_WORLD);
    MPI_Allreduce(&pre_mat_E, &g_pre_mat_E, 1, MPI_DOUBLE, MPI_SUM, 
      MPI_COMM_WORLD);
    MPI_Allreduce(&post_census_E, &g_post_census_E, 1, MPI_DOUBLE, MPI_SUM, 
      MPI_COMM_WORLD);
    MPI_Allreduce(&post_mat_E, &g_post_mat_E, 1, MPI_DOUBLE, MPI_SUM, 
      MPI_COMM_WORLD);
    MPI_Allreduce(&exit_E, &g_exit_E, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    //reduce timestep values
    MPI_Allreduce(&rank_transport_runtime, &max_transport_time, 1, MPI_DOUBLE, 
      MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&rank_transport_runtime, &min_transport_time, 1, MPI_DOUBLE, 
      MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&rank_mpi_time, &max_mpi_time, 1, MPI_DOUBLE, MPI_MAX, 
      MPI_COMM_WORLD);
    MPI_Allreduce(&rank_mpi_time, &min_mpi_time, 1, MPI_DOUBLE, MPI_MIN, 
      MPI_COMM_WORLD);

    // reduce diagnostic values
    // 64 bit integer reductions
    MPI_Allreduce(&trans_particles, &g_trans_particles, 1, MPI_UNSIGNED_LONG, 
      MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_particles_sent, &g_step_particles_sent, 1, 
      MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&census_size, &g_census_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 
      MPI_COMM_WORLD);

    // 32 bit integer reductions
    MPI_Allreduce(&step_cells_requested, &g_step_cells_requested, 1, 
      MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_particle_messages, &g_step_particle_messages, 1, 
      MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_cell_messages, &g_step_cell_messages, 1, MPI_UNSIGNED, 
      MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_cells_sent, &g_step_cells_sent, 1, MPI_UNSIGNED, 
      MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_sends_posted, &g_step_sends_posted, 1, MPI_UNSIGNED, 
      MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_sends_completed, &g_step_sends_completed, 1, 
      MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_receives_posted, &g_step_receives_posted, 1, 
      MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&step_receives_completed, &g_step_receives_completed, 1, 
      MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    
    double rad_conservation = (g_absorbed_E + g_post_census_E + g_exit_E) - 
      (g_pre_census_E + g_emission_E + source_E);

    double mat_conservation = g_post_mat_E - (g_pre_mat_E + g_absorbed_E -
      g_emission_E);

    // update total simulation counters
    total_cells_requested+=g_step_cells_requested;
    total_cells_sent+=g_step_cells_sent;
    total_cell_messages+=g_step_cell_messages;
    total_particles_sent+=g_step_particles_sent;
    total_particle_messages+=g_step_particle_messages;
    

    if (rank == 0) {
      cout<<"Total Photons transported: "<<g_trans_particles<<endl;
      cout<<"Emission E: "<<g_emission_E<<", Absorption E: "<<g_absorbed_E;
      cout<<", Exit E: "<<g_exit_E<<endl;
      cout<<"Pre census E: "<<g_pre_census_E<<" Post census E: ";
      cout<<g_post_census_E<<" Post census Size: "<< g_census_size<<endl;
      cout<<"Pre mat E: "<<g_pre_mat_E<<" Post mat E: "<<g_post_mat_E<<endl;
      cout<<"Radiation conservation: "<<rad_conservation<<endl;
      cout<<"Material conservation: "<<mat_conservation<<endl;
      cout<<"Sends posted: "<<g_step_sends_posted;
      cout<<", sends completed: "<<g_step_sends_completed<<endl;
      cout<<"Receives posted: "<<g_step_receives_posted;
      cout<<", receives completed: "<<g_step_receives_completed<<endl;
      if (dd_type == PARTICLE_PASS) {
        cout<<"Step particles messages sent: "<<g_step_particle_messages;
        cout<<", Step particles sent: "<<g_step_particles_sent<<endl;
      }
      else {
        cout<<"Step cell messages sent: "<<g_step_cell_messages;
        cout<<", Step cells sent: "<<g_step_cells_sent<<endl;
        cout<<"Step cells requested: "<<g_step_cells_requested<<endl;
        cout<<"Load balance time: "<<rank_load_balance_time<<endl;
      }
      cout<<"Transport time max/min: "<<max_transport_time<<"/";
      cout<<min_transport_time<<endl;
      cout<<"MPI time max/min: "<<max_mpi_time<<"/";
      cout<<min_mpi_time<<endl;
    } // if rank==0
  }

  //! Return random number generator
  RNG* get_rng(void) {return m_RNG;}

  //! Increment time and step counter
  void next_time_step(void) {
    m_time += m_dt;
    m_dt = get_next_dt();
    m_step++;
  }

  //! Set pre-transport census energy  (diagnostic)
  void set_pre_census_E(double _pre_census_E) {pre_census_E = _pre_census_E;}

  //! Set post-transport census energy (diagnostic)
  void set_post_census_E(double _post_census_E) {
    post_census_E = _post_census_E;
  }

  //! Set pre-transport material energy (diagnostic)
  void set_pre_mat_E(double _pre_mat_E) {pre_mat_E = _pre_mat_E;}

  //! Set post-transport material energy (diagnostic)
  void set_post_mat_E(double _post_mat_E) {post_mat_E = _post_mat_E;}

  //! Set timestep emission energy (diagnostic)
  void set_emission_E(double _emission_E) {emission_E = _emission_E;}

  //! Set source energy for current timestep (diagnostic)
  void set_source_E(double _source_E) {source_E = _source_E;}

  //! Set absorbed energy for current timestep (diagnostic)
  void set_absorbed_E(double _absorbed_E) {absorbed_E = _absorbed_E;}

  //! Set exit energy from transport (diagnostic) 
  void set_exit_E(double _exit_E) {exit_E = _exit_E;}

  //! set particles transported for current timestep (diagnostic, 64 bit)
  void set_transported_particles(uint64_t _trans_particles) {
    trans_particles = _trans_particles;
  }

  //! Set number of census particles for current timestep (diagnostic, 64 bit)
  void set_census_size(uint64_t _census_size) {census_size = _census_size;}

  //! Set the network message counters used in diagnostics
  void set_network_message_counts(Message_Counter& mctr) {
    step_particles_sent=mctr.n_particles_sent;
    step_particle_messages=mctr.n_particle_messages;
    step_cell_messages=mctr.n_cell_messages;
    step_cells_sent=mctr.n_cells_sent;
    step_sends_posted=mctr.n_sends_posted;
    step_sends_completed=mctr.n_sends_completed;
    step_receives_posted=mctr.n_receives_posted;
    step_receives_completed=mctr.n_receives_completed;
  }

  //! Set the number of cells requested in mesh passing method this timestep
  void set_step_cells_requested(uint32_t _step_cells_requested) {
    step_cells_requested = _step_cells_requested;
  }

  //! Set transport runtime for this timestep
  void set_rank_transport_runtime(double _rank_transport_runtime) {
    rank_transport_runtime = _rank_transport_runtime;
  }

  //! Set MPI time for this timestep
  void set_rank_mpi_time(double _rank_mpi_time) {
    rank_mpi_time = _rank_mpi_time;
  }

  //! Set load balance time for this timestep
  void set_load_balance_time(double _load_balance_time) {
   rank_load_balance_time = _load_balance_time; 
  }

  //--------------------------------------------------------------------------//
  // member data                                                              //
  //--------------------------------------------------------------------------//
  private:
  uint32_t rank; //! Rank owning this states object
  //time
  double m_dt; //! Current time step size (sh)
  double m_time; //! Current time (sh)
  double m_time_stop; //! End time (sh)
  uint32_t m_step; //! Time step (start at 1)
  double m_dt_mult; //! Time step multiplier
  double m_dt_max;  //! Max time step

  //conservation
  double pre_census_E; //! Census energy at the beginning of the timestep
  double post_census_E; //! Census energy at the end of the timestep
  double pre_mat_E; //! Material energy at the beginning of timestep
  double post_mat_E; //! Material energy at the end of timestep
  double emission_E; //! Energy emitted this timestep
  double exit_E; //! Energy exiting problem
  double absorbed_E; //! Total absorbed energy 
  double source_E; //! Sourced energy

  //diagnostic 64 bit integers relating to particle counts
  uint64_t trans_particles; //! Particles transported
  uint64_t census_size; //! Number of particles in census

  uint64_t step_particles_sent; //! Number of particles passed

  //! Total number of particles sent for simulation
  uint64_t total_particles_sent; 

  //diagnostic 32 bit integers relating to messages and cell counts
  //! Total number of cells requested for simulation
  uint32_t total_cells_requested; 

  //! Total number of cells sent for simulation
  uint32_t total_cells_sent;

  //! Total number of cell message for simulation
  uint32_t total_cell_messages;

  //! Total number of particle messages sent for simulation
  uint32_t total_particle_messages;
 
  uint32_t step_particle_messages; //! Number of particle messages
  uint32_t step_cells_requested; //! Number of cells requested by this rank
  uint32_t step_cell_messages; //! Number of cell messages
  uint32_t step_cells_sent; //! Number of cells passed
  uint32_t step_sends_posted; //! Number of sent messages posted
  uint32_t step_sends_completed; //! Number of sent messages completed
  uint32_t step_receives_posted; //! Number of received messages completed
  uint32_t step_receives_completed;  //! Number of received messages completed

  double rank_transport_runtime; //! Transport step runtime for this rank
  double rank_mpi_time; //! Time set in MPI related calls for this rank

  //! Time to load balance particles this timestep
  double rank_load_balance_time; 

  //! RNG
  RNG* m_RNG;
};

#endif // imc_state_h_
//---------------------------------------------------------------------------//
// end of imc_state.h
//---------------------------------------------------------------------------//
