/*
  Author: Alex Long
  Date: 7/24/2014
  Name: imc_state.h
*/
#ifndef imc_state_h_
#define imc_state_h_

#include <iostream>
#include <vector>
#include <cmath>

#include "input.h"
#include "RNG.h"

using std::pow;
using std::cout;
using std::endl;

class IMC_State
{
  public:
  IMC_State(Input *input)
    : m_dt(input->get_dt()),
      m_time(input->get_time_start()),
      m_time_stop(input->get_time_finish()),
      m_step(1),
      m_dt_mult(input->get_time_mult()),
      m_exit_E(0.0),
      m_pre_mat_E(0.0),
      m_post_mat_E(0.0),
      m_pre_census_E(0.0),
      m_post_census_E(0.0),
      m_total_photons(input->get_number_photons()),
      use_tilt(input->get_tilt_bool()),
      m_RNG(new RNG()),
      m_l_bound(input->get_l_bound()),
      m_r_bound(input->get_r_bound()),
      m_lump_emission(input->get_lump_emission_bool()),
      m_lump_time(input->get_lump_time_bool())
    {
      set_rng_seed(input->get_rng_seed());
    }

  ~IMC_State() { delete m_RNG;}

  //member functions
  // const functions
  double        get_dt(void) const {return m_dt;}
  unsigned int  get_step(void) const {return m_step;}
  unsigned int  get_total_step_photons(void) const {return m_total_photons;}
  double        get_exit_E(void) const {return m_exit_E;}
  RNG*          get_rng(void) const { return m_RNG;}
  bool          get_tilt_bool() {return use_tilt;}
  bool          get_lump_emission_bool() {return m_lump_emission;}
  bool          get_lump_time_bool() {return m_lump_time;}
  
  bool finished(void) const {
    if (m_time >= m_time_stop) return true;
    else return false;
  }

  double get_pre_census_E(void) const {return m_pre_census_E;}
  double get_post_census_E(void) const {return m_post_census_E;}
  double get_pre_mat_E(void) const {return m_pre_mat_E;}
  double get_post_mat_E(void) const {return m_post_mat_E;}
  unsigned int get_L_boundary(void) {return m_l_bound;}
  unsigned int get_R_boundary(void) {return m_r_bound;}

  // non-const functions
  void tally_exit_E(double phtn_E) {m_exit_E += phtn_E;}
  void set_rng_seed(unsigned int seed) { m_RNG->set_seed(seed);}
  void set_pre_census_E(double pre_census_E) {m_pre_census_E = pre_census_E;}
  void set_post_census_E(double post_census_E) {m_post_census_E = post_census_E;}
  void set_pre_mat_E(double pre_mat_E) {m_pre_mat_E = pre_mat_E;}
  void set_post_mat_E(double post_mat_E) {m_post_mat_E = post_mat_E;}

  void next_time_step(void) {
    //update time
    m_time += m_dt;
    m_dt*=m_dt_mult;
    m_step++;

    //reset exit energy tally
    m_exit_E = 0.0;
    //cout<<"-------------------------------------------------------------------";
    //cout<<"-------------"<<endl;
  }

  void print_time_info(void) {
    cout<<endl;
    cout<<"-------------------------------------------------------------------";
    cout<<"-------------"<<endl;
    cout<<" Time: "<<m_time<<"     dt: "<<m_dt<<endl;
    cout<<"-------------------------------------------------------------------";
    cout<<"-------------"<<endl;
  }

  private:
  //time
  double m_dt; //!< Current time step size (sh)
  double m_time; //!< Current time (sh)
  double m_time_stop; //!< End time (sh)
  unsigned int m_step; //!< Time step (start at 1)
  double m_dt_mult; //!< Time step multiplier

  //conservation check
  double m_exit_E; //!< Enery leaving problem domain, for conservation check
  double m_pre_mat_E; //!< Total energy in the material at beginning of timestep
  double m_post_mat_E; //!< Total energy in the material at beginning of timestep
  double m_pre_census_E; //!< Total energy in the census at beginning of timestep
  double m_post_census_E; //!< Total energy in the census at end of timestep

  //photons
  unsigned int m_total_photons;

  bool use_tilt;
  //RNG
  RNG *m_RNG;

  //bounds
  unsigned int m_l_bound;
  unsigned int m_r_bound;

  bool m_lump_emission;
  bool m_lump_time;
};

#endif
