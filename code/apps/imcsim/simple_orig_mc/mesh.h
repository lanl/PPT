/*
  Author: Alex Long
  Date: 7/18/2014
  Name: mesh.h
*/

#ifndef mesh_h_
#define mesh_h_

#include <iostream>
#include <vector>

#include "constants.h"
#include "imc_state.h"
#include "input.h"
#include "RNG.h"
#include "sampling_functions.h"

using std::cout;
using std::endl;
using std::min;
using std::pow;
using std::vector;
using std::abs;

using Constants::c;
using Constants::a;

class Mesh
{
  //I added these counters
    double total_loops;
    double br1;
    double br2;
    double br2_1;
    double br2_2;
    double br2_3;
    double br3;
    double br4;
  public:
  Mesh(Input *input)  
    : m_elements(input->get_n_x_cell()),
      m_x_start(input->get_x_start()),
      m_dx(input->get_dx())
  {
    m_T = vector<double>(m_elements, input->get_initial_Tm());
    m_Tr = vector<double>(m_elements, input->get_initial_Tr());
    m_Ts = vector<double>(m_elements, 0.0);
    m_op = vector<double>(m_elements, 0.0); 
    m_f = vector<double>(m_elements, 0.0); 

    //source setting
    m_source_elem = input->get_source_element();
    m_Ts[m_source_elem] = input->get_source_T();

    m_source_E = vector<double>(m_elements, 0.0);
    m_census_E = vector<double>(m_elements, 0.0);
    m_emission_E = vector<double>(m_elements, 0.0);
    m_abs_tally = vector<double>(m_elements, 0.0); 

    m_density = input->get_density();
    m_CV = input->get_CV();
    m_opA = input->get_opacity_A();
    m_opB = input->get_opacity_B();
    m_opC = input->get_opacity_C();


    //set boundary values
    m_l_bound = vector<int>(m_elements, 0);
    m_r_bound = vector<int>(m_elements, 0);
    for (unsigned int i=0; i<m_elements; ++i) m_l_bound[i] = i-1;
    for (unsigned int i=0; i<m_elements; ++i) m_r_bound[i] = i+1;
    m_l_bound[0] = input->get_l_bound();
    m_r_bound[m_elements-1] = input->get_r_bound();

    // counters
    m_scatter_event = vector<unsigned int>(m_elements, 0); 
    m_exit_event = 0;
  }

  ~Mesh(void) {}

  //--------------------------------------------------------------------------//
  // const functions                                                   
  //--------------------------------------------------------------------------//
  
  double get_total_abs_E(void) const {
    double total_abs_E = 0.0;
    for (unsigned int i = 0; i<m_elements;++i) total_abs_E+=m_abs_tally[i];
    return total_abs_E;
  }

  double get_distance_to_scatter(unsigned int elem_id, RNG* rng, 
    Photon* phtn) const 
  {
    return -log(rng->generate_random_number())/
      (m_op[elem_id]*(1.0-m_f[elem_id]));
  }
  
  double position_in_elem(unsigned int elem_id, bool use_tilt, RNG* rng) const {
    if (use_tilt) {
      double T, T_left, T_right;
      double T_left_int, T_right_int;
      double m, b;

      T = m_T[elem_id];
      //left temeprature
      if(elem_id == 0) T_left = T;
      else T_left = m_T[elem_id-1];

      //right temperature
      if(elem_id == m_elements-1) T_right = T;
      else T_right = m_T[elem_id+1];

      //interface temperature
      T_left_int = 0.5*(T_left + T);
      T_right_int = 0.5*(T_right + T);

      //slope between interfaces
      m =(T_right_int - T_left_int)/m_dx;
      b = T_left_int;

      /*
      if ( abs(T_right_int - T) > abs(T - T_left_int)) {
        m =(T_right_int - T)/m_dx;
        b = T - 0.5*m_dx;
      }
      else {
        m = (T - T_left_int);
        b = T_left_int;
      }
      */

      //don't use slopes that are almost zero
      if (abs(m) < 1.0e-8) {
        return elem_id*m_dx + m_dx*rng->generate_random_number();
      }
      else {
        double norm = m*(m_dx*m_dx)/2.0 + b*m_dx;
        double a_quad = (m/2.0) / norm;
        double b_quad = b / norm;
        double c_quad = -rng->generate_random_number();
        double x =( -b_quad + sqrt(b_quad*b_quad - 4.0*a_quad*c_quad))/(2.0*a_quad); 
        return m_dx*elem_id + x;
      }
    } 
    else return elem_id*m_dx + m_dx*rng->generate_random_number();
  }

  double get_total_mat_E(void) const {
    double tot_mat_E = 0.0;
    for (unsigned int i=0; i<m_elements;++i) tot_mat_E+=m_T[i]*m_CV*m_density*m_dx;
    return tot_mat_E;
  }
  
  void print_mesh_info(void) const {
    cout<<"------------------ Piecewise Constant Mesh ------------------------";
    cout<<"-------------"<<endl;
    cout<<"Element  x   T   op   emission   census   source abs_E"<<endl;
    for (unsigned int i=0; i<Mesh::m_elements; ++i) {
      cout<<i<<" "<<Mesh::m_x_start + m_dx*i+m_dx*0.5<<" "<<m_T[i]<<" "
          <<m_op[i]<<" "<<m_emission_E[i]<<" "<<m_census_E[i]<<" "
          <<m_source_E[i]<<" "<<m_abs_tally[i]<<endl;
    }
    cout<<"-------------------------------------------------------------------";
    cout<<"-------------"<<endl;
  }

  unsigned int get_num_elems(void) const {return m_elements;}
  const vector<double>& get_source_E_vector(void) const { return m_source_E;}
  const vector<double>& get_emission_E_vector(void) const {return m_emission_E;}
  const vector<double>& get_census_E_vector(void) const {return m_census_E;}
  const vector<double>& get_abs_E_vector(void) const {return m_abs_tally;}

  double get_total_photon_E(void) const {
    double tot_photon_E = 0.0;
    for(unsigned int i=0; i<m_elements; ++i)
      tot_photon_E += (m_source_E[i] + m_census_E[i] + m_emission_E[i]);
    return tot_photon_E;
  }

  double get_total_census_E(void) const {
    double tot_census_E = 0.0;
    for (unsigned int i=0; i<m_elements;++i) tot_census_E+=m_census_E[i];
    return tot_census_E;
  }

  double get_distance_to_bound(double pos, double angle,unsigned int elem_id,
    int& next_elem) const
  {
    if (angle > 0.0) {
      next_elem = m_r_bound[elem_id];
      return (m_dx*(elem_id+1) - pos)/angle;
    }
    else {
      next_elem = m_l_bound[elem_id];
      return (pos - m_dx*elem_id)/abs(angle);
    }
  }

  //--------------------------------------------------------------------------//
  // non-const functions                                                   
  //--------------------------------------------------------------------------//

  void calculate_photon_energy(IMC_State* imc_s) {
    double dt = imc_s->get_dt();
    unsigned int step = imc_s->get_step();
    for (unsigned int i=0; i<m_elements;++i) {
      m_op[i] = m_density*(m_opA + m_opB*pow(m_T[i], m_opC));
      m_f[i] = 1.0/(1.0 + dt*m_op[i]*c*4.0*a*pow(m_T[i],3)/(m_CV*m_density) );
      m_emission_E[i] = dt*m_dx*m_f[i]*m_op[i]*a*c*pow(m_T[i],4);
      if (step > 1) m_census_E[i] = 0.0;  
      else m_census_E[i] =m_dx*a*pow(m_Tr[i],4); 
      m_source_E[i] = (1.0/4.0)*dt*a*c*pow(m_Ts[i],4);
    }
  }
 
  void tally_exit_event(void) { m_exit_event++;} 

  void tally_line_abs_E(double dist, unsigned int elem_ID, Photon* phtn) {
    double start_phtn_E = phtn->get_E();
    double abs_E = start_phtn_E*(1.0 - exp(-m_f[elem_ID]*m_op[elem_ID]*dist));
    m_abs_tally[elem_ID] += abs_E;
    phtn->set_E(start_phtn_E-abs_E);
  }

  void update_temperature(bool lump_time) {
    for (unsigned int i=0; i<m_elements;++i) 
      m_T[i] = m_T[i] + (m_abs_tally[i] - 
        m_emission_E[i])/(m_CV*m_dx*m_density);
  }

  //Olena code
  void set_counters()
  {
    br1, br2, br3, br2_1, br2_2, br2_3, br4, total_loops= 0;
    br3=0;
  }
  void print_counters()
  {
    cout<<"br1:"<<br1<<"\n"<<"br2="<< br2<<" br2_1="<<br2_1<<" br2_2="<<br2_2<<" br2_3="<<br2_3<<"\nbr3="<<br3<<"\nbr4="<<br4<<"\ntotal_loops"<<total_loops<<"\n";
  }

  void transport_photons(vector<Photon>& photon_list, IMC_State* imc_state) {
    unsigned int e_ID;
    bool active;
    double dist_to_scatter, dist_to_boundary, dist_to_census, dist_to_event;
    int e_next;
    double cutoff_fraction = 0.0001; //note: get this from IMC_state
    RNG* rng = imc_state->get_rng();


    //transport photons
    for (vector<Photon>::iterator iphtn=photon_list.begin(); iphtn!=photon_list.end(); iphtn++) {
      active=true;
      while(active) {
        //set element ID
        e_ID=iphtn->get_element();

        //get distance to event
        dist_to_scatter = get_distance_to_scatter(e_ID, rng, &(*iphtn) );
        dist_to_boundary = get_distance_to_bound(iphtn->get_position(), iphtn->get_angle(), e_ID, e_next);
        dist_to_census = iphtn->get_distance_remaining();
        dist_to_event = min(dist_to_scatter, min(dist_to_boundary, dist_to_census));

        //Attenuate photon energy(if applicable)
        tally_line_abs_E(dist_to_event, e_ID, &(*iphtn));

        //update position and distance to census
        iphtn->move(dist_to_event);
        total_loops +=1;
        //Apply event
        //EVENT TYPE: SCATTER
        //Olena: branch1= br1 counter
        if(dist_to_event == dist_to_scatter) {
          iphtn->set_angle(get_uniform_angle(rng));
          tally_scatter_event(e_ID);
          br1 +=1;
        }
        //EVENT TYPE: BOUNDARY CROSS
        else if(dist_to_event == dist_to_boundary) {
          br2 +=1;
          //VACUUM
          if(e_next == -1) {
            active=false;
            tally_exit_event();
            imc_state->tally_exit_E(iphtn->get_E());
            br2_1 +=1;
          }
          //REFLECT
          else if (e_next == -2)
            {iphtn->reflect();
              br2_2 +=1;
            }
          //SET NEXT ELEMENT
          else
            {iphtn->set_element(e_next);
              br2_3 +=1;}
          
        }
        //EVENT TYPE: REACH CENSUS
        else if(dist_to_event == dist_to_census) {
          br3 +=1;
          iphtn->set_census_flag(true);
          active=false;

        }

        //Apply variance/runtime reduction if photon did not exit
        if(iphtn->below_cutoff(cutoff_fraction) && active) {
          br4 +=1;
          tally_point_abs_E(iphtn->get_E(), e_ID);
          active=false;
        }

      } // end while alive
    } // end for iphtn
  }
  
  vector<Photon> make_photons(IMC_State* imc_state) const {
    unsigned int total_phtns = imc_state->get_total_step_photons();
    double delta_t = imc_state->get_dt();
    RNG* rng = imc_state->get_rng();

    vector<Photon> phtn_vec;
    double total_E = get_total_photon_E();
    
    for (unsigned int elem_ID = 0; elem_ID<get_num_elems(); elem_ID++) {
      double pos, angle;

      //Census photons scope
      {
        unsigned int elem_census_phtns = int(total_phtns*m_census_E[elem_ID]/total_E);
        if (elem_census_phtns == 0 && m_census_E[elem_ID]>0.0) elem_census_phtns=1;
        double census_phtn_E = m_census_E[elem_ID] / elem_census_phtns;
        for (unsigned int iphtn = 0; iphtn<elem_census_phtns; ++iphtn) {
          pos = position_in_elem(elem_ID, false, rng);
          angle = get_uniform_angle(rng);
          Photon census_photon(pos, angle);
          census_photon.set_E0(census_phtn_E);
          census_photon.set_distance_to_census(c*delta_t);
          census_photon.set_element(elem_ID);
          phtn_vec.push_back(census_photon);
        }
      } //end census photons scope

      //Emission photons scope
      { 
        unsigned int elem_emission_phtns = int(total_phtns*m_emission_E[elem_ID]/total_E);
        if (elem_emission_phtns == 0 && m_emission_E[elem_ID]>0.0) elem_emission_phtns=1;
        double emission_phtn_E = m_emission_E[elem_ID] / elem_emission_phtns;
        for (unsigned int iphtn = 0; iphtn<elem_emission_phtns; ++iphtn) {
          pos = position_in_elem(elem_ID, imc_state->get_tilt_bool(), rng);
          angle = get_uniform_angle(rng);
          Photon emission_photon(pos, angle);
          emission_photon.set_E0(emission_phtn_E);
          emission_photon.set_distance_to_census(rng->generate_random_number()*c*delta_t);
          emission_photon.set_element(elem_ID);
          phtn_vec.push_back(emission_photon);
        }
      } //end scope of emission photons

      //Source Photons scope
      {
        unsigned int elem_source_phtns = int(total_phtns*m_source_E[elem_ID]/total_E);
        double source_phtn_E = m_source_E[elem_ID] / elem_source_phtns;
        for (unsigned int iphtn = 0; iphtn<elem_source_phtns; ++iphtn) {
          pos = 0.0;
          angle = get_source_angle(rng);
          Photon source_photon(pos, angle);
          source_photon.set_E0(source_phtn_E);
          source_photon.set_distance_to_census(rng->generate_random_number()*c*delta_t);
          source_photon.set_element(elem_ID);
          phtn_vec.push_back(source_photon);
        }
      } //end source photons scope

    } //end element loop

    return phtn_vec;
  }
  void tally_point_abs_E(double abs_E, unsigned int elem_ID) { 
    m_abs_tally[elem_ID] += abs_E;
  }

  void zero_tally(void) {
    for (unsigned int i=0; i<m_abs_tally.size() ;++i) m_abs_tally[i]=0.0;
  }

  void tally_scatter_event(unsigned int elem_ID) {
    m_scatter_event[elem_ID]++;
  } 

  private:
  unsigned int m_elements;
  unsigned int m_source_elem;
  double m_density;
  double m_CV;
  double m_opA;
  double m_opB;
  double m_opC;
  double m_dx;
  double m_x_start;
  vector<double> m_T;
  vector<double> m_Tr;
  vector<double> m_Ts;
  vector<double> m_source_E;
  vector<double> m_census_E;
  vector<double> m_emission_E;
  vector<double> m_abs_tally;
  vector<double> m_op;
  vector<double> m_f;
  vector<int> m_l_bound;
  vector<int> m_r_bound;

  //event counters
  vector<unsigned int> m_scatter_event;
  unsigned int m_exit_event;

};


#endif //mesh_h
