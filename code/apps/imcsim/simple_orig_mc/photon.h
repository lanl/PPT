/*
  Author: Alex Long
  Date: 7/18/2014
  Name: photon.h
*/

#ifndef photon_h_
#define photon_h_

#include "constants.h"

#include <iostream>
#include <vector>
#include <cmath>

using std::pow;
using std::cout;
using std::endl;
using Constants::c;

class Photon
{
  public:
  Photon(double pos, double angle) {
    set_pos(pos);
    set_angle(angle);
    m_census_flag = false;
    m_E = 0.0;
    m_E0 = 0.0;
    m_elem_ID = 0;
    m_life_dx = 0.0; 
  }

  ~Photon(void) {}

  //non-const functions
  void move(const double& distance) { 
    m_pos += m_angle*distance;
    m_life_dx -= distance;
  }

  bool below_cutoff(const double& cutoff_fraction) const {
    bool return_bool = false;
    if (m_E / m_E0 < cutoff_fraction) {
      return_bool = true;
    }
    return return_bool;
  }
 
  void set_birth_time(double dt) { m_life_dx = c*dt;}
  void set_element(const unsigned int& new_elem) { m_elem_ID = new_elem;}
  void set_E0(const double& E) { 
    m_E0 = E;
    m_E = E;
  }
  void set_E(const double& E) {m_E = E;}
  void reflect(void) {m_angle *=-1.0;}
  

  void set_census_flag(const bool& census_flag) {m_census_flag = census_flag;}
  void set_distance_to_census(const double& dist_remain) {m_life_dx = dist_remain;}
  void set_angle(double angle) { m_angle = angle;}// m_angle[1] = angle[1]; m_angle[2] = angle[2];

  //const functions
  unsigned int get_element(void) const { return m_elem_ID; }
  double get_position(void) const { return m_pos; }
  double get_angle(void) const { return m_angle; }
  double get_E(void) const { return m_E;}
  bool get_census_flag(void) const {return m_census_flag;}
  double get_distance_remaining(void) const {return m_life_dx;}  

  void print_info(void ) const {
    cout<<"----Photon Info----\n";
    cout<<"position: "<<m_pos<<endl;
    cout<<"angle: "<<m_angle<<endl;
    cout<<"Energy: "<<m_E<<" , Initial energy: "<<m_E0<<endl;
    cout<<"Element ID: "<<m_elem_ID<<" , Census Flag: ";
    if (m_census_flag) cout<<"True"<<endl;
    else cout<<"False"<<endl;
  }
  
  //member variables  
  private:
  double m_pos; //!< photon position
  double m_angle; //!< photon angle vector

  double m_E; //!< current photon energy
  double m_E0; //!< photon energy at creation

  double m_elem_ID; //!< Element ID
  bool m_census_flag; //!< Flag for census, true if photon reaches census
  double m_life_dx; //!< Distance remaining this time step

  //private member functions
  private:

  void set_pos(double pos) {
    m_pos = pos;// m_pos[1] = pos[1]; m_pos[2] = pos[2];
  }

};

#endif
