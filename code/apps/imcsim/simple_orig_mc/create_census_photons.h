/*
  Author: Alex Long
  Date: 6/29/2014
  Name: transport.h
*/

#ifndef transport_h_
#define transport_h_

#include "constants.h"
#include "sampling_functions.h"
#include "RNG.h"

#include <vector>


using Constants::c;
using std::vector;
using std::min;


void print_photon_list(vector<Photon>& photon_list)
{
  double tot_E = 0.0;
  for (vector<Photon>::iterator iphtn=photon_list.begin(); iphtn!=photon_list.end(); iphtn++) {
    iphtn->print_info();
    tot_E += iphtn->get_E();
  }
  cout<<"\nTotal List Energy: "<<tot_E<<endl;
}


double get_photon_list_E(vector<Photon>& photon_list) 
{
  //NOTE: does not check for census flag
  double tot_E=0.0;
  for (vector<Photon>::iterator iphtn=photon_list.begin(); iphtn!=photon_list.end(); iphtn++) {
    tot_E += iphtn->get_E();
  }
  return tot_E;
}


vector<Photon> build_census_list(vector<Photon>& photon_list)
{
  vector<Photon> census_list;
  for (vector<Photon>::iterator iphtn=photon_list.begin(); iphtn!=photon_list.end(); iphtn++) {
    if (iphtn->get_census_flag()) census_list.push_back(*iphtn);
  }
  return census_list;
}


void set_census_birth_time(vector<Photon>& photon_list, double dt)
{
  for (vector<Photon>::iterator iphtn=photon_list.begin(); iphtn!=photon_list.end(); iphtn++) {
    iphtn->set_birth_time(dt);
    iphtn->set_census_flag(false);
  }
}

#endif // transport_h_
