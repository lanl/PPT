/*
  Author: Alex Long
  Date: 10/22/2014
  Name: conservation.h
*/

#ifndef conservation_h_
#define conservation_h_

#include <iostream>
#include <vector>
#include <cmath>

using std::pow;
using std::cout;
using std::endl;


void print_conservation(Mesh* mesh, IMC_State* imc_state) {
  vector<double> census_E = mesh->get_census_E_vector();
  vector<double> emission_E = mesh->get_emission_E_vector();
  vector<double> source_E = mesh->get_source_E_vector();
  
  double tot_emission_E = 0.0;
  double tot_source_E = 0.0;
  double tot_abs_E = mesh->get_total_abs_E();

  for (unsigned int elem_ID = 0; elem_ID<mesh->get_num_elems(); elem_ID++) {
    tot_emission_E += emission_E[elem_ID];
    tot_source_E += source_E[elem_ID];
  }

  double rad_conservation = (tot_abs_E + imc_state->get_post_census_E() + imc_state->get_exit_E()) - 
                            (imc_state->get_pre_census_E() + tot_emission_E + tot_source_E);

  double mat_conservation = imc_state->get_post_mat_E() - (imc_state->get_pre_mat_E() + tot_abs_E 
                            - tot_emission_E);

  cout<<"Energy Emitted: "<<tot_emission_E<<endl;
  cout<<"Energy Absorbed: "<<tot_abs_E<<endl;
  cout<<"Energy Sourced: "<<tot_source_E<<endl;
  cout<<"Start Census Energy: "<<imc_state->get_pre_census_E();
  cout<< " End Census Energy: "<<imc_state->get_post_census_E()<<endl;
  
  cout<<"Exit Energy: "<<imc_state->get_exit_E()<<endl;
  cout<<"Start Mat Energy: "<<imc_state->get_pre_mat_E();
  cout<<" End Mat Energy: "<<imc_state->get_post_mat_E()<<endl;

  cout<<"Radiation Conservation: "<<rad_conservation<<endl;
  cout<<"Material Conservation: "<<mat_conservation<<endl;
}

#endif
