/*
  Author: Alex Long
  Date: 7/18/2014
  Name: main.cpp
*/

#include <iostream>
#include <sstream>

#include "constants.h"
#include "photon.h"
#include "mesh.h"
#include "imc_state.h"
#include "create_census_photons.h"
#include "input.h"
#include "conservation.h"

using std::cout;
using std::endl;
using std::stringstream;

int main(int argc, char* argv[])
{ 
   
  // input
  stringstream fName;
  fName << argv[1];  
  std::string filename = fName.str();
  Input *input;
  input = new Input(argc, argv);
  input->print_problem_info();

  //IMC state setup
  IMC_State *imc_state;
  imc_state = new IMC_State(input);

  Mesh* mesh = new Mesh(input);
  mesh->set_counters();
  
  vector<Photon> photon_list;
  vector<Photon> census_list;

  while (!imc_state->finished())
  {
    //imc_state->print_time_info();
  
    //set opacity, Fleck factor all energy to source
    mesh->calculate_photon_energy(imc_state);

    //get conservation quantities
    imc_state->set_pre_census_E(get_photon_list_E(census_list) + 
      mesh->get_total_census_E());

    imc_state->set_pre_mat_E(mesh->get_total_mat_E());

    //print temperatures and sources
    //mesh->print_mesh_info();

    //make photons from energy vectors on mesh
    photon_list = mesh->make_photons(imc_state);

    //push census list on to the stack of photons to transport
    set_census_birth_time(census_list, imc_state->get_dt());
    photon_list.insert(photon_list.end(), census_list.begin(), 
      census_list.end());

    //transport photons
    mesh->transport_photons(photon_list, imc_state);

    //make census list
    census_list = build_census_list(photon_list);
    imc_state->set_post_census_E(get_photon_list_E(census_list));

    //update time for next step
    mesh->update_temperature(imc_state->get_lump_time_bool());
    imc_state->set_post_mat_E(mesh->get_total_mat_E());

    //print out conservation data 
    //print_conservation(mesh, imc_state);

    //get ready for next time step
    mesh->zero_tally();
    imc_state->next_time_step();
  }

  //Print final output
  cout<<endl<<endl;
  cout<<"----------------------------------------------------------"<<endl;
  cout<<"----------------------------------------------------------"<<endl;
  cout<<"------------------- SIMULATION COMPLETE ------------------"<<endl;
  cout<<"----------------------------------------------------------"<<endl;
  cout<<"----------------------------------------------------------"<<endl;
  imc_state->print_time_info();
  mesh->print_mesh_info();
  cout<<"----------------------------------------------------------"<<endl;
  mesh->print_counters();

  delete mesh;
  delete imc_state;
  delete input;

  return 0;
}
