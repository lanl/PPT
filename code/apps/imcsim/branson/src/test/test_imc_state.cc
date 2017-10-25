//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_imc_state.cc
 * \author Alex Long
 * \date   February 11 2016
 * \brief  Test IMC state get and set functions and 64 bit reductions
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <string>
#include "../imc_state.h"
#include "../input.h"
#include "../message_counter.h"
#include "testing_functions.h"

using std::cout;
using std::endl;
using std::string;

int main (int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  
  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

  int nfail = 0;
  
  // test get functions
  {
    bool get_functions_pass = true;

    //setup imc_state
    string filename("simple_input.xml");
    Input *input = new Input(filename);
    IMC_State imc_state(input, rank);

    if (imc_state.get_dt() != input->get_dt()) get_functions_pass = false;
    //no time multiplier so next dt should be the same
    if (imc_state.get_next_dt() != imc_state.get_dt()) get_functions_pass = false;
    if (imc_state.get_step() != 1) get_functions_pass = false;
    if (imc_state.get_census_size() != 0) get_functions_pass = false;
    if (imc_state.get_pre_census_E() != 0.0) get_functions_pass = false;
    if (imc_state.get_emission_E() != 0.0) get_functions_pass = false;
    if (imc_state.finished()) get_functions_pass = false; 

    if (get_functions_pass) cout<<"TEST PASSED: IMC_State get functions "<<endl;
    else { 
      cout<<"TEST FAILED: IMC_State get functions"<<endl; 
      nfail++;
    }
    delete input;
  }

  //test time increment functions
  {
    bool time_functions_pass = true;
    double tolerance = 1.0e-8;

    //setup imc_state
    string large_filename("large_particle_input.xml");
    Input *input = new Input(large_filename);
    IMC_State imc_state(input, rank);

    if (imc_state.get_dt() != input->get_dt()) time_functions_pass = false;
    if (imc_state.get_step() != 1) time_functions_pass = false;
    //time multiplier is 1.5 so next dt should be the 1.5 times larger
    if (imc_state.get_next_dt() != (1.5*imc_state.get_dt()) ) time_functions_pass = false;
    if (imc_state.get_next_dt() == input->get_dt()) time_functions_pass = false;

    //increment step and check values
    imc_state.next_time_step();
    if (imc_state.get_step() != 2) time_functions_pass = false;
    if (!soft_equiv(imc_state.get_next_dt(), (1.5*imc_state.get_dt()), tolerance))
      time_functions_pass = false;

    //increment many times to check that dt does not exceed maximum dt
    for (uint32_t i=0; i<10; i++) imc_state.next_time_step();

    if (!soft_equiv(imc_state.get_next_dt(), input->get_dt_max(), tolerance))
      time_functions_pass = false;
    if (!soft_equiv(imc_state.get_dt(),input->get_dt_max(), tolerance))
      time_functions_pass = false;
 
    if (time_functions_pass) cout<<"TEST PASSED: IMC_State time increment functions"<<endl;
    else { 
      cout<<"TEST FAILED: IMC_State time increment functions"<<endl; 
      nfail++;
    }
    delete input;
  }

  //test get and set of 64 bit quantities
  {
    bool large_quantity_pass = true;
    //setup imc_state
    string filename("simple_input.xml");
    Input *input = new Input(filename);
    IMC_State imc_state(input, rank);

    uint64_t big_64_bit_number_1 = 7000000000;
    uint64_t big_64_bit_number_2 = 8000000000;
    uint64_t big_64_bit_number_3 = 9000000000;

    Message_Counter mctr;
    mctr.n_particles_sent = big_64_bit_number_1;

    imc_state.set_network_message_counts(mctr);
    imc_state.set_census_size(big_64_bit_number_2);
    imc_state.set_transported_particles(big_64_bit_number_3);
    
    if (imc_state.get_step_particles_sent() != big_64_bit_number_1)
      large_quantity_pass = false;

    if (imc_state.get_census_size() != big_64_bit_number_2)
      large_quantity_pass = false;

    if (imc_state.get_transported_particles() != big_64_bit_number_3)
      large_quantity_pass = false;

    if (large_quantity_pass) cout<<"TEST PASSED: IMC_State 64 bit value get and set"<<endl;
    else { 
      cout<<"TEST FAILED: IMC_State 64 bit value get and set"<<endl; 
      nfail++;
    }
    delete input;
  }

  

  //test reduction of 64 bit diagnostic quantities
  {
    bool large_reduction_pass = true;
    //setup imc_state
    string filename("simple_input.xml");
    Input *input = new Input(filename);
    IMC_State imc_state(input, rank);

    uint32_t big_32_bit_number = 3500000000;
    uint64_t combined_64_bit_number = 7000000000;

    Message_Counter mctr;
    mctr.n_particles_sent = big_32_bit_number;

    imc_state.set_network_message_counts(mctr);
    
    imc_state.print_conservation(0);
    
    if (imc_state.get_total_particles_sent() != combined_64_bit_number)
      large_reduction_pass = false;

    if (large_reduction_pass) cout<<"TEST PASSED: IMC_State 64 bit value reduction"<<endl;
    else { 
      cout<<"TEST FAILED: IMC_State 64 bit value reduction"<<endl; 
      nfail++;
    }
    delete input;
  }

  MPI_Finalize();

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_imc_state.cc
//---------------------------------------------------------------------------//
