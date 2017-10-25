//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   main.cc
 * \author Alex Long
 * \date   July 24 2014
 * \brief  Reads input file, sets up mesh and runs transport 
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

#include "constants.h"
#include "decompose_mesh.h"
#include "imc_drivers.h"
#include "imc_state.h"
#include "imc_parameters.h"
#include "input.h"
#include "mesh.h"
#include "info.h"
#include "mpi_types.h"
#include "timer.h"

using std::vector;
using std::endl;
using std::cout;
using std::string;
using Constants::PARTICLE_PASS;
using Constants::CELL_PASS;
using Constants::CELL_PASS_RMA;

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  //check to see if number of arguments is correct
  if (argc != 2) {
    cout<<"Usage: BRANSON <path_to_input_file>"<<endl;
    exit(EXIT_FAILURE); 
  }

  // get MPI parmeters and set them in mpi_info
  const Info mpi_info;

  // make MPI types object
  MPI_Types* mpi_types= new MPI_Types();

  // get input object from filename
  std::string filename( argv[1]);
  Input *input;
  input = new Input(filename);
  if(mpi_info.get_rank()==0) input->print_problem_info();

  // IMC paramters setup
  IMC_Parameters *imc_p;
  imc_p = new IMC_Parameters(input);

  // IMC state setup
  IMC_State *imc_state;
  imc_state = new IMC_State(input, mpi_info.get_rank());

  //timing 
  Timer * timers = new Timer();

  // make mesh from input object and decompose mesh with ParMetis
  timers->start_timer("Total setup");
  Mesh *mesh = new Mesh(input, mpi_types, mpi_info);
  decompose_mesh(mesh, mpi_types, mpi_info, imc_p->get_grip_size());
  timers->stop_timer("Total setup");

  MPI_Barrier(MPI_COMM_WORLD);
  //print_MPI_out(mesh, rank, n_rank);

  //--------------------------------------------------------------------------//
  // TRT PHYSICS CALCULATION
  //--------------------------------------------------------------------------//

  timers->start_timer("Total transport");

  if (input->get_dd_mode() == PARTICLE_PASS)
    imc_particle_pass_driver(mesh, imc_state, imc_p, mpi_types, mpi_info);
  else if (input->get_dd_mode() == CELL_PASS)
    imc_cell_pass_driver(mesh, imc_state, imc_p, mpi_types, mpi_info);
  else if (input->get_dd_mode() == CELL_PASS_RMA) 
    imc_rma_cell_pass_driver(mesh, imc_state, imc_p, mpi_types, mpi_info);

  timers->stop_timer("Total transport");
  
  if (mpi_info.get_rank()==0) {
    cout<<"****************************************";
    cout<<"****************************************"<<endl;
    imc_state->print_simulation_footer(input->get_dd_mode());
    timers->print_timers();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  delete mesh;
  delete timers;
  delete imc_state;
  delete imc_p;
  delete input;
  delete mpi_types;

  MPI_Finalize();
}
//---------------------------------------------------------------------------//
// end of main.cc
//---------------------------------------------------------------------------//
