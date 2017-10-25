//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_mpi_types.cc
 * \author Alex Long
 * \date   May 12 2016
 * \brief  Test custom MPI types for consistency with their datatypes
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <string>
#include "../cell.h"
#include "../mpi_types.h"
#include "../photon.h"
#include "../work_packet.h"

using std::cout;
using std::endl;
using std::string;

int main (int argc, char *argv[]) {

  MPI_Init(&argc, &argv);
  
  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

  int nfail = 0;
  
  MPI_Types mpi_types;
  // test get and size functions
  {
    bool size_functions_pass = true;

    // test particle size
    Photon photon;
    MPI_Datatype MPI_Particle = mpi_types.get_particle_type();

    int particle_size;
    MPI_Type_size(MPI_Particle, &particle_size);

    // copy should be the same size as size recorded in class
    if (particle_size != mpi_types.get_particle_size()  ) size_functions_pass = false;

    // copy should be the same size as actual Photon class
    if (particle_size != sizeof(Photon) ) size_functions_pass = false;


    // test cell size
    Cell cell;
    MPI_Datatype MPI_Cell = mpi_types.get_cell_type();

    int cell_size;
    MPI_Type_size(MPI_Cell, &cell_size);

    // copy should be the same size as size recorded in class
    if (cell_size != mpi_types.get_cell_size() ) size_functions_pass = false;

    // copy should be the same size as actual Photon class
    if (cell_size != sizeof(Cell) ) size_functions_pass = false;


    // test work packet size
    Work_Packet work_packet;
    MPI_Datatype MPI_Work_Packet = mpi_types.get_work_packet_type();

    int work_packet_size;
    MPI_Type_size(MPI_Work_Packet, &work_packet_size);

    // copy should be the same size as size recorded in class
    if (work_packet_size != mpi_types.get_work_packet_size() ) size_functions_pass = false;

    // copy should be the same size as actual Photon class
    if (work_packet_size != sizeof(Work_Packet) ) size_functions_pass = false;


    if (size_functions_pass) cout<<"TEST PASSED: MPI_Types size functions "<<endl;
    else { 
      cout<<"TEST FAILED: MPI_Types size functions"<<endl; 
      nfail++;
    }
  }

  MPI_Finalize();

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_mpi_types.cc
//---------------------------------------------------------------------------//
