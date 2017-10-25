//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   imc_parameters.h
 * \author Alex Long
 * \date   December 3 2015
 * \brief  Holds parameters needed in IMC simulation
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef imc_parameters_h_
#define imc_parameters_h_

#include "input.h"

//==============================================================================
/*!
 * \class IMC_Parameters
 * \brief Holds parameters used in IMC simulation
 * 
 * Initialized with the input class and then data members are invariant
 * \example no test yet
 */
//==============================================================================

class IMC_Parameters
{
  public:
  //! constructor
  IMC_Parameters(Input *input)
    : n_user_photon(input->get_number_photons()),
      grip_size(input->get_grip_size()),
      map_size(input->get_map_size()),
      dd_mode(input->get_dd_mode()),
      completion_method(input->get_completion_routine()),
      batch_size(input->get_batch_size()),
      particle_message_size(input->get_particle_message_size()),
      write_silo_flag(input->get_write_silo_bool())
    {}

  // destructor
  ~IMC_Parameters() {}

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//

  //! Return total photons specified by the user
  uint64_t get_n_user_photon(void) const {return n_user_photon;}

  //! Return the preferred number of cells in a parallel communication 
  uint32_t get_grip_size(void) const {return grip_size;}

  //! Return maximum size of stored remote mesh
  uint32_t get_map_size(void) const {return map_size;}

  //! Return domain decomposition algorithm
  uint32_t get_dd_mode(void) const {return dd_mode;}

  //! Return completion 
  uint32_t get_completion_method(void) const {return completion_method;}

  //! Get the number of particles to run between MPI message processing
  uint32_t get_batch_size(void) const {return batch_size;}

  //! Get the desired number of particles in messages (particle passing only)
  uint32_t get_particle_message_size(void) const {return particle_message_size;}

  //! Get SILO write flag
  bool get_write_silo_flag(void) const {return write_silo_flag;}

  //--------------------------------------------------------------------------//
  // member data                                                              //
  //--------------------------------------------------------------------------//
  private:

  uint64_t n_user_photon; //! User requested number of photons per timestep

  //! Preferred number of cells in a grip, the number of cells that are sent 
  // in a message together
  uint32_t grip_size; 

  uint32_t map_size; //! Size of stored off-rank mesh cells
  uint32_t dd_mode; //! Mode of domain decomposed transport algorithm
  uint32_t completion_method; //! Method for handling completion messages
  uint32_t batch_size; //! How often to check for MPI passed data
  uint32_t particle_message_size; //! Preferred number of particles in MPI sends
  bool write_silo_flag; //! Write SILO output files flag
};

#endif // imc_parameters_h_
//----------------------------------------------------------------------------//
// end of imc_parameters.h
//----------------------------------------------------------------------------//
