//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   constants.h
 * \author Alex Long
 * \date   July 18 2014
 * \brief  Physical constants, custom enumerations and MPI tags
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef constants_h_
#define constants_h_

namespace Constants {
const double pi(3.1415926535897932384626433832795); //! Pi
const double c(299.792458); //! speed of light in cm/shake
const double c_SO(1.0); //! speed of light for Su-Olson problem
const double h(6.62606957e-34 * 1.0e-9/1.0e-8); //! Planck's constant in GJ/sh
const double k(1.60219e-31); //! energy conversion constant GJ/keV
const double a(0.01372); //! Boltzmann constant in GJ/cm^3/keV^4
const double a_SO(1.0); //! Boltzmann constant for SO problems

enum bc_type {REFLECT, VACUUM, ELEMENT, PROCESSOR}; //! Boundary conditions
enum dir_type {X_NEG, X_POS, Y_NEG, Y_POS, Z_NEG, Z_POS}; //! Directions
enum event_type {KILL, EXIT, PASS, CENSUS, WAIT}; //! Events
enum {PARTICLE_PASS, CELL_PASS, CELL_PASS_RMA}; //! DD types
enum {RMA_COMPLETION, MILAGRO_COMPLETION}; //! Completion manager types
enum {EMISSION, INITIAL_CENSUS}; //! Particle type for work packets
const int grip_id_tag(1); //! MPI tag for grip ID messages
const int cell_id_tag(2); //! MPI tag for requested cell ID messages
const int count_tag(3); //! MPI tag for completion count messages
const int photon_tag(4); //! MPI tag for photon messages
const int work_tag(5); //! MPI tag for work packet messages
const int tally_tag(6); //! MPI tag for tally messages
const int n_tally_tag(7); //! MPI tag for number of tally messages

//! MPI tag for cell messages NOTE: the number of grips in the message will
// added to the end of this number
const int cell_tag(8);
};

#endif // constants_h_
//---------------------------------------------------------------------------//
// end of constants.h
//---------------------------------------------------------------------------//
