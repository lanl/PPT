//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   census_creation.h
 * \author Alex Long
 * \date   January 1 2015
 * \brief  Function for creating initial census particles 
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//
// $Id$
//---------------------------------------------------------------------------//

#ifndef census_creation_h_
#define census_creation_h_

#include <vector>

#include "photon.h"

double get_photon_list_E(std::vector<Photon> photons) {
  double total_E = 0.0;
  for (std::vector<Photon>::iterator iphtn=photons.begin(); 
      iphtn<photons.end(); 
      iphtn++)
    total_E += iphtn->get_E();
  return total_E;
}

#endif // def census_creation_h_
//---------------------------------------------------------------------------//
// end of census_creation.h
//---------------------------------------------------------------------------//
