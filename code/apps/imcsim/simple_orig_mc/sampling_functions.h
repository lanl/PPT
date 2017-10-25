/*
  Author: Alex Long
  Date: 9/17/2014
  Name: sampling_functions.h
*/

#ifndef sampling_functions_h_
#define sampling_functions_h_

#include <iostream>
#include <vector>
#include <stdlib.h>

#include "RNG.h"
#include "constants.h"

using Constants::pi;

using std::sqrt;
using std::sin;
using std::cos;

double get_uniform_angle(RNG* rng) {
  return rng->generate_random_number()*2.0-1.0;
}

double get_source_angle(RNG* rng) {
  //return std::max(rng->generate_random_number(), rng->generate_random_number());
  return sqrt(rng->generate_random_number());
}

void get_uniform_angle(double* angle, RNG* rng) {
  double mu =rng->generate_random_number()*2.0-1.0; 
  double phi = rng->generate_random_number()*2.0*pi;
  double sin_theta = sqrt(1.0 - mu*mu);
  angle[0] = sin_theta*cos(phi);
  angle[1] = sin_theta*sin(phi);
  angle[2] = mu;
}

void get_source_angle(double* angle, RNG* rng) {
  double mu =sqrt(rng->generate_random_number());
  double phi = rng->generate_random_number()*2.0*pi;
  double sin_theta = sqrt(1.0 - mu*mu);
  angle[0] = sin_theta*cos(phi);
  angle[1] = sin_theta*sin(phi);
  angle[2] = mu;
}


#endif
