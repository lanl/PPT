//Copyright 2010-2011, D. E. Shaw Research.
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are
//met:
//
//* Redistributions of source code must retain the above copyright
//  notice, this list of conditions, and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright
//  notice, this list of conditions, and the following disclaimer in the
//  documentation and/or other materials provided with the distribution.
//
//* Neither the name of D. E. Shaw Research nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


/**
* \file RNG.h
* \brief This file contains the class RNG.
* First created by: Carolyn McGraw and Daniel Holladay
* First created on: January 2013
*
* Update history
* 1. generates a single # in every call of the function
*/

#ifndef RNG_h
#define RNG_h

#include <stdlib.h>
#include "random123/threefry.h"

//===========================================================================//
/*!
* \class RNG
  * \brief
* Implements a counter based PRNG based on the threefish method.
* This is perfect for parallel generation of random numbers.
*/
//===========================================================================//
class RNG
{
public:

  // CREATORS
  //! Default constructors.
  RNG(int size=10000) :
    array_size(size), index(0), counter(0), seed(0) {
    rn_bank = new double[array_size];
  }

  //! Destructor.
  ~RNG()
  {
    delete [] rn_bank;
  }

  void set_seed(int new_seed){
    seed = new_seed;
		regenerate_random_numbers();
}

double generate_random_number(){

  if(counter == array_size){
    regenerate_random_numbers();
  }

  return rn_bank[counter++];
}

private:

  void regenerate_random_numbers() {

    counter = 0;  
    threefry2x64_ctr_t c = {{}};
    threefry2x64_key_t k = {{}};
    k.v[0] = seed;
    threefry2x64_ctr_t r;

    for(size_t i = 0; i < array_size; ++i){
      c.v[0] = index++;
      r = threefry2x64(c,k);
      rn_bank[i] = r.v[0]/(double(std::numeric_limits<unsigned long>::max())) ;
    }
  }

  size_t array_size;
  size_t index;
  size_t counter;
  size_t seed;
  double *rn_bank;

};

#endif // RNG_h
