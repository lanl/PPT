//----------------------------------*-C++-*-----------------------------------//
/*!
 * \file   message_counter.h
 * \author Alex Long
 * \date   July 20 2016
 * \brief  Struct that holds all network message counters
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//----------------------------------------------------------------------------//

#ifndef message_counter_h_
#define message_counter_h_

#include <vector>

struct Message_Counter {

  public:
  Message_Counter()
  : n_cell_messages(0),
    n_cells_sent(0),
    n_particle_messages(0),
    n_particles_sent(0),
    n_sends_posted(0),
    n_sends_completed(0),
    n_receives_posted(0),
    n_receives_completed(0)
    {}
  ~Message_Counter() {}

  void reset_counters(void) {
    n_cell_messages = 0;
    n_cells_sent = 0;
    n_particle_messages = 0;
    n_particles_sent = 0;
    n_sends_posted = 0;
    n_sends_completed = 0;
    n_receives_posted = 0;
    n_receives_completed = 0;
  }


  uint32_t n_cell_messages; //! Number of cell messages
  uint32_t n_cells_sent; //! Number of cells passed
  uint32_t n_particle_messages; //! Number of particle messages
  uint64_t n_particles_sent; //! Number of particles sent (64 bit)
  uint32_t n_sends_posted; //! Number of sent messages posted
  uint32_t n_sends_completed; //! Number of sent messages completed
  uint32_t n_receives_posted; //! Number of received messages completed
  uint32_t n_receives_completed; //! Number of received messages completed


};

#endif // message_counter_h_
//----------------------------------------------------------------------------//
// end of message_counter.h
//----------------------------------------------------------------------------//
