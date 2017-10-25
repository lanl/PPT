//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   timer.h
 * \author Alex Long
 * \date   August 4 2016
 * \brief  Class for tracking multiple timers
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef timer_h_
#define timer_h_

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>

class Timer
{
  public:

  Timer(void) {}
  ~Timer(void) {}

  //! Start or continute timer with name
  void start_timer(std::string name) {
    // start timer if new
    if (times.find(name) == times.end())
      times[name] = 0.0;
    temp_start = std::chrono::high_resolution_clock::now();
  }

  //! Stop timer with name (must be the last active timer)
  void stop_timer(std::string name) {
    double time_seconds = 
      std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - temp_start).count() / 1.0e6;
    times[name] += time_seconds;
  }

  //! Print all timers that have been measured with this clsas
  void print_timers(void) const {
    typedef std::unordered_map<std::string, double>::const_iterator map_itr;
    for (map_itr i = times.cbegin(); i != times.cend(); i++) {
      std::cout<<i->first<<": "<<i->second<<std::endl;
    }
  }

  //! Get the current elapsed time for a timer
  double get_time(std::string name) {
    return times[name];
  }

  private:
  //! Map of timer names to times
  std::chrono::high_resolution_clock::time_point temp_start;
  std::unordered_map<std::string, double> times;
};

#endif // timer_h_
