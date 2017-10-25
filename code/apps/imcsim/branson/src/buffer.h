//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   buffer.h
 * \author Alex Long
 * \date   December 15 2015
 * \brief  Class for simplifying data management in MPI messaging
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef buffer_h_
#define buffer_h_

#include <vector>
#include <mpi.h>

template<class T>
class Buffer {

  public:
  Buffer()
  : status(EMPTY),
    buffer_recv_size(0),
    rank(MPI_PROC_NULL)
  {}
  ~Buffer() {}

  void fill (std::vector<T> _object) {
    object = _object;
    status = READY;
  }

  void *get_buffer (void) {
    if (object.size()>0) return &object[0];
    else return &object;
  }

  std::vector<T>& get_object(void) {return object;}

  void resize(uint32_t new_size) {object.resize(new_size);}
  void clear(void) {object.clear(); status=EMPTY;}
  void reset(void) {status=EMPTY;}
  void set_sent(void) {status = SENT;}
  void set_received(void) {status = RECEIVED;}
  void set_awaiting(void) {status = AWAITING;}

  bool sent(void) const {return status == SENT;}
  bool awaiting(void) const {return status == AWAITING;}
  bool ready(void) const {return status == READY;}
  bool received(void) const  {return status == RECEIVED;}
  bool empty(void) const {return status == EMPTY;}
  const std::vector<uint32_t>& get_grip_IDs(void) const {return grip_IDs;}
  uint32_t get_grip_ID(void) const {return grip_IDs[0];}
  int32_t get_rank(void) const {return rank;}
  uint32_t get_receive_size(void) const {return buffer_recv_size;}

  void set_grip_ID(const uint32_t _grip_ID) {
    grip_IDs = std::vector<uint32_t>(1,_grip_ID);
  }

  void set_grip_IDs(std::vector<uint32_t> _grip_IDs) {grip_IDs = _grip_IDs;}

  void set_rank(uint32_t _rank) {rank=_rank;}
  void set_receive_size(uint32_t _recv_size) {buffer_recv_size = _recv_size;}

  private:
  uint32_t status; //! Current status of the buffer

  //! Actual elements sent over MPI (buffer is generally oversized)
  uint32_t buffer_recv_size;

  //! Rank (source for receive, destination for send) used for convenience in
  // mesh passing
  int32_t rank;

  //! Grips received or sent, used for convenience in mesh passing
  std::vector<uint32_t> grip_IDs;

  std::vector<T> object; //! Where sent/received data is stored

  //! Buffer statuses, used in completion routine
  enum {EMPTY, READY, SENT, AWAITING, RECEIVED};
};

#endif // buffer_h_
//---------------------------------------------------------------------------//
// end of buffer.h
//---------------------------------------------------------------------------//
