//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   mpi_types.h
 * \author Alex Long
 * \date   May 12 2016
 * \brief  Creates and provides access to custom MPI dataypes
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef mpi_types_h_
#define mpi_types_h_

//==============================================================================
/*!
 * \class MPI_Types
 * \brief Creates, commits and provides access to custom MPI types
 * 
 * MPI allows custom types to be specified and then used directly in MPI send
 * receive and get calls. This class creates MPI types needed for particles, 
 * cells and work packets
 * \example no test yet
 */
//==============================================================================

class MPI_Types
{
  public:
  //! constructor
  MPI_Types(void) {

    // make and commit the MPI cell type
    {
      MPI_Datatype og_MPI_Cell;

      // remake the MPI cell datatype from mesh
      const int cell_entry_count = 3 ; 
      // 7 uint32_t, 6 int, 13 double
      int cell_array_of_block_length[4] = {16, 6, 14};
      // Displacements of each type in the cell
      MPI_Aint cell_array_of_block_displace[3] = 
        {0, 16*sizeof(uint32_t),  16*sizeof(uint32_t)+6*sizeof(int)};
      //Type of each memory block
      MPI_Datatype cell_array_of_types[3] = {MPI_UNSIGNED, MPI_INT, MPI_DOUBLE}; 

      MPI_Type_create_struct(cell_entry_count, cell_array_of_block_length,
        cell_array_of_block_displace, cell_array_of_types, &og_MPI_Cell);

      // Commit the type to MPI so it recognizes it in communication calls
      MPI_Type_commit(&og_MPI_Cell);

      MPI_Type_size(og_MPI_Cell, &mpi_cell_size);
      // Duplicate the type so it's recognized when returned out of this
      // context (I don't know why this is necessary)
      MPI_Type_dup(og_MPI_Cell, &MPI_Cell);
    }


    // make and commit the MPI particle type
    {
      MPI_Datatype og_MPI_Particle;

      const int particle_entry_count = 2 ;

      // 2 uint32_t, 9 double
      int particle_array_of_block_length[3] = { 2, 9};

      // Displacements of each type in the cell
      MPI_Aint particle_array_of_block_displace[2] = 
        {0, 2*sizeof(uint32_t)};

      //Type of each memory block
      MPI_Datatype particle_array_of_types[2] = {MPI_UNSIGNED, MPI_DOUBLE};

      MPI_Type_create_struct(particle_entry_count, 
        particle_array_of_block_length, particle_array_of_block_displace, 
        particle_array_of_types, &og_MPI_Particle);

      // Commit the type to MPI so it recognizes it in communication calls
      MPI_Type_commit(&og_MPI_Particle);

      MPI_Type_size(og_MPI_Particle, &mpi_particle_size);
      // Duplicate the type so it's recognized when returned out of this
      // context (I don't know why this is necessary)
      MPI_Type_dup(og_MPI_Particle, &MPI_Particle);
    }

    // make and commit the MPI tally type
    {
      // make the Work Packet 
      const int tally_entry_count = 2; 

      // 1 uint32_t, 1 double
      int tally_array_of_block_length[2] = { 2, 1};

      // displacements of each type in the cell
      MPI_Aint tally_array_of_block_displace[2] = 
        {0, 2*sizeof(uint32_t)};

      // type of each memory block
      MPI_Datatype tally_array_of_types[2] = {MPI_UNSIGNED, MPI_DOUBLE};

      MPI_Datatype og_MPI_Tally;
      MPI_Type_create_struct(tally_entry_count, tally_array_of_block_length,
        tally_array_of_block_displace, tally_array_of_types, &og_MPI_Tally);

      // Commit the type to MPI so it recognizes it in communication calls
      MPI_Type_commit(&og_MPI_Tally);

      MPI_Type_size(og_MPI_Tally, &mpi_tally_size);

      // duplicate the type so it's recognized when returned out of this
      // context (I don't know why this is necessary)
      MPI_Type_dup(og_MPI_Tally, &MPI_Tally);
    }

    // make and commit the MPI work packet type
    {
      // make the Work Packet 
      const int wp_entry_count = 2 ; 
      // 7 uint32_t, 7 doubles
      int wp_array_of_block_length[3] = { 8, 7};
      // Displacements of each type in the cell
      MPI_Aint wp_array_of_block_displace[2] = 
        {0, 8*sizeof(uint32_t)};
      //Type of each memory block
      MPI_Datatype wp_array_of_types[2] = {MPI_UNSIGNED, MPI_DOUBLE};

      MPI_Datatype og_MPI_Work_Packet;
      MPI_Type_create_struct(wp_entry_count, wp_array_of_block_length, 
        wp_array_of_block_displace, wp_array_of_types, &og_MPI_Work_Packet);

      // Commit the type to MPI so it recognizes it in communication calls
      MPI_Type_commit(&og_MPI_Work_Packet);

      MPI_Type_size(og_MPI_Work_Packet, &mpi_work_packet_size);
      // Duplicate the type so it's recognized when returned out of this
      // context (I don't know why this is necessary)
      MPI_Type_dup(og_MPI_Work_Packet, &MPI_Work_Packet);
    }
  }

  // destructor
  ~MPI_Types() {}

  //--------------------------------------------------------------------------//
  // const functions                                                          //
  //--------------------------------------------------------------------------//

  //! Return reference to MPI_Particle for use in communication calls
  MPI_Datatype get_particle_type(void) const {return MPI_Particle;}

  //! Return reference to MPI_Cell for use in communication calls
  MPI_Datatype get_cell_type(void) const {return MPI_Cell;}

  //! Return reference to MPI_Work_Packet for use in communication calls
  MPI_Datatype get_work_packet_type(void) const {return MPI_Work_Packet;}

  //! Return reference to MPI_abs_E for use in communication calls
  MPI_Datatype get_tally_type(void) const {return MPI_Tally;}

  //! Return size of MPI particle in bytes
  int get_particle_size(void) const {return mpi_particle_size;}

  //! Return size of MPI cell in bytes
  int get_cell_size(void) const {return mpi_cell_size;}

  //! Return size of MPI work packet in bytes
  int get_work_packet_size(void) const {return mpi_work_packet_size;}

  //! Return size of MPI abs_E tally in bytes
  int get_tally_size(void) const {return mpi_tally_size;}

  //--------------------------------------------------------------------------//
  // member data                                                              //
  //--------------------------------------------------------------------------//
  private:
  MPI_Datatype MPI_Particle; //! Custom MPI datatype for particles
  MPI_Datatype MPI_Cell; //! Custom MPI datatype for mesh cell
  MPI_Datatype MPI_Tally; //! Custom MPI datatype for tally
  MPI_Datatype MPI_Work_Packet; //! Custom MPI datatype for work packet
  int mpi_cell_size; //! Size of MPI_Cell datatype
  int mpi_particle_size; //! Size of MPI_Particle datatype
  int mpi_tally_size; //! Size of MPI_Tally datatype
  int mpi_work_packet_size; //! Size of MPI_Work_Packet datatype
};

#endif // mpi_types_h_
//---------------------------------------------------------------------------//
// end of mpi_types.h
//---------------------------------------------------------------------------//
