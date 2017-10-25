# Unit tests
cmake_minimum_required (VERSION 2.8)
# Set Intel MPI Compiler
#set(CMAKE_CXX_COMPILER mpiicpc)
# Set OpenMPI compiler
#set(CMAKE_CXX_COMPILER mpic++)

if ("${SITENAME}" STREQUAL "Moonlight" OR ${SITENAME} MATCHES "ccscs[0-9]+")
  set(MPI_EXEC mpirun)
elseif (${SITENAME} STREQUAL "Trinitite")
  set(MPI_EXEC aprun)
endif ()

include_directories(${Boost_INCLUDE_DIR})
include_directories($ENV{METIS_INC_DIR})
link_directories($ENV{PARMETIS_LIB_DIR})
link_directories($ENV{SILO_ROOT}/lib)
link_directories($ENV{HDF5_ROOT}/lib)

# Cell class test
add_executable(test_cell_exe test_cell.cc)
add_test(NAME test_cell COMMAND ${MPI_EXEC} -n 1 ./test_cell_exe)

# Photon class test
add_executable(test_photon_exe test_photon.cc)
add_test(NAME test_photon COMMAND ${MPI_EXEC} -n 1 ./test_photon_exe)

# Buffer class test
add_executable(test_buffer_exe test_buffer.cc)
add_test(NAME test_buffer COMMAND ${MPI_EXEC} -n 1 ./test_buffer_exe)

# Work_Packet class test
add_executable(test_work_packet_exe test_work_packet.cc)
add_test(NAME test_work_packet COMMAND ${MPI_EXEC} -n 1 ./test_work_packet_exe)

# MPI_Types class test
add_executable(test_mpi_types_exe test_mpi_types.cc)
add_test(NAME test_mpi_types COMMAND ${MPI_EXEC} -n 1 ./test_mpi_types_exe)

# Particle Completion classes test
add_executable(test_completion_manager_exe test_completion_manager.cc)
add_test(NAME test_completion_manager COMMAND ${MPI_EXEC} -n 7 ./test_completion_manager_exe)

# remap_census functions test
#add_executable(test_remap_census_exe test_remap_census.cc)
#add_test(NAME test_remap_census COMMAND ${MPI_EXEC} -n 8 ./test_remap_census_exe)

# Tally_Manager class test
add_executable(test_tally_manager_exe test_tally_manager.cc)
add_test(NAME test_tally_manager COMMAND ${MPI_EXEC} -n 8 ./test_tally_manager_exe)

# Parmetis Test with 1 processor
add_executable(test_parmetis_1_exe test_parmetis.cc)
target_link_libraries(test_parmetis_1_exe parmetis)
target_link_libraries(test_parmetis_1_exe metis)
add_test(NAME test_parmetis_1 COMMAND ${MPI_EXEC} -n 1 ./test_parmetis_1_exe)

# Parmetis Test with 4 processors
add_executable(test_parmetis_4_exe test_parmetis.cc)
target_link_libraries(test_parmetis_4_exe parmetis)
target_link_libraries(test_parmetis_4_exe metis)
add_test(NAME test_parmetis_4 COMMAND ${MPI_EXEC} -n 4 ./test_parmetis_4_exe)

# copy these input files for Input, IMC_State, Mesh and write_silo tests
configure_file(simple_input.xml ${CMAKE_CURRENT_BINARY_DIR}/simple_input.xml COPYONLY)
configure_file(large_particle_input.xml ${CMAKE_CURRENT_BINARY_DIR}/large_particle_input.xml COPYONLY)
configure_file(three_region_mesh_input.xml ${CMAKE_CURRENT_BINARY_DIR}/three_region_mesh_input.xml COPYONLY)

# Input class test
add_executable(test_input_exe test_input.cc)
add_test(NAME test_input COMMAND ${MPI_EXEC} -n 1 ./test_input_exe)

# Mesh class test
add_executable(test_mesh_exe test_mesh.cc)
add_test(NAME test_mesh COMMAND ${MPI_EXEC} -n 1 ./test_mesh_exe)

# IMC_State class test
add_executable(test_imc_state_exe test_imc_state.cc)
add_test(NAME test_imc_state COMMAND ${MPI_EXEC} -n 2 ./test_imc_state_exe)

# write_silo function test
if (VIZ_LIBRARIES_FOUND)
  add_executable(test_write_silo_exe test_write_silo.cc)
  target_link_libraries(test_write_silo_exe parmetis)
  target_link_libraries(test_write_silo_exe metis)
  target_link_libraries(test_write_silo_exe hdf5)
  target_link_libraries(test_write_silo_exe siloh5)
  add_test(NAME test_write_silo COMMAND ${MPI_EXEC} -n 1 ./test_write_silo_exe)
endif ()
