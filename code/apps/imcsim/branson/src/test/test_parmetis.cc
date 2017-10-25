//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   test_parmetis.cc
 * \author Alex Long
 * \date   February 11 2016
 * \brief  Make sure ParMetis can be called with a simple graph
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <numeric>
#include <mpi.h>
#include <parmetis.h>
#include <vector>

int main (int argc, char *argv[]) {

  using std::cout;
  using std::endl;
  using std::vector;
  using std::partial_sum;

  MPI_Init(&argc, &argv);

  int rank, n_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &n_rank);

  int nfail = 0;

  // Try to decompose a simple graph with Parmetis
  {

    bool simple_partitioning_pass=true;

    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    //get the number of cells on each processor
    //vtxdist has number of vertices on each rank, same for all ranks
    const int n_cell_on_rank = 100;
    vector<int> start_ncells(n_rank, n_cell_on_rank); //cells on each rank
    vector<int> vtxdist(n_rank, 0);  //prefix sum for cells on rank

    partial_sum(start_ncells.begin(),
                start_ncells.end(),
                vtxdist.begin());
    vtxdist.insert(vtxdist.begin(), 0);

    //build adjacency list needed for ParMetis call for each rank
    vector<int> xadj;
    vector<int> adjncy;
    int adjncy_counter = 0;
    uint32_t g_ID; //! Global ID
    for (uint32_t i=0; i<n_cell_on_rank;i++) {

      //get a global id for this cell
      g_ID = i + rank*n_cell_on_rank;

      //starting index in xadj for this cell's nodes
      xadj.push_back(adjncy_counter);

      //first cell in row
      if (i == 0) {
        adjncy.push_back(g_ID+1);
        adjncy_counter++;
        if (rank != 0) {
          adjncy.push_back(g_ID-n_cell_on_rank);
          adjncy_counter++;
        }
        if (rank != n_rank-1) {
          adjncy.push_back(g_ID+n_cell_on_rank);
          adjncy_counter++;
        }
      }
      //last cell in row
      else if (i == n_cell_on_rank-1) {
        adjncy.push_back(g_ID-1);
        adjncy_counter++;
        if (rank != 0) {
          adjncy.push_back(g_ID-n_cell_on_rank);
          adjncy_counter++;
        }
        if (rank != n_rank-1) {
          adjncy.push_back(g_ID+n_cell_on_rank);
          adjncy_counter++;
        }
      }
      //other cells in row
      else {
        adjncy.push_back(g_ID-1);
        adjncy_counter++;
        adjncy.push_back(g_ID+1);
        adjncy_counter++;
        if (rank != 0) {
          adjncy.push_back(g_ID-n_cell_on_rank);
          adjncy_counter++;
        }
        if (rank != n_rank-1) {
          adjncy.push_back(g_ID+n_cell_on_rank);
          adjncy_counter++;
        }
      }
    }  // end loop over cells to build adjacency lists
    xadj.push_back(adjncy_counter);

    int wgtflag = 0; //no weights for cells
    int numflag = 0; //C-style numbering
    int ncon = 1;
    int nparts = n_rank; //sub-domains = n_rank

    float *tpwgts = new float[nparts];
    for (int i=0; i<nparts; i++) tpwgts[i]=1.0/nparts;

    float *ubvec = new float[ncon];
    for (int i=0; i<ncon; i++) ubvec[i]=1.05;

    int options[3];
    options[0] = 1; // 0--use default values, 1--use the values in 1 and 2
    options[1] = 3; //output level
    options[2] = 1242; //random number seed

    int edgecut =0;
    int *part = new int[n_cell_on_rank];

    ParMETIS_V3_PartKway( &vtxdist[0],   // array describing how cells are distributed
                          &xadj[0],   // how cells are stored locally
                          &adjncy[0], // how cells are stored loccaly
                          NULL,       // weight of vertices
                          NULL,       // weight of edges
                          &wgtflag,   // 0 means no weights for node or edges
                          &numflag,   // numbering style, 0 for C-style
                          &ncon,      // weights per vertex
                          &nparts,    // number of sub-domains
                          tpwgts,     // weight per sub-domain
                          ubvec,      // unbalance in vertex weight
                          options,    // options array
                          &edgecut,   // OUTPUT: Number of edgecuts
                          part,       // OUTPUT: partition of each vertex
                          &comm); // MPI communicator

    // ParMetis should not partition mesh if one rank is used (this is the
    // edgecut parameter)
    if (n_rank == 1) {
      if (edgecut != 0) simple_partitioning_pass = false;
    }
    // ParMetis should partition the mesh if more than one rank is used
    else {
      if (edgecut == 0) simple_partitioning_pass = false;
    }

    vector<int> n_on_rank(n_rank);
    for (uint32_t i =0; i<n_cell_on_rank; i++)
      n_on_rank[part[i]]++;

    cout<<"Cells on rank "<<rank<<": "<<n_on_rank[rank]<<endl;
    // Number of cells on this rank should be greater than zero
    if (n_on_rank[rank] <= 0) simple_partitioning_pass = false;

    if (simple_partitioning_pass) cout<<"TEST PASSED: simple ParMetis partitioning"<<endl;
    else {
      cout<<"TEST FAILED: simple ParMetis partitioning"<<endl;
      nfail++;
    }

    delete[] part;
  } //end simple_partitioning test scope

  MPI_Finalize();

  return nfail;
}
//---------------------------------------------------------------------------//
// end of test_parmetis.cc
//---------------------------------------------------------------------------//
