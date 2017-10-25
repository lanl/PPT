//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   input.h
 * \author Alex Long
 * \date   July 18 2014
 * \brief  Reads data from XML input file and makes mesh skeleton
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef input_h_
#define input_h_

#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <stdlib.h>
#include <string>

#include "config.h"
#include "constants.h"
#include "region.h"

//==============================================================================
/*!
 * \class Input
 * \brief Reads input data from an XML file and stores that data
 *
 * Boost's XML parser is used to read an XML input file. This class stores that
 * information and provides functions to access it. This class also prints the
 * problem information.
 */
//==============================================================================
class Input
{
  public:
  Input( std::string fileName)
    : using_simple_mesh(false),
      using_detailed_mesh(false)
  {
    using boost::property_tree::ptree;
    using Constants::VACUUM; using Constants::REFLECT; using Constants::ELEMENT;
    using Constants::X_POS;  using Constants::Y_POS; using Constants::Z_POS;
    using Constants::X_NEG;  using Constants::Y_NEG; using Constants::Z_NEG;
    using Constants::PARTICLE_PASS;
    using Constants::CELL_PASS;
    using Constants::CELL_PASS_RMA;
    using Constants::RMA_COMPLETION;
    using Constants::MILAGRO_COMPLETION;
    using std::cout;
    using std::endl;
    using std::vector;

    // default completion routine is RMA
    completion_routine = RMA_COMPLETION;

    uint32_t x_key, y_key, z_key, key;
    // initialize nunmber of divisions in each dimension to zero
    uint32_t nx_divisions = 0;
    uint32_t ny_divisions = 0;
    uint32_t nz_divisions = 0;
    uint32_t region_ID;

    vector<float> x;
    vector<float> y;
    vector<float> z;

    ptree pt;
    read_xml(fileName, pt);
    std::string tempString;
    // traverse pt
    BOOST_FOREACH( ptree::value_type const& v, pt.get_child("prototype") )
    {
      //read in basic problem parameters
      if(v.first =="common")
      {
        tFinish = v.second.get<double>("t_stop");
        dt = v.second.get<double>("dt_start");
        tStart= v.second.get<double>("t_start");
        tMult = v.second.get<double>("t_mult" , 1.0);
        dtMax = v.second.get<double>("dt_max" , dt);
        n_photons =v.second.get<uint64_t>("photons");
        seed = v.second.get<int>("seed");
        grip_size = v.second.get<int>("grip_size", 10);
        map_size = v.second.get<int>("map_size");
        output_freq = v.second.get<int>("output_frequency",1);
        tempString = v.second.get<std::string>("tilt", std::string("FALSE"));
        if (tempString == "TRUE") use_tilt = 1;
        else use_tilt = 0;
        tempString = v.second.get<std::string>("use_combing",
                                                std::string("TRUE"));
        if (tempString == "TRUE") use_comb= 1;
        else use_comb = 0;

        // stratified sampling
        tempString = v.second.get<std::string>("stratified_sampling",
                                                std::string("FALSE"));
        if (tempString == "TRUE") use_strat= true;
        else use_strat = false;

        // write silo flag
        tempString = v.second.get<std::string>("write_silo",
          std::string("FALSE"));
        if (tempString == "TRUE") write_silo = true;
        else write_silo = false;

        // completion message type
        tempString = v.second.get<std::string>("completion_routine",
          std::string("RMA"));
        if (tempString == "RMA") completion_routine = RMA_COMPLETION;
        else if (tempString == "MILAGRO")
          completion_routine = MILAGRO_COMPLETION;
        else {
          cout<<"Completion routine not recognized, setting to MILAGRO"<<endl;
          completion_routine = MILAGRO_COMPLETION;
        }

        //number of particles to run between MPI message checks
        batch_size = v.second.get<uint32_t>("batch_size", 100);

        //preferred number of particles per MPI send
        particle_message_size = v.second.get<uint32_t>("particle_message_size", 100);

        // domain decomposed transport aglorithm
        tempString = v.second.get<std::string>("dd_transport_type",
                                               std::string("CELL_PASS"));
        if (tempString == "CELL_PASS") dd_mode = CELL_PASS;
        else if (tempString == "CELL_PASS_RMA") dd_mode = CELL_PASS_RMA;
        else if (tempString == "PARTICLE_PASS") dd_mode = PARTICLE_PASS;
        else {
          cout<<"WARNING: Domain decomposition method not recognized... ";
          cout<<"setting to PARTICLE PASSING method"<<endl;
          dd_mode = PARTICLE_PASS;
        }
      } //end common

      // read in basic problem parameters
      else if(v.first =="debug_options")
      {
        tempString = v.second.get<std::string>("print_verbose", "FALSE");
        if (tempString == "TRUE") print_verbose = true;
        else print_verbose = false;
        tempString = v.second.get<std::string>("print_mesh_info", "FALSE");
        if (tempString == "TRUE") print_mesh_info = true;
        else print_mesh_info = false;
      } //end common

      // read in detailed spatial information
      else if(v.first == "spatial") {
        using_detailed_mesh = true;
        double d_x_start, d_x_end, d_y_start, d_y_end, d_z_start, d_z_end;
        uint32_t d_x_cells, d_y_cells, d_z_cells;
        BOOST_FOREACH( ptree::value_type const& g, v.second )
        {
          if(g.first == "x_division") {
            //x information for this region
            d_x_start =g.second.get<double>("x_start");
            d_x_end = g.second.get<double>("x_end");
            d_x_cells = g.second.get<uint32_t>("n_x_cells");
            x_start.push_back(d_x_start);
            x_end.push_back(d_x_end);
            n_x_cells.push_back(d_x_cells);
            nx_divisions++;
            // push back the master x points for silo
            for (uint32_t i=0;i<d_x_cells;i++)
              x.push_back(d_x_start+i*(d_x_end-d_x_start)/d_x_cells);
          }

          if(g.first == "y_division") {
            //y information for this region
            d_y_start =g.second.get<double>("y_start");
            d_y_end = g.second.get<double>("y_end");
            d_y_cells = g.second.get<uint32_t>("n_y_cells");
            y_start.push_back(d_y_start);
            y_end.push_back(d_y_end);
            n_y_cells.push_back(d_y_cells);
            ny_divisions++;
            // push back the master y points for silo
            for (uint32_t i=0;i<d_y_cells;i++)
              y.push_back(d_y_start+i*(d_y_end-d_y_start)/d_y_cells);
          }

          if(g.first == "z_division") {
            //z information for this region
            d_z_start =g.second.get<double>("z_start");
            d_z_end = g.second.get<double>("z_end");
            d_z_cells = g.second.get<uint32_t>("n_z_cells");
            z_start.push_back(d_z_start);
            z_end.push_back(d_z_end);
            n_z_cells.push_back(d_z_cells);
            nz_divisions++;
            // push back the master z points for silo
            for (uint32_t i=0;i<d_z_cells;i++)
              z.push_back(d_z_start+i*(d_z_end-d_z_start)/d_z_cells);
          }

          if(g.first == "region_map") {
            x_key = (g.second.get<uint32_t>("x_div_ID"));
            y_key = (g.second.get<uint32_t>("y_div_ID"));
            z_key = (g.second.get<uint32_t>("z_div_ID"));
            region_ID = (g.second.get<uint32_t>("region_ID"));
            // make a unique key using the division ID of x,y and z
            // this mapping allows for 1000 unique divisions in
            // each dimension (way too many)
            key = z_key*1000000 + y_key*1000 + x_key;
            region_map[key] = region_ID;
          }
        }
      }

      //read in simple spatial data
      else if(v.first =="simple_spatial")
      {
        using_simple_mesh = true;

        x_start.push_back(v.second.get<double>("x_start"));
        x_end.push_back(v.second.get<double>("x_end"));
        n_x_cells.push_back(v.second.get<uint32_t>("n_x_cells"));

        y_start.push_back(v.second.get<double>("y_start"));
        y_end.push_back(v.second.get<double>("y_end"));
        n_y_cells.push_back(v.second.get<uint32_t>("n_y_cells"));

        z_start.push_back(v.second.get<double>("z_start"));
        z_end.push_back(v.second.get<double>("z_end"));
        n_z_cells.push_back(v.second.get<uint32_t>("n_z_cells"));

        region_ID = (v.second.get<uint32_t>("region_ID",0));

        // add spatial information to SILO information
        for (uint32_t i=0;i<n_x_cells[0];i++)
          x.push_back(x_start[0]+i*(x_end[0]-x_start[0])/n_x_cells[0]);
        for (uint32_t i=0;i<n_y_cells[0];i++)
          y.push_back(y_start[0]+i*(y_end[0]-y_start[0])/n_y_cells[0]);
        for (uint32_t i=0;i<n_z_cells[0];i++)
          z.push_back(z_start[0]+i*(z_end[0]-z_start[0])/n_z_cells[0]);

        // map zero key to region_ID
        region_map[0] = region_ID;
      }


      else if(v.first =="boundary")
      {
        //read in boundary conditions
        bool b_error = false;
        tempString = v.second.get<std::string>("bc_right");
        if      (tempString == "REFLECT") bc[X_POS] = REFLECT;
        else if (tempString == "VACUUM")  bc[X_POS] = VACUUM;
        else    b_error = true;

        tempString = v.second.get<std::string>("bc_left");
        if      (tempString == "REFLECT") bc[X_NEG] = REFLECT;
        else if (tempString == "VACUUM")  bc[X_NEG] = VACUUM;
        else    b_error = true;

        tempString = v.second.get<std::string>("bc_up");
        if      (tempString == "REFLECT") bc[Y_POS] = REFLECT;
        else if (tempString == "VACUUM")  bc[Y_POS] = VACUUM;
        else    b_error = true;

        tempString = v.second.get<std::string>("bc_down");
        if      (tempString == "REFLECT") bc[Y_NEG] = REFLECT;
        else if (tempString == "VACUUM")  bc[Y_NEG] = VACUUM;
        else    b_error = true;

        tempString = v.second.get<std::string>("bc_top");
        if      (tempString == "REFLECT") bc[Z_POS] = REFLECT;
        else if (tempString == "VACUUM")  bc[Z_POS] = VACUUM;
        else    b_error = true;

        tempString = v.second.get<std::string>("bc_bottom");
        if      (tempString == "REFLECT") bc[Z_NEG] = REFLECT;
        else if (tempString == "VACUUM")  bc[Z_NEG] = VACUUM;
        else    b_error = true;

        if (b_error) {
          cout<<"ERROR: Boundary type not reconginzed. Exiting..."<<endl;
          exit(EXIT_FAILURE);
        }
      }

      // read in region data
      else if(v.first == "regions") {
        BOOST_FOREACH( ptree::value_type const& g, v.second ) {
          if(g.first == "region") {
            Region temp_region;
            temp_region.set_ID(g.second.get<uint32_t>("ID"));
            temp_region.set_cV(g.second.get<double>("CV", 0.0));
            temp_region.set_rho(g.second.get<double>("density", 0.0));
            temp_region.set_opac_A(g.second.get<double>("opacA", 0.0));
            temp_region.set_opac_B(g.second.get<double>("opacB", 0.0));
            temp_region.set_opac_C(g.second.get<double>("opacC", 0.0));
            temp_region.set_opac_S(g.second.get<double>("opacS", 0.0));
            temp_region.set_T_e(g.second.get<double>("initial_T_e", 0.0));
            // default T_r to T_e if not specified
            temp_region.set_T_r(g.second.get<double>("initial_T_r", temp_region.get_T_e()));
            // map user defined ID to index in region vector
            region_ID_to_index[temp_region.get_ID()] = regions.size();
            // add to list of regions
            regions.push_back(temp_region);
          }
        }
      }

      // if both simple_spatial and detailed_spatial are true, exit with an
      // error message
      if ( using_detailed_mesh && using_simple_mesh) {
        cout<<"ERROR: Spatial information cannot be specified in both";
        cout<<" simple and detailed XML regions. Exiting...\n";
        exit(EXIT_FAILURE);
      }
    } //end xml parse

    //get global cell counts
    n_global_x_cells = std::accumulate(n_x_cells.begin(), n_x_cells.end(), 0);
    n_global_y_cells = std::accumulate(n_y_cells.begin(), n_y_cells.end(), 0);
    n_global_z_cells = std::accumulate(n_z_cells.begin(), n_z_cells.end(), 0);

    // set total number of divisions
    if (using_simple_mesh) n_divisions=1;
    else n_divisions = nx_divisions*ny_divisions*nz_divisions;

    //make sure at least one region is specified
    if (!regions.size()) {
      cout<<"ERROR: No regions were specified. Exiting..."<<endl;
      exit(EXIT_FAILURE);
    }

    // only one region can be specified in simple mesh mode
    if (using_simple_mesh && regions.size() != 1) {
      cout<<"ERROR: Only one region may be specified in simple mesh mode. ";
      cout<<" Exiting..."<<endl;
      exit(EXIT_FAILURE);
    }

    // for simple spatial input, region ID must match region ID in region map
    if (using_simple_mesh) {
      if(regions[0].get_ID() != region_map[0]) {
        cout<<"ERROR: Region ID in simple spatial blocl must match region ID ";
        cout<<"in region block. Exiting..."<<endl;
        exit(EXIT_FAILURE);
      }
    }

    // for detailed meshes, the total number of divisions  must equal the
    // number of unique region maps
    if (using_detailed_mesh && n_divisions != region_map.size()) {
      cout<<"ERROR: Number of total divisions must match the number of ";
      cout<<"unique region maps. Exiting..."<<endl;
      exit(EXIT_FAILURE);
    }

    // append the last point values and allocate SILO array
    x.push_back(x_end.back());
    y.push_back(y_end.back());
    z.push_back(z_end.back());
    silo_x = new float[x.size()];
    silo_y = new float[y.size()];
    silo_z = new float[z.size()];
    for (uint32_t i=0;i<x.size();i++) silo_x[i] = x[i];
    for (uint32_t j=0;j<y.size();j++) silo_y[j] = y[j];
    for (uint32_t k=0;k<z.size();k++) silo_z[k] = z[k];
  }

  ~Input() {
    delete[] silo_x;
    delete[] silo_y;
    delete[] silo_z;
  };

  void print_problem_info(void) const {
    using Constants::a;
    using Constants::c;
    using std::cout;
    using std::endl;
    using Constants::PARTICLE_PASS;
    using Constants::CELL_PASS;
    using Constants::CELL_PASS_RMA;
    using Constants::RMA_COMPLETION;
    using Constants::MILAGRO_COMPLETION;

    cout<<"Problem Specifications:";
    cout<<"Constants -- c: "<<c<<" (cm/sh) , a: "<<a <<endl;
    cout<<"Run Parameters-- Photons: "<<n_photons<<", time finish: "<<tFinish;
    cout<<" (sh), time step: "<<dt<<" (sh) ,"<<endl;

    cout<<" time multiplier: "<<tMult<<" , max dt:"<<dtMax;
    cout<<" (sh), Random number seed: "<<seed;
    cout<<" , output frequency: "<<output_freq<<endl;

    cout<<"material temperature: "<<Tm_initial;
    cout<<" (keV), radiation temperature: "<<Tr_initial<<" (keV)"<<endl;

    //cout<<"Sampling -- Emission Position: ";
    //if (use_tilt) cout<<"source tilting (x only), ";
    //else cout<<"uniform (default), ";
    //cout<<" Angle: ";
    cout<<"Stratified sampling in angle: ";
    if(use_strat) cout<<"TRUE"<<endl;
    else cout<<"FALSE"<<endl;

    if (use_comb) cout<<"Combing census enabled (default)"<<endl;
    else cout<<"No combing"<<endl;

    if (print_verbose) cout<<"Verbose printing mode enabled"<<endl;
    else cout<<"Terse printing mode (default)"<<endl;

    #ifdef VIZ_LIBRARIES_FOUND
      if (write_silo) cout<<"SILO output enabled"<<endl;
      else cout<<"SILO output disabled (default)"<<endl;
    #else
      if (write_silo)
        cout<<"NOTE: SILO libraries not linked... no visualization"<<endl;
    #endif
    cout<<"Spatial Information -- cells x,y,z: "<<n_global_x_cells<<" ";
    cout<<n_global_y_cells<<" "<<n_global_z_cells<<endl;
    if (using_simple_mesh) {
      cout<<"dx: "<< (x_end[0] - x_start[0])/n_x_cells[0]
        <<" dy: "<< (y_end[0] - y_start[0])/n_y_cells[0]
        <<" dz: "<< (z_end[0] - z_start[0])/n_z_cells[0]<<endl;
    }

    cout<<"--Material Information--"<<endl;
    for(uint32_t r=0; r<regions.size();r++) {
      cout<<" heat capacity: "<<regions[r].get_cV();
      cout<<" opacity constants: "<<regions[r].get_opac_A()
        <<" + "<<regions[r].get_opac_B()<<"^"<<regions[r].get_opac_C();
      cout<<", scattering opacity: "<<regions[r].get_scattering_opacity()<<endl;
    }

    cout<<"--Parallel Information--"<<endl;
    cout<<"DD algorithm: ";
    if (dd_mode == CELL_PASS) {
      cout<<"CELL PASSING"<<endl;
      cout<<"grip size: "<<grip_size;
      cout<<", map size: "<<map_size;
      cout<<", batch size: "<<batch_size;
      cout<<endl;
    }
    else if (dd_mode ==CELL_PASS_RMA) {
      cout<<"CELL PASSING (with RMA on MPI windows)"<<endl;
      cout<<"COMPILE PARAMETER: maximum number of RMA requests"<<endl;
      cout<<"grip size: "<<grip_size;
      cout<<", map size: "<<map_size;
      cout<<", batch size: "<<batch_size;
      cout<<endl;
    }
    else if (dd_mode == PARTICLE_PASS) {
      cout<<"PARTICLE PASSING"<<endl;
      cout<<"Batch size: "<<batch_size;
      cout<<", particle message size: "<<particle_message_size;
      cout<<endl;
    }
    else {
      cout<<"ERROR: Parallel method not specific correctly";
      cout<<" Exiting..."<<endl;
      exit(EXIT_FAILURE);
    }

    // cell pass RMA does not use completion routines
    if (dd_mode != CELL_PASS_RMA) {
      cout<<"Particle completion communication: ";
      if (completion_routine == MILAGRO_COMPLETION) {
        cout<<"Milagro binary tree"<<endl;
      }
      else if (completion_routine == RMA_COMPLETION) {
        cout<<"One-sided RMA binary tree"<<endl;
      }
      else {
        cout<<"ERROR: Particle completion method not specific correctly";
        cout<<" Exiting..."<<endl;
        exit(EXIT_FAILURE);
      }
    }
    else {
      cout<<"Completion messages are not used in CELL_PASS_RMA"<<endl;
    }

    cout<<endl;
  }

  uint32_t get_global_n_x_cells(void) const {return n_global_x_cells;}
  uint32_t get_global_n_y_cells(void) const {return n_global_y_cells;}
  uint32_t get_global_n_z_cells(void) const {return n_global_z_cells;}

  uint32_t get_x_division_cells(const uint32_t& div) const {
    return n_x_cells[div];
  }
  uint32_t get_y_division_cells(const uint32_t& div) const {
    return n_y_cells[div];
  }
  uint32_t get_z_division_cells(const uint32_t& div) const {
    return n_z_cells[div];
  }

  float* get_silo_x_ptr(void) {return silo_x;}
  float* get_silo_y_ptr(void) {return silo_y;}
  float* get_silo_z_ptr(void) {return silo_z;}

  uint32_t get_n_x_divisions(void) const {return n_x_cells.size();}
  uint32_t get_n_y_divisions(void) const {return n_y_cells.size();}
  uint32_t get_n_z_divisions(void) const {return n_z_cells.size();}

  double get_dx(const uint32_t& div) const {
    return (x_end[div] - x_start[div])/n_x_cells[div];
  }
  double get_dy(const uint32_t& div) const {
    return (y_end[div] - y_start[div])/n_y_cells[div];
  }
  double get_dz(const uint32_t& div) const {
    return (z_end[div] - z_start[div])/n_z_cells[div];
  }

  double get_x_start(const uint32_t& div) const { return x_start[div];}
  double get_y_start(const uint32_t& div) const { return y_start[div];}
  double get_z_start(const uint32_t& div) const { return z_start[div];}

  bool get_tilt_bool(void) const {return use_tilt;}
  bool get_comb_bool(void) const {return use_comb;}
  bool get_stratified_bool(void) const {return use_strat;}
  bool get_write_silo_bool(void) const {return write_silo;}

  bool get_verbose_print_bool(void) const {return print_verbose;}
  bool get_print_mesh_info_bool(void) const {return print_mesh_info;}
  int get_output_freq(void) const {return output_freq;}

  double get_dt(void) const {return dt;}
  double get_time_start(void) const {return tStart;}
  double get_time_finish(void) const {return tFinish;}
  double get_time_mult(void) const {return tMult;}
  double get_dt_max(void) const {return dtMax;}
  int get_rng_seed(void) const {return seed;}
  uint64_t get_number_photons(void) const {return n_photons;}
  uint32_t get_completion_routine(void) const {return completion_routine;}
  uint32_t get_batch_size(void) const {return batch_size;}
  uint32_t get_particle_message_size(void) const {return particle_message_size;}
  uint32_t get_grip_size(void) const {return grip_size;}
  uint32_t get_map_size(void) const {return map_size;}
  uint32_t get_dd_mode(void) const {return dd_mode;}

  //source functions
  double get_source_T(void) const {return T_source;}

  //material functions
  uint32_t get_n_regions(void) {return regions.size();}
  std::vector<Region> get_regions(void) {return regions;}
  uint32_t get_region_index(const uint32_t& x_div, const uint32_t& y_div,
    const uint32_t& z_div) {
    // make a unique key using the division ID of x,y and z
    // this mapping allows for 1000 unique divisions in
    // each dimension (way too many)
    uint32_t key = z_div*1000000 + y_div*1000 + x_div;
    return  region_ID_to_index[region_map[key]];
  }

  Constants::bc_type get_bc(const Constants::dir_type& direction) const
  {
    return bc[direction];
  }

  private:

  // flags
  bool using_simple_mesh;
  bool using_detailed_mesh;
  bool write_silo;

  Constants::bc_type bc[6]; //! Boundary condition array

  // timing
  double tStart; //! Starting time
  double dt; //! Timestep size
  double tFinish; //! Finish time
  double tMult; //! Timestep multiplier
  double dtMax; //! Maximum timestep size

  // initial conditions
  double Tm_initial; //! Initial material temperature
  double Tr_initial; //! Initial radiation temperature

  //material
  std::vector<Region> regions;
  std::unordered_map<uint32_t, uint32_t> region_map;
  std::unordered_map<uint32_t, uint32_t> region_ID_to_index;

  //source
  double T_source; //! Temperature of source

  // Monte Carlo parameters
  uint64_t n_photons; //! Photons to source each timestep
  uint32_t seed; //! Random number seed

  // Method parameters
  bool use_tilt; //! Use tilting for emission sampling
  bool use_comb; //! Comb census photons
  bool use_strat; //! Use strafifed sampling
  uint32_t dd_mode; //! Mode of domain decomposed transport algorithm
  uint32_t completion_routine; //! Method for handling transport completion

  // Debug parameters
  int output_freq; //! How often to print temperature information
  bool print_verbose; //! Verbose printing flag
  bool print_mesh_info; //! Mesh information printing flag

  // parallel performance parameters
  uint32_t grip_size; //! Preferred number of cells in a parallel communication
  uint32_t map_size; //! Size of stored off-rank mesh cells
  uint32_t batch_size; //! Particles to run between MPI message checks
  uint32_t particle_message_size; //! Preferred number of particles in MPI sends

  // detailed mesh specifications
  uint32_t n_divisions;
  std::vector<double> x_start;
  std::vector<double> x_end;
  std::vector<double> y_start;
  std::vector<double> y_end;
  std::vector<double> z_start;
  std::vector<double> z_end;
  std::vector<uint32_t> n_x_cells;
  std::vector<uint32_t> n_y_cells;
  std::vector<uint32_t> n_z_cells;
  uint32_t n_global_x_cells;
  uint32_t n_global_y_cells;
  uint32_t n_global_z_cells;

  // arrays for silo
  float *silo_x;
  float *silo_y;
  float *silo_z;
};

#endif // input_h_
//---------------------------------------------------------------------------//
// end of input.h
//---------------------------------------------------------------------------//
