#ifndef Input_h_
#define Input_h_
 
#include <stdlib.h>
#include <string>

#include "constants.h"
#include "KeyValueReader.h"
 
using std::cout;
using std::endl;
using std::vector;
using Constants::a;
using Constants::c;
 
class Input
{
  public:
  Input( int argc, char* argv[])
  {
    t_start = strtod(argv[1], NULL);
    t_stop = strtod(argv[2], NULL);
    dt = strtod(argv[3], NULL);
    dt_mult = strtod(argv[4], NULL);
    dt_max = strtod(argv[5], NULL);
    Tm_initial=strtod(argv[6], NULL);
    Tr_initial=strtod(argv[7], NULL);
    n_photons = strtod(argv[8], NULL);
    seed = strtod(argv[9], NULL);
    output_freq = strtod(argv[11], NULL);
    use_tilt = get_bool(argv[10][0]);
    print_verbose = get_bool(argv[12][0]);
    print_mesh_info= get_bool(argv[13][0]);
    x_start= strtod(argv[14], NULL);
    x_end= strtod(argv[15], NULL);
    n_x_cell= strtod(argv[16], NULL);

    // source
    source_cell = atoi(argv[17]);
    T_source = atoi(argv[18]);

    density= strtod(argv[19], NULL);
    CV= strtod(argv[20], NULL);
    opac_A= strtod(argv[21], NULL);
    opac_B= strtod(argv[22], NULL);
    opac_C= strtod(argv[23], NULL);
    // boundary conditions
    // default values
    l_bound =0;
    r_bound =0;
    std::string temp_string;
    temp_string = argv[24];
    if (temp_string == "VACUUM") l_bound = -1;
    else if (temp_string == "REFLECT")  l_bound = -2;
    else cout<<"Boundary condition not recognized"<<endl;

    // right face
    temp_string = argv[25];
    if (temp_string == "VACUUM") r_bound = -1;
    else if (temp_string == "REFLECT")  r_bound = -2;
    else cout<<"Boundary condition not recognized"<<endl;

    

/*
    kvr.getDouble("t_start", t_start);
    kvr.getDouble("t_stop", t_stop);
    kvr.getDouble("dt_start", dt);
    kvr.getDouble("dt_mult", dt_mult);
    kvr.getDouble("dt_max", dt_max);
    kvr.getDouble("Tm_initial", Tm_initial);
    kvr.getDouble("Tr_initial", Tr_initial);
    kvr.getInt("photons", n_photons);
    kvr.getInt("seed", seed);
    kvr.getInt("output_frequency",output_freq);
    kvr.getBool("use_tilt",use_tilt);

    // verbose print options
    kvr.getBool("print_verbose",print_verbose);
    kvr.getBool("print_mesh_info",print_mesh_info);

    // 1D geometry 
    kvr.getDouble("x_start", x_start);
    kvr.getDouble("x_end", x_end);
    kvr.getInt("n_x_cell", n_x_cell);

    // physical properties
    kvr.getDouble("density", density);
    kvr.getDouble("CV", CV);
    kvr.getDouble("opacity_A", opac_A);
    kvr.getDouble("opacity_B", opac_B);
    kvr.getDouble("opacity_C", opac_C);

    // boundary conditions
    // default values
    l_bound =0;
    r_bound =0;
    // left face
    std::string temp_string;
    kvr.getString("left_boundary", temp_string);
    if (temp_string == "VACUUM") l_bound = -1;
    else if (temp_string == "REFLECT")  l_bound = -2;
    else cout<<"Boundary condition not recognized"<<endl;

    // right face
    kvr.getString("right_boundary", temp_string);
    if (temp_string == "VACUUM") r_bound = -1;
    else if (temp_string == "REFLECT")  r_bound = -2;
    else cout<<"Boundary condition not recognized"<<endl;

    // source
    kvr.getDouble("T_source", T_source);
    kvr.getInt("source_cell", source_cell);*/
  }
   
  ~Input() {};
   
  void print_problem_info(void)
  {
    cout<<"Problem Specifications: ";
    cout<<"Method = piecewise constant IMC"<<endl;
 
    cout<<"Constants -- c: "<<c<<" (cm/sh) , a: "<<a <<endl;
    cout<<"Run Parameters-- Photons: "<<n_photons<<", time finish: "
        <<t_stop<<" (sh), time step: "<<dt<<" (sh) ,"<<endl;
    cout<<" timestep size multiplier: "<<dt_mult<<" , max dt:"<<dt_max
        <<" (sh), Random number seed: "<<seed
        <<" , output frequency: "<<output_freq<<endl;
     
    cout<<"material temperature: "<<Tm_initial
        <<" (keV), radiation temperature: "
        <<Tr_initial<<" (keV)"<<endl;
 
    cout<<"Sampling -- Emission Position: "; 
    if (use_tilt) cout<<"source tilting (x only), ";
    else cout<<"uniform (default), ";
 
    if (print_verbose) cout<<"Verbose printing mode enabled"<<endl;
    else cout<<"Terse printing mode (default)"<<endl;
 
    cout<<"Spatial Information -- elements: "<<n_x_cell<<" , x_start: "<<x_start
        <<" , x_end: "<< x_end<<" , dx: "<<get_dx()<<endl;
 
    cout<<"Material Information -- heat capacity: "<<CV
        <<" opacity constants: "<<opac_A<<" + "<<opac_B<<"^"<<opac_C<<endl;
 
    cout<<"Boundary Information -- Left: "<<l_bound<<" Right: "<<r_bound<<endl;
     
    cout<<endl;
  }

  bool get_bool(char letter)
  {
    if (letter=='t')
      return true;
    else 
      false;
  }
 
  int get_n_x_cell(void) {return n_x_cell;}
  double get_dx(void) {return (x_end-x_start)/n_x_cell;}
  double get_x_start(void) {return x_start;}
  double get_initial_Tm(void) {return Tm_initial;}
  double get_initial_Tr(void) {return Tr_initial;}
  int get_output_freq(void) {return output_freq;}
 
  bool get_tilt_bool(void) {return use_tilt;}
  bool get_lump_emission_bool(void) {return lump_emission;}
  bool get_lump_time_bool(void) {return lump_time;}
  bool get_verbose_print_bool(void) {return print_verbose;}
  bool get_print_mesh_info_bool(void) {return print_mesh_info;}
 
  double get_dt(void) {return dt;}
  double get_time_start(void) {return t_start;}
  double get_time_finish(void) {return t_stop;}
  double get_time_mult(void) {return dt_mult;}
  double get_dt_max(void) {return dt_max;}
  int get_number_photons(void) {return n_photons;}
  int get_rng_seed(void) {return seed;}
 
  unsigned int get_l_bound(void) const {return l_bound;}
  unsigned int get_r_bound(void) const {return r_bound;}
 
  //source functions
  unsigned int get_source_element(void) {return source_cell;}
  double get_source_T(void) {return T_source;}
 
  //material functions
  double get_density(void) {return density;}
  double get_CV(void) {return CV;}
  double get_opacity_A(void) {return opac_A;}
  double get_opacity_B(void) {return opac_B;}
  double get_opacity_C(void) {return opac_C;}
  
 
  private:
 
  double t_start;
  double dt;
  double t_stop;
  double dt_mult;
  double dt_max;
  double Tm_initial;
  double Tr_initial;
  int n_photons;
  int seed;
  int output_freq;
  bool use_tilt;
  bool lump_emission;
  bool lump_time;
  bool print_verbose;
  bool print_mesh_info;
 
  //spatial
  int n_x_cell;
  double x_start;
  double x_end; 
 
  //source
  double T_source;
  int source_cell;
 
  //material
  double density; //g/cc
  double CV;  // jk/g/keV
  double opac_A;
  double opac_B;
  double opac_C;
 
  //bounds
  int l_bound;
  int r_bound;
};
 
#endif // Input_h_
