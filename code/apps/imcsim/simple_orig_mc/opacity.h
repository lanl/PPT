/*
  Author: Alex Long
  Date: 6/29/2014
  Name: opacity.h
*/

#include <stdio>
#include <vector>
#include <cmath>

using std::power;
using std::cout;
using std::vector;

class Opacity
{
  Opacity(void) {}
  ~Opacity(void) {}

  virtual double evaluate_opacity(const vector<double>& shape_vals, const vector<double>& mat_T) = 0;

  void check_mat_T(const vector<double>& mat_T) {
    // check that there is a temperature value in the vector
    if (mat_T.size() <= 1) cout<<"ERROR: no temepratures passed to opacity.\n";

    // check that all temperatures are greater than zero
    for (unsigned int i=0; i<mat_T.size(); i++)
      if (mat_T[i] < 0.0) cout<<"ERROR: temperature less than zero.\n";
  }

  void check_shape_vals(const vector<double>& shape_vals) {
    // check that there is a shape value in the vector
    if (shape_vals.size() <= 1) cout<<"ERROR: no shape values passed to opacity.\n";

    // check that all shape values are greater than zero
    for (unsigned int i=0; i<mat_T.size(); i++)
      if (shape_vals[i] < 0.0) cout<<"ERROR: shape value less than zero.\n";
  }
  
  double get_opacity(matT) {
    return 300.0/pow(matT,3);
  } 

};


// child of Opacity class for piecewise constant opacity
class PC_Opacity : public Opacity
{
  PC_Opacity(void) { 
    Opacity();
  }

  ~PC_Opacity(void) {}

  virtual double evaluate_opacity(const vector<double>& shape_vals, const vector<double>& mat_T) {
    check_mat_T(mat_T);
    double op = Opacity::get_opacity(mat_T[0]);
    return op;
  }

};
