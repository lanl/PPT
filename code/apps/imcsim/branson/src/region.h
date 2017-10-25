//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   region.h
 * \author Alex Long
 * \date   April 4 2016
 * \brief  Describes a mesh region with independent physical properties
 * \note   ***COPYRIGHT_GOES_HERE****
 */
//---------------------------------------------------------------------------//

#ifndef region_h_
#define region_h_

#include <cmath>

//==============================================================================
/*!
 * \class Region
 * \brief Contains data necessary to physically define a region of the problem
 * 
 * This class holds heat capcaity, density, opacity and initial temperature 
 * conditions. All of these are necessary to initialize cells created in 
 * this spatial region.
 */
//==============================================================================
class Region
{
  public:
  Region(void) {
    T_s = 0.0;
  }
  ~Region(void) {}

/*****************************************************************************/
/* const functions                                                           */
/*****************************************************************************/
  uint32_t get_ID(void) const {return ID;}
  double get_cV(void) const {return cv;}
  double get_rho(void) const {return rho;}
  double get_opac_A(void) const {return opacA;}
  double get_opac_B(void) const {return opacB;}
  double get_opac_C(void) const {return opacC;}
  double get_T_e(void) const {return T_e;}
  double get_T_r(void) const {return T_r;}
  double get_T_s(void) const {return T_s;}
  double get_absorption_opacity(double T) const {
    return opacA + opacB*std::pow(T, opacC);
  }
  double get_scattering_opacity(void) const {return opacS;}
/*****************************************************************************/
/* non-const functions                                                       */
/*****************************************************************************/
  void set_ID(const uint32_t& _ID) {ID = _ID;}
  void set_cV(const double& _cv) {cv = _cv;}
  void set_rho(const double& _rho) {rho = _rho;}
  void set_opac_A(const double& _opacA) {opacA = _opacA;}
  void set_opac_B(const double& _opacB) {opacB = _opacB;}
  void set_opac_C(const double& _opacC) {opacC = _opacC;}
  void set_opac_S(const double& _opacS) {opacS = _opacS;}
  void set_T_e(const double& _T_e) {T_e = _T_e;}
  void set_T_r(const double& _T_r) {T_r = _T_r;}
  void set_T_s(const double& _T_s) {T_s = _T_s;}

/*****************************************************************************/
/* member variables and private functions                                    */
/*****************************************************************************/
  private:
  uint32_t ID; //! User defined ID of this region 
  double cv; //! Heat capacity in this region
  double rho; //! Density in this region (g/cc)
  double opacA; //! A in A + B * T ^ C
  double opacB; //! B in A + B * T ^ C
  double opacC; //! C in A + B * T ^ C
  double opacS; //! Physical scattering constant coefficient
  double T_e; //! Initial electron temperature in region
  double T_r; //! Initial radiation temperature in region
  double T_s; //! Temperature of source in region
};

#endif
//---------------------------------------------------------------------------//
// end of region.h
//---------------------------------------------------------------------------//
