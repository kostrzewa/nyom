/**********************************************************************
 * Copyright (C) 2018 Bartosz Kostrzewa
 *
 * This file is part of nyom.
 *
 * nyom is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * nyom is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with nyom.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/

#pragma once


#include "Core.hpp"

#include <string.h>

namespace nyom {

constexpr int NISC_ndim = 9;
constexpr int NIC_ndim = 7;

// this enum controls the ordering of the dimensions of the NucleonIntermediateSpinColour tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum NucleonIntermediateSpinColour_dims_t {
  NISC_DIM_T_SNK = 0,
  NISC_DIM_X_SNK,
  NISC_DIM_Y_SNK,
  NISC_DIM_Z_SNK,
  NISC_DIM_D1,
  NISC_DIM_D2,
  NISC_DIM_C1,
  NISC_DIM_C2,
  NISC_DIM_C3
} NucleonIntermediateSpinColour_dims_t;

typedef enum NucleonIntermediateColour_dims_t {
  NIC_DIM_T_SNK = 0,
  NIC_DIM_X_SNK,
  NIC_DIM_Y_SNK,
  NIC_DIM_Z_SNK,
  NIC_DIM_C1,
  NIC_DIM_C2,
  NIC_DIM_C3
} NucleonIntermediateColour_dims_t;

class NucleonIntermediateColour
{
public:
  NucleonIntermediateColour(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  NucleonIntermediateColour() = delete;

  void set_src_coords(const int src_coords_in[])
  {
    src_coords[0] = src_coords_in[0];
    src_coords[1] = src_coords_in[1];
    src_coords[2] = src_coords_in[2];
    src_coords[3] = src_coords_in[3];
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;
  int src_coords[4];

private:
  const nyom::Core & core;
  int shapes[NIC_ndim];
  int sizes[NIC_ndim];

  void init()
  {
    for(int d = 0; d < NIC_ndim; ++d){
      shapes[d] = NS;
    }
    sizes[NIC_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[NIC_DIM_X_SNK] = core.input_node["Nx"].as<int>();
    sizes[NIC_DIM_Y_SNK] = core.input_node["Ny"].as<int>();
    sizes[NIC_DIM_Z_SNK] = core.input_node["Nz"].as<int>();
    sizes[NIC_DIM_C1] = 3;
    sizes[NIC_DIM_C2] = 3;
    sizes[NIC_DIM_C3] = 3;
    tensor = CTF::Tensor< complex<double> >(NIC_ndim, sizes, shapes, core.geom.get_world(), "NucleonIntermediateColour" );
  }
};

class NucleonIntermediateSpinColour
{
public:
  NucleonIntermediateSpinColour(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  NucleonIntermediateSpinColour() = delete;

  void set_src_coords(const int src_coords_in[])
  {
    src_coords[0] = src_coords_in[0];
    src_coords[1] = src_coords_in[1];
    src_coords[2] = src_coords_in[2];
    src_coords[3] = src_coords_in[3];
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;
  int src_coords[4];

private:
  const nyom::Core & core;
  int shapes[NISC_ndim];
  int sizes[NISC_ndim];

  void init()
  {
    for(int d = 0; d < NISC_ndim; ++d){
      shapes[d] = NS;
    }
    sizes[NISC_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[NISC_DIM_X_SNK] = core.input_node["Nx"].as<int>();
    sizes[NISC_DIM_Y_SNK] = core.input_node["Ny"].as<int>();
    sizes[NISC_DIM_Z_SNK] = core.input_node["Nz"].as<int>();
    sizes[NISC_DIM_D1] = 4;
    sizes[NISC_DIM_D2] = 4;
    sizes[NISC_DIM_C1] = 3;
    sizes[NISC_DIM_C2] = 3;
    sizes[NISC_DIM_C3] = 3;
    tensor = CTF::Tensor< complex<double> >(NISC_ndim, sizes, shapes, core.geom.get_world(), "NucleonIntermediateSpinColour" );
  }
};

} // namespace

