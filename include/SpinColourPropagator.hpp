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

#include <tmLQCD.h>
#include <string.h>

// struct_accessors.h sits in the tmLQCD source directory
// and contains static inline functions for accessing the individual
// elements in su3 and spinor structs via colour and spin indices 
#include <struct_accessors.h>

namespace nyom {

// this enum controls the ordering of the dimensions of the SpinColourPropagator tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum SpinColourPropagator_dims_t {
  SCP_DIM_T_SNK = 0,
  SCP_DIM_D_SNK,
  SCP_DIM_D_SRC,
  SCP_DIM_C_SNK,
  SCP_DIM_C_SRC,
} SpinColourPropagator_dims_t;

class SpinColourPropagator
{
public:
  SpinColourPropagator(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  SpinColourPropagator() = delete;

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

  CTF::Tensor< std::complex<double> > tensor;
  int src_coords[4];

private:
  const nyom::Core & core;
  int shapes[5];
  int sizes[5];

  void init()
  {
    for(int d = 0; d < 5; ++d){
      shapes[d] = NS;
    }
    sizes[SCP_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[SCP_DIM_D_SNK] = 4;
    sizes[SCP_DIM_D_SRC] = 4;
    sizes[SCP_DIM_C_SNK] = 3;
    sizes[SCP_DIM_C_SRC] = 3;
    tensor = CTF::Tensor< std::complex<double> >(5, sizes, shapes, core.geom.get_world(), "SpinColourPropagator" );
  }
};

} // namespace

