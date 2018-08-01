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

// this enum controls the ordering of the dimensions of the ColourDipolePropagator tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum ColourDipolePropagator_dims_t {
  CDP_DIM_T_SNK = 0,
  CDP_DIM_X_SNK,
  CDP_DIM_Y_SNK,
  CDP_DIM_Z_SNK,
  CDP_DIM_C1_SNK,
  CDP_DIM_C1_SRC,
  CDP_DIM_C2_SNK,
  CDP_DIM_C2_SRC
} ColourDipolePropagator_dims_t;

class ColourDipolePropagator
{
public:
  ColourDipolePropagator(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  ColourDipolePropagator() = delete;

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
  int shapes[8];
  int sizes[8];

  void init()
  {
    for(int d = 0; d < 8; ++d){
      shapes[d] = NS;
    }
    sizes[CDP_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[CDP_DIM_X_SNK] = core.input_node["Nx"].as<int>();
    sizes[CDP_DIM_Y_SNK] = core.input_node["Ny"].as<int>();
    sizes[CDP_DIM_Z_SNK] = core.input_node["Nz"].as<int>();
    sizes[CDP_DIM_C1_SNK] = 3;
    sizes[CDP_DIM_C1_SRC] = 3;
    sizes[CDP_DIM_C2_SNK] = 3;
    sizes[CDP_DIM_C2_SRC] = 3;
    tensor = CTF::Tensor< complex<double> >(8, sizes, shapes, core.geom.get_world(), "ColourDipolePropagator" );
  }
};

} // namespace

