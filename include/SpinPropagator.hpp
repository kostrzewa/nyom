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

// this enum controls the ordering of the dimensions of the SpinPropagator tensor
// The ordering can be adjusted at will by simply adjusting the ordering here
// For convenience, we want the source Dirac index to run fastest and T to run slowest
typedef enum SpinPropagator_dims_t {
  SP_DIM_D_SRC = 0,
  SP_DIM_D_SNK,
  SP_DIM_T_SNK
} SpinPropagator_dims_t;

class SpinPropagator
{
public:
  SpinPropagator(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  SpinPropagator() = delete;

  void set_src_coords(const int src_coords_in[])
  {
    src_coords[0] = src_coords_in[0];
    src_coords[1] = src_coords_in[1];
    src_coords[2] = src_coords_in[2];
    src_coords[3] = src_coords_in[3];
  }

  void get_idx_coords(std::vector<int> & idx_coords, int64_t idx){
    if( idx_coords.size() != 3 ){
      idx_coords.resize(3);
    }
    for(int i = 0; i < 3; ++i){
      idx_coords[i] = idx % sizes[i];
      idx /= sizes[i];
    }
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;
  int src_coords[4];
  int shapes[3];
  int sizes[3];

private:
  const nyom::Core & core;

  void init()
  {
    for(int i = 0; i < 3; ++i ){
      shapes[i] = NS;
    }
    sizes[SP_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[SP_DIM_D_SNK] = 4;
    sizes[SP_DIM_D_SRC] = 4;
    tensor = CTF::Tensor< complex<double> >(3, sizes, shapes, core.geom.get_world(), "SpinPropagator" );
  }
};

} // namespace

