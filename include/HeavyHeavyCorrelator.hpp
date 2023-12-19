/**********************************************************************
 * Copyright (C) 2023 Bartosz Kostrzewa
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

#include <string>

namespace nyom {

// this enum controls the ordering of the dimensions of the HeavyHeavyCorrelator tensor
// Their ordering can be adjusted at will by simply adjusting the ordering here
// however the index computation in the "push" and "fill" methods (if any) needs to be adjusted
typedef enum HeavyHeavyCorrelator_dims_t {
  HHC_DIM_F1 = 0,
  HHC_DIM_F2,
  HHC_DIM_F3,
  HHC_DIM_F4,
  HHC_DIM_S1,
  HHC_DIM_S2,
  HHC_DIM_T,
  HHC_NDIM
} HeavyHeavyCorrelator_dims_t;

template <int Nf1, int Nf2, int Nsmearings>
class HeavyHeavyCorrelator
{
public:
  HeavyHeavyCorrelator(const nyom::Core &core_in, const std::string _name = "HeavyHeavyCorrelator") :
    core(core_in),
    name(_name)
  {
    init();
  }

  // we don't want this to be default-constructible
  HeavyHeavyCorrelator() = delete;

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
  int shapes[HHC_NDIM];
  int sizes[HHC_NDIM];
  const std::string name;

  void init()
  {
    for(int d = 0; d < HHC_NDIM; ++d){
      shapes[d] = NS;
    }
    sizes[HHC_DIM_F1] = Nf1; 
    sizes[HHC_DIM_F2] = Nf1;
    sizes[HHC_DIM_F3] = Nf2;
    sizes[HHC_DIM_F4] = Nf2;
    sizes[HHC_DIM_S1] = Nsmearings;
    sizes[HHC_DIM_S2] = Nsmearings;
    sizes[HHC_DIM_T] = core.input_node["Nt"].as<int>();
    tensor = CTF::Tensor< complex<double> >(HHC_NDIM, sizes, shapes, core.geom.get_world(), name.c_str() );
  }
};

} // namespace

