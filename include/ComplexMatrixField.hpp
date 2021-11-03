/**********************************************************************
 * Copyright (C) 2021 Bartosz Kostrzewa
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

namespace nyom {

// this enum controls the ordering of the dimensions of the ComplexMatrixField tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum ComplexMatrixField_dims_s {
  CM_DIM_R = 0,
  CM_DIM_C,
  CM_DIM_T,
  CM_DIM_X,
  CM_DIM_Y,
  CM_DIM_Z,
  CM_NDIM
} ComplexMatrixField_dims_t;

template <int nr, int nc>
class ComplexMatrixField
{
public:
  ComplexMatrixField(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  ComplexMatrixField() = delete;

  // we probably don't need this
  ComplexMatrixField<nr,nc> & operator=(ComplexMatrixField<nr,nc> && rhs){
    *this = std::move(rhs);
    return *this;
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< std::complex<double> > tensor;
  
  int sizes[CM_NDIM];
  int shapes[CM_NDIM];

private:
  const nyom::Core & core;

  void init()
  {
    for( int i = 0; i < CM_NDIM; ++i ){
      shapes[i] = NS;
    }
    sizes[CM_DIM_R] = nr;
    sizes[CM_DIM_C] = nc;
    sizes[CM_DIM_T] = core.input_node["Nt"].as<int>();
    sizes[CM_DIM_X] = core.input_node["Nx"].as<int>();
    sizes[CM_DIM_Y] = core.input_node["Ny"].as<int>();
    sizes[CM_DIM_Z] = core.input_node["Nz"].as<int>();
    
    tensor = CTF::Tensor< std::complex<double> >(CM_NDIM, sizes, shapes, core.geom.get_world(), "ComplexMatrixField");
  }
};

} // namespace

