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
#include "constexpr.hpp"

#include <stdexcept>
#include <sstream>

namespace nyom {

class LeviCivita
{
public:
  LeviCivita(const nyom::Core &core_in,
             const int ndim) :
    core(core_in)
  {
    init(ndim);
  }
  
  // we don't want this to be default-constructible
  LeviCivita() = delete;

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;

private:
  const nyom::Core & core;

  void init(const int ndim)
  {
    std::stringstream name;
    name << "LeviCivita" << ndim << "d";
    // technically, it should be possible to exploit CTFs AS tensor shape, but
    // it doesn't seem to work... We thus use NS
    std::vector<int> shapes( ndim, NS );
    std::vector<int> sizes( ndim, ndim );
    tensor = CTF::Tensor< complex<double> >(ndim, sizes.data(), shapes.data(), core.geom.get_world(), name.str().c_str() );

    // for each local index, we will need to compute the tensor coordinates below
    std::vector<int> idx_coords(ndim, 0);

    // we implement the n-dimensional LeviCivita symbol as
    // \eps_{i_1,i_2,...,i_d} = (-1)^d \prod_{i_j > i_i} (i_i - i_j)
    int sign = -1;
    for( int d = 1; d < ndim; ++d ){
      sign *= -1;
    }
    
    int64_t* indices;
    complex<double>* values;
    int64_t nval;
    tensor.read_local(&nval, &indices, &values);
    for( int i = 0; i < nval; ++i ){
      int idx = indices[i];
      for( int d = 0; d < ndim; ++d ){
        idx_coords[d] = idx % ndim;
        idx /= ndim;
      }
      int p = sign;
      for( int i1 = 0; i1 < ndim; ++i1 ){
        for( int i2 = i1+1; i2 < ndim; ++i2 ){
          p *= ( idx_coords[i1] - idx_coords[i2] );
        }
      }
      // when the product is zero, at least two indices coincide and the LeviCivita is zero
      values[i] = p;
      if( p != 0 ){
        values[i] = p > 0 ? 1 : -1;
      }
    }
    tensor.write(nval, indices, values);
    free(values);
    free(indices);
  }
};

} // namespace

