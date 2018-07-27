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
    std::vector<int> shapes( ndim, NS );
    std::vector<int> sizes( ndim, ndim );
    tensor = CTF::Tensor< complex<double> >(ndim, sizes.data(), shapes.data(), core.geom.get_world(), name.str().c_str() );

    //if( core.geom.get_myrank() == 0 ){
    //  int64_t* indices;
    //  complex<double>* values;
    //  int64_t nval;
    //  tensor.read_all(&nval, &indices, &values);


    //  free(values);
    //  free(indices);
    //}
   
    int index = 0;
    int offset = ndim; 
    complex<double> one = complex<double>(1.0, 0.0);
    for( int d = 1; d < ndim; ++d ){
      index += d*offset;
      offset *= ndim;
    }
    tensor.write((int64_t)1, (const int64_t*)&index, (const complex<double>*)&one); 
  }
};

} // namespace

