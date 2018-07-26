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

namespace nyom {

// this enum controls the ordering of the dimensions of the MomentumTensor tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum MomentumTensor_dims_t {
  MOMT_DIM_X = 0,
  MOMT_DIM_Y,
  MOMT_DIM_Z
} MomentumTensor_dims_t;

class MomentumTensor
{
public:
  MomentumTensor(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }
  
  MomentumTensor(const nyom::Core &core_in,
                 const std::vector<int> & momenta_in) :
    MomentumTensor(core_in)
  {
    set_momenta(momenta_in);
  }

  // we don't want this to be default-constructible
  MomentumTensor() = delete;

  void set_momenta(const std::vector<int> & momenta_in){
    if( momenta_in.size() < 3 ){
      throw( std::domain_error("nyom::MomentumTensor::set_momenta: input momenta "
                               "vector must have at least 3 elements!" ) );
    }
    momenta[0] = momenta_in[0];
    momenta[1] = momenta_in[1];
    momenta[2] = momenta_in[2];

    const int64_t Nx = core.input_node["Nx"].as<int64_t>();
    const int64_t Ny = core.input_node["Ny"].as<int64_t>();
    const int64_t Nz = core.input_node["Nz"].as<int64_t>();

    const double px_pi_ov_two_Nx = 0.5*momenta[0]*nyom::pi/Nx;
    const double py_pi_ov_two_Ny = 0.5*momenta[1]*nyom::pi/Ny;
    const double pz_pi_ov_two_Nz = 0.5*momenta[2]*nyom::pi/Nz;

    const complex<double> imag_unit = complex<double>(0.0, 1.0);

    complex<double> *values;
    int64_t *indices;
    int64_t nval;
    tensor.read_local(&nval, &indices, &values);
#pragma omp parallel for
    for(int64_t idx = 0; idx < nval; ++idx){
      int64_t x = indices[idx] % Nx;
      int64_t y = (indices[idx] / Nx) % Ny;
      int64_t z = indices[idx] / Nx / Nz;
      values[idx] = exp(- imag_unit * ( x * px_pi_ov_two_Nx +
                                        y * py_pi_ov_two_Ny +
                                        z * pz_pi_ov_two_Nz)
                       );
    }
    tensor.write(nval, indices, values); 

    free(indices);
    free(values);
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;

private:
  const nyom::Core & core;

  std::vector<int> momenta;
  
  void init()
  {
    momenta.reserve(3);
    int shapes[3] = {NS, NS, NS};
    int sizes[3];
    sizes[MOMT_DIM_X] = core.input_node["Nx"].as<int>();
    sizes[MOMT_DIM_Y] = core.input_node["Ny"].as<int>();
    sizes[MOMT_DIM_Z] = core.input_node["Nz"].as<int>();
    tensor = CTF::Tensor< complex<double> >(3, sizes, shapes, core.geom.get_world(), "MomentumTensor" );
  }
};

} // namespace

