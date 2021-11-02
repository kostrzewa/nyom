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

  // BaKo: comment out fill function copied from ScalarFieldVector (2021-11-02)
  //void fill(const double * const * const scalar_field){

  //  const int Nt = core.input_node["Nt"].as<int>();
  //  const int Nx = core.input_node["Nx"].as<int>();
  //  const int Ny = core.input_node["Ny"].as<int>();
  //  const int Nz = core.input_node["Nz"].as<int>();

  //  const int lt = core.geom.tmlqcd_lat.T;
  //  const int lx = core.geom.tmlqcd_lat.LX;
  //  const int ly = core.geom.tmlqcd_lat.LY;
  //  const int lz = core.geom.tmlqcd_lat.LZ;

  //  const int local_volume = ly * lx * ly * lz;

  //  nyom::Stopwatch sw(core.geom.get_nyom_comm());
  //  std::vector<int64_t> indices(local_volume);

  //  for(int comp = 0; comp < n_components; ++comp){
  //    int64_t counter = 0;
  //    // The scalar field on the tmLQCD side is a two-dim array
  //    // with the scalar component running slowest and then
  //    // (slowest to fastest) TXYZ
  //    // On the other hand, the CTF::Tensor has the ordering
  //    // given by the index translation below, with the component
  //    // running fastest.
  //    for(int64_t t = 0; t < core.geom.tmlqcd_lat.T; ++t){
  //      int64_t gt = ly * core.geom.tmlqcd_mpi.proc_coords[0] + t;

  //      for(int64_t x = 0; x < core.geom.tmlqcd_lat.LX; ++x){
  //        int64_t gx = lx * core.geom.tmlqcd_mpi.proc_coords[1] + x;

  //        for(int64_t y = 0; y < core.geom.tmlqcd_lat.LY; ++y){
  //          int64_t gy = ly * core.geom.tmlqcd_mpi.proc_coords[2] + y;

  //          for(int64_t z = 0; z < core.geom.tmlqcd_lat.LZ; ++z){
  //            int64_t gz = lz * core.geom.tmlqcd_mpi.proc_coords[3] + z;
  //            
  //            indices[counter] = gt    *             +
  //                               gx    * (Nt)        +
  //                               gy    * (Nt*Nx)     +
  //                               gz    * (Nt*Nx*Ny);
  //            counter++;
  //          }
  //        }
  //      }
  //    }
  //    // it is not at all guaranteed that this will always work due to possible padding
  //    // techically, we should allocate a temporary buffer and extract the components from
  //    // the struct
  //    tensor[comp].write(counter, indices.data(), reinterpret_cast<const double * const >(&scalar_field[comp][0]) );
  //  }
  //  sw.elapsed_print("ComplexMatrixField fill");
  //}

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

