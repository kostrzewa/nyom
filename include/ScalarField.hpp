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

// this enum controls the ordering of the dimensions of the ScalarField tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum ScalarField_dims_s {
  SF_ELEM = 0,
  SF_DIM_T,
  SF_DIM_X,
  SF_DIM_Y,
  SF_DIM_Z
} ScalarField_dims_t;

class ScalarField
{
public:
  ScalarField(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  ScalarField() = delete;

  void fill(const double ** const scalar_field){

    const int Nt = core.input_node["Nt"].as<int>();
    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();

    const int lt = core.geom.tmlqcd_lat.T;
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ;

    const int local_volume = ly * lx * ly * lz;

    nyom::Stopwatch sw(core.geom.get_nyom_comm());
    std::vector<int64_t> indices(local_volume);

    for(int comp = 0; comp < n_components; ++comp){
      int64_t counter = 0;
      // The scalar field on the tmLQCD is a two-dim array
      // with the scalar index running slowest and then
      // (slowest to fastest) TXYZ
      // On the other hand, the CTF::Tensor has the ordering
      // given by the index translation below, with T
      // running fastest.
      for(int64_t t = 0; t < core.geom.tmlqcd_lat.T; ++t){
        int64_t gt = ly * core.geom.tmlqcd_mpi.proc_coords[0] + t;

        for(int64_t x = 0; x < core.geom.tmlqcd_lat.LX; ++x){
          int64_t gx = lx * core.geom.tmlqcd_mpi.proc_coords[1] + x;

          for(int64_t y = 0; y < core.geom.tmlqcd_lat.LY; ++y){
            int64_t gy = ly * core.geom.tmlqcd_mpi.proc_coords[2] + y;

            for(int64_t z = 0; z < core.geom.tmlqcd_lat.LZ; ++z){
              int64_t gz = lz * core.geom.tmlqcd_mpi.proc_coords[3] + z;
              
              indices[counter] = gt    *               +
                                 gx    * (Nt)          +
                                 gy    * (Nt*Nx)       +
                                 gz    * (Nt*Nx*Ny)    +
                                 comp  * (Nt*Nx*Ny*Nz);
              counter++;
            }
          }
        }
      }
      // it is not at all guaranteed that this will always work due to possible padding
      // techically, we should allocate a temporary buffer and extract the components from
      // the struct
      tensor.write(counter, indices.data(), reinterpret_cast<const double*>(&scalar_field[comp][0]) );
    }
    sw.elapsed_print("ScalarField fill");
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;
  
  int sizes[n_idcs];
  int shapes[n_idcs];

private:
  const nyom::Core & core;

  constexpr int n_idcs = 5;
  constexpr int n_components = 4;

  void init()
  {
    for( int i = 0; i < n_idcs; ++i ){
      shapes[i] = NS;
    }
    sizes[SF_ELEM] = n_components;
    sizes[SF_DIM_T] = core.input_node["Nt"].as<int>();
    sizes[SF_DIM_X] = core.input_node["Nx"].as<int>();
    sizes[SF_DIM_Y] = core.input_node["Ny"].as<int>();
    sizes[SF_DIM_Z] = core.input_node["Nz"].as<int>();
    tensor = CTF::Tensor<double>(n_idcs, sizes, shapes, core.geom.get_world(), "ScalarField" );
  }
};

} // namespace

