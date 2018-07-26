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

// this enum controls the ordering of the dimensions of the PointSourcePropagator tensor
// Ther ordering can be adjusted at will by simply adjusting the ordering here
typedef enum PointSourcePropagator_dims_t {
  PSP_DIM_T_SNK = 0,
  PSP_DIM_X_SNK,
  PSP_DIM_Y_SNK,
  PSP_DIM_Z_SNK,
  PSP_DIM_D_SNK,
  PSP_DIM_D_SRC,
  PSP_DIM_C_SNK,
  PSP_DIM_C_SRC,
} PointSourcePropagator_dims_t;

class PointSourcePropagator
{
public:
  PointSourcePropagator(const nyom::Core &core)
  {
    init(core);
  }

  // we don't want this to be default-constructible
  PointSourcePropagator() = delete;

  void set_src_coords(const int src_coords_in[])
  {
    src_coords[0] = src_coords_in[0];
    src_coords[1] = src_coords_in[1];
    src_coords[2] = src_coords_in[2];
    src_coords[3] = src_coords_in[3];
  }

  void fill(const spinor * const propagator,
            const int src_d,
            const int src_c,
            const nyom::Core &core){

    int Nt = core.input_node["Nt"].as<int>();
    int Nx = core.input_node["Nx"].as<int>();
    int Ny = core.input_node["Ny"].as<int>();
    int Nz = core.input_node["Nz"].as<int>();

    int local_volume = core.geom.tmlqcd_lat.T  *
                       core.geom.tmlqcd_lat.LX *
                       core.geom.tmlqcd_lat.LY *
                       core.geom.tmlqcd_lat.LZ;

    nyom::Stopwatch sw(core.geom.get_nyom_comm());
    int64_t npair = 4*3*local_volume;
    std::vector<int64_t> indices( 4*3*local_volume );
    int64_t counter = 0;

    // The propagator vector on the tmLQCD side is ordered
    // (slowest to fastest) TXYZ Dirac colour complex
    // On the other hand, the CTF::Tensor has the ordering
    // given by the index translation below, with T
    // running fastest.
    for(int64_t t = 0; t < core.geom.tmlqcd_lat.T; ++t){

      int64_t gt = core.geom.tmlqcd_lat.T*core.geom.tmlqcd_mpi.proc_coords[0] + t;

      for(int64_t x = 0; x < core.geom.tmlqcd_lat.LX; ++x){
        int64_t gx = core.geom.tmlqcd_lat.LX*core.geom.tmlqcd_mpi.proc_coords[1] + x;

        for(int64_t y = 0; y < core.geom.tmlqcd_lat.LY; ++y){
          int64_t gy = core.geom.tmlqcd_lat.LY*core.geom.tmlqcd_mpi.proc_coords[2] + y;

          for(int64_t z = 0; z < core.geom.tmlqcd_lat.LZ; ++z){
            int64_t gz = core.geom.tmlqcd_lat.LZ*core.geom.tmlqcd_mpi.proc_coords[3] + z;

            for(int64_t snk_d = 0; snk_d < 4; ++snk_d){
              for(int64_t snk_c = 0; snk_c < 3; ++snk_c){
                indices[counter] = gt                         +
                                   gx    * (Nt)               +
                                   gy    * (Nt*Nx)            +
                                   gz    * (Nt*Nx*Ny)         +
                                   snk_d * (Nt*Nx*Ny*Nz)      +
                                   src_d * (Nt*Nx*Ny*Nz*4)    +
                                   snk_c * (Nt*Nx*Ny*Nz*4*4)  +
                                   src_c * (Nt*Nx*Ny*Nz*4*4*3);
                counter++;
              }
            }
          }
        }
      }
    }
    tensor.write(counter, indices.data(), reinterpret_cast<const complex<double>*>(&propagator[0]) );
    sw.elapsed_print("PointSourcePropagator fill");
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
  void init(const nyom::Core &core)
  {
    int shapes[8] = {NS, NS, NS, NS, NS, NS, NS, NS};
    int sizes[8];
    sizes[PSP_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[PSP_DIM_X_SNK] = core.input_node["Nx"].as<int>();
    sizes[PSP_DIM_Y_SNK] = core.input_node["Ny"].as<int>();
    sizes[PSP_DIM_Z_SNK] = core.input_node["Nz"].as<int>();
    sizes[PSP_DIM_D_SRC] = 4;
    sizes[PSP_DIM_D_SNK] = 4;
    sizes[PSP_DIM_C_SRC] = 3;
    sizes[PSP_DIM_C_SNK] = 3;
    tensor = CTF::Tensor< complex<double> >(8, sizes, shapes, core.geom.get_world(), "PointSourcePropagator" );
  }
};

} // namespace

