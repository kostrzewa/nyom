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
  PSP_NDIM
} PointSourcePropagator_dims_t;

class PointSourcePropagator
{
public:
  PointSourcePropagator(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
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

  void fill(const double * const propagator,
            const int src_d,
            const int src_c)
  {

    const int Nt = core.input_node["Nt"].as<int>();
    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();

    const int lt = core.geom.tmlqcd_lat.T; 
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ; 

    const int local_volume = lt * lx * ly * lz;

    nyom::Stopwatch sw(core.geom.get_nyom_comm());
    const int64_t npair = 4*3*local_volume;
    std::vector<int64_t> indices( npair );

    // The propagator vector on the tmLQCD side is ordered
    // (slowest to fastest) flavour TXYZ Dirac colour complex  (space-time MPI-local)
    // On the other hand, the CTF::Tensor has the ordering
    // given by the index translation below, with T (global!)
    // running fastest.
    # pragma omp parallel for
    for(int64_t t = 0; t < lt; ++t){
      int64_t gt = lt*core.geom.tmlqcd_mpi.proc_coords[0] + t;
      for(int64_t x = 0; x < lx; ++x){
        int64_t gx = lx*core.geom.tmlqcd_mpi.proc_coords[1] + x;
        for(int64_t y = 0; y < ly; ++y){
          int64_t gy = ly*core.geom.tmlqcd_mpi.proc_coords[2] + y;
          for(int64_t z = 0; z < lz; ++z){
            int64_t gz = lz*core.geom.tmlqcd_mpi.proc_coords[3] + z;
            for(int64_t snk_d = 0; snk_d < 4; ++snk_d){
              for(int64_t snk_c = 0; snk_c < 3; ++snk_c){
                int64_t counter = snk_c                                  +
                                  snk_d *  (3)                           +
                                  z     *  (3*4)                         +
                                  y     *  (3*4*lz)                      +
                                  x     *  (3*4*lz*ly)                   +
                                  t     *  (3*4*lz*ly*lx);

                indices[counter] = gt                            +
                                   gx    * (Nt)                  +
                                   gy    * (Nt*Nx)               +
                                   gz    * (Nt*Nx*Ny)            +
                                   snk_d * (Nt*Nx*Ny*Nz)         +
                                   src_d * (Nt*Nx*Ny*Nz*4)       +
                                   snk_c * (Nt*Nx*Ny*Nz*4*4)     +
                                   src_c * (Nt*Nx*Ny*Nz*4*4*3);
              }
            }
          }
        }
      }
    }
    // it is not at all guaranteed that this will always work due to possible padding
    // techically, we should allocate a temporary buffer and extract the components from
    // the struct
    tensor.write(npair, indices.data(), reinterpret_cast<const std::complex<double>*>(&propagator[0]) );
    sw.elapsed_print("PointSourcePropagator fill");
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< std::complex<double> > tensor;
  int src_coords[4];
  int sizes[PSP_NDIM];
  int shapes[PSP_NDIM];

private:
  const nyom::Core & core;

  void init()
  {
    for( int i = 0; i < PSP_NDIM; ++i ){
      shapes[i] = NS;
    }
    sizes[PSP_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[PSP_DIM_X_SNK] = core.input_node["Nx"].as<int>();
    sizes[PSP_DIM_Y_SNK] = core.input_node["Ny"].as<int>();
    sizes[PSP_DIM_Z_SNK] = core.input_node["Nz"].as<int>();
    sizes[PSP_DIM_D_SNK] = 4;
    sizes[PSP_DIM_D_SRC] = 4;
    sizes[PSP_DIM_C_SNK] = 3;
    sizes[PSP_DIM_C_SRC] = 3;
    tensor = CTF::Tensor< std::complex<double> >(PSP_NDIM, sizes, shapes, core.geom.get_world(), "PointSourcePropagator" );
  }
};

} // namespace

