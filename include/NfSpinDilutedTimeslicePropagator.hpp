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

namespace nyom {

// this enum controls the ordering of the dimensions of the NfSpinDilutedTimeslicePropagator tensor
// Their ordering can be adjusted at will by simply adjusting the ordering here
// however the index computation in the "push" and "fill" methods (if any) needs to be adjusted
typedef enum NfSpinDilutedTimeslicePropagator_dims_t {
  NF_SDTSP_DIM_T_SNK = 0,
  NF_SDTSP_DIM_X_SNK,
  NF_SDTSP_DIM_Y_SNK,
  NF_SDTSP_DIM_Z_SNK,
  NF_SDTSP_DIM_D_SNK,
  NF_SDTSP_DIM_D_SRC,
  NF_SDTSP_DIM_C_SNK,
  NF_SDTSP_DIM_F_SNK,
  NF_SDTSP_DIM_F_SRC,
  NF_SDTSP_NDIM
} NfSpinDilutedTimeslicePropagator_dims_t;

template <int Nf>
class NfSpinDilutedTimeslicePropagator
{
public:
  NfSpinDilutedTimeslicePropagator(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  NfSpinDilutedTimeslicePropagator() = delete;

  void set_src_ts(const int ts)
  {
    src_ts = ts;
  }

  // because of the way that the doublet inversion interface is implemented, we will
  // fill the propagator one sink flavour at a time even though the inversion
  // of course produces both
  void fill(double * propagator,
            const int snk_f,
            const int src_f,
            const int src_d){

    const int Nt = core.input_node["Nt"].as<int>();
    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();

    const int pt = core.geom.tmlqcd_mpi.proc_coords[0]; 
    const int px = core.geom.tmlqcd_mpi.proc_coords[1]; 
    const int py = core.geom.tmlqcd_mpi.proc_coords[2]; 
    const int pz = core.geom.tmlqcd_mpi.proc_coords[3]; 

    const int lt = core.geom.tmlqcd_lat.T; 
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ; 

    const int local_volume = lt * lx * ly * lz;

    nyom::Stopwatch sw(core.geom.get_nyom_comm());
    int64_t npair = Nf*4*3*static_cast<int64_t>(local_volume);
    std::vector<int64_t> indices( npair );

    // The propagator vector on the tmLQCD side is ordered
    // (slowest to fastest) flavour TXYZ Dirac colour complex
    // On the other hand, the CTF::Tensor has the ordering
    // given by the index translation below, with T
    // running fastest.
    # pragma omp parallel
    {
      # pragma omp for
      for(int64_t t = 0; t < lt; ++t){
        const int64_t gt = lt*pt + t;
        for(int64_t x = 0; x < lx; ++x){
          const int64_t gx = lx*px + x;
          for(int64_t y = 0; y < ly; ++y){
            const int64_t gy = ly*py + y;
            for(int64_t z = 0; z < lz; ++z){
              const int64_t gz = lz*pz + z;
              for(int64_t snk_d = 0; snk_d < 4; ++snk_d){
                for(int64_t snk_c = 0; snk_c < 3; ++snk_c){
                  const int64_t counter = snk_c                  +
                                          snk_d *  (3)           +
                                          z     *  (3*4)         +
                                          y     *  (3*4*lz)      +
                                          x     *  (3*4*lz*ly)   +
                                          t     *  (3*4*lz*ly*lx);
                
                  indices[counter] = gt                             +
                                     gx    * (Nt)                   +
                                     gy    * (Nt*Nx)                +
                                     gz    * (Nt*Nx*Ny)             +
                                     snk_d * (Nt*Nx*Ny*Nz)          +
                                     src_d * (Nt*Nx*Ny*Nz*4)        +
                                     snk_c * (Nt*Nx*Ny*Nz*4*4)      +
                                     snk_f * (Nt*Nx*Ny*Nz*4*4*3)    +
                                     src_f * (Nt*Nx*Ny*Nz*4*4*3*Nf);
                }
              }
            }
          }
        }
      }
    } // parallel section closing brace

    // it is not at all guaranteed that this will always work due to possible padding
    // techically, we should allocate a temporary buffer and extract the components from
    // the struct one-by-one first
    tensor.write(counter, indices.data(), reinterpret_cast<const std::complex<double>*>(&propagator[0]) );
    sw.elapsed_print("NfSpinDilutedTimeslicePropagator fill");
  }

  // by overloading the square bracket operator, we can give convenient access to the underlying
  // tensor
  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< std::complex<double> > tensor;
  int src_ts;
  int sizes[NF_SDTSP_NDIM];
  int shapes[NF_SDTSP_NDIM];

private:
  const nyom::Core & core;

  void init()
  {
    for( int i = 0; i < NF_SDTSP_NDIM; ++i ){
      shapes[i] = NS;
    }
    sizes[NF_SDTSP_DIM_T_SNK] = core.input_node["Nt"].as<int>();
    sizes[NF_SDTSP_DIM_X_SNK] = core.input_node["Nx"].as<int>();
    sizes[NF_SDTSP_DIM_Y_SNK] = core.input_node["Ny"].as<int>();
    sizes[NF_SDTSP_DIM_Z_SNK] = core.input_node["Nz"].as<int>();
    sizes[NF_SDTSP_DIM_D_SNK] = 4;
    sizes[NF_SDTSP_DIM_D_SRC] = 4;
    sizes[NF_SDTSP_DIM_C_SNK] = 3;
    sizes[NF_SDTSP_DIM_F_SNK] = Nf;
    sizes[NF_SDTSP_DIM_F_SRC] = Nf;
    tensor = CTF::Tensor< std::complex<double> >(NF_SDTSP_NDIM, sizes, shapes, core.geom.get_world(), "NfSpinDilutedTimeslicePropagator" );
  }
};

} // namespace

