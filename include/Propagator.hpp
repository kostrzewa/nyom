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

typedef enum SpinDilutedTimeslicePropagatorVector_dims_t {
  SDTPV_DIM_TSNK = 0,
  SDTPV_DIM_XSNK,
  SDTPV_DIM_YSNK,
  SDTPV_DIM_ZSNK,
  SDTPV_DIM_DSNK,
  SDTPV_DIM_CSNK
} SpinDilutedTimeslicePropagatorVector_dims_t;

class SpinDilutedTimeslicePropagatorVector
{
public:
  SpinDilutedTimeslicePropagatorVector(const nyom::Core &core)
  {
    init(core);
  }

  // we don't want this to be default-constructible
  SpinDilutedTimeslicePropagatorVector() = delete;

  void fill(const spinor * const propagator,
            const int dsrc_in,
            const int tsrc_in,
            const nyom::Core &core){
    dsrc = dsrc_in;
    tsrc = tsrc_in;

    int Nt = core.input_node["Nt"].as<int>();
    int Nx = core.input_node["Nx"].as<int>();
    int Ny = core.input_node["Ny"].as<int>();
    int Nz = core.input_node["Nz"].as<int>();

    int local_volume = core.geom.tmlqcd_lat.T  *
                       core.geom.tmlqcd_lat.LX *
                       core.geom.tmlqcd_lat.LY *
                       core.geom.tmlqcd_lat.LZ;

    int64_t npair = 4*3*local_volume;
    std::vector<int64_t> indices( 4*3*local_volume );
    std::vector<complex<double>> pairs( 4*3*local_volume );
    int64_t counter = 0;

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
                indices[counter] = snk_c * (Nt*Nx*Ny*Nz*4) +
                                   snk_d * (Nt*Nx*Ny*Nz)   +
                                   gz    * (Nt*Nx*Ny)      +
                                   gy    * (Nt*Nx)         +
                                   gx    * (Nt)            +
                                   gt;

                pairs[counter] = complex<double>( spinor_get_elem(propagator, snk_d, snk_c, 0),
                                                  spinor_get_elem(propagator, snk_d, snk_c, 1) );
                counter++;
              }
            }
          }
        }
      }
    }
  }

  void push(spinor * const propagator,
            const nyom::Core &core){
    
    int Nt = core.input_node["Nt"].as<int>();
    int Nx = core.input_node["Nx"].as<int>();
    int Ny = core.input_node["Ny"].as<int>();
    int Nz = core.input_node["Nz"].as<int>();
    
    int local_volume = core.geom.tmlqcd_lat.T  *
                       core.geom.tmlqcd_lat.LX *
                       core.geom.tmlqcd_lat.LY *
                       core.geom.tmlqcd_lat.LZ;

    int64_t npair = 4*3*local_volume;
    std::vector<int64_t> indices( 4*3*local_volume );
    int64_t counter = 0;

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
                indices[counter] = snk_c * (Nt*Nx*Ny*Nz*4) +
                                   snk_d * (Nt*Nx*Ny*Nz)   +
                                   gz    * (Nt*Nx*Ny)      +
                                   gy    * (Nt*Nx)         +
                                   gx    * (Nt)            +
                                   gt;
                counter++;
              }
            }
          }
        }
      }
    }
    tensor.read(counter, indices.data(), reinterpret_cast<complex<double>*>(&propagator[0]) );
  }

  CTF::Tensor< complex<double> > tensor;
  int dsrc;
  int tsrc;

private:
  void init(const nyom::Core &core)
  {
    int shapes[6] = {NS, NS, NS, NS, NS, NS};
    int sizes[6];
    sizes[SDTPV_DIM_TSNK] = core.input_node["Nt"].as<int>();
    sizes[SDTPV_DIM_XSNK] = core.input_node["Nx"].as<int>();
    sizes[SDTPV_DIM_YSNK] = core.input_node["Ny"].as<int>();
    sizes[SDTPV_DIM_ZSNK] = core.input_node["Nz"].as<int>();
    sizes[SDTPV_DIM_DSNK] = 4;
    sizes[SDTPV_DIM_CSNK] = 3;
    tensor = CTF::Tensor< complex<double> >(6, sizes, shapes, core.geom.get_world() );
  }
};

} // namespace

