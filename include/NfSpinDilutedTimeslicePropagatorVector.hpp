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

// C running fastest in the tensor
typedef enum SpinDilutedTimesliceSourceVector_dims_t {
  SDTSV_DIM_C = 0,
  SDTSV_DIM_Z,
  SDTSV_DIM_Y,
  SDTSV_DIM_X
} SpinDilutedTimesliceSourceVector_dims_t;

class SpinDilutedTimesliceSourceVector
{
public:
  SpinDilutedTimesliceSourceVector(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  SpinDilutedTimesliceSourceVector() = delete;

  void fill(const double * source,
            const int dsrc_in,
            const int tsrc_in){
    dsrc = dsrc_in;
    tsrc = tsrc_in;

    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();

    nyom::Stopwatch sw;

    const int lt = core.geom.tmlqcd_lat.T;
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ;

    const int local_volume = lt*lx*ly*lz;

    const int pt = core.geom.tmlqcd_mpi.proc_coords[0];
    const int px = core.geom.tmlqcd_mpi.proc_coords[1];
    const int py = core.geom.tmlqcd_mpi.proc_coords[2];
    const int pz = core.geom.tmlqcd_mpi.proc_coords[3];

    // global indices of time slices residing on this process
    const int gt_min = lt*pt;
    const int gt_max = gt_min + lt;

    if( tsrc_in >= gt_min && tsrc_in < gt_max ){
      const int n_elem = 3*local_volume/lt;

      std::vector<int64_t> indices( n_elem );
      std::vector<complex<double>> pairs( n_elem );
      const int t_local = tsrc_in % lt;
     
      # pragma omp parallel for
      for(int x = 0; x < lx; ++x){
        const int64_t gx = static_cast<int64_t>(lx)*px + x;

        for(int y = 0; y < ly; ++y){
          const int64_t gy = static_cast<int64_t>(ly)*py + y;

          for(int z = 0; z < lz; ++z){
            const int64_t gz = static_cast<int64_t>(lz)*pz + z;

            for(int c = 0; c < 3; ++c){
              const int counter = c +
                                  z * (3) +
                                  y * (3*lz) + 
                                  x * (3*ly*lz);

              indices[counter] = gx * (3*Nz*Ny) +
                                 gy * (3*Nz)    +
                                 gz * (3)       +
                                 c;

              const int tmlqcd_idx = tmLQCD_idx(t_local,x,y,z);

              pairs[counter] = complex<double>( tmLQCD_spinor_get_elem(&source[tmlqcd_idx],
                                                                       dsrc_in,
                                                                       c,
                                                                       0 ),
                                                tmLQCD_spinor_get_elem(&source[tmlqcd_idx],
                                                                       dsrc_in,
                                                                       c,
                                                                       1 )
                                              );
            } // c
          } // z
        } // y
      } // x
      tensor.write(n_elem, indices.data(), pairs.data() );
    } else {
      tensor.write(0, NULL, NULL);

    }
    sw.elapsed_print("SpinDilutedTimesliceSourceVector fill");
  }

  void push(double * source,
            const int d_in,
            const int t_in){
    
    const int lt = core.geom.tmlqcd_lat.T;
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ;

    const int pt = core.geom.tmlqcd_mpi.proc_coords[0];
    const int px = core.geom.tmlqcd_mpi.proc_coords[1];
    const int py = core.geom.tmlqcd_mpi.proc_coords[2];
    const int pz = core.geom.tmlqcd_mpi.proc_coords[3];

    // global indices of time slices residing on this process
    const int gt_min = lt*pt;
    const int gt_max = gt_min + lt;
    
    const int Nt = core.input_node["Nt"].as<int>();
    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();
    
    const int local_volume = lt*lx*ly*lz;

    const int n_elem = 3*local_volume/lt;
    
    nyom::Stopwatch sw;

    // zero out the target spinor
    memset(reinterpret_cast<void*>(&source[0]), 0, 4*3*local_volume*sizeof(complex<double>));

    if( t_in >= gt_min && t_in < gt_max ){
      // read out the tensor into a buffer
      std::vector<int64_t> indices( 3*local_volume/core.geom.tmlqcd_lat.T );
      std::vector<complex<double>> pairs( 3*local_volume/core.geom.tmlqcd_lat.T );
      const int gt = t_in;
      const int t_local = t_in % lt;
     
      # pragma omp parallel for
      for(int x = 0; x < lx; ++x){
        const int64_t gx = static_cast<int64_t>(lx)*px + x;

        for(int y = 0; y < ly; ++y){
          const int64_t gy = static_cast<int64_t>(ly)*py + y;

          for(int z = 0; z < lz; ++z){
            const int64_t gz = static_cast<int64_t>(lz)*pz + z;

            for(int c = 0; c < 3; ++c){
              const int counter = c +
                                  z * (3) +
                                  y * (3*lz) +
                                  x * (3*lz*ly);

              indices[counter] = gx    * (3*Nz*Ny)    +
                                 gy    * (3*Nz)       +
                                 gz    * (3)          +
                                 c;
            } // c
          } // z
        } // y
      } // x
      tensor.read(n_elem, indices.data(), pairs.data() );

      // map the tensor to the spinor
      # pragma omp parallel for
      for(int x = 0; x < lx; ++x){
        for(int y = 0; y < ly; ++y){
          for(int z = 0; z < lz; ++z){
            const int tmlqcd_idx = tmLQCD_idx(t_local,x,y,z);
            for(int c = 0; c < 3; ++c){
              const int tensor_idx =  x * 3*ly*lz +
                                      y * 3*lz    +
                                      z * 3 +
                                      c;
              tmLQCD_spinor_set_elem(&source[tmlqcd_idx],
                                     d_in,
                                     c,
                                     pairs[tensor_idx].real(),
                                     pairs[tensor_idx].imag());
            } // c
          } // z
        } // y
      } // x
    } else {
      tensor.read(0, NULL, NULL);
    }
    sw.elapsed_print("SpinDilutedTimesliceSourceVector push");
  }

  CTF::Tensor< complex<double> > tensor;
  int dsrc;
  int tsrc;

private:
  const nyom::Core & core;

  void init()
  {
    int shapes[4] = {NS, NS, NS, NS};
    int sizes[4];
    sizes[SDTSV_DIM_X] = core.input_node["Nx"].as<int>();
    sizes[SDTSV_DIM_Y] = core.input_node["Ny"].as<int>();
    sizes[SDTSV_DIM_Z] = core.input_node["Nz"].as<int>();
    sizes[SDTSV_DIM_C] = 3;
    tensor = CTF::Tensor< complex<double> >(4, sizes, shapes, core.geom.get_world(), "SpinDilutedTimesliceSourceVector" );
  }
};

// C running fastest in the tensor
typedef enum SpinDilutedTimeslicePropagatorVector_dims_t {
  SDTPV_DIM_C = 0,
  SDTPV_DIM_D,
  SDTPV_DIM_Z,
  SDTPV_DIM_Y,
  SDTPV_DIM_X,
  SDTPV_DIM_T
} SpinDilutedTimeslicePropagatorVector_dims_t;

class SpinDilutedTimeslicePropagatorVector
{
public:
  SpinDilutedTimeslicePropagatorVector(const nyom::Core &core_in) :
    core(core_in)
  {
    init();
  }

  // we don't want this to be default-constructible
  SpinDilutedTimeslicePropagatorVector() = delete;

  void fill(const double * const propagator,
            const int dsrc_in,
            const int tsrc_in){
    dsrc = dsrc_in;
    tsrc = tsrc_in;

    const int Nt = core.input_node["Nt"].as<int>();
    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();
    
    const int lt = core.geom.tmlqcd_lat.T;
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ;

    const int pt = core.geom.tmlqcd_mpi.proc_coords[0];
    const int px = core.geom.tmlqcd_mpi.proc_coords[1];
    const int py = core.geom.tmlqcd_mpi.proc_coords[2];
    const int pz = core.geom.tmlqcd_mpi.proc_coords[3];

    const int local_volume = lt*lx*ly*lz;

    const int64_t n_elem = 4*3*static_cast<int64_t>(local_volume);
    
    std::vector<int64_t> indices( n_elem );
    
    nyom::Stopwatch sw;

    # pragma omp parallel for 
    for(int t = 0; t < lt; ++t){
      const int64_t gt = static_cast<int64_t>(lt)*pt + t;

      for(int x = 0; x < lx; ++x){
        const int64_t gx = static_cast<int64_t>(lx)*px + x;

        for(int y = 0; y < ly; ++y){
          const int64_t gy = static_cast<int64_t>(ly)*py + y;

          for(int64_t z = 0; z < core.geom.tmlqcd_lat.LZ; ++z){
            const int64_t gz = static_cast<int64_t>(lz)*pz + z;

            for(int64_t snk_d = 0; snk_d < 4; ++snk_d){
              for(int64_t snk_c = 0; snk_c < 3; ++snk_c){
                const int64_t counter = static_cast<int64_t>(t) * (3*4*lz*ly*lx) +
                                        static_cast<int64_t>(x) * (3*4*lz*ly)    +
                                        static_cast<int64_t>(y) * (3*4*lz)       +
                                        static_cast<int64_t>(z) * (3*4)          +
                                        snk_d                   * (3)            +
                                        snk_c;

                indices[counter] = gt    * (3*4*Nz*Ny*Nx) +
                                   gx    * (3*4*Nz*Ny)    +
                                   gy    * (3*4*Nz)       +
                                   gz    * (3*4)          +
                                   snk_d * (3)            +
                                   snk_c;
              }
            }
          }
        }
      }
    }
    tensor.write(n_elem, indices.data(), reinterpret_cast<const complex<double>*>(&propagator[0]) );
    sw.elapsed_print("SpinDilutedTimeslicePropagatorVector fill");
  }

  void push(double * const propagator){
    
    const int Nt = core.input_node["Nt"].as<int>();
    const int Nx = core.input_node["Nx"].as<int>();
    const int Ny = core.input_node["Ny"].as<int>();
    const int Nz = core.input_node["Nz"].as<int>();
    
    const int lt = core.geom.tmlqcd_lat.T;
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ;

    const int pt = core.geom.tmlqcd_mpi.proc_coords[0];
    const int px = core.geom.tmlqcd_mpi.proc_coords[1];
    const int py = core.geom.tmlqcd_mpi.proc_coords[2];
    const int pz = core.geom.tmlqcd_mpi.proc_coords[3];

    const int local_volume = lt*lx*ly*lz;

    const int64_t n_elem = 4*3*static_cast<int64_t>(local_volume);
    
    std::vector<int64_t> indices( n_elem );
    
    nyom::Stopwatch sw;
      
    for(int t = 0; t < lt; ++t){
      const int64_t gt = static_cast<int64_t>(lt)*pt + t;
      for(int x = 0; x < lx; ++x){
        int64_t gx = static_cast<int64_t>(lx)*px + x;

        for(int y = 0; y < ly; ++y){
          const int64_t gy = static_cast<int64_t>(ly)*py + y;

          for(int z = 0; z < core.geom.tmlqcd_lat.LZ; ++z){
            const int64_t gz = static_cast<int64_t>(lz)*pz + z;

            for(int d = 0; d < 4; ++d){
              for(int c = 0; c < 3; ++c){
                const int64_t counter = static_cast<int64_t>(t) * (3*4*lz*ly*lx) +
                                        static_cast<int64_t>(x) * (3*4*lz*ly)    +
                                        static_cast<int64_t>(y) * (3*4*lz)       +
                                        static_cast<int64_t>(z) * (3*4)          +
                                        d                       * (3)            +
                                        c;

                indices[counter] = gt * (3*4*Nz*Ny*Nx) +
                                   gx * (3*4*Nz*Ny)    +
                                   gy * (3*4*Nz)       +
                                   gz * (3*4)          +
                                   d  * (3)            +
                                   c;
              } // c
            } // d
          } // z
        } // y
      } // x
    } // t
    tensor.read(n_elem, indices.data(), reinterpret_cast<complex<double>*>(&propagator[0]) );
    sw.elapsed_print("SpinDilutedTimeslicePropagatorVector push");
  }

  CTF::Tensor< complex<double> > tensor;
  int dsrc;
  int tsrc;

private:
  const nyom::Core & core;

  void init()
  {
    int shapes[6] = {NS, NS, NS, NS, NS, NS};
    int sizes[6];
    sizes[SDTPV_DIM_T] = core.input_node["Nt"].as<int>();
    sizes[SDTPV_DIM_X] = core.input_node["Nx"].as<int>();
    sizes[SDTPV_DIM_Y] = core.input_node["Ny"].as<int>();
    sizes[SDTPV_DIM_Z] = core.input_node["Nz"].as<int>();
    sizes[SDTPV_DIM_D] = 4;
    sizes[SDTPV_DIM_C] = 3;
    tensor = CTF::Tensor< complex<double> >(6, sizes, shapes, core.geom.get_world(), "SpinDilutedTimeslicePropagatorVector" );
  }
};

} // namespace

