/***********************************************************************
 * Copyright (C) 2018 Bartosz Kostrzewa
 *
 * This file is part of nyom.
 *
 * nyom is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any.tmlqcd_lat.r version.
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
#include "Stopwatch.hpp"
#include "Geometry.hpp"

#include <ctf.hpp>

extern "C" {
#include <tmLQCD.h>
}

#include <cfloat>

namespace nyom {
  typedef enum Gaugefield_dims_t {
    GF_DIM_CC = 0,
    GF_DIM_CR,
    GF_DIM_Z,
    GF_DIM_Y,
    GF_DIM_X,
    GF_DIM_T
  } Gaugefield_dims_t;

  typedef enum Gaugefield_dirs_t {
    GF_DIR_T = 0,
    GF_DIR_X,
    GF_DIR_Y,
    GF_DIR_Z
  } Gaugefield_dirs_t;

  typedef struct Gaugefield {
    CTF::Tensor< complex<double> > U[4];
  } Gaugefield;

  Gaugefield make_Gaugefield(int Nt,
                             int Nx,
                             int Ny,
                             int Nz,
                             int Nc,
                             const nyom::Core & core){
    const int shapes[6] = { NS, NS, NS, NS, NS, NS };
    const int sizes[6] = {Nc, Nc, Nz, Ny, Nx, Nt};

    
    nyom::Gaugefield gf;
    for( auto & gfmu : gf.U ){
      gfmu = CTF::Tensor< complex<double> >(6, sizes, shapes, core.geom.get_world(), "Gaugefield" );
    }

    return( gf );
  }

  void read_Gaugefield_from_file(Gaugefield &gf,
                                 const int cid,
                                 const nyom::Core & core){
    nyom::Stopwatch sw;

    sw.reset();
    tmLQCD_read_gauge( cid );
    sw.elapsed_print("tmLQCD gauge field reading");

    const int Nd = 4;
    const int Nc = gf.U[0].lens[GF_DIM_CR];

    // global dimensions
    const int Nt = gf.U[0].lens[GF_DIM_T];
    const int Nx = gf.U[0].lens[GF_DIM_X];
    const int Ny = gf.U[0].lens[GF_DIM_Y];
    const int Nz = gf.U[0].lens[GF_DIM_Z];

    const int lt = core.geom.tmlqcd_lat.T;
    const int lx = core.geom.tmlqcd_lat.LX;
    const int ly = core.geom.tmlqcd_lat.LY;
    const int lz = core.geom.tmlqcd_lat.LZ;

    const int local_volume = lt*lx*ly*lz;

    const int pt = core.geom.tmlqcd_mpi.proc_coords[0];
    const int px = core.geom.tmlqcd_mpi.proc_coords[1];
    const int py = core.geom.tmlqcd_mpi.proc_coords[2];
    const int pz = core.geom.tmlqcd_mpi.proc_coords[3];

    int64_t * indices;
    complex<double> * pairs;

    double ** gauge_field_ptr = NULL;
    tmLQCD_get_gauge_field_pointer(gauge_field_ptr);

    for( int dir = 0; dir < 4; ++dir ){
      sw.reset();
      gf.U[dir].read_local(&npair, &indices, &pairs);
      sw.elapsed_print_and_reset("gf.read_local");

      # pragma omp parallel for
      for( int t = 0; t < lt; ++t){
        const int64_t gt = static_cast<int64_t>(lt)*pt + t;
        
        for( int x = 0; x < lx; ++x){
          const int64_t gx = static_cast<int64_t>(lx)*px + x;

          for( int y = 0; y < ly; ++y){
            const int64_t gy = static_cast<int64_t>(ly)*py + y;

            for( int z = 0; z < lz; ++z){
              const int gz = static_cast<int64_t>(lz)*pz + z;
              const int idx = tmLQCD_idx(t,x,y,z);

              for( int cr = 0; cr < Nc; ++cr ){
                for( int cc = 0; cc < Nc; ++cc ){

                  const int64_t counter = cc
                                          + cr * Nc
                                          + z  * Nc*Nc
                                          + y  * Nc*Nc*lz
                                          + x  * Nc*Nc*lz*ly
                                          + t  * Nc*Nc*lz*ly*lx;
                 
                  // already provided by read_local above
                  //indices[counter] = cc
                  //                   + cr * Nc
                  //                   + gz * Nc*Nc
                  //                   + gy * Nc*Nc*Nz
                  //                   + gx * Nc*Nc*Nz*Ny
                  //                   + gt * Nc*Nc*Nz*Ny*Nx;

                  pairs[counter] = std::complex<double>(tmLQCD_su3_get_elem(&gauge_field_ptr[idx][dir],
                                                                            cr,
                                                                            cc,
                                                                            0),
                                                        tmLQCD_su3_get_elem(&gauge_field_ptr[idx][dir], 
                                                                            cr,
                                                                            cc,
                                                                            1) 
                                                        );
                } // cc
              } // cr
            } // z
          } // y
        } // x
      } // t
      sw.elapsed_print_and_reset("gf reshuffle");
      gf.U[dir].write(npair, indices, pairs);
      sw.elapsed_print("gf.write");
      free(indices);
      free(pairs);
    } // dir
  }

} // namespace(nyom)

