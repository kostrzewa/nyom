/***********************************************************************
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

    int Nd = 4;
    int Nc = gf.U[0].lens[GF_DIM_CR];

    // global dimensions
    int Nt = gf.U[0].lens[GF_DIM_T];
    int Nx = gf.U[0].lens[GF_DIM_X];
    int Ny = gf.U[0].lens[GF_DIM_Y];
    int Nz = gf.U[0].lens[GF_DIM_Z];

    int Nt_local = core.geom.lat.T;
    int Nx_local = core.geom.lat.LX;
    int Ny_local = core.geom.lat.LY;
    int Nz_local = core.geom.lat.LZ;

    int64_t npair;
    int64_t * indices;
    complex<double> * pairs;

    for( int dir = 0; dir < 4; ++dir ){
      sw.reset();
      gf.U[dir].read_local(&npair, &indices, &pairs);
      sw.elapsed_print_and_reset("gf.read_local");

      int64_t counter = 0;
      for( int t = 0; t < Nt_local; ++t){
        int gt = Nt_local*core.geom.tmlqcd_mpi.proc_coords[0] + t;
        
        for( int x = 0; x < Nx_local; ++x){
          int gx = Nx_local*core.geom.tmlqcd_mpi.proc_coords[1] + x;

          for( int y = 0; y < Ny_local; ++y){
            int gy = Ny_local*core.geom.tmlqcd_mpi.proc_coords[2] + y;

            for( int z = 0; z < Nz_local; ++z){
              int gz = Nz_local*core.geom.tmlqcd_mpi.proc_coords[3] + z;

              for( int cr = 0; cr < Nc; ++cr ){
                for( int cc = 0; cc < Nc; ++cc ){
                  indices[counter] = cc
                                     + cr  * Nc
                                     + gz  * Nc*Nc
                                     + gy  * Nc*Nc*Nz
                                     + gx  * Nc*Nc*Nz*Ny
                                     + gt  * Nc*Nc*Nz*Ny*Nx;

                  pairs[counter] = std::complex<double>(su3_get_elem(&g_gauge_field[ g_ipt[t][x][y][z] ][dir],
                                                                     cr,
                                                                     cc,
                                                                     0),
                                                        su3_get_elem(&g_gauge_field[ g_ipt[t][x][y][z] ][dir], 
                                                                     cr,
                                                                     cc,
                                                                     1) 
                                                        );
                  counter++;
                } // cc
              } // cr
            } // z
          } // y
        } // x
      } // t
      sw.elapsed_print_and_reset("gf reshuffle");
      gf.U[dir].write(npair, indices, pairs);
      sw.elapsed_print("gf.write");
      free(indices); free(pairs);
    } // dir
  }

} // namespace(nyom)

