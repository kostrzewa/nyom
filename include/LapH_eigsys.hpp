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

#include "Geometry.hpp"
#include "Stopwatch.hpp"

#include <ctf.hpp>

#include <sstream>
#include <fstream>
#include <iomanip>

namespace nyom {

typedef enum LapH_eigsys_dims_t {
  LAPH_DIM_C = 0,
  LAPH_DIM_EV,
  LAPH_DIM_Z,
  LAPH_DIM_Y,
  LAPH_DIM_X,
  LAPH_DIM_T
} LapH_eigsys_dims_t;

typedef CTF::Tensor< complex<double> > LapH_eigsys;

nyom::LapH_eigsys make_LapH_eigsys(const int Nev,
                                   const int Nt,
                                   const int Nx,
                                   const int Ny,
                                   const int Nz,
                                   const int Nc,
                                   CTF::World &world){
  int shapes[6] = {NS, NS, NS, NS, NS, NS};
  int sizes[6];
  sizes[LAPH_DIM_T]  = Nt;
  sizes[LAPH_DIM_X]  = Nx;
  sizes[LAPH_DIM_Y]  = Ny;
  sizes[LAPH_DIM_Z]  = Nz;
  sizes[LAPH_DIM_EV] = Nev;
  sizes[LAPH_DIM_C]  = Nc;
  return( CTF::Tensor< complex<double> >(6, sizes, shapes, world) );
}

void read_LapH_eigsys_from_files(nyom::LapH_eigsys & V,
                                 const std::string path,
                                 const int cid,
                                 const nyom::Geometry & geom){

  const int Nev = V.lens[LAPH_DIM_EV];
  const int Nc = V.lens[LAPH_DIM_C];
  
  const int Nt = V.lens[LAPH_DIM_T];
  const int Nx = V.lens[LAPH_DIM_X];
  const int Ny = V.lens[LAPH_DIM_Y];
  const int Nz = V.lens[LAPH_DIM_Z];

  const int Nt_local = geom.lat.T;
  const int Nx_local = geom.lat.LX;
  const int Ny_local = geom.lat.LY;
  const int Nz_local = geom.lat.LZ;

  nyom::Stopwatch sw;

  std::vector<complex<double>> buffer( Nev*Nc*Nx*Ny*Nz );

  /* the reasonable thing to do here would be MPI I/O, but it would take
   * a while to implement. Since this is a test code, we will read the entire
   * time slice worth of eigenvectors and just grab what we need. At a later
   * stage we can implement and proper reading routine */

  int64_t * indices;
  complex<double> * pairs;
  int64_t npair;
  int64_t buffer_index;
  sw.reset();
  V.read_local(&npair, &indices, &pairs);
  sw.elapsed_print_and_reset("V.local_read");
  
  int64_t counter = 0;
  for( int t = 0; t < Nt_local; ++t ){
    int gt = Nt_local*geom.mpi.proc_coords[0] + t;
    std::stringstream filename;
    filename << path << "/eigenvectors."
      << std::setfill('0') << std::setw(4) << cid << '.'
      << std::setfill('0') << std::setw(3) << gt;

    std::ifstream ev_file(filename.str(), 
                          std::ios::in | std::ios::binary );
    ev_file.read(reinterpret_cast<char*>(buffer.data()), 
                 buffer.size()*sizeof( complex<double> ) );
    if( !ev_file ){
      std::cout << "Error reading from " <<
        filename.str() << std::endl;
      std::abort();
    }
    ev_file.close();

    for( int x = 0; x < Nx_local; ++x ){
      int gx = Nx_local*geom.mpi.proc_coords[1] + x;
      for( int y = 0; y < Ny_local; ++y ){
        int gy = Ny_local*geom.mpi.proc_coords[2] + y;
        for( int z = 0; z < Nz_local; ++z ){
          int gz = Nz_local*geom.mpi.proc_coords[3] + z;
          for( int ev = 0; ev < Nev; ++ev ){
            for( int c = 0; c < Nc; ++c ){
              buffer_index = c 
                             + gz * Nc
                             + gy * Nz*Nc
                             + gx * Ny*Nz*Nc
                             + ev * Nx*Ny*Nz*Nc;

              indices[counter] = c
                                 + ev * Nc
                                 + gz * Nev*Nc
                                 + gy * Nz*Nev*Nc
                                 + gx * Ny*Nz*Nev*Nc
                                 + gt * Nx*Ny*Nz*Nev*Nc;

              pairs[counter] = buffer[buffer_index];
              counter++;
            } // c
          } // ev
        } // z
      } // y
    } // x
  } // t
  sw.elapsed_print_and_reset("V I/O and V.reshuffle");
  V.write(npair, indices, pairs);
  sw.elapsed_print_and_reset("V.write");
  free(indices); free(pairs);
}

} // namespace(nyom)
