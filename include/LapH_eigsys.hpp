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

#include "Geometry.hpp"
#include "Stopwatch.hpp"

#include <ctf.hpp>

#include <sstream>
#include <fstream>
#include <iomanip>

namespace nyom {

typedef enum LapH_eigsys_dims_t {
  LAPH_DIM_C = 0,
  LAPH_DIM_Z,
  LAPH_DIM_Y,
  LAPH_DIM_X,
  LAPH_DIM_EV,
  LAPH_DIM_T
} LapH_eigsys_dims_t;

// well, this is kind of a disappointing type, since it basically does not give
// any compile-time checking of the CTF::Tensor rank or dimensions
// Maybe one should wrap these tensors into classes after all, inheriting
// from CTF::Tensor to automatically provide all the overloaded operators
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
                                 const int gauge_conf_id,
                                 const nyom::Core & core){

  const int Nev = V.lens[LAPH_DIM_EV];
  const int Nc = V.lens[LAPH_DIM_C];
  
  const int Nt = V.lens[LAPH_DIM_T];
  const int Nx = V.lens[LAPH_DIM_X];
  const int Ny = V.lens[LAPH_DIM_Y];
  const int Nz = V.lens[LAPH_DIM_Z];

  const int Nt_local = core.geom.lat.T;
  const int Nx_local = core.geom.lat.LX;
  const int Ny_local = core.geom.lat.LY;
  const int Nz_local = core.geom.lat.LZ;

  nyom::Stopwatch sw;

  /* the reasonable thing to do here would be MPI I/O, but it would take
   * a while to implement. Since this is a test code, we will read the entire
   * time slice worth of eigenvectors and just grab what we need. At a later
   * stage we can implement and proper reading routine 
   * What is implemented below is still parallel in T, however. */
  
  MPI_Comm ts_comm;
  int ts_rank;
  int ts_Nranks;

  /* We split the communicator into groups which hold the same time slices.
   * Within the group, we determine the rank ordering with Z process location
   * running fastest, then Y, then X. */
  MPI_Comm_split(MPI_COMM_WORLD,
                 core.geom.mpi.proc_coords[0],
// Z running fastest
                 core.geom.mpi.proc_coords[3] +
                 core.geom.mpi.proc_coords[2]*core.geom.mpi.nproc_z +
                 core.geom.mpi.proc_coords[1]*core.geom.mpi.nproc_z*core.geom.mpi.nproc_y,
                 &ts_comm);
  MPI_Comm_rank(ts_comm,
                &ts_rank);
  MPI_Comm_size(ts_comm,
                &ts_Nranks);

  for( int proc = 0; proc < core.geom.get_Nranks(); ++proc ){
    MPI_Barrier(MPI_COMM_WORLD);
    if( proc == core.geom.get_myrank() ){
      printf("Process %d in MPI_COMM_WORLD has coords (txyz) %d %d %d %d\n"
             "Process %d in ts_comm\n", 
             core.geom.get_myrank(),
             core.geom.mpi.proc_coords[0],
             core.geom.mpi.proc_coords[1],
             core.geom.mpi.proc_coords[2],
             core.geom.mpi.proc_coords[3],
             ts_rank );
    }
  }

  sw.reset();

  std::vector<complex<double>> buffer; 
  std::vector<int64_t> indices; 
  
  if( ts_rank == 0 ){
    buffer.resize( Nev*Nx*Ny*Nz*Nc ); 
    indices.resize( Nev*Nx*Ny*Nz*Nc );
  } 

  for( int t = 0; t < Nt_local; ++t ){
    if( ts_rank == 0 ){
      int gt = Nt_local*core.geom.mpi.proc_coords[0] + t;
      std::stringstream filename;
      filename << path << "/eigenvectors."
        << std::setfill('0') << std::setw(4) << gauge_conf_id << '.'
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
      
      int64_t counter = 0;
      for( int ev = 0; ev < Nev; ++ev ){
        for( int gx = 0; gx < Nx; ++gx ){
          for( int gy = 0; gy < Ny; ++gy ){
            for( int gz = 0; gz < Nz; ++gz ){
                for( int c = 0; c < Nc; ++c ){
                  indices[counter] = c
                                     + gz * Nc
                                     + gy * Nz*Nc
                                     + gx * Ny*Nz*Nc
                                     + ev * Nx*Ny*Nz*Nc
                                     + gt * Nev*Nx*Ny*Nz*Nc;
                  counter++;
                } // c
              } // gz
            } // gy
          } // gx
        } // ev
      V.write(counter, indices.data(), buffer.data());
    } else {
      // all processes need to execute "write", but only those with data
      // actually perform one
      V.write(0, NULL, NULL);
    }
  } // t
  MPI_Barrier(MPI_COMM_WORLD);
  sw.elapsed_print_and_reset("V read from file");
} // read_LapH_eigsys_from_files

void write_LapH_eigsys_to_files(nyom::LapH_eigsys & V,
                                const std::string path,
                                const int gauge_conf_id,
                                const nyom::Core & core){

  const int Nev = V.lens[LAPH_DIM_EV];
  const int Nc = V.lens[LAPH_DIM_C];
  
  const int Nt = V.lens[LAPH_DIM_T];
  const int Nx = V.lens[LAPH_DIM_X];
  const int Ny = V.lens[LAPH_DIM_Y];
  const int Nz = V.lens[LAPH_DIM_Z];

  const int Nt_local = core.geom.lat.T;
  const int Nx_local = core.geom.lat.LX;
  const int Ny_local = core.geom.lat.LY;
  const int Nz_local = core.geom.lat.LZ;

  nyom::Stopwatch sw;

  /* the reasonable thing to do here would be MPI I/O but it would take a
   * while to implement. As a result, we are going to split MPI_COMM_WORLD
   * into groups on mpi.proc_coords[0].
   * For each group, we designate one process for writing and collect all
   * data for each timeslice there and then write. */

  MPI_Comm ts_comm;
  int ts_rank;
  int ts_Nranks;

  /* We split the communicator into groups which hold the same time slices.
   * Within the group, we determine the rank ordering with Z process location
   * running fastest, then Y, then X. */
  MPI_Comm_split(MPI_COMM_WORLD,
                 core.geom.mpi.proc_coords[0],
// Z running fastest
                 core.geom.mpi.proc_coords[3] +
                 core.geom.mpi.proc_coords[2]*core.geom.mpi.nproc_z +
                 core.geom.mpi.proc_coords[1]*core.geom.mpi.nproc_z*core.geom.mpi.nproc_y,
                 &ts_comm);
  MPI_Comm_rank(ts_comm,
                &ts_rank);
  MPI_Comm_size(ts_comm,
                &ts_Nranks);

  for( int proc = 0; proc < core.geom.get_Nranks(); ++proc ){
    MPI_Barrier(MPI_COMM_WORLD);
    if( proc == core.geom.get_myrank() ){
      printf("Process %d in MPI_COMM_WORLD has coords (txyz) %d %d %d %d\n"
             "Process %d in ts_comm\n", 
             core.geom.get_myrank(),
             core.geom.mpi.proc_coords[0],
             core.geom.mpi.proc_coords[1],
             core.geom.mpi.proc_coords[2],
             core.geom.mpi.proc_coords[3],
             ts_rank );
    }
  }

  sw.reset();  
  std::vector<complex<double>> buffer; 
  std::vector<int64_t> indices; 
  
  if( ts_rank == 0 ){
    buffer.resize( Nev*Nx*Ny*Nz*Nc ); 
    indices.resize( Nev*Nx*Ny*Nz*Nc );
  } 
  
  for( int64_t t = 0; t < Nt_local; ++t ){
    if( ts_rank == 0 ){
      int64_t gt = t + core.geom.mpi.proc_coords[0]*Nt_local;
      std::stringstream filename;
      filename << path << "/eigenvectors."
        << std::setfill('0') << std::setw(4) << gauge_conf_id << '.'
        << std::setfill('0') << std::setw(3) << gt;
      
      std::ofstream ev_file(filename.str(),
                            std::ios::out | std::ios::binary );
      
      int64_t counter = 0;
      for( int64_t ev = 0; ev < Nev; ++ev ){
        for( int64_t gx = 0; gx < Nx; ++gx ){
          for( int64_t gy = 0; gy < Ny; ++gy ){
            for( int64_t gz = 0; gz < Nz; ++gz ){
              for( int64_t c = 0; c < Nc; ++c ){
                indices[counter] = c
                                   + gz * Nc
                                   + gy * Nz*Nc
                                   + gx * Ny*Nz*Nc
                                   + ev * Nx*Ny*Nz*Nc
                                   + gt * Nev*Nx*Ny*Nz*Nc;
                counter++;
              } // c
            } // gz
          } // gy
        } // gx
      } // ev
      V.read(counter, indices.data(), buffer.data() );
      
      ev_file.write(reinterpret_cast<char*>( buffer.data() ),
                    Nev*Nx*Ny*Nz*Nc*sizeof( complex<double> ) );
      ev_file.close();
    } else {
      // all processes need to exectute read, but only ts_rank==0 ones
      // actually read data from the tensor
      V.read(0, NULL, NULL);
    }
  } // t
  MPI_Barrier(MPI_COMM_WORLD);
  sw.elapsed_print("Writing of eigenvectors");
} // write_LapH_eigsys_to_files

} // namespace(nyom)
