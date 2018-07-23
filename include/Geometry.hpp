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

//#include "utility_functions.hpp"

#include <ctf.hpp>
#include <tmLQCD.h>

#include <exception>
#include <memory>
#include <sstream>

namespace nyom {

/**
 * @brief The geometry class serves as a link between the CTF world
 * and the parallelisation we inherit from tmLQCD.
 */
class Geometry {
  public:
    Geometry(int argc,
             char ** argv){
      
      int thread_requested = MPI_THREAD_SERIALIZED;
      int thread_provided;
      MPI_Init_thread(&argc, &argv, thread_requested, &thread_provided);
      if( thread_provided != thread_requested ){
        throw std::runtime_error("Provided MPI thread level does not match requested thread level in MPI_Init_thread.\n");
      }

      int tm_init_state = tmLQCD_invert_init(argc,argv,1,0);
      if( tm_init_state != 0 ){
        throw std::runtime_error("tmLQCD_invert_init had nonzero exit status!\n");
      }
      tmLQCD_get_mpi_params(&tmlqcd_mpi);
      tmLQCD_get_lat_params(&tmlqcd_lat);

      MPI_Comm_size(MPI_COMM_WORLD, &Nranks);
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

      world = new CTF::World(argc,argv);
    }

  ~Geometry(){
    finalise();
  }

  void finalise(void){
    delete world;
    tmLQCD_finalise();
    MPI_Finalize();
  }

  // print tmLQCD process information in order
  void print_tmLQCD_geometry(){
    for(int r = 0; r < tmlqcd_mpi.nproc; ++r){
      MPI_Barrier(MPI_COMM_WORLD);
      if(r == myrank){ 
        printf("tmLQCD mpi params  cart_id: %5d proc_id: %5d time_rank: %3d\n", 
                  tmlqcd_mpi.cart_id, tmlqcd_mpi.proc_id, tmlqcd_mpi.time_rank);
        printf("tmLQCD mpi params  proc[t]: %3d proc[x]: %3d proc[y]: %3d proc_z: %3d\n", 
               tmlqcd_mpi.proc_coords[0],
               tmlqcd_mpi.proc_coords[1],
               tmlqcd_mpi.proc_coords[2],
               tmlqcd_mpi.proc_coords[3]);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  CTF::World& get_world() const {
    return *world;
  }

  int get_myrank() const {
    return myrank;
  }

  int get_Nranks() const {
    return Nranks;
  }

  tmLQCD_lat_params tmlqcd_lat;
  tmLQCD_mpi_params tmlqcd_mpi;

  private:
    CTF::World* world;
    int Nranks;
    int myrank;

};

}
