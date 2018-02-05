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

#include <stdexcept>
#include <memory>
#include <sstream>

namespace nyom {

/**
 * @brief The geometry class serves as a link between the CTF world
 * and the parallelisation considerations for LapH eigensystems
 * and perambulators for the purpose of efficient parallel I/O.
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

      tmLQCD_invert_init(argc,argv,1,0);
      tmLQCD_get_mpi_params(&mpi);
      tmLQCD_get_lat_params(&lat);

      MPI_Comm_size(MPI_COMM_WORLD, &Nranks);

      //// partition the ranks in two dimensions along time and eigenvector extents
      //int ranks_remaining;
      //Nranks_t = gcd(Nranks, Nt);
      //ranks_remaining = Nranks / Nranks_t;
      //Nranks_ev = gcd(ranks_remaining, Nev);
      //ranks_remaining = ranks_remaining / Nranks_ev;
      //if( ranks_remaining != 1 ){
      //  std::stringstream ss;
      //  ss << "Number of MPI ranks " <<
      //    Nranks <<
      //    " cannot be partitioned into dimensions " <<
      //    "Nt = " << Nt << " and " <<
      //    "Nev = " << Nev << "!\n";
      //  throw std::runtime_error(ss.str());
      //}

      //local_Nt = Nt / Nranks_t;
      //local_Nev = Nev / Nranks_ev;
      //dim_Nranks[LAPH_DIM_T] = Nranks_t;
      //dim_Nranks[LAPH_DIM_EV] = Nranks_ev;

      //MPI_Dims_create(Nranks, 2, dim_Nranks);

      world = new CTF::World(argc,argv);
    }

  ~Geometry(){
    delete world;
    tmLQCD_finalise();
    MPI_Finalize();
  }

  CTF::World& get_world() const {
    return *world;
  }

  tmLQCD_lat_params lat;
  tmLQCD_mpi_params mpi;

  private:
    CTF::World* world;
    int Nranks;

};

}
