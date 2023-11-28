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

#include <chrono>
#include <mpi.h>

namespace nyom {

  typedef struct duration {
    double min;
    double max;
    double mean;
  } duration;

class Stopwatch {
  public:
    Stopwatch() {
      comm = MPI_COMM_WORLD;
      init();
    }

    Stopwatch(MPI_Comm comm_in){
      comm = comm_in;
      init();
    }

    void init(){
      MPI_Comm_rank(comm,
                    &rank);
      MPI_Comm_size(comm,
                    &Nranks);
      reset();
    }

    void reset(void){
      MPI_Barrier(comm);
      time = std::chrono::steady_clock::now();
    }
   
    nyom::duration elapsed(void) {
      nyom::duration duration;
      double seconds;

      std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - time;
      seconds = elapsed_seconds.count();

      MPI_Allreduce(&seconds,
                    &(duration.mean),
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    comm);
      duration.mean = duration.mean / Nranks;

      MPI_Allreduce(&seconds,
                    &(duration.min),
                    1,
                    MPI_DOUBLE,
                    MPI_MIN,
                    comm);
      MPI_Allreduce(&seconds,
                    &(duration.max),
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    comm);
      return(duration);
    }

    nyom::duration elapsed_print(const char* const name){
      nyom::duration duration = elapsed();
      if(rank==0){ 
        std::cout << name << " " << duration.mean 
          << " seconds" << std::endl
          << "min(" << duration.min << ") max(" 
          << duration.max << ")" << std::endl;
      }
      return(duration);
    }
    
    nyom::duration elapsed_print_and_reset(const char* const name){
      nyom::duration duration = elapsed_print(name);
      reset();
      return(duration);
    }

    void measure_flops_per_second(CTF::Flop_counter& flp,
                                  const char* const name,
                                  const int np ){
      int64_t flops = flp.count( MPI_COMM_WORLD );
      double fps = (double)(flops)/(elapsed().mean*1e6/np);
      if(rank==0){
        printf("'Performance in '%s': %.6e mflop/s\n", name, fps);
      }
    }
  
  private:
    std::chrono::time_point<std::chrono::steady_clock> time;
    int rank;
    int Nranks;
    MPI_Comm comm;
};



} //namespace(nyom)

