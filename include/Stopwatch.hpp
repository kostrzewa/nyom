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

namespace nyom {

  typedef struct duration {
    double min;
    double max;
    double mean;
  } duration;

class Stopwatch {
  public:
    Stopwatch() {
      MPI_Comm_rank(MPI_COMM_WORLD,
                    &rank);
      MPI_Comm_size(MPI_COMM_WORLD,
                    &Nranks); 
    }

    void reset(void){
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
                    MPI_COMM_WORLD);
      duration.mean = duration.mean / Nranks;

      MPI_Allreduce(&seconds,
                    &(duration.min),
                    1,
                    MPI_DOUBLE,
                    MPI_MIN,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&seconds,
                    &(duration.max),
                    1,
                    MPI_DOUBLE,
                    MPI_MAX,
                    MPI_COMM_WORLD);
      return(duration);
    }

    nyom::duration elapsed_print(const char* const name){
      nyom::duration duration = elapsed();
      if(rank==0){ 
        cout << "Time for " << name << " " << duration.mean 
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
  
  private:
    std::chrono::time_point<std::chrono::steady_clock> time;
    int rank;
    int Nranks;
};

} //namespace(nyom)

