/***********************************************************************                                                                                                            
 * Copyright (C) 2016 Bartosz Kostrzewa
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

#include <cstdio>

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>

#include <ctf.hpp>

#include "tmLQCD.h"

extern "C" {
#include "global.h"
#include "solver/solver_field.h"
#include "start.h"
#include "io/spinor.h"
#include "linalg/convert_eo_to_lexic.h"
}

#include "gammas.hpp"

using namespace CTF;
using namespace std;
using namespace nyom;

void timeReset( std::chrono::time_point<std::chrono::steady_clock>& time ){
  MPI_Barrier(MPI_COMM_WORLD);
  time = std::chrono::steady_clock::now();
}

double timeDiff(
        const std::chrono::time_point<std::chrono::steady_clock>& time, 
        const char* const name,
        const int rank){
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::duration<double> elapsed_seconds = std::chrono::steady_clock::now() - time;
  if(rank==0){ 
    cout << "Time for " << name << " " << elapsed_seconds.count() <<
      " seconds" << std::endl;
  }
  return(elapsed_seconds.count());
}

double timeDiffAndUpdate(
        std::chrono::time_point<std::chrono::steady_clock>& time, 
        const char* const name,
        const int rank){
  double rval = timeDiff(time,name,rank);
  time = std::chrono::steady_clock::now();
  return(rval);
}

double measureFlopsPerSecond( double local_time, 
                              CTF::Flop_counter& flp,
                              const char* const name,
                              const int rank, const int np ){
  double global_time;
  MPI_Allreduce( &local_time, &global_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  int64_t flops = flp.count( MPI_COMM_WORLD );
  double fps = (double)(flops)/(global_time*1e6/np);
  if(rank==0){
    printf("'Performance in '%s': %.6e mflop/s\n", name, fps);
  }
}

int main(int argc, char ** argv) {
  int rank, np, d;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  tmLQCD_invert_init(argc,argv,1,0);
  
  tmLQCD_lat_params lat;
  tmLQCD_mpi_params mpi;

  tmLQCD_get_lat_params(&lat);
  tmLQCD_get_mpi_params(&mpi);

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;
  
  if(rank==0){ 
    printf("tmLQCD local lattice volume T:%02d x X:%02d x Y:%02d x Z:%02d\n", 
              lat.T, lat.LX, lat.LY, lat.LZ);
    printf("tmLQCD mpi params    nproc: %05d nproc_t: %02d nproc_x: %02d nproc_y: %02d nproc_z: %02d\n", 
              mpi.nproc, mpi.nproc_t, mpi.nproc_x, mpi.nproc_y, mpi.nproc_z);
  }

  // print tmLQCD process information in order
  for(int r = 0; r < mpi.nproc; ++r){
      if(r == rank){ 
        printf("tmLQCD mpi params  cart_id: %05d proc_id: %05d time_rank: %02d\n", 
                  mpi.cart_id, mpi.proc_id, mpi.time_rank);
        printf("tmLQCD mpi params  proc[t]: %02d proc[x]: %02d proc[y]: %02d proc_z: %02d\n", 
                  mpi.proc_coords[0], mpi.proc_coords[1], mpi.proc_coords[2], mpi.proc_coords[3]);
        MPI_Barrier(MPI_COMM_WORLD);
      }else{
        MPI_Barrier(MPI_COMM_WORLD);
      }
  }

  World dw(argc, argv);

  timeReset(moment);
  init_gammas(dw);
  timeDiff(moment,"gamma initialisation", rank);
  
  // print out all gamma matrices for cross-check
  //for( std::string g1 : { "I", "0", "1", "2", "3", "5", "Ip5", "Im5" } ){ 
  //  if(rank==0) cout << "gamma" << g1 << endl;
  //  MPI_Barrier(MPI_COMM_WORLD);
  //  g[g1].print();
  //}
  //
  //for( std::string g1 : { "0", "1", "2", "3", "5" } ){ 
  //  for( std::string g2 : { "0", "1", "2", "3" } ){
  //    if( g1 == g2 ) continue;
  //    if(rank==0) cout << "gamma" << g1+g2 << endl;
  //    MPI_Barrier(MPI_COMM_WORLD);
  //    g[g1+g2].print();
  //  }
  //}

  // prevent integer overflows further down the line
  size_t Ds = 4;
  size_t Cs = 3;
  size_t Fs = 2;
  size_t Ls = mpi.nproc_x * LX;
  size_t Ts = mpi.nproc_t * T;

  int prop_sizes[8] = { Ts, Ls, Ls, Ls, Ds, Ds, Cs, Cs };
  //int prop_sizes[8] = { Cs, Cs, Ds, Ds, Ls, Ls, Ls, Ts };
  int prop_shapes[8] = { NS, NS, NS, NS, NS, NS, NS, NS };
  //int prop_sizes[6] = { Ts, Ls*Ls*Ls, Ds, Ds, Cs, Cs };
 
  Tensor< std::complex<double> > S(8, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > Sconj(8, prop_sizes, prop_shapes, dw);
  
  Tensor< std::complex<double> > Ssrc(8, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > Ssnk(8, prop_sizes, prop_shapes, dw);

  const int nsources = 5;

  timeReset(start);

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,VOLUMEPLUSRAND,2);

  // read propagators from file(s)? if yes, don't need to load gauge field
  bool read = false;
  bool write = false;
  if(read) {
    write = false; 
  } else {
    tmLQCD_read_gauge(lat.nstore); 
  }
  
  std::random_device r;
  std::mt19937 mt_gen(12345);

  // uniform distribution in space coordinates
  std::uniform_int_distribution<int> ran_space_idx(0, Ls-1);
  // uniform distribution in time coordinates
  std::uniform_int_distribution<int> ran_time_idx(0, Ts-1);

  int64_t* prop_tensor_indices = (int64_t*) calloc(4*3*VOLUME, sizeof(int64_t));
  std::complex<double>* prop_tensor_pairs = (std::complex<double>*) calloc(4*3*VOLUME, sizeof(std::complex<double>));
  for(int src_id = 0; src_id < nsources; src_id++){
    int src_coords[4];
    
    // for certainty, we broadcast the source coordinates from rank 0 
    if(rank==0){
      src_coords[0] = ran_time_idx(mt_gen);
      for(int i = 1; i < 4; ++i){
        src_coords[i] = ran_space_idx(mt_gen);
      }
    }
    MPI_Bcast(src_coords,
              4,
              MPI_INT,
              0,
              MPI_COMM_WORLD);

    for(size_t src_s = 0; src_s < Ds; ++src_s){
      for(size_t src_c = 0; src_c < Cs; ++src_c){
        timeReset(moment);
        if(read == false){
          full_source_spinor_field_point(temp_field[1], src_s, src_c, src_coords); 
          //source_spinor_field(g_spinor_field[0],g_spinor_field[1], src_s, src_c);
          //convert_eo_to_lexic(temp_field[1],g_spinor_field[0],g_spinor_field[1]);
          tmLQCD_invert((double*)temp_field[0],(double*)temp_field[1],0,0);
          timeDiffAndUpdate(moment,"inversion",rank);

          if(write){
            convert_lexic_to_eo(g_spinor_field[0], g_spinor_field[1], temp_field[0]);
            WRITER *writer = NULL;
            char fname[200];
            snprintf(fname,200,"source.%04d.00.00.inverted",lat.nstore);
            construct_writer(&writer, fname, 1);
            write_spinor(writer, &g_spinor_field[0], &g_spinor_field[1], 1, 64);
            destruct_writer(writer);
          }
          timeDiffAndUpdate(moment,"propagator io",rank);
        
        }else{
          char fname[200]; snprintf(fname,200,"source.%04d.00.00.inverted",lat.nstore);
          read_spinor(g_spinor_field[0],g_spinor_field[1],fname, (int)(src_s*Cs+src_c) );
          convert_eo_to_lexic(temp_field[0],g_spinor_field[0],g_spinor_field[1]);
          timeDiffAndUpdate(moment,"propagator io",rank);
        }
        // globally translate from tmLQCD to Cyclops
        timeReset(moment);
        int64_t counter = 0;
        for(size_t t = 0; t < T; ++t){
          size_t gt = T*mpi.proc_coords[0] + t;

          for(size_t x = 0; x < LX; ++x){
            size_t gx = LX*mpi.proc_coords[1] + x;

            for(size_t y = 0; y < LY; ++y){
              size_t gy = LY*mpi.proc_coords[2] + y;

              for(size_t z = 0; z < LZ; ++z){
                size_t gz = LZ*mpi.proc_coords[3] + z;
                size_t x_3d = Ls*Ls*gx + Ls*gy + gz;

                for(size_t prop_s = 0; prop_s < Ds; ++prop_s){
                  for(size_t prop_c = 0; prop_c < Cs; ++prop_c){
                    // global Cyclops index for S, leftmost tensor index runs fastest
                    prop_tensor_indices[counter] = src_c  * (Ts*Ls*Ls*Ls*Ds*Ds*Cs) +
                                                   prop_c * (Ts*Ls*Ls*Ls*Ds*Ds)    +
                                                   src_s  * (Ts*Ls*Ls*Ls*Ds)       +
                                                   prop_s * (Ts*Ls*Ls*Ls)          +
                                                   gz     * (Ts*Ls*Ls)             +
                                                   gy     * (Ts*Ls)                +
                                                   gx     * (Ts)                   +
                                                   gt;
                    //indices[counter] = src_c  * (Ts*Ls*Ls*Ls*Ds*Ds*Cs) +
                    //                   prop_c * (Ts*Ls*Ls*Ls*Ds*Ds)    +
                    //                   src_s  * (Ts*Ls*Ls*Ls*Ds)       +
                    //                   prop_s * (Ts*Ls*Ls*Ls)          +
                    //                   x_3d   * (Ts)                   + // x_3d runs from 0 to Ls^3
                    //                   gt;

                    // need to write a clean wrapper for this which
                    // deals with the struct directly instead of doing two different pointer arithmetics in one line...
                    prop_tensor_pairs[counter] = *( ((_Complex double*)(temp_field[0] + g_ipt[t][x][y][z])) + prop_s*Cs + prop_c );
                    counter++;
                  }
                }
              }
            }
          }
        }
        // global write with automatic MPI communication
        S.write(4*3*VOLUME,prop_tensor_indices,prop_tensor_pairs);
        timeDiffAndUpdate(moment,"copying propagator to tensor",rank);
      }
    }
 
    // transpose colour indices and take complex conjugate for gamma5  hermiticity
    // the transpose in spin for the gamma_5 S^dag gamma_5 identity will be taken
    // below
    Sconj["txyzijba"] = S["txyzijab"];
    ((Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sconj["txyzijab"]);

    // perform contractions
    Flop_counter flp;
    Vector< std::complex<double> > C(Ts,dw);
    double norm = 1.0/((double)Ls*Ls*Ls);

    std::complex<double>* correl_pairs;
    int64_t correl_npair;
    // many of these are non-sensical or zero, but for now, let's loop over all of them
    for( std::string g_src : {"5", "05"} ){
      for( std::string g_snk : {"5", "05"} ){
          timeReset(moment);
          flp.zero();

        Ssrc["txyzijab"] = S["txyziLab"] * (g[g_src])["LK"] * (g["5"])["Kj"];
          timeDiffAndUpdate(moment,"source gamma insertion",rank);

        Ssnk["txyzijab"] = Sconj["txyzLiab"] * (g["5"])["LK"]  * (g[g_snk])["Kj"];
          timeDiffAndUpdate(moment,"sink gamma insertion",rank);

        C["t"] = Ssnk["tXYZIJAB"] * Ssrc["tXYZJIBA"];
          measureFlopsPerSecond( timeDiffAndUpdate(moment,"2-pt contraction",rank), flp, "meson 2-pt function", rank, np );

        C.read_all(&correl_npair,&correl_pairs);
        if(rank==0){
          ofstream correl;
          char fname[200]; 
          snprintf(fname, 200,
                   "pion_2pt_conf%05d_srcidx%03d_srct%03d_srcx%03d_srcy%03d_srcz%03d_snkG%s_srcG%s.txt",
                   lat.nstore, src_id,
                   src_coords[0], src_coords[1], src_coords[2], src_coords[3],
                   g_snk.c_str(), g_src.c_str());
          correl.open(fname);
          for(int64_t i = 0; i < correl_npair; ++i){
            correl << i << "\t" << 
              std::setprecision(16) << norm*correl_pairs[i].real() << "\t" << 
              std::setprecision(16) << norm*correl_pairs[i].imag() << 
              endl;
          }
          correl.close();
        }
        free(correl_pairs);
        timeDiffAndUpdate(moment,"Correlator IO",rank);
      }
    }
  } // loop over sources
  free(prop_tensor_indices); free(prop_tensor_pairs);
  finalize_solver(temp_field,2);
  MPI_Barrier(MPI_COMM_WORLD);
  elapsed_seconds = std::chrono::steady_clock::now()-start;
  if(rank==0)
    cout << "All meson 2-pt functions took " << elapsed_seconds.count() << " seconds" << std::endl;

  tmLQCD_finalise();  
  MPI_Finalize(); // probably need to destroy CTF::World before calling this because it throws lots of errors
  return 0;
}
