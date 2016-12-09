
#include <cstdio>

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include <ctf.hpp>

#include "tmLQCD.h"

using namespace CTF;
using namespace std;

double timeDiff(
        const std::chrono::time_point<std::chrono::steady_clock>& time, 
        const char* const name,
        const int rank){
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

  // prevent integer overflows further down the line
  size_t Ds = 4;
  size_t Cs = 3;
  size_t Fs = 2;
  size_t Ls = mpi.nproc_x * LX;
  size_t Ts = mpi.nproc_t * T;

  //int prop_sizes[8] = { Ts, Ls, Ls, Ls, Ds, Ds, Cs, Cs };
  //int prop_sizes[8] = { Cs, Cs, Ds, Ds, Ls, Ls, Ls, Ts };
  int prop_shapes[6] = { NS, NS, NS, NS, NS, NS };
  int prop_sizes[6] = { Ts, Ls*Ls*Ls, Ds, Ds, Cs, Cs };
 
  Tensor< std::complex<double> > S(6, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > Sconj(6, prop_sizes, prop_shapes, dw);

  std::complex<double>* pairs;
  int64_t *indices;
  int64_t  npair;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  start = std::chrono::steady_clock::now();

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,VOLUMEPLUSRAND,2);

  // read propagators from file(s)? if yes, don't need to load gauge field
  bool read = false;
  bool write = false;
  if(read==false) tmLQCD_read_gauge(lat.nstore);
  
  indices = (int64_t*) calloc(4*3*VOLUME, sizeof(int64_t));
  pairs = (std::complex<double>*) calloc(4*3*VOLUME, sizeof(std::complex<double>));
  for(size_t src_s = 0; src_s < Ds; ++src_s){
    for(size_t src_c = 0; src_c < Cs; ++src_c){
      if(read == false){ 
        source_spinor_field(g_spinor_field[0],g_spinor_field[1], src_s, src_c);
        convert_eo_to_lexic(temp_field[1],g_spinor_field[0],g_spinor_field[1]);
        moment = std::chrono::steady_clock::now();
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
        read_spinor(g_spinor_field[0],g_spinor_field[1],"source.0000.00.inverted", (int)(src_s*Cs+src_c) );
        convert_eo_to_lexic(temp_field[0],g_spinor_field[0],g_spinor_field[1]);
      }
      // globally translate from tmLQCD to Cyclops
      moment = std::chrono::steady_clock::now();
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
//                  indices[counter] = src_c  * (Ts*Ls*Ls*Ls*Ds*Ds*Cs) +
//                                     prop_c * (Ts*Ls*Ls*Ls*Ds*Ds)    +
//                                     src_s  * (Ts*Ls*Ls*Ls*Ds)       +
//                                     prop_s * (Ts*Ls*Ls*Ls)          +
//                                     gz     * (Ts*Ls*Ls)             +
//                                     gy     * (Ts*Ls)                +
//                                     gx     * (Ts)                   +
//                                     gt;
                  indices[counter] = src_c  * (Ts*Ls*Ls*Ls*Ds*Ds*Cs) +
                                     prop_c * (Ts*Ls*Ls*Ls*Ds*Ds)    +
                                     src_s  * (Ts*Ls*Ls*Ls*Ds)       +
                                     prop_s * (Ts*Ls*Ls*Ls)          +
                                     x_3d   * (Ts)                   + // x_3d runs from 0 to Ls^3
                                     gt;

                  // need to write a clean wrapper for this which
                  // deals with the struct directly instead of doing two different pointer arithmetics in one line...
                  pairs[counter] = *( ((_Complex double*)(temp_field[0] + g_ipt[t][x][y][z])) + prop_s*Cs + prop_c );
                  counter++;
                }
              }
            }
          }
        }
      }
      // global write with automatic MPI communication
      S.write(4*3*VOLUME,indices,pairs);
      timeDiffAndUpdate(moment,"copying propagator to tensor",rank);
    }
  }
  free(indices); free(pairs);
  finalize_solver(temp_field,2);
 
  // take complex conjugate of S
  Sconj["txijab"] = S["txijab"];
  ((Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sconj["txijab"]);

  // perform contraction
  Vector< std::complex<double> > C(Ts,dw);
  MPI_Barrier( MPI_COMM_WORLD );
  moment = std::chrono::steady_clock::now();
  Flop_counter flp;
  flp.zero();
  C["t"] = Sconj["tXIJAB"]*S["tXIJAB"];
  MPI_Barrier( MPI_COMM_WORLD );
  double cntr_time = timeDiffAndUpdate(moment,"2-pt contraction",rank);
  double global_time;
  
  MPI_Allreduce( &cntr_time, &global_time, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );
  cntr_time = global_time / np;
  int64_t flops = flp.count( MPI_COMM_WORLD );

  // naive number of FP ops
  // 4 FP ops per complex multiplication
  // 2*(3*(L-1) + (C-1) + (D-1) ) additions to sum contraction of all indices
  // 12 additions : (7 indices are contracted -> 7-1 complex additions for the sum)
  // repeated T times
  if(rank==0){
    printf("Naive Performance: %.6e mflop/s\n", (double)Ts*( 12 + 4*(double)Ls*Ls*Ls*Ds*Ds*Cs*Cs + 2*( 3*(Ls-1) + (Cs-1) + (Ds-1) )  )/(cntr_time*1e6));
    printf("'True' (Cyclops) Performance: %.6e mflop/s\n", (double)(flops)/(cntr_time*1e6));
  }

  elapsed_seconds = std::chrono::steady_clock::now()-start;
  
  if(rank==0)
    cout << "Pion 2-pt function took " << elapsed_seconds.count() << " seconds" << std::endl;

  C.print();

  double norm = 1.0/((double)Ls*Ls*Ls);
  C.read_all(&npair,&pairs);
  if(rank==0){
    ofstream correl;
    char fname[200]; snprintf(fname,200,"pion_2pt_%05d.txt", lat.nstore);
    correl.open(fname);
    for(int64_t i = 0; i < npair; ++i){
      correl << i << "\t" << norm*pairs[i].real() << "\t" << norm*pairs[i].imag() << endl;
    }
    correl.close();
  } 
  
  //MPI_Finalize();
  return 0;
}
