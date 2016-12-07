
#include <cstdio>

#include <vector>
#include <iostream>
#include <chrono>

#include <ctf.hpp>

#include "../../external/ranlxd.h"

#include "tmLQCD.h"

#define Ds 4
#define Cs 3
#define Fs 2

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

  int Ls = mpi.nproc_x * LX;
  int Ts = mpi.nproc_t * T;

  int prop_sizes[8] = { Ts, Ls, Ls, Ls, Ds, Ds, Cs, Cs };
  int prop_shapes[8] = { NS, NS, NS, NS, NS, NS, NS, NS };
 
  Tensor< std::complex<double> > S(8, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > Sconj(8, prop_sizes, prop_shapes, dw);

  std::complex<double>* pairs;
  int64_t *indices;
  int64_t  npair;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  start = std::chrono::steady_clock::now();

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,VOLUMEPLUSRAND,2);
  tmLQCD_read_gauge(lat.nstore);
  indices = (int64_t*) calloc(4*3*VOLUME, sizeof(int64_t));
  pairs = (std::complex<double>*) calloc(4*3*VOLUME, sizeof(std::complex<double>));
  for(int src_s = 0; src_s < 4; ++src_s){
    for(int src_c = 0; src_c < 3; ++src_c){ 
      source_spinor_field(g_spinor_field[0],g_spinor_field[1], src_s, src_c);
      convert_eo_to_lexic(temp_field[1],g_spinor_field[0],g_spinor_field[1]);
      moment = std::chrono::steady_clock::now();
      tmLQCD_invert((double*)temp_field[0],(double*)temp_field[1],0,0);
      timeDiffAndUpdate(moment,"inversion",rank);
      // globally translate from tmLQCD to Cyclops
      int counter = 0;
      for(int t = 0; t < T; ++t){
        int gt = T*mpi.proc_coords[0] + t;
        for(int x = 0; x < LX; ++x){
          int gx = LX*mpi.proc_coords[1] + x;
          for(int y = 0; y < LY; ++y){
            int gy = LY*mpi.proc_coords[2] + y;
            for(int z = 0; z < LZ; ++z){
              int gz = LZ*mpi.proc_coords[3] + z;
              for(int prop_s = 0; prop_s < Ds; ++prop_s){
                for(int prop_c = 0; prop_c < Cs; ++prop_c){
                  // global Cyclops index for S
                  indices[counter] = src_c  * (Ts*Ls*Ls*Ls*Ds*Ds*Cs) +
                                     prop_c * (Ts*Ls*Ls*Ls*Ds*Ds)    +
                                     src_s  * (Ts*Ls*Ls*Ls*Ds)       +
                                     prop_s * (Ts*Ls*Ls*Ls)          +
                                     gz     * (Ts*Ls*Ls)             +
                                     gy     * (Ts*Ls)                +
                                     gx     * (Ts)                   +
                                     gt;

                  pairs[counter] = *((_Complex double*)(temp_field[0] + g_ipt[t][x][y][z] + prop_s*Cs + prop_c));
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
  Sconj["txyzijab"] = S["txyzijab"];
  ((Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sconj["txyzijab"]);
  
  // perform contraction
  Vector< std::complex<double> > C(Ts,dw);
  moment = std::chrono::steady_clock::now();
  C["t"] = Sconj["tXYZJIBA"]*S["tXYZIJAB"];
  double cntr_time = timeDiffAndUpdate(moment,"2-pt contraction",rank);
  // 4 FP ops per complex multiplication
  // 2*(3*(L-1) + (C-1) + (D-1) ) additions to sum contraction of each index
  // 12 additions : (7 indices are contracted -> 7-1 complex additions for the sum)
  // repeated T times
  if(rank==0)
    printf("Performance: %.6e mflops\n", (double)Ts*( 12 + 4*(double)Ls*Ls*Ls*Ds*Ds*Cs*Cs + 2*( 3*(Ls-1) + (Cs-1) + (Ds-1) )  )/(cntr_time*1e6));

  elapsed_seconds = std::chrono::steady_clock::now()-start;
  
  if(rank==0)
    cout << "Pion 2-pt function took " << elapsed_seconds.count() << " seconds" << std::endl;

  C.print();
  
  MPI_Finalize();
  return 0;
}
