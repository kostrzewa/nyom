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

#include "Core.hpp"
#include "Logfile.hpp"
#include "Stopwatch.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>
#include <memory>

#include "gammas.hpp"

#include "PointSourcePropagator.hpp"
#include "SpinColourPropagator.hpp"
#include "MomentumTensor.hpp"

extern "C" {
#include "global.h"
#include "solver/solver_field.h"
#include "start.h"
#include "io/spinor.h"
#include "linalg/convert_eo_to_lexic.h"
} // extern "C"

#include <omp.h>

#define printf0 if(core.geom.get_myrank()==0) printf

double measureFlopsPerSecond( double local_time, 
                              CTF::Flop_counter& flp,
                              const char* const name,
                              const int rank,
                              const int np,
                              MPI_Comm comm)
{
  double global_time;
  MPI_Allreduce( &local_time, &global_time, 1, MPI_DOUBLE, MPI_SUM, comm );
  int64_t flops = flp.count( comm );
  double fps = (double)(flops)/(global_time*1e6/np);
  if(rank==0){
    printf("Performance in '%s': %.6e mflop/s\n", name, fps);
  }
  return fps;
}

typedef enum flav_idx_t {
  UP = 0,
  DOWN
} flav_idx_t;

int main(int argc, char ** argv) {
  nyom::Core core(argc,argv);
  
  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  const int rank = core.geom.get_myrank();
  const int np = core.geom.get_Nranks();

  const int Nt = core.input_node["Nt"].as<int>();
  const int Ns = core.input_node["Nx"].as<int>();
  const int conf_start = core.input_node["conf_start"].as<int>();
  const int conf_stride = core.input_node["conf_stride"].as<int>();
  const int conf_end = core.input_node["conf_end"].as<int>();
  const int n_src = core.input_node["n_src"].as<int>();

  const int nyom_threads = core.input_node["threaded"].as<bool>() ? 2 : 1;

  const int Nd = 4;
  const int Nc = 3;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  nyom::init_gammas( core.geom.get_world() );

  std::unique_ptr<double> src_spinor(new double[2*4*3*VOLUME]);
  std::unique_ptr<double> prop_inv(new double[2*4*3*VOLUME]);
  std::unique_ptr<double> prop_fill(new double[2*4*3*VOLUME]);

  // read propagators from file(s)? if yes, don't need to load gauge field
  bool read = false;
  bool write = false;
  if(read) {
    write = false; 
  } else {
    tmLQCD_read_gauge( core.geom.tmlqcd_lat.nstore ); 
  }
  
  std::random_device r;
  std::mt19937 mt_gen(core.input_node["seed"].as<int>());

  // uniform distribution in space coordinates
  std::uniform_int_distribution<int> ran_space_idx(0, Ns-1);
  // uniform distribution in time coordinates
  std::uniform_int_distribution<int> ran_time_idx(0, Nt-1);

  nyom::PointSourcePropagator<1> S(core);
  nyom::PointSourcePropagator<1> Sconj(core);

  nyom::PointSourcePropagator<1> Ssnk(core);
  nyom::PointSourcePropagator<1> Ssrc(core);

  sw.reset();

  for(int src_id = 0; src_id < n_src; src_id++){
    int src_coords[4];
    
    // for certainty, we broadcast the source coordinates from rank 0 
    if(rank == 0){
      src_coords[0] = ran_time_idx(mt_gen);
      for(int i = 1; i < 4; ++i){
        src_coords[i] = ran_space_idx(mt_gen);
      }
    }
    MPI_Bcast(src_coords,
              4,
              MPI_INT,
              0,
              core.geom.get_nyom_comm());

    for(int flav_idx : {UP} ){
      S.set_src_coords(src_coords);
      for(int src_d = 0; src_d < Nd; ++src_d){
        for(int src_c = 0; src_c < Nc; ++src_c){
          #pragma omp parallel num_threads(nyom_threads)
          {
            if( (nyom_threads == 1) ||
                (nyom_threads != 1 && omp_get_thread_num() == 0) ){
              printf0("Thread id %d of %d doing inversion\n", omp_get_thread_num(), omp_get_num_threads());
              if(read == false){
                tmLQCD_full_source_spinor_field_point(src_spinor.get(), src_d, src_c, src_coords); 
                tmLQCD_invert(prop_inv.get(),
                              src_spinor.get(),
                              flav_idx,
                              0);

                if(write){
                  char fname[200];
                  snprintf(fname,
                           200,
                           "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
                           core.geom.tmlqcd_lat.nstore,
                           flav_idx,
                           src_coords[0],
                           src_coords[1],
                           src_coords[2],
                           src_coords[3]);
                  tmLQCD_write_spinor(prop_inv.get(), fname, 1, 1);
                }
              
              }else{
                char fname[200];
                snprintf(fname,
                         200,
                         "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
                         core.geom.tmlqcd_lat.nstore,
                         flav_idx,
                         src_coords[0],
                         src_coords[1],
                         src_coords[2],
                         src_coords[3]);
                tmLQCD_read_spinor(prop_inv.get(), fname, (int)(src_d*Nc+src_c) );
              }
            }
          
            #pragma omp barrier
            #pragma omp single
            {
              printf0("Thread %d switching pointers\n", omp_get_thread_num());
              prop_inv.swap(prop_fill);
            }
          
            if( (nyom_threads == 1) || 
                (nyom_threads != 1 && omp_get_thread_num() == 1) ){ 
              printf0("Thread %d filling tensor\n", omp_get_thread_num());
              S.fill(prop_fill.get(),
                     src_d,
                     src_c,
                     flav_idx);
            }
          } // OpenMP parallel closing brace
        }
      }
    }
 
    // transpose colour indices and take complex conjugate for gamma5  hermiticity
    // the transpose in spin for the gamma_5 S^dag gamma_5 identity will be taken
    // below
    sw.reset();
    Sconj["txyzijbafg"] = S["txyzijabfg"];
    ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sconj["txyzijabfg"]);
    sw.elapsed_print_and_reset("Complex conjugation and colour transpose");

    // perform contractions
    CTF::Flop_counter flp;
    CTF::Vector< std::complex<double> > C(Nt,core.geom.get_world());
    double norm = 1.0/((double)Ns*Ns*Ns);
    
    flp.zero();

    std::complex<double>* correl_values;
    int64_t correl_nval;
    // PP PA AP AA
    for( std::string g_src : {"5", "05"} ){
      for( std::string g_snk : {"5", "05"} ){

        Ssrc["txyzijabfg"] = S["txyziLabfg"] * (nyom::g[g_src])["LK"] * (nyom::g["5"])["Kj"];
        sw.elapsed_print_and_reset("source gamma insertion");

        Ssnk["txyzijabfg"] = Sconj["txyzLiabfg"] * (nyom::g["5"])["LK"]  * (nyom::g[g_snk])["Kj"];
        sw.elapsed_print_and_reset("sink gamma insertion");

        C["t"] = Ssnk["tXYZIJABFG"] * Ssrc["tXYZJIBAGF"];
        measureFlopsPerSecond(sw.elapsed_print_and_reset("meson 2-pt function").mean,
                              flp, 
                              "meson 2-pt function",
                              core.geom.get_myrank(),
                              core.geom.get_Nranks(),
                              core.geom.get_nyom_comm() );

        C.read_all(&correl_nval,&correl_values);
        if(rank==0){
          ofstream correl;
          char fname[200]; 
          snprintf(fname, 200,
                   "pion_2pt_conf%05d_srcidx%03d_srct%03d_srcx%03d_srcy%03d_srcz%03d_snkG%s_srcG%s.txt",
                   core.geom.tmlqcd_lat.nstore, src_id,
                   src_coords[0], src_coords[1], src_coords[2], src_coords[3],
                   g_snk.c_str(), g_src.c_str());
          correl.open(fname);
          for(int64_t i = 0; i < correl_nval; ++i){
            correl << i << "\t" << 
              std::setprecision(16) << norm*correl_values[i].real() << "\t" << 
              std::setprecision(16) << norm*correl_values[i].imag() << 
              endl;
          }
          correl.close();
        }
        free(correl_values);
        sw.elapsed_print_and_reset("Correlator I/O");
      }
    }
  } // loop over sources

  MPI_Barrier( core.geom.get_nyom_comm() );

  return 0;
}
