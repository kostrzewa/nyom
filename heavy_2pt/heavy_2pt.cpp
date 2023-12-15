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

#include <cstring>

#include "gammas.hpp"

#include "NfSpinDilutedTimeslicePropagator.hpp"
#include "SpinDilutedTimeslicePropagator.hpp"
#include "PointSourcePropagator.hpp"
#include "SpinColourPropagator.hpp"
#include "MomentumTensor.hpp"

//extern "C" {
//#include "global.h"
//#include "solver/solver_field.h"
//#include "start.h"
//#include "io/spinor.h"
//#include "linalg/convert_eo_to_lexic.h"
//} // extern "C"

#include <omp.h>

#define printf0 if(core.geom.get_myrank()==0) printf

double measureFlopsPerSecond(double local_time, 
                             CTF::Flop_counter& flp,
                             const char* const name,
                             const int rank,
                             const int np,
                             const nyom::Core & core)
{
  double global_time;
  MPI_Allreduce( &local_time, &global_time, 1, MPI_DOUBLE, MPI_SUM, core.geom.get_nyom_comm() );
  int64_t flops = flp.count( core.geom.get_nyom_comm() );
  double fps = (double)(flops)/(global_time*1e6/np);
  printf0("Performance in '%s': %.6e mflop/s\n", name, fps);
  return fps;
}

typedef enum nd_src_flav_idx_t {
  UP = 0,
  DOWN,
  ND_UP,
  ND_DOWN
} nd_src_flav_idx_t;

int main(int argc, char ** argv) {
  nyom::Core core(argc,argv);
  
  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  const int rank = core.geom.get_myrank();
  const int np = core.geom.get_Nranks();

  const int seed = core.input_node["seed"].as<int>();
  const int conf_start = core.input_node["conf_start"].as<int>();
  const int conf_stride = core.input_node["conf_stride"].as<int>();
  const int conf_end = core.input_node["conf_end"].as<int>();
  const int n_src = core.input_node["n_src_per_gauge"].as<int>();

  const int Nd = 4;
  const int Nc = 3;

  const int Nt = core.input_node["Nt"].as<int>();
  const int Nx = core.input_node["Nx"].as<int>();    
  const int Ny = core.input_node["Ny"].as<int>();    
  const int Nz = core.input_node["Nz"].as<int>();
  const int Ns = Nx;
   
  const int pt = core.geom.tmlqcd_mpi.proc_coords[0];     
  const int px = core.geom.tmlqcd_mpi.proc_coords[1];     
  const int py = core.geom.tmlqcd_mpi.proc_coords[2];     
  const int pz = core.geom.tmlqcd_mpi.proc_coords[3];     
   
  const int lt = core.geom.tmlqcd_lat.T;     
  const int lx = core.geom.tmlqcd_lat.LX;    
  const int ly = core.geom.tmlqcd_lat.LY;    
  const int lz = core.geom.tmlqcd_lat.LZ;     
   
  const int local_volume = lt * lx * ly * lz;    

  const size_t spinor_elem = 4*3*2*local_volume;
  const size_t spinor_bytes = 8*spinor_elem;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  nyom::init_gammas( core.geom.get_world() );

  std::unique_ptr<double> s0(new double[spinor_elem]);
  std::unique_ptr<double> s1(new double[spinor_elem]);
  std::unique_ptr<double> p0(new double[spinor_elem]);
  std::unique_ptr<double> p1(new double[spinor_elem]);
  std::unique_ptr<double> tmp(new double[spinor_elem]);

  std::random_device r;
  std::mt19937 mt_gen(seed);

  // uniform distribution in space coordinates
  std::uniform_int_distribution<int> ran_space_idx(0, Ns-1);
  // uniform distribution in time coordinates
  std::uniform_int_distribution<int> ran_time_idx(0, Nt-1);

  nyom::SpinDilutedTimeslicePropagator Sup(core);
  nyom::SpinDilutedTimeslicePropagator Sdown(core);
  
  nyom::SpinDilutedTimeslicePropagator Sup_conj(core);
  nyom::SpinDilutedTimeslicePropagator Sdown_conj(core);
  
  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd(core);
  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd_conj(core);

  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd_snk(core);
  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd_src(core);

  sw.reset();

  for(int cid = conf_start; cid <= conf_end; cid += conf_stride){
    // read propagators from file(s)? if yes, don't need to load gauge field
    bool write = false;
    bool read = false;
    if(read) {
      write = false; 
    } else {
      tmLQCD_read_gauge(cid); 
    }

    for(int src_id = 0; src_id < n_src; src_id++){
      int src_ts;
      
      // for certainty, we broadcast the source coordinates from rank 0 
      if(rank == 0){
        src_ts = ran_time_idx(mt_gen);
      }
      MPI_Bcast(&src_ts, 1, MPI_INT, 0, core.geom.get_nyom_comm());

      Sup.set_src_ts(src_ts);
      Sdown.set_src_ts(src_ts);
      Sup_conj.set_src_ts(src_ts);
      Sdown_conj.set_src_ts(src_ts);
      S_nd.set_src_ts(src_ts);
      S_nd_conj.set_src_ts(src_ts);
      S_nd_snk.set_src_ts(src_ts);
      S_nd_src.set_src_ts(src_ts);

      for(int src_flav_idx : {UP,DOWN,ND_UP,ND_DOWN} ){
        for(int src_d = 0; src_d < Nd; ++src_d){
          if(!read){
            tmLQCD_full_source_spinor_field_spin_diluted_oet_ts(s0.get(), src_ts, src_d, src_id, cid, seed);
            if( src_flav_idx == UP || src_flav_idx == DOWN ){
              tmLQCD_invert(p0.get(), s0.get(), static_cast<int>(src_flav_idx), 0);
              if( src_flav_idx == UP ){
                Sup.fill(p0.get(), src_d);
              } else {
                Sdown.fill(p0.get(), src_d);
              }
            } else {
              memset((void*)s1.get(), 0, spinor_bytes);
              if( src_flav_idx == ND_DOWN ) s0.swap(s1);
              tmLQCD_invert_doublet(p0.get(), p1.get(), s0.get(), s1.get(), 3, 0);
              S_nd.fill(p0.get(), 0, static_cast<int>(src_flav_idx), src_d);
              S_nd.fill(p1.get(), 0, static_cast<int>(src_flav_idx), src_d);
            }
          }
        }
      }

  //            

  //                    
  //                    tmLQCD_doublet_invert(prop_inv.get(),
  //      src_spinor.get(),
  //                                src_flav_idx,
  //                                0);

  //                  if(write){
  //                    char fname[200];
  //                    snprintf(fname,
  //                             200,
  //                             "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
  //                             core.geom.tmlqcd_lat.nstore,
  //                             src_flav_idx,
  //                             src_coords[0],
  //                             src_coords[1],
  //                             src_coords[2],
  //                             src_coords[3]);
  //                    tmLQCD_write_spinor(prop_inv.get(), fname, 1, 1);
  //                  }
  //                
  //                }else{
  //                  char fname[200];
  //                  snprintf(fname,
  //                           200,
  //                           "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
  //                           core.geom.tmlqcd_lat.nstore,
  //                           src_flav_idx,
  //                           src_coords[0],
  //                           src_coords[1],
  //                           src_coords[2],
  //                           src_coords[3]);
  //                  tmLQCD_read_spinor(prop_inv.get(), fname, (int)(src_d*Nc+src_c) );
  //                }
  //              }
  //            
  //              #pragma omp barrier
  //              #pragma omp single
  //              {
  //                printf0("Thread %d switching pointers\n", omp_get_thread_num());
  //                prop_inv.swap(prop_fill);
  //              }
  //            
  //              if( (nyom_threads == 1) || 
  //                  (nyom_threads != 1 && omp_get_thread_num() == 1) ){ 
  //                printf0("Thread %d filling tensor\n", omp_get_thread_num());
  //                S.fill(prop_fill.get(),
  //                       src_d,
  //                       src_c);
  //              }
  //            } // OpenMP parallel closing brace
  //          }
  //        }
  //      }
  //     
 
  //    // transpose colour indices and take complex conjugate for gamma5  hermiticity
  //    // the transpose in spin for the gamma_5 S^dag gamma_5 identity will be taken
  //    // below
  //    sw.reset();
  //    Sconj["txyzijba"] = S["txyzijab"];
  //    ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sconj["txyzijab"]);
  //    sw.elapsed_print_and_reset("Complex conjugation and colour transpose");

  //    // perform contractions
  //    CTF::Flop_counter flp;
  //    CTF::Vector< std::complex<double> > C(Nt,core.geom.get_world());
  //    double norm = 1.0/((double)Ns*Ns*Ns);
  //    
  //    flp.zero();

  //    std::complex<double>* correl_values;
  //    int64_t correl_nval;
  //    // PP PA AP AA
  //    for( std::string g_src : {"5", "05"} ){
  //      for( std::string g_snk : {"5", "05"} ){

  //        Ssrc["txyzijab"] = S["txyziLab"] * (nyom::g[g_src])["LK"] * (nyom::g["5"])["Kj"];
  //        sw.elapsed_print_and_reset("source gamma insertion");

  //        Ssnk["txyzijab"] = Sconj["txyzLiab"] * (nyom::g["5"])["LK"]  * (nyom::g[g_snk])["Kj"];
  //        sw.elapsed_print_and_reset("sink gamma insertion");

  //        C["t"] = Ssnk["tXYZIJAB"] * Ssrc["tXYZJIBA"];
  //        measureFlopsPerSecond(sw.elapsed_print_and_reset("meson 2-pt function").mean,
  //                              flp, 
  //                              "meson 2-pt function",
  //                              core.geom.get_myrank(),
  //                              core.geom.get_Nranks(),
  //                              core.geom.get_nyom_comm() );

  //        C.read_all(&correl_nval,&correl_values);
  //        if(rank==0){
  //          ofstream correl;
  //          char fname[200]; 
  //          snprintf(fname, 200,
  //                   "pion_2pt_conf%05d_srcidx%03d_srct%03d_srcx%03d_srcy%03d_srcz%03d_snkG%s_srcG%s.txt",
  //                   cid, src_id,
  //                   src_coords[0], src_coords[1], src_coords[2], src_coords[3],
  //                   g_snk.c_str(), g_src.c_str());
  //          correl.open(fname);
  //          for(int64_t i = 0; i < correl_nval; ++i){
  //            correl << i << "\t" << 
  //              std::setprecision(16) << norm*correl_values[i].real() << "\t" << 
  //              std::setprecision(16) << norm*correl_values[i].imag() << 
  //              endl;
  //          }
  //          correl.close();
  //        }
  //        free(correl_values);
  //        sw.elapsed_print_and_reset("Correlator I/O");
  //      }
  //    }
    } // src loop
  } // cid loop

  MPI_Barrier( core.geom.get_nyom_comm() );

  return 0;
}
