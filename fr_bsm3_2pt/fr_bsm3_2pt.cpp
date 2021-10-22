/***********************************************************************                                                                                                            
 * Copyright (C) 2021 Bartosz Kostrzewa
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

#include "gammas.hpp"

#include "PointSourcePropagator.hpp"
#include "SpinColourPropagator.hpp"
#include "MomentumTensor.hpp"

#include "contractions.hpp"

extern "C" {
#include "global.h"
#include "source_generation.h"
#include "prepare_source.h"
#include "solver/solver_field.h"
#include "start.h"
#include "io/spinor.h"
#include "linalg/convert_eo_to_lexic.h"
#include "linalg/square_norm.h"
#include "io/scalar.h"
} // extern "C"

#include <omp.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>

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
  core.geom.print_tmLQCD_geometry();
  
  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  const int rank = core.geom.get_myrank();
  const int np = core.geom.get_Nranks();

  const int Nt = core.input_node["Nt"].as<int>();
  const int Ns = core.input_node["Nx"].as<int>();
  const int conf_start = core.input_node["conf_start"].as<int>();
  const int conf_stride = core.input_node["conf_stride"].as<int>();
  const int conf_end = core.input_node["conf_end"].as<int>();

  const int n_src = core.input_node["n_src"].as<int>();

  const int nscalarstep = core.input_node["nscalarstep"].as<int>();
  const int npergauge = core.input_node["npergauge"].as<int>();

  const int nyom_threads = core.input_node["threaded"].as<bool>() ? 2 : 1;

  const int Nd = 4;
  const int Nc = 3;

  int gauge_conf_id = conf_start;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  //nyom::init_gammas( core.geom.get_world() );

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,2*VOLUME,3);

  spinor ** temp_eo_spinors = NULL;
  init_solver_field(&temp_eo_spinors,VOLUME/2,2);

  spinor * src_spinor = temp_field[0];
  spinor * prop_inv = temp_field[1];
  spinor * prop_fill = temp_field[2];
  spinor * prop_tmp;

  // read propagators from file(s)? if yes, don't need to load gauge field
  bool read = false;
  bool write = false;
  if(read) {
    write = false; 
  }
 
  // for now we don't  
  // std::random_device r;
  // std::mt19937 mt_gen(12345);

  // // uniform distribution in space coordinates
  // std::uniform_int_distribution<int> ran_space_idx(0, Ns-1);
  // // uniform distribution in time coordinates
  // std::uniform_int_distribution<int> ran_time_idx(0, Nt-1);

  nyom::PointSourcePropagator<2> S(core);
  nyom::PointSourcePropagator<2> Stilde(core);

  // unfortunately the way that sources are handled in tmLQCD is kind of brain-dead
  // with a global "SourceInfo" struct (instead of passing this as a parameter to the
  // inverter driver...)
  SourceInfo.sample = 0;
  SourceInfo.ix = 0;
  SourceInfo.no_flavours = 1;

  int src_coords[4] = {0, 0, 0, 0};

  for(int gauge_conf_id = conf_start; gauge_conf_id <= conf_end; gauge_conf_id += conf_stride){
    if(read) {
      write = false; 
    } else {
      // read unsmeared gauge field
      tmLQCD_read_gauge(gauge_conf_id, 0);
      // read smeared gauge field
      tmLQCD_read_gauge(gauge_conf_id, 1);
      SourceInfo.nstore = gauge_conf_id;
    }

    const int imeas = (conf_start-gauge_conf_id)/conf_stride;

    for(int iscalar = 0; iscalar < npergauge; iscalar++){
      const int scalar_idx = tmLQCD_compute_scalar_idx(imeas, iscalar, npergauge, nscalarstep);
      tmLQCD_read_scalar_field(scalar_idx);
      // TODO: smear scalar field for contractions

      for(int src_f : {UP,DOWN} ){
        for(int src_d = 0; src_d < 4; src_d++){
          for(int src_c = 0; src_c < 3; src_c++){
            for(int dagger_inv = 0; dagger_inv < 2; dagger_inv++){
              printf0("Performing inversion for gauge: %d scalar: %d f: %d, d: %d c: %d, dagger: %d\n",
                      gauge_conf_id, scalar_idx, src_f, src_d, src_c, dagger_inv); 
              if(!read){
                zero_spinor_field(src_spinor, 2*VOLUME);
                full_source_spinor_field_point(src_spinor + src_f*VOLUME, src_d, src_c, src_coords);
                tmLQCD_invert((double*)prop_inv, (double*)src_spinor, 0, dagger_inv, src_f, 0);
              } else {
              }
              if(dagger_inv == 0){
                S.fill(prop_inv, src_d, src_c, src_f);
              } else{
                Stilde.fill(prop_inv, src_d, src_c, src_f);
              }
            }
          }
        }
      }
      // take the complex conjugate of Stilde
      sw.reset();
      ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Stilde["txyzijabfg"]);
      sw.elapsed_print_and_reset("Complex conjugation of Stilde");
      // TODO call contraction function(s)
      local_2pt(S, Stilde, core);
      sw.elapsed_print_and_reset("Local contractions");
    } // scalar field loop
  } // gauge conf loop

  //    for(int flav_idx : {UP} ){
  //      props[flav_idx].set_src_coords(src_coords);
  //      for(size_t src_d = 0; src_d < Nd; ++src_d){
  //        for(size_t src_c = 0; src_c < Nc; ++src_c){
  //          #pragma omp parallel num_threads(nyom_threads)
  //          {
  //            if( (nyom_threads == 1) ||
  //                (nyom_threads != 1 && omp_get_thread_num() == 0) ){
  //              printf0("Thread id %d of %d doing inversion\n", omp_get_thread_num(), omp_get_num_threads());
  //              if(read == false){
  //                full_source_spinor_field_point(temp_field[2], src_d, src_c, src_coords); 
  //                tmLQCD_invert((double*)prop_inv,
  //                              (double*)src_spinor,
  //                              flav_idx,
  //                              0);

  //                if(write){
  //                  convert_lexic_to_eo(temp_eo_spinors[0], temp_eo_spinors[1], prop_inv);
  //                  WRITER *writer = NULL;
  //                  char fname[200];
  //                  snprintf(fname,
  //                           200,
  //                           "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
  //                           core.geom.tmlqcd_lat.nstore,
  //                           flav_idx,
  //                           src_coords[0],
  //                           src_coords[1],
  //                           src_coords[2],
  //                           src_coords[3]);
  //                  construct_writer(&writer, fname, 1);
  //                  write_spinor(writer, &temp_eo_spinors[0], &temp_eo_spinors[1], 1, 64);
  //                  destruct_writer(writer);
  //                }
  //              
  //              }else{
  //                char fname[200];
  //                snprintf(fname,
  //                         200,
  //                         "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
  //                         core.geom.tmlqcd_lat.nstore,
  //                         flav_idx,
  //                         src_coords[0],
  //                         src_coords[1],
  //                         src_coords[2],
  //                         src_coords[3]);
  //                read_spinor(temp_eo_spinors[0],temp_eo_spinors[1], fname, (int)(src_d*Nc+src_c) );
  //                convert_eo_to_lexic(prop_inv, temp_eo_spinors[0], temp_eo_spinors[1]);
  //              }
  //            }
  //          
  //            #pragma omp barrier
  //            #pragma omp single
  //            {
  //              printf0("Thread %d switching pointers\n", omp_get_thread_num());
  //              prop_tmp = prop_inv;
  //              prop_inv = prop_fill;
  //              prop_fill = prop_tmp;
  //            }
  //          
  //            if( (nyom_threads == 1) || 
  //                (nyom_threads != 1 && omp_get_thread_num() == 1) ){ 
  //              printf0("Thread %d filling tensor\n", omp_get_thread_num());
  //              props[flav_idx].fill(prop_fill,
  //                                   src_d,
  //                                   src_c);
  //            }
  //          } // OpenMP parallel closing brace
  //        }
  //      }
  //    }
 
  //    //// transpose colour indices and take complex conjugate for gamma5  hermiticity
  //    //// the transpose in spin for the gamma_5 S^dag gamma_5 identity will be taken
  //    //// below
  //    //sw.reset();
  //    //Sconj["txyzijba"] = props[UP]["txyzijab"];
  //    //((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sconj["txyzijab"]);
  //    //sw.elapsed_print_and_reset("Complex conjugation and colour transpose");

  //    //// perform contractions
  //    //CTF::Flop_counter flp;
  //    //CTF::Vector< std::complex<double> > C(Nt,core.geom.get_world());
  //    //double norm = 1.0/((double)Ns*Ns*Ns);
  //    //
  //    //flp.zero();

  //    //sw.reset();
  //    //Ssrc["tijab"] = mom_src["XYZ"] * props[UP]["tXYZijab"];
  //    //Ssnk["tijab"] = mom_snk["XYZ"] * Sconj["tXYZijab"];
  //    //sw.elapsed_print_and_reset("source and sink momentum projection");

  //    //std::complex<double>* correl_values;
  //    //int64_t correl_nval;
  //    //// many of these are non-sensical or zero, but for now, let's loop over all of them
  //    //for( std::string g_src : {"5", "05"} ){
  //    //  for( std::string g_snk : {"5", "05"} ){

  //    //    Ssrc["tijab"] = Ssrc["tiLab"] * (nyom::g[g_src])["LK"] * (nyom::g["5"])["Kj"];
  //    //    sw.elapsed_print_and_reset("source gamma insertion");

  //    //    Ssnk["tijab"] = Ssnk["tLiab"] * (nyom::g["5"])["LK"]  * (nyom::g[g_snk])["Kj"];
  //    //    sw.elapsed_print_and_reset("sink gamma insertion");

  //    //    C["t"] = Ssnk["tIJAB"] * Ssrc["tJIBA"];
  //    //    measureFlopsPerSecond(sw.elapsed_print_and_reset("meson 2-pt function").mean,
  //    //                          flp, 
  //    //                          "meson 2-pt function",
  //    //                          core.geom.get_myrank(),
  //    //                          core.geom.get_Nranks(),
  //    //                          core.geom.get_nyom_comm() );

  //    //    C.read_all(&correl_nval,&correl_values);
  //    //    if(rank==0){
  //    //      ofstream correl;
  //    //      char fname[200]; 
  //    //      snprintf(fname, 200,
  //    //               "pion_2pt_conf%05d_srcidx%03d_srct%03d_srcx%03d_srcy%03d_srcz%03d_snkG%s_srcG%s.txt",
  //    //               core.geom.tmlqcd_lat.nstore, src_id,
  //    //               src_coords[0], src_coords[1], src_coords[2], src_coords[3],
  //    //               g_snk.c_str(), g_src.c_str());
  //    //      correl.open(fname);
  //    //      for(int64_t i = 0; i < correl_nval; ++i){
  //    //        correl << i << "\t" << 
  //    //          std::setprecision(16) << norm*correl_values[i].real() << "\t" << 
  //    //          std::setprecision(16) << norm*correl_values[i].imag() << 
  //    //          endl;
  //    //      }
  //    //      correl.close();
  //    //    }
  //    //    free(correl_values);
  //    //    sw.elapsed_print_and_reset("Correlator I/O");
  //    //  }
  //    //}
  //  } // loop over sources

  MPI_Barrier( core.geom.get_nyom_comm() );

  finalize_solver(temp_field,3);
  finalize_solver(temp_eo_spinors,2);
  return 0;
}
