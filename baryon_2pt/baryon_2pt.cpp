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

#include "Core.hpp"
#include "Logfile.hpp"
#include "Stopwatch.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>
#include <complex>

#include "gammas.hpp"

#include "PointSourcePropagator.hpp"
#include "SpinColourPropagator.hpp"
#include "SpinPropagator.hpp"
#include "MomentumTensor.hpp"
#include "LeviCivita.hpp"

extern "C" {
#include "global.h"
#include "solver/solver_field.h"
#include "start.h"
#include "io/spinor.h"
#include "linalg/convert_eo_to_lexic.h"
} // extern "C"

#include <omp.h>

#define printf0 if(core.geom.get_myrank()==0) printf

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

  const int nyom_threads = core.input_node["threaded"].as<bool>() ? 2 : 1;

  const int Nd = 4;
  const int Nc = 3;

  nyom::init_gammas( core.geom.get_world() );

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,VOLUMEPLUSRAND,3);

  spinor ** temp_eo_spinors = NULL;
  init_solver_field(&temp_eo_spinors,VOLUME/2,2);

  spinor * src_spinor = temp_field[2];
  spinor * prop_inv = temp_field[0];
  spinor * prop_fill = temp_field[1];
  spinor * prop_tmp;

  // read propagators from file(s)? if yes, don't need to load gauge field
  bool read = false;
  bool write = false;
  
  std::random_device r;
  std::mt19937 mt_gen(12345);

  // uniform distribution in space coordinates
  std::uniform_int_distribution<int> ran_space_idx(0, Ns-1);
  // uniform distribution in time coordinates
  std::uniform_int_distribution<int> ran_time_idx(0, Nt-1);

  std::vector<nyom::PointSourcePropagator> props;
  props.emplace_back(core);
  props.emplace_back(core);

  nyom::SpinColourPropagator up(core);
  nyom::SpinColourPropagator down(core);
  
  nyom::SpinColourPropagator up_mom(core);
  nyom::SpinColourPropagator down_mom(core);

  sw.reset();
  nyom::MomentumTensor mom_snk(core,
                               {0, 0, 0});
  sw.elapsed_print("Momentum tensor creation");

  nyom::SpinPropagator C(core);

  nyom::LeviCivita eps_abc(core, 3);

  double norm = 1.0/((double)Ns*Ns*Ns); 

  for( int conf_idx = conf_start; conf_idx <= conf_end; conf_idx += conf_stride ){
    if(read) {
      write = false; 
    } else {
      tmLQCD_read_gauge( conf_idx ); 
    }

    for(int src_id = 0; src_id < 1; src_id++){
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

      for(int flav_idx : {UP, DOWN} ){
        props[flav_idx].set_src_coords(src_coords);
        for(size_t src_d = 0; src_d < Nd; ++src_d){
          for(size_t src_c = 0; src_c < Nc; ++src_c){
            #pragma omp parallel num_threads(nyom_threads)
            {
              if( (nyom_threads == 1) ||
                  (nyom_threads != 1 && omp_get_thread_num() == 0) ){
                printf0("Thread id %d of %d doing inversion\n", omp_get_thread_num(), omp_get_num_threads());
                if(read == false){
                  full_source_spinor_field_point(src_spinor, src_d, src_c, src_coords); 
                  tmLQCD_invert((double*)prop_inv,
                                (double*)src_spinor,
                                flav_idx,
                                0);

                  if(write){
                    convert_lexic_to_eo(temp_eo_spinors[0], temp_eo_spinors[1], prop_inv);
                    WRITER *writer = NULL;
                    char fname[200];
                    snprintf(fname,
                             200,
                             "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
                             conf_idx,
                             flav_idx,
                             src_coords[0],
                             src_coords[1],
                             src_coords[2],
                             src_coords[3]);
                    construct_writer(&writer, fname, 1);
                    write_spinor(writer, &temp_eo_spinors[0], &temp_eo_spinors[1], 1, 64);
                    destruct_writer(writer);
                  }
                
                }else{
                  char fname[200];
                  snprintf(fname,
                           200,
                           "source.conf%04d.flav%1d.srct%3d.srcx%3d.srcy%3d.srcz%3d.inverted",
                           conf_idx,
                           flav_idx,
                           src_coords[0],
                           src_coords[1],
                           src_coords[2],
                           src_coords[3]);
                  read_spinor(temp_eo_spinors[0],temp_eo_spinors[1], fname, (int)(src_d*Nc+src_c) );
                  convert_eo_to_lexic(prop_inv, temp_eo_spinors[0], temp_eo_spinors[1]);
                }
              }
              
              #pragma omp barrier
              #pragma omp single
              {
                printf0("Thread %d switching pointers\n", omp_get_thread_num());
                prop_tmp = prop_inv;
                prop_inv = prop_fill;
                prop_fill = prop_tmp;
              }
            
              if( (nyom_threads == 1) || 
                  (nyom_threads != 1 && omp_get_thread_num() == 1) ){ 
                printf0("Thread %d filling tensor\n", omp_get_thread_num());
                props[flav_idx].fill(prop_fill,
                                     src_d,
                                     src_c);
              }
            } // OpenMP parallel closing brace
          }
        }
      }
      
      sw.reset();

      // first we do the momentum projections, allowing us to work with
      // small tensors below
      down_mom["tijab"] = mom_snk["XYZ"] * props[DOWN]["tXYZKLab"];
      sw.elapsed_print_and_reset("down prop mom projection");
      
      up_mom["tijab"] = mom_snk["XYZ"] * props[UP]["tXYZKLab"];
      sw.elapsed_print_and_reset("up prop mom projection");

      up["tijab"] = nyom::g["Ip5"]["iK"] * up_mom["tKLab"] * nyom::g["Ip5"]["Lj"];
      sw.elapsed_print_and_reset("up twist rotation");

      down["tijab"] = nyom::g["Im5"]["iK"] * down_mom["tKLab"] * nyom::g["Im5"]["Lj"];
      sw.elapsed_print_and_reset("down twist rotation");
    
      // we build a 3x3 correlator matrix for the proton two-point function
      // with 
      // \Gamma = C\gamma_5           , \tilde{\Gamma} = 1
      // \Gamma = C                   , \tilde{\Gamma} = \gamma_5
      // \Gamma = Ci\gamma_0\gamma_5  , \tilde{\Gamma} = 1
      // \Gamma = C\gamma_0           , \tilde{\Gamma} = \gamma_5
      
      // note that below, \Gamma == g1, \tilde{\Gamma} = g2
      for( std::string g1_src : { "C", "C5", "Ci05", "C0" } ){
        std::string g2_src = "I";
        if( g1_src == std::string("C") || g1_src == std::string("C0") ){
          g2_src = std::string("5");
        }
        for( std::string g1_snk : { "C", "C5", "Ci05", "C0" } ){ 
          std::string g2_snk = "I";
          if( g1_snk == std::string("C") || g1_snk == std::string("C0") ){
            g2_snk = std::string("5");
          }
          // note index convention for Spin propagator
          C["jit"] = nyom::g0_sign[g1_src] * nyom::g0_sign[g2_src] *
                   // ^ signs on source gamma structures due to gamma_0 insertions 
                     eps_abc["ABC"] * eps_abc["EFG"] * 
                     nyom::g[g1_snk]["IJ"] * nyom::g[g2_snk]["iK"] *
                     nyom::g[g1_src]["LM"] * nyom::g[g2_src]["Nj"] *
                     ( up["tIMAE"] * up["tKNCG"] - up["tINAG"] * up["tKMCE"] ) *
                     down["tJLBF"];
          
          sw.elapsed_print_and_reset("Spin-colour contraction");

          std::complex<double>* correl_values;
          int64_t correl_nval;
          C.tensor.read_all(&correl_nval, &correl_values);
          if(rank == 0){
            std::vector<int> idx_coords(3, 0);
            ofstream correl; 
            char fname[200]; 
            snprintf(fname, 200,
                     "baryon_2pt_conf%05d_srcidx%03d"
                     "_srct%03d_srcx%03d_srcy%03d_srcz%03d_"
                     "snkG1-%s_snkG2-%s_srcG1-%s_srcG2-%s.txt",
                     conf_idx, src_id,
                     src_coords[0], src_coords[1], src_coords[2], src_coords[3],
                     g1_snk.c_str(),
                     g2_snk.c_str(),
                     g1_src.c_str(),
                     g2_src.c_str());
            correl.open(fname);
            for(int64_t i = 0; i < correl_nval; ++i){
              C.get_idx_coords(idx_coords, i);
              correl << idx_coords[2] << " " << idx_coords[1] << " " << idx_coords[0] << "\t" <<
                std::setprecision(16) << norm*correl_values[i].real() << "\t" << 
                std::setprecision(16) << norm*correl_values[i].imag() << 
                endl;
            }
            correl.close();
          }
        } // loop over g1_snk
      } // loop over g1_src
    } // loop over sources
  } // loop over configurations

  finalize_solver(temp_field,3);
  finalize_solver(temp_eo_spinors,2);
  return 0;
}
