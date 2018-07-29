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
  if(read) {
    write = false; 
  } else {
    tmLQCD_read_gauge( core.geom.tmlqcd_lat.nstore ); 
  }
  
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

  sw.reset();
  nyom::MomentumTensor mom_snk(core,
                               {0, 0, 0});
  sw.elapsed_print("Momentum tensor creation");

  nyom::SpinPropagator C(core);

  nyom::LeviCivita eps_abc(core, 3);

  double norm = 1.0/((double)Ns*Ns*Ns); 

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
                           core.geom.tmlqcd_lat.nstore,
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
                         core.geom.tmlqcd_lat.nstore,
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
    up["tijab"] = mom_snk["XYZ"] * props[UP]["tXYZijab"];
    down["tijab"] = mom_snk["XYZ"] * props[DOWN]["tXYZijab"];
    sw.elapsed_print_and_reset("Sink momentum projection");
   
    for( std::string g_src : { "C5", "C05" } ){
      for( std::string g_snk : { "C5", "C05" } ){ 
        // note index convention for Spin propagator
        C["jit"] = eps_abc["CDE"] * eps_abc["FGH"] * nyom::g[g_snk]["IJ"] * nyom::g[g_src]["KL"] *
                   up["tijCF"] * up["tIKDG"] * down["tJLGH"];
        sw.elapsed_print_and_reset("Spin-colour contraction");

        std::complex<double>* correl_values;
        int64_t correl_nval;
        C.tensor.read_all(&correl_nval, &correl_values);
        if(rank == 0){
          std::vector<int> idx_coords(3, 0);
          ofstream correl; 
          char fname[200]; 
          snprintf(fname, 200,
                   "baryon_2pt_conf%05d_srcidx%03d_srct%03d_srcx%03d_srcy%03d_srcz%03d_snkG%s_srcG%s.txt",
                   core.geom.tmlqcd_lat.nstore, src_id,
                   src_coords[0], src_coords[1], src_coords[2], src_coords[3],
                   g_snk.c_str(), g_src.c_str());
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
      } // loop over g_snk
    } // loop over g_src
  } // loop over sources

  finalize_solver(temp_field,3);
  finalize_solver(temp_eo_spinors,2);
  return 0;
}
