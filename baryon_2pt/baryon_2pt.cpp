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
#include "constexpr.hpp"

#include "PointSourcePropagator.hpp"
#include "SpinColourPropagator.hpp"
#include "ColourDipolePropagator.hpp"
#include "SpinPropagator.hpp"
#include "MomentumTensor.hpp"
#include "LeviCivita.hpp"
#include "Shifts.hpp"

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

  const int n_src_per_gauge = core.input_node["n_src_per_gauge"].as<int>();

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
  std::mt19937 mt_gen(conf_start+12345);

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

  nyom::SpinColourPropagator q1(core);
  nyom::SpinColourPropagator q2(core);
  nyom::SpinColourPropagator q3(core);

  nyom::ColourDipolePropagator colour_dipole(core);


  sw.reset();
  nyom::MomentumTensor mom_snk(core,
                               {0, 0, 0});
  sw.elapsed_print("Momentum tensor creation");

  nyom::SpinPropagator C(core);
  nyom::SpinPropagator C_translated(core);

  nyom::LeviCivita eps_abc(core, 3);

  // this can be elevated to include the source phase factor at some point
  double norm = 1.0/((double)Ns*Ns*Ns); 

  for( int conf_idx = conf_start; conf_idx <= conf_end; conf_idx += conf_stride ){
    if(read) {
      write = false; 
    } else {
      tmLQCD_read_gauge( conf_idx ); 
    }

    for(int src_id = 0; src_id < n_src_per_gauge; src_id++){
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
      
      // we will time translate the correlator such that the indices are always
      // relative to the source
      // we also undo the twisted temporal boundary conditions on the quark
      // fields (beware ifnaive anti-periodic boundary conditions are used in
      // the inversion instead)
      nyom::SimpleShift time_translation(core, Nt, -src_coords[0], 3*nyom::pi/Nt);

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
      up_mom["tijab"] = mom_snk["XYZ"] * props[UP]["tXYZijab"];
      sw.elapsed_print_and_reset("up prop mom projection");

      down_mom["tijab"] = mom_snk["XYZ"] * props[DOWN]["tXYZijab"];
      sw.elapsed_print_and_reset("down prop mom projection");

      up["tijab"] = nyom::g["TwistPlus"]["iK"] * up_mom["tKLab"] * nyom::g["TwistPlus"]["Lj"];
      sw.elapsed_print_and_reset("up twist rotation");

      down["tijab"] = nyom::g["TwistMinus"]["iK"] * down_mom["tKLab"] * nyom::g["TwistMinus"]["Lj"];
      sw.elapsed_print_and_reset("down twist rotation");
    
      // we build a 4x4 correlator matrix for the proton two-point function
      // where we have \Gamma_1 in the dipole and \Gamma_2 multiplying the
      // third quark field in the following combinations
      //
      // \Gamma_1 = C\gamma_5           , \Gamma_2 = 1
      // \Gamma_1 = C                   , \Gamma_2 = \gamma_5
      // \Gamma_1 = Ci\gamma_0\gamma_5  , \Gamma_2 = 1
      // \Gamma_1 = C\gamma_0           , \Gamma_2 = \gamma_5
      //
      // Note that at the source we use the identity
      //
      // \gamma_0 \Gamma_i^\dagger \gamma_0 = (sign_\Gamma_i) \Gamma_i
      //
      // where (sign_\Gamma_i) is simply 1 or -1 and encoded in g0_sign below
      
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

          // This is the "direct" baryon contraction
          //
          //  C["jit"] = nyom::g0_sign[g1_src] * nyom::g0_sign[g2_src] *
          //             eps_abc["ABC"] * eps_abc["EFG"] * 
          //             nyom::g[g1_snk]["LM"] * nyom::g[g2_snk]["iN"] *
          //             nyom::g[g1_src]["OP"] * nyom::g[g2_src]["Qj"] *
          //             up["tLPAE"] * up["tNQCG"] * down["tMOBF"];
          //
          // Doing the full contraction in one go leads to issues when more than
          // four MPI tasks are used. Technically, one can even combine
          // the direct and exchange diagrams into a single line. It works
          // up to four MPI tasks..
          //
          // Hence, we first take care of gamma structures
          // first the third quark
          // note that on the LHS below, we keep lower-case indices which will later
          // be contractred as upper-case indices, such that we can keep track of what
          // we are doing
          q3["tijcg"] = nyom::g0_sign[g2_src] * nyom::g[g2_snk]["iN"] * up["tNQcg"] * nyom::g[g2_src]["Qj"];
          sw.elapsed_print_and_reset("first term q3 construction");
          
          // now the colour dipole "propagator"
          colour_dipole["taebf"] = nyom::g0_sign[g1_src] * nyom::g[g1_snk]["LM"] * up["tLPae"] * nyom::g[g1_src]["OP"] * down["tMObf"];
          sw.elapsed_print_and_reset("first term colour dipole construction");

          // note index convention for Spin propagator, chosen such that it can be
          // serialised easily below
          C["jit"] = eps_abc["ABC"] * eps_abc["EFG"] *
                     q3["tijCG"] * colour_dipole["tAEBF"];
          sw.elapsed_print_and_reset("first term colour anti-symmetrisation");

          // now for the second term with up-quark exchange
          // 
          // C["jit"] -= nyom::g0_sign[g1_src] * nyom::g0_sign[g2_src] *
          //            eps_abc["ABC"] * eps_abc["EFG"] * 
          //            nyom::g[g1_snk]["LM"] * nyom::g[g2_snk]["iN"] *
          //            nyom::g[g1_src]["OP"] * nyom::g[g2_src]["Qj"] *
          //            up["tLQAG"] * up["tNPCE"] * down["tMOBF"];
          
          q3["tljag"] = nyom::g0_sign[g2_src] * up["tlQag"] * nyom::g[g2_src]["Qj"];
          sw.elapsed_print_and_reset("second term q3 construction");
          
          q1["tipce"] = nyom::g[g2_snk]["iN"] * up["tNpce"];
          sw.elapsed_print_and_reset("second term q1 construction");

          q2["tlpbf"] = nyom::g0_sign[g1_src] * nyom::g[g1_snk]["lM"] * nyom::g[g1_src]["Op"] * down["tMObf"];
          sw.elapsed_print_and_reset("second term q2 construction");

          C["jit"] -= eps_abc["ABC"] * eps_abc["EFG"] *
                      q3["tLjAG"] * q2["tLPBF"] * q1["tiPCE"];
          sw.elapsed_print_and_reset("second term three-quark contraction and colour anti-symmetrisation");

          C_translated["jit"] = time_translation["tT"] * C["jiT"];
          sw.elapsed_print_and_reset("Time translation");

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
            correl << "t_snk d_snk d_src re im" << std::endl;
            for(int64_t i = 0; i < correl_nval; ++i){
              C.get_idx_coords(idx_coords, i);
              correl << idx_coords[2] << " " << idx_coords[1] << " " << idx_coords[0] << "\t" <<
                std::setprecision(16) << norm*correl_values[i].real() << "\t" << 
                std::setprecision(16) << norm*correl_values[i].imag() << 
                std::endl;
            }
            correl.close();
          }
          free(correl_values);

          C_translated.tensor.read_all(&correl_nval, &correl_values);
          if(rank == 0){
            std::vector<int> idx_coords(3, 0);
            ofstream correl; 
            char fname[200]; 
            snprintf(fname, 200,
                     "translated_baryon_2pt_conf%05d_srcidx%03d"
                     "_srct%03d_srcx%03d_srcy%03d_srcz%03d_"
                     "snkG1-%s_snkG2-%s_srcG1-%s_srcG2-%s.txt",
                     conf_idx, src_id,
                     src_coords[0], src_coords[1], src_coords[2], src_coords[3],
                     g1_snk.c_str(),
                     g2_snk.c_str(),
                     g1_src.c_str(),
                     g2_src.c_str());
            correl.open(fname);
            correl << "t_snk d_snk d_src re im" << std::endl;
            for(int64_t i = 0; i < correl_nval; ++i){
              C.get_idx_coords(idx_coords, i);
              correl << idx_coords[2] << " " << idx_coords[1] << " " << idx_coords[0] << "\t" <<
                std::setprecision(16) << norm*correl_values[i].real() << "\t" << 
                std::setprecision(16) << norm*correl_values[i].imag() << 
                std::endl;
            }
            correl.close();
          }
          free(correl_values);

        } // loop over g1_snk
      } // loop over g1_src
    } // loop over sources
  } // loop over configurations

  finalize_solver(temp_field,3);
  finalize_solver(temp_eo_spinors,2);
  return 0;
}
