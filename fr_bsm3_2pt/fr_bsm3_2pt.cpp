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
#include "ScalarFieldVector.hpp"
#include "ComplexMatrixField.hpp"

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

  bool read_props = core.input_node["read_props"].as<bool>();
  bool write_props = core.input_node["write_props"].as<bool>();
  // prevent propagators from being overwritten
  if(read_props) write_props = false;

  const int Nd = 4;
  const int Nc = 3;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  constexpr int no_temp_spinor_fields = 3;
  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,2*VOLUME,no_temp_spinor_fields);

  constexpr int no_temp_eo_spinors = 2;
  spinor ** temp_eo_spinors = NULL;
  init_solver_field(&temp_eo_spinors,VOLUME/2,no_temp_eo_spinors);

  spinor * src_spinor = temp_field[0];
  spinor * prop_inv = temp_field[1];
  spinor * prop_fill = temp_field[2];

  double ** tmlqcd_scalar_field;

  // for now we don't  
  // std::random_device r;
  // std::mt19937 mt_gen(12345);

  // // uniform distribution in space coordinates
  // std::uniform_int_distribution<int> ran_space_idx(0, Ns-1);
  // // uniform distribution in time coordinates
  // std::uniform_int_distribution<int> ran_time_idx(0, Nt-1);

  nyom::PointSourcePropagator<2> S(core);
  nyom::PointSourcePropagator<2> Sbar(core);

  nyom::ScalarFieldVector<4> sf(core);
  std::map< std::string, nyom::ComplexMatrixField<2,2> > theta;
  std::map< std::string, nyom::ComplexMatrixField<2,2> > theta_tilde;

  for( std::string theta_name : {"1", "2", "3"} ){
    // man C++ sucks sometimes... our CTF::Tensor wrappers are currently not default constructible
    // so we have to go the long way around to put them into a map
    theta.emplace( std::piecewise_construct, std::forward_as_tuple(theta_name), 
                                             std::forward_as_tuple(core) );
    theta_tilde.emplace( std::piecewise_construct, std::forward_as_tuple(theta_name), 
                                                   std::forward_as_tuple(core) );
  }

  // unfortunately the way that sources are handled in tmLQCD is kind of brain-dead
  // with a global "SourceInfo" struct (instead of passing this as a parameter to the
  // inverter driver...)
  SourceInfo.sample = 0;
  SourceInfo.t = 0;
  SourceInfo.no_flavours = 1;

  int src_coords[4] = {0, 0, 0, 0};

  for(int gauge_conf_id = conf_start; gauge_conf_id <= conf_end; gauge_conf_id += conf_stride){
    if( !read_props ){
      // read unsmeared gauge field
      tmLQCD_read_gauge(gauge_conf_id, 0);
      // read smeared gauge field
      tmLQCD_read_gauge(gauge_conf_id, 1);
    }
    SourceInfo.nstore = gauge_conf_id;

    const int imeas = (conf_start-gauge_conf_id)/conf_stride;

    for(int iscalar = 0; iscalar < npergauge; iscalar++){
      const int scalar_idx = tmLQCD_compute_scalar_idx(imeas, iscalar, npergauge, nscalarstep);

      tmLQCD_read_scalar_field(scalar_idx);

      tmLQCD_get_scalar_field_pointer(&tmlqcd_scalar_field);

      sf.fill(tmlqcd_scalar_field);

      // because nyom::ComplexMatrixField<nr,nc> is not default-constrible, we can't use
      // the square brackets to access the elemetns of the corresponding map
      // well... that's a design failure right there, but let's go with it for now

      sw.reset();
      theta.at("1")["ijtxyz"] = nyom::tau["1"]["ij"] * sf[0]["txyz"] + nyom::tau["iI"]["ij"] * sf[1]["txyz"]; 
      theta_tilde.at("1")["ijtxyz"] = nyom::tau["1"]["ij"] * sf[0]["txyz"] - nyom::tau["iI"]["ij"] * sf[1]["txyz"];

      theta.at("2")["ijtxyz"] = nyom::tau["2"]["ij"] * sf[0]["txyz"] + nyom::tau["iI"]["ij"] * sf[2]["txyz"]; 
      theta_tilde.at("2")["ijtxyz"] = nyom::tau["2"]["ij"] * sf[0]["txyz"] - nyom::tau["iI"]["ij"] * sf[2]["txyz"];

      theta.at("3")["ijtxyz"] = nyom::tau["3"]["ij"] * sf[0]["txyz"] + nyom::tau["iI"]["ij"] * sf[3]["txyz"]; 
      theta_tilde.at("3")["ijtxyz"] = nyom::tau["3"]["ij"] * sf[0]["txyz"] - nyom::tau["iI"]["ij"] * sf[3]["txyz"]; 
      sw.elapsed_print_and_reset("theta and theta_tilde field construction");


      for(int src_f : {UP,DOWN} ){
        for(int src_d = 0; src_d < 4; src_d++){
          for(int src_c = 0; src_c < 3; src_c++){
            // this is so stupid as an interface...
            SourceInfo.ix = 3*src_d + src_c;

            for(int dagger_inv = 0; dagger_inv < 2; dagger_inv++){
              printf0("Performing inversion for gauge: %d scalar: %d f: %d, d: %d c: %d, dagger: %d\n",
                      gauge_conf_id, scalar_idx, src_f, src_d, src_c, dagger_inv); 
              if(!read_props){

                zero_spinor_field(src_spinor, 2*VOLUME);
                full_source_spinor_field_point(src_spinor + src_f*VOLUME, src_d, src_c, src_coords);
                tmLQCD_invert((double*)prop_inv, (double*)src_spinor, 0, dagger_inv, src_f, static_cast<int>(write_props) );

              } else {

                char spinor_filename[500];
                snprintf(spinor_filename, 500,
                         "%s.%04d.%02d.%02d.%08d.%s",
                         PropInfo.basename, SourceInfo.nstore, SourceInfo.t,
                         SourceInfo.ix, iscalar, "inverted");

                // the second sink flavour comes first in the file, so let's read
                // the records the other way around 
                read_spinor(temp_eo_spinors[0], temp_eo_spinors[1], spinor_filename,
                            4*src_f + 2*dagger_inv + 1);
                convert_eo_to_lexic(prop_inv, temp_eo_spinors[0], temp_eo_spinors[1]);

                read_spinor(temp_eo_spinors[0], temp_eo_spinors[1], spinor_filename,
                            4*src_f + 2*dagger_inv);
                convert_eo_to_lexic(prop_inv + VOLUME, temp_eo_spinors[0], temp_eo_spinors[1]);

              }

              if(dagger_inv == 0){
                S.fill(prop_inv, src_d, src_c, src_f);
              } else{
                Sbar.fill(prop_inv, src_d, src_c, src_f);
              }

            } // for(dagger_inv)
          } // for(src_c)
        } // for(src_d)
      } // for(src_f)
      // take the complex conjugate transpose of Sbar to get S(0,x)
      sw.reset();
      ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sbar["txyzijabfg"]);
      Sbar["txyzijabfg"] = Sbar["txyzjibagf"];
      sw.elapsed_print_and_reset("Conjugate transpose of Sbar");
      
      local_2pt(S, Sbar, core, gauge_conf_id, scalar_idx);
      PDP(S, Sbar, theta, theta_tilde, core, gauge_conf_id, scalar_idx);

      sw.elapsed_print_and_reset("Local contractions");
    } // scalar field loop
  } // gauge conf loop

  MPI_Barrier( core.geom.get_nyom_comm() );

  finalize_solver(temp_field,no_temp_spinor_fields);
  finalize_solver(temp_eo_spinors,no_temp_eo_spinors);
  return 0;
}
