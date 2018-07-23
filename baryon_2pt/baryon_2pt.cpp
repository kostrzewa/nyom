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

#include <yaml-cpp/yaml.h>

#include "gammas.hpp"

#include "PointSourcePropagator.hpp"

extern "C" {
#include "global.h"
#include "solver/solver_field.h"
#include "io/spinor.h"
#include "linalg/convert_eo_to_lexic.h"
} // extern "C"

typedef enum flavour_idx_t {
  UP = 0,
  DOWN
} flavour_idx_t;

int main(int argc, char ** argv) {
  nyom::Core core(argc,argv);

  int rank = core.geom.get_myrank();

  int Nt = core.input_node["Nt"].as<int>();
  int Ns = core.input_node["Nx"].as<int>();
  int conf_start = core.input_node["conf_start"].as<int>();
  int conf_stride = core.input_node["conf_stride"].as<int>();
  int conf_end = core.input_node["conf_end"].as<int>();

  const int Nd = 4;
  const int Nc = 3;

  nyom::init_gammas( core.geom.get_world() );

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field,VOLUMEPLUSRAND,2);

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
              MPI_COMM_WORLD);

    for(int flav_idx : {UP, DOWN} ){
      props[flav_idx].set_src_coords(src_coords);
      for(size_t src_d = 0; src_d < Nd; ++src_d){
        for(size_t src_c = 0; src_c < Nc; ++src_c){
          if(read == false){
            full_source_spinor_field_point(temp_field[1], src_d, src_c, src_coords); 
            tmLQCD_invert((double*)temp_field[0],
                          (double*)temp_field[1],
                          flav_idx,
                          0);

            if(write){
              convert_lexic_to_eo(g_spinor_field[0], g_spinor_field[1], temp_field[0]);
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
              write_spinor(writer, &g_spinor_field[0], &g_spinor_field[1], 1, 64);
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
            read_spinor(g_spinor_field[0],g_spinor_field[1],fname, (int)(src_d*Nc+src_c) );
            convert_eo_to_lexic(temp_field[0],g_spinor_field[0],g_spinor_field[1]);
          }
          
          props[flav_idx].fill(temp_field[0],
                               src_d,
                               src_c,
                               core);
        }
      }
    }
  }

  finalize_solver(temp_field,2);
  return 0;
}
