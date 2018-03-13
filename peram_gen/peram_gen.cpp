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
#include "Perambulator.hpp"
#include "Propagator.hpp"
#include "LapH_eigsys.hpp"
#include "One.hpp"
#include "Stopwatch.hpp"
#include "Functors.hpp"

#include <sstream>

int main(int argc, char ** argv){
  nyom::Core core(argc,argv);

  int Nt  = core.input_node["Nt"].as<int>();
  int Nx  = core.input_node["Nx"].as<int>();
  int Ny  = core.input_node["Ny"].as<int>();
  int Nz  = core.input_node["Nz"].as<int>();

  int Nev = core.input_node["Nev"].as<int>();

  int conf_start = core.input_node["conf_start"].as<int>();
  int conf_stride = core.input_node["conf_stride"].as<int>();
  int conf_end = core.input_node["conf_end"].as<int>(); 

  std::string ev_path = core.input_node["ev_path"].as<std::string>();

  nyom::Stopwatch sw;

  nyom::Perambulator peram = nyom::make_Perambulator(Nev,
                                                     /* Nt_src */ Nt,
                                                     /* Nt_snk */ Nt,
                                                     core);

  nyom::SpinDilutedTimesliceSourceVector     src(core);
  nyom::SpinDilutedTimeslicePropagatorVector prop(core);

  nyom::LapH_eigsys V = nyom::make_LapH_eigsys(Nev,
                                               Nt,
                                               Nx,
                                               Ny,
                                               Nz,
                                               /* Nc  */ 3,
                                               core.geom.get_world());

  int64_t local_volume = Nt*Nx*Ny*Nz / core.geom.tmlqcd_mpi.nproc;

  spinor* source = new spinor[local_volume];
  spinor* propagator = new spinor[local_volume];
 
  // these two tensors will serve as projectors from the eigensystem
  // to the source spinor
  // and for the projection of the resulting propagator spinor
  // to the perambulator tensor
  int src_shapes[2] = {NS, NS};
  int src_sizes[2];
  src_sizes[0] = Nev;
  src_sizes[1] = Nt;

  int src_dof_shapes[3] = {NS, NS, NS};
  int src_dof_sizes[3];
  src_dof_sizes[0] = Nev;
  src_dof_sizes[1] = 4;
  src_dof_sizes[2] = Nt; 

  int prop_shapes[3] = {NS, NS, NS};
  int prop_sizes[3];
  prop_sizes[0] = Nev;
  prop_sizes[1] = 4;
  prop_sizes[2] = Nt;
  CTF::Tensor<complex<double>> proj_prop(3, prop_sizes, prop_shapes, core.geom.get_world());

  for(int cid = conf_start; cid <= conf_end; cid += conf_stride){ 
    tmLQCD_read_gauge( cid );
    read_LapH_eigsys_from_files(/* &LapH_eigsys */ V,
                                ev_path,
                                cid,
                                core);

    sw.reset();
    nyom::LapH_eigsys Vdagger(V);
    ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Vdagger["cizyxt"]);
    sw.elapsed_print("Vdagger");

    CTF::Flop_counter flop_counter;
    int64_t flops;
    nyom::duration elapsed;

    std::stringstream msg;

    for(int tsrc = 0; tsrc < Nt; ++tsrc){
      // vector with a single 1.0 for the source time slice
      nyom::One one_tsrc = nyom::make_One(Nt, tsrc, core.geom.get_world() );
      for(int esrc = 0; esrc < Nev; ++esrc){
        nyom::One one_esrc = nyom::make_One(Nev, esrc, core.geom.get_world() );

        //outer product of the "one" vectors in time, Dirac and LapH eigenvector space
        //this is a three-index tensor with a single 1.0 for the given combination
        CTF::Tensor<complex<double>> src_proj(2, src_sizes, src_shapes, core.geom.get_world(), "src_proj");
        src_proj["et"] =  one_esrc["e"] * one_tsrc["t"];
        
        sw.reset();
        nyom::V_project_SpinDilutedTimesliceSourceVector src_project =
          nyom::make_V_project_SpinDilutedTimesliceSourceVector(src,
                                                                V,
                                                                src_proj,
                                                                tsrc,
                                                                esrc,
                                                                core);
        sw.elapsed_print("Functor overhead V_project_SpinDilutedTimesliceSourceVector");
        
        sw.reset();
        src_project();
        elapsed = sw.elapsed_print("LapH eigenvector to source projection");
        msg << "src_project(time): " << elapsed.mean << std::endl;
        core.Logger("src_project",
                    nyom::log_perf,
                    msg.str());
        msg.str("");
        
        for(int dsrc = 0; dsrc < 4; ++dsrc){
          // vector with a single 1.0 for the source dirac component
          nyom::One one_dsrc = nyom::make_One(4, dsrc, core.geom.get_world() );

          if( core.geom.get_myrank() == 0 ){
            std::cout << "inverting t=" << tsrc << " esrs=" <<
              esrc << " dsrc=" << dsrc << std::endl;
          }

          CTF::Tensor<complex<double>> src_dof(3, src_dof_sizes, src_dof_shapes, core.geom.get_world(), "src_dof");
          src_dof["edt"] = one_esrc["e"] * one_dsrc["d"] * one_tsrc["t"];
          // sparsification seems to hurt performance for some reason...
          // src_dof.sparsify();

          // reshape source to tmLQCD format into correct source Dirac index
          src.push(source,
                   /* d_in */ dsrc,
                   /* t_in */ tsrc,
                   core);

          sw.reset();
          invert_quda_direct(reinterpret_cast<double*>(&propagator[0]),
                             reinterpret_cast<double*>(&source[0]),
                             0, 1);
          //tmLQCD_invert(reinterpret_cast<double*>(&propagator[0]),
          //              reinterpret_cast<double*>(&source[0]),
          //              0, 0);
          sw.elapsed_print("Inversion");   

          // propagator to tensor (has full complement of indices)
          // 2.5 seconds...
          prop.fill(propagator, dsrc, tsrc, core);
          
          flop_counter.zero();
          sw.reset();
          
          sw.reset();
          nyom::Vdagger_project_SpinDilutedTimeslicePropagatorVector prop_project =
            nyom::make_Vdagger_project_SpinDilutedTimeslicePropagatorVector(prop,
                                                                  Vdagger,
                                                                  proj_prop,
                                                                  core);
          sw.elapsed_print("Functor overhead Vdagger_project_SpinDilutedTimeslicePropagatorVector");
        
          // project the propagator with the sink eigenvectors
          sw.reset();
          prop_project();
          elapsed = sw.elapsed_print("Propagator to perambulator sink projection");
          flops = flop_counter.count();
          msg << "Vdagger_project(time,flops,Gflop/s): " << elapsed.mean << " "  << 
            flops << " " << (1.0e-9)*flops/elapsed.mean << std::endl;
          core.Logger("Vdagger_project",
                      nyom::log_perf,
                      msg.str());
          msg.str("");

          flop_counter.zero();
          sw.reset();
          // outer product with non-zero source indices
          // * peram["ijpqts"] += one_esrc["j"] * one_dsrc["q"] * one_tsrc["s"] * proj_prop["ipt"]; // this is awfully ( 23 sec )
          // * peram["ijpqts"] += src_dof["jqs"] * proj_prop["ipt"]; // this is awfully slow ( 17 sec )
          
          // this is reasonable, but still too slow for comfort... 
          // ( 4.8 sec with src_dof dense, 11 sec with src_dof sparse... )
          //peram["ijpqts"] += proj_prop["ipt"] * src_dof["jqs"];
          
          // this executes a manual write into the right slice of peram
          // and is much faster
          nyom::add_to_Perambulator( peram, proj_prop, tsrc, esrc, dsrc, core ); // 0.68 seconds...
          elapsed = sw.elapsed_print("Perambulator source index outer product");
          flops = flop_counter.count();
          msg << "Peram_outer(time,flops,Gflop/s): " << elapsed.mean << " "  << 
            flops << " " << (1.0e-9)*flops/elapsed.mean << std::endl;
          core.Logger("Peram_outer",
                      nyom::log_perf,
                      msg.str());
          msg.str("");
        } // esrc
      } // dsrc 
    } // tsrc
    // TODO: PERAMBULATOR I/O
  } // cid

  delete[] source;
  delete[] propagator;

  return 0;
}
