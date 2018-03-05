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

  nyom::SpinDilutedTimeslicePropagatorVector src(core);
  nyom::SpinDilutedTimeslicePropagatorVector prop(core);

  nyom::LapH_eigsys V = nyom::make_LapH_eigsys(Nev,
                                               Nt,
                                               Nx,
                                               Ny,
                                               Nz,
                                               /* Nc  */ 3,
                                               core.geom.get_world());

  spinor* source = new spinor[VOLUME];
  spinor* propagator = new spinor[VOLUME];
 
  // these two tensors will serve as projectors from the eigensystem
  // to the source spinor (to compute the elements of the perambulator)
  // and for the projection of the resulting propagator spinor
  // to the perambulator tensor
  int shapes[3] = {NS, NS, NS};
  int sizes[3];
  sizes[0] = Nev;
  sizes[1] = 4;
  sizes[2] = Nt;
  CTF::Tensor<complex<double>> prop_proj(3, sizes, shapes, core.geom.get_world());
  CTF::Tensor<complex<double>> src_proj(3, sizes, shapes, core.geom.get_world());

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

    for(int tsrc = 0; tsrc < Nt; ++tsrc){
      // vector with a single 1.0 for the source time slice
      nyom::One one_tsrc = nyom::make_One(Nt, tsrc, core.geom.get_world() );
      for(int dsrc = 0; dsrc < 4; ++dsrc){
        // vector with a single 1.0 for the source dirac component
        nyom::One one_dsrc = nyom::make_One(4, dsrc, core.geom.get_world() );
        for(int esrc = 0; esrc < Nev; ++esrc){
          // vector with a single 1.0 for the source LapH eigenvector
          nyom::One one_esrc = nyom::make_One(Nev, esrc, core.geom.get_world() );

          // outer product of the "one" vectors in time, Dirac and LapH eigenvector space
          // this is a three-index tensor with a single 1.0 for the given combination
          src_proj["jqs"] =  one_esrc["j"] * one_dsrc["q"] * one_tsrc["s"];
          
          // project source eigenvector on source time slice and with source Dirac
          // index into a spinor
          sw.reset();
          src.tensor["txyzdc"] = V["czyxEt"] * src_proj["Edt"];
          // reshape to tmLQCD format
          src.push(source, core);
          sw.elapsed_print("LapH eigenvector to source projection");

          sw.reset();
          tmLQCD_invert(reinterpret_cast<double*>(&propagator[0]),
                        reinterpret_cast<double*>(&source[0]),
                        0, 0);
          sw.elapsed_print("Inversion");   

          sw.reset();
          // fully spin-diluted timeslice propagator to tensor
          prop.fill(propagator, dsrc, tsrc, core);
          sw.elapsed_print("Perambulator prop to tensor");
          
          sw.reset();
          // project the propagator with the sink eigenvectors
          prop_proj["ipt"] = prop.tensor["tXYZpC"] * Vdagger["CZYXit"];
          // outer product with source projector 
          peram["ijpqts"] += prop_proj["ipt"] * src_proj["jqs"];
          sw.elapsed_print("Propagator to perambulator projection");
        } // esrc
      } // dsrc 
    } // tsrc
    // TODO: PERAMBULATOR I/O
  } // cid

  delete[] source;
  delete[] propagator;

  return 0;
}
