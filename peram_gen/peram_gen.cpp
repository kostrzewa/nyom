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
  tmLQCD_read_gauge( 400 );

  nyom::Stopwatch sw;

  // loop over configurations
  //   init empty perambulator
  //   tmLQCD read gauge field
  //   nyom read eigensystem
  //
  //   loop over time-slices
  //     create_source
  //     add to perambulator
  //   
  //   write perambulator to disk
  
  nyom::Perambulator peram = nyom::make_Perambulator(/* Nev */    48,
                                                     /* Nt_src */ 24,
                                                     /* Nt_snk */ 24,
                                                     core);

  nyom::SpinDilutedTimeslicePropagatorVector src(core);
  nyom::SpinDilutedTimeslicePropagatorVector prop(core);

  nyom::LapH_eigsys V = nyom::make_LapH_eigsys(/* Nev */ 48,
                                               /* Nt  */ 24,
                                               /* Nx  */ 12,
                                               /* Ny  */ 12,
                                               /* Nz  */ 12,
                                               /* Nc  */ 3,
                                               core.geom.get_world());

  read_LapH_eigsys_from_files(V,
                              "ev",
                              400,
                              core);

  sw.reset();
  nyom::LapH_eigsys Vdagger(V);
  ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Vdagger["cizyxt"]);
  sw.elapsed_print("Vdagger");

  spinor* source = new spinor[VOLUME];
  spinor* propagator = new spinor[VOLUME];

  for(int64_t tsrc = 0; tsrc < 24; ++tsrc){
    nyom::One one_tsrc = nyom::make_One(24, tsrc, core.geom.get_world() );
    for(int64_t dsrc = 0; dsrc < 4; ++dsrc){
      nyom::One one_dsrc = nyom::make_One(4, dsrc, core.geom.get_world() );
      for(int64_t esrc = 0; esrc < 48; ++esrc){
        nyom::One one_esrc = nyom::make_One(48, esrc, core.geom.get_world() );

        int shapes[3] = {NS, NS, NS};
        int sizes[3] = {48, 4, 24};
        CTF::Tensor<complex<double>> prop_proj(3, sizes, shapes, core.geom.get_world());
        CTF::Tensor<complex<double>> src_proj(3, sizes, shapes, core.geom.get_world());
        src_proj["jqs"] =  one_esrc["j"] * one_dsrc["q"] * one_tsrc["s"];
         
        sw.reset();
        src.tensor["txyzdc"] = V["czyxEt"] * src_proj["Edt"];
        src.push(source, core);
        sw.elapsed_print("LapH eigenvector to source projection");

        sw.reset();
        tmLQCD_invert(reinterpret_cast<double*>(&propagator[0]),
                      reinterpret_cast<double*>(&source[0]),
                      0, 0);
        sw.elapsed_print("Inversion");   

        sw.reset();
        prop.fill(propagator, dsrc, tsrc, core);
        sw.elapsed_print("Perambulator prop to tensor");
        
        sw.reset();
        // project the propagator with the sink eigenvectors
        prop_proj["ipt"] = prop.tensor["tXYZpC"] * Vdagger["CZYXit"];
        // outer product with source projector 
        peram["ijpqts"] += prop_proj["ipt"] * src_proj["jqs"];
        sw.elapsed_print("Propagator to perambulator projection");
      }
    }
  }

  delete[] source;
  delete[] propagator;

  return 0;
}
