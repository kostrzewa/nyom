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

#pragma once

#include "Core.hpp"
#include "Stopwatch.hpp"
#include "LapH_eigsys.hpp"
#include "SpinDilutedTimeslicePropagatorVector.hpp"
#include "One.hpp"
#include "AutoTunable.hpp"

#include <sstream>
#include <string>

namespace nyom {

class Vdagger_project_SpinDilutedTimeslicePropagatorVector : public nyom::AutoTunable {
public:
  Vdagger_project_SpinDilutedTimeslicePropagatorVector(
      const std::string name_in,
      const std::string identifier_in,
      nyom::SpinDilutedTimeslicePropagatorVector & prop_in,
      nyom::LapH_eigsys & Vdagger_in,
      CTF::Tensor<complex<double>> & proj_prop_in,
      const nyom::Core & core_in) :

    AutoTunable(name_in, identifier_in, core_in),
      name(name_in),
      identifier(identifier_in),
      core(core_in),
      Vdagger(Vdagger_in),
      prop(prop_in),
      proj_prop(proj_prop_in)
  {
    variants.emplace("P_Vdagger", std::mem_fun(&Vdagger_project_SpinDilutedTimeslicePropagatorVector::P_Vdagger));
    variants.emplace("Vdagger_P", std::mem_fun(&Vdagger_project_SpinDilutedTimeslicePropagatorVector::Vdagger_P));
    MPI_Barrier( core.geom.get_nyom_comm() );
  }

  Vdagger_project_SpinDilutedTimeslicePropagatorVector() = delete;

  void P_Vdagger()
  {
    proj_prop["ipt"] = prop.tensor["CpZYXt"] * Vdagger["CZYXit"];
  }

  void Vdagger_P()
  {
    proj_prop["ipt"] = Vdagger["CZYXit"] * prop.tensor["CpZYXt"] ;
  }

  void operator()(void)
  {
    // the second condition for tuning takes care of the situation
    // in which a new variant was added but tuning results already exist
    // so the new variant would be completely ignored
    if( this->get_optimal_variant().size() == 0 || variants.size() != this->get_foms_size() ){
      tune();
      if( core.geom.get_myrank() == 0 ){
        std::cout << this->get_optimal_variant() << " has won: " << 
          this->get_fom( this->get_optimal_variant() ) << std::endl;
      }
    } else {
      // for some reason, it is not possible to call variants["name"](this) directly...
      // doing so through the iterator works, however
      std::unordered_map< std::string, std::mem_fun_t<void,nyom::Vdagger_project_SpinDilutedTimeslicePropagatorVector> >::iterator 
        it = variants.find( this->get_optimal_variant() );
      it->second(this);
    }
  }
  
  void tune(void) 
  {
    if( core.geom.get_myrank() == 0 ){
		  std::cout << "Tuning " << name << std::endl;
      fflush(stdout);
    }
    std::unordered_map< std::string, std::mem_fun_t<void,nyom::Vdagger_project_SpinDilutedTimeslicePropagatorVector> >::iterator it = variants.begin();
    nyom::duration elapsed;
    while( it != variants.end() ){
      sw.reset();
      if( core.geom.get_myrank() == 0 ){
        std::cout << "Calling variant " << it->first << std::endl;
        fflush(stdout);
      }
      it->second(this);
      elapsed = sw.elapsed();
      add_measurement( /* variant */ it->first,
                       /* fom */ elapsed.mean );
      ++it;
    }
  }

private:
  nyom::Stopwatch sw;

  std::unordered_map<std::string, std::mem_fun_t<void,nyom::Vdagger_project_SpinDilutedTimeslicePropagatorVector> > variants;

  const std::string name;
  const std::string identifier;

  const nyom::Core & core;
  nyom::LapH_eigsys & Vdagger;
  nyom::SpinDilutedTimeslicePropagatorVector & prop;
  CTF::Tensor<complex<double>> & proj_prop;
  
};

Vdagger_project_SpinDilutedTimeslicePropagatorVector
make_Vdagger_project_SpinDilutedTimeslicePropagatorVector(
  nyom::SpinDilutedTimeslicePropagatorVector & prop,
  nyom::LapH_eigsys & Vdagger,
  CTF::Tensor<complex<double>> & proj_prop,
  const nyom::Core & core)
{
  int     Nt = core.input_node["Nt"].as<int>();
  int     Nx = core.input_node["Nx"].as<int>();  
  int     Ny = core.input_node["Ny"].as<int>(); 
  int     Nz = core.input_node["Nz"].as<int>(); 
  int    Nev = core.input_node["Nev"].as<int>(); 
  int Nranks = core.geom.get_Nranks();

  std::string name("Vdagger_project_SpinDilutedTimeslicePropagatorVector");
  std::stringstream identifier;

  identifier << "Nt" << Nt << "_"
             << "Nx" << Nx << "_"
             << "Ny" << Ny << "_"
             << "Nz" << Nz << "_"
             << "Nev" << Nev << "_"
             << "Nranks" << Nranks;

  return( Vdagger_project_SpinDilutedTimeslicePropagatorVector(name,
                                                               identifier.str(),
                                                               prop,
                                                               Vdagger,
                                                               proj_prop,
                                                               core) );
}


class V_project_SpinDilutedTimesliceSourceVector : public nyom::AutoTunable {                                                                     
public:
  V_project_SpinDilutedTimesliceSourceVector(
      const std::string name_in,
      const std::string identifier_in,
      nyom::SpinDilutedTimesliceSourceVector & src_in,
      nyom::LapH_eigsys & V_in,
      const int tsrc_in,
      const int esrc_in, 
      const nyom::Core & core_in) :

    AutoTunable(name_in, identifier_in, core_in),
      name(name_in),
      identifier(identifier_in),
      core(core_in),
      V(V_in),
      src(src_in),
      tsrc(tsrc_in),
      esrc(esrc_in),
      Nt( core_in.input_node["Nt"].as<int>() ),
      Nev( core_in.input_node["Nev"].as<int>() )

  {
    variants.emplace("proj_sparse_V", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::proj_sparse_V));
    variants.emplace("V_proj_sparse", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::V_proj_sparse));
    variants.emplace("proj_V", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::proj_V));
    variants.emplace("V_proj", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::V_proj));
    variants.emplace("oneE_V_oneT", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::oneE_V_oneT));
    variants.emplace("oneE_oneT_V", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::oneE_oneT_V));
    variants.emplace("V_oneE_oneT", std::mem_fun(&V_project_SpinDilutedTimesliceSourceVector::V_oneE_oneT));

    one_tsrc = nyom::make_One(Nt, tsrc, core.geom.get_world() );
    one_esrc = nyom::make_One(Nev, esrc, core.geom.get_world() );
    int src_shapes[2] = {NS, NS};
    int src_sizes[2] = {Nev, Nt};
    src_proj = CTF::Tensor<complex<double>>(2, src_sizes, src_shapes, core.geom.get_world(), "src_proj");
    src_proj_sparse = CTF::Tensor<complex<double>>(2, src_sizes, src_shapes, core.geom.get_world(), "src_proj");
    src_proj["et"] = one_esrc["e"] * one_tsrc["t"];
    src_proj_sparse["et"] = src_proj["et"];
    src_proj_sparse.sparsify(); // Note: this is probably broken because sparse*dense leads to (silent) errors...

    MPI_Barrier( core.geom.get_nyom_comm() );
  }

  V_project_SpinDilutedTimesliceSourceVector() = delete;

  void proj_V()
  {
    src.tensor["czyx"] = src_proj["ET"] * V["czyxET"];
  }

  void V_proj()
  {
    src.tensor["czyx"] = V["czyxET"] * src_proj["ET"];
  }
  
  void proj_sparse_V()
  {
    src.tensor["czyx"] = src_proj_sparse["ET"] * V["czyxET"];
  }

  void V_proj_sparse()
  {
    src.tensor["czyx"] = V["czyxET"] * src_proj_sparse["ET"];
  }

  void oneE_oneT_V() 
  {
    src.tensor["czyx"] = one_esrc["E"] * one_tsrc["T"] * V["czyxET"];
  }
  
  void oneE_V_oneT() 
  {
    src.tensor["czyx"] = one_esrc["E"] * V["czyxET"] * one_tsrc["T"];
  }
  
  void V_oneE_oneT() 
  {
    src.tensor["czyx"] = V["czyxET"] * one_esrc["E"] * one_tsrc["T"];
  }

  void operator()(void)
  {
    // the second condition for tuning takes care of the situation
    // in which a new variant was added but tuning results already exist
    // so the new variant would be completely ignored
    if( this->get_optimal_variant().size() == 0 || variants.size() != this->get_foms_size() ){
      tune();
      if( core.geom.get_myrank() == 0 ){
        std::cout << this->get_optimal_variant() << " has won: " << 
          this->get_fom( this->get_optimal_variant() ) << std::endl;
      }
    } else {
      // for some reason, it is not possible to call variants["name"](this) directly...
      // doing so through the iterator works, however
      std::unordered_map< std::string, std::mem_fun_t<void,nyom::V_project_SpinDilutedTimesliceSourceVector> >::iterator 
        it = variants.find( this->get_optimal_variant() );
      it->second(this);
    }
  }
  
  void tune(void) 
  {
    if( core.geom.get_myrank() == 0 ){
		  std::cout << "Tuning " << name << std::endl;
      fflush(stdout);
    }
    std::unordered_map< std::string, std::mem_fun_t<void,nyom::V_project_SpinDilutedTimesliceSourceVector> >::iterator it = variants.begin();
    nyom::duration elapsed;
    while( it != variants.end() ){
      sw.reset();
      if( core.geom.get_myrank() == 0 ){
        std::cout << "Calling variant " << it->first << std::endl;
        fflush(stdout);
      }
      it->second(this);
      elapsed = sw.elapsed();
      add_measurement( /* variant */ it->first,
                       /* fom */ elapsed.mean );
      ++it;
    }
  }

private:
  nyom::Stopwatch sw;

  std::unordered_map<std::string, std::mem_fun_t<void,nyom::V_project_SpinDilutedTimesliceSourceVector> > variants;

  const std::string name;
  const std::string identifier;

  const nyom::Core & core;
  nyom::LapH_eigsys & V;
  nyom::SpinDilutedTimesliceSourceVector & src;
  CTF::Tensor<complex<double>> src_proj;
  CTF::Tensor<complex<double>> src_proj_sparse;
  
  nyom::One one_tsrc;
  nyom::One one_esrc;

  const int tsrc;
  const int esrc;
  const int Nt;
  const int Nev;
};

V_project_SpinDilutedTimesliceSourceVector
make_V_project_SpinDilutedTimesliceSourceVector(
  nyom::SpinDilutedTimesliceSourceVector & src,
  nyom::LapH_eigsys & V,
  const int tsrc_in,
  const int esrc_in,
  const nyom::Core & core)
{
  int     Nt = core.input_node["Nt"].as<int>();
  int     Nx = core.input_node["Nx"].as<int>();  
  int     Ny = core.input_node["Ny"].as<int>(); 
  int     Nz = core.input_node["Nz"].as<int>(); 
  int    Nev = core.input_node["Nev"].as<int>(); 
  int Nranks = core.geom.get_Nranks();

  std::string name("V_project_SpinDilutedTimesliceSourceVector");
  std::stringstream identifier;

  identifier << "Nt" << Nt << "_"
             << "Nx" << Nx << "_"
             << "Ny" << Ny << "_"
             << "Nz" << Nz << "_"
             << "Nev" << Nev << "_"
             << "Nranks" << Nranks;

  return( V_project_SpinDilutedTimesliceSourceVector(name,
                                                     identifier.str(),
                                                     src,
                                                     V,
                                                     tsrc_in,
                                                     esrc_in,
                                                     core) );
}

}
