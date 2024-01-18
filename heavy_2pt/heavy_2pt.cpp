/***********************************************************************                                                                                                            
 * Copyright (C) 2016 Bartosz Kostrzewa
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
#include "h5utils.hpp"

#include "gammas.hpp"

#include "NfSpinDilutedTimeslicePropagator.hpp"
#include "SpinDilutedTimeslicePropagator.hpp"
#include "PointSourcePropagator.hpp"
#include "SpinColourPropagator.hpp"
#include "MomentumTensor.hpp"
#include "HeavyLightCorrelator.hpp"
#include "HeavyHeavyCorrelator.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <iomanip>
#include <memory>

#include <cstring>

#include <omp.h>

#define printf0 if(core.geom.get_myrank()==0) printf

double measureFlopsPerSecond(double local_time, 
                             CTF::Flop_counter& flp,
                             const char* const name,
                             const nyom::Core & core)
{
  double global_time;
  MPI_Allreduce( &local_time, &global_time, 1, MPI_DOUBLE, MPI_SUM, core.geom.get_nyom_comm() );
  int64_t flops = flp.count( core.geom.get_nyom_comm() );
  double fps = (double)(flops)/( (global_time/core.geom.get_Nranks()) * 1.e6 );
  printf0("Performance in '%s': %.6e mflop/s\n", name, fps);
  return fps;
}

typedef enum nd_src_flav_idx_t {
  UP = 0,
  DOWN,
  ND_UP,
  ND_DOWN
} nd_src_flav_idx_t;

int main(int argc, char ** argv) {
  nyom::Core core(argc,argv);
  
  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  const int rank = core.geom.get_myrank();
  const int np = core.geom.get_Nranks();

  const int seed = core.input_node["seed"].as<int>();
  const int conf_start = core.input_node["conf_start"].as<int>();
  const int conf_stride = core.input_node["conf_stride"].as<int>();
  const int conf_end = core.input_node["conf_end"].as<int>();
  const int n_src = core.input_node["n_src_per_gauge"].as<int>();

  const int Nd = 4;
  const int Nc = 3;

  const int Nt = core.input_node["Nt"].as<int>();
  const int Nx = core.input_node["Nx"].as<int>();    
  const int Ny = core.input_node["Ny"].as<int>();    
  const int Nz = core.input_node["Nz"].as<int>();
  const int Ns = Nx;
   
  const int pt = core.geom.tmlqcd_mpi.proc_coords[0];     
  const int px = core.geom.tmlqcd_mpi.proc_coords[1];     
  const int py = core.geom.tmlqcd_mpi.proc_coords[2];     
  const int pz = core.geom.tmlqcd_mpi.proc_coords[3];     
   
  const int lt = core.geom.tmlqcd_lat.T;     
  const int lx = core.geom.tmlqcd_lat.LX;    
  const int ly = core.geom.tmlqcd_lat.LY;    
  const int lz = core.geom.tmlqcd_lat.LZ;     
   
  const int local_volume = lt * lx * ly * lz;    

  const size_t spinor_elem = 4*3*2*local_volume;
  const size_t spinor_bytes = 8*spinor_elem;

  std::chrono::time_point<std::chrono::steady_clock> start;
  std::chrono::time_point<std::chrono::steady_clock> moment;
  std::chrono::duration<float> elapsed_seconds;

  nyom::init_gammas( core.geom.get_world() );

  std::unique_ptr<double> s0(new double[spinor_elem]);
  std::unique_ptr<double> s1(new double[spinor_elem]);
  std::unique_ptr<double> p0(new double[spinor_elem]);
  std::unique_ptr<double> p1(new double[spinor_elem]);
  std::unique_ptr<double> tmp(new double[spinor_elem]);

  std::random_device r;
  std::mt19937 mt_gen(seed);

  // uniform distribution in space coordinates
  std::uniform_int_distribution<int> ran_space_idx(0, Ns-1);
  // uniform distribution in time coordinates
  std::uniform_int_distribution<int> ran_time_idx(0, Nt-1);

  nyom::SpinDilutedTimeslicePropagator Sup(core);
  nyom::SpinDilutedTimeslicePropagator Sdown(core);
  
  nyom::SpinDilutedTimeslicePropagator Sup_conj(core);
  nyom::SpinDilutedTimeslicePropagator Sdown_conj(core);
  
  nyom::SpinDilutedTimeslicePropagator Sup_fwd(core);
  nyom::SpinDilutedTimeslicePropagator Sup_bwd(core);
  nyom::SpinDilutedTimeslicePropagator Sdown_fwd(core);
  nyom::SpinDilutedTimeslicePropagator Sdown_bwd(core);
  
  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd(core);
  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd_conj(core);

  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd_fwd(core);
  nyom::NfSpinDilutedTimeslicePropagator<2> S_nd_bwd(core);

  sw.reset();

  for(int cid = conf_start; cid <= conf_end; cid += conf_stride){
    // read propagators from file(s)? if yes, don't need to load gauge field
    bool write = false;
    bool read = false;
    if(read) {
      write = false; 
    } else {
      tmLQCD_read_gauge(cid); 
    }
    
    char filename[100];
    snprintf(filename, 100, "hl_hh_%04d.h5", cid);

    for(int src_id = 0; src_id < n_src; src_id++){
      int src_ts;
      
      // for certainty, we broadcast the source coordinates from rank 0 
      if(rank == 0){
        src_ts = ran_time_idx(mt_gen);
      }
      MPI_Bcast(&src_ts, 1, MPI_INT, 0, core.geom.get_nyom_comm());

      Sup.set_src_ts(src_ts);
      Sdown.set_src_ts(src_ts);
      Sup_conj.set_src_ts(src_ts);
      Sdown_conj.set_src_ts(src_ts);
      S_nd.set_src_ts(src_ts);
      S_nd_conj.set_src_ts(src_ts);

      Sup_fwd.set_src_ts(src_ts);
      Sup_bwd.set_src_ts(src_ts);
      Sdown_fwd.set_src_ts(src_ts);
      Sdown_bwd.set_src_ts(src_ts);
      S_nd_fwd.set_src_ts(src_ts);
      S_nd_bwd.set_src_ts(src_ts);

      for(int src_flav_idx : {UP,DOWN,ND_UP,ND_DOWN} ){
        for(int src_d = 0; src_d < Nd; ++src_d){
          if(!read){
            tmLQCD_full_source_spinor_field_spin_diluted_oet_ts(s0.get(), src_ts, src_d, src_id, cid, seed);
            if( src_flav_idx == UP || src_flav_idx == DOWN ){
              if ( tmLQCD_invert(p0.get(), s0.get(), static_cast<int>(src_flav_idx), 0) != 0 ){
                throw( std::runtime_error("light inversion failed") );
              }
              if( src_flav_idx == UP ){
                Sup.fill(p0.get(), src_d);
              } else {
                Sdown.fill(p0.get(), src_d);
              }
            } else {
              memset((void*)s1.get(), 0, spinor_bytes);
              if( src_flav_idx == ND_DOWN ) s0.swap(s1);
              
              if( tmLQCD_invert_doublet(p0.get(), p1.get(), s0.get(), s1.get(), 2, 0) != 0 ){
                throw( std::runtime_error("heavy inversion failed") );
              }

              S_nd.fill(p0.get(), 0, static_cast<int>(src_flav_idx), src_d);
              S_nd.fill(p1.get(), 1, static_cast<int>(src_flav_idx), src_d);
            }
          }
        }
      }

      // transpose colour indices and take complex conjugate for gamma5  hermiticity
      // the transpose in spin for the gamma_5 S^dag gamma_5 identity will be taken
      // below
      sw.reset();
      Sup_conj["txyzija"] = Sup["txyzija"];
      Sdown_conj["txyzija"] = Sdown["txyzija"];
      S_nd_conj["txyzijagf"] = S_nd["txyzijafg"];
      ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sup_conj["txyzija"]);
      ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Sdown_conj["txyzija"]);
      ((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(S_nd_conj["txyzijafg"]);
      sw.elapsed_print_and_reset("Complex conjugation");

      // perform contractions
      CTF::Flop_counter flp;
      CTF::Vector< std::complex<double> > C(Nt,core.geom.get_world());
      nyom::HeavyLightCorrelator<2,1> C_lhh(core);
      nyom::HeavyHeavyCorrelator<2,2,1> C_ffhh(core);
       
      const double norm = 1.0/((double)Ns*Ns*Ns);
      
      flp.zero();

      std::complex<double>* correl_values;
      int64_t correl_nval;

      std::list<std::string> path_list;

      // PP PA AP AA
      for( std::string g_src : {"5", "I"} ){
        for( std::string g_snk : {"5", "I"} ){

          Sup_fwd["txyzija"] = Sup["txyziLa"] * (nyom::g[g_src])["LK"] * (nyom::g["5"])["Kj"];
          Sdown_fwd["txyzija"] = Sdown["txyziLa"] * (nyom::g[g_src])["LK"] * (nyom::g[g_snk])["Kj"];
          S_nd_fwd["txyzijafg"] = S_nd["txyziLafg"] * (nyom::g["5"])["LK"] * (nyom::g[g_snk])["Kj"];
          sw.elapsed_print_and_reset("source gamma insertion");

          Sup_bwd["txyzija"] = Sup_conj["txyzLia"] * (nyom::g["5"])["LK"]  * (nyom::g[g_snk])["Kj"];
          Sdown_bwd["txyzija"] = Sdown_conj["txyziLa"] * (nyom::g["5"])["LK"] * (nyom::g[g_snk])["Kj"];
          S_nd_bwd["txyzijafg"] = S_nd_conj["txyziLafg"] * (nyom::g["5"])["LK"] * (nyom::g[g_snk])["Kj"];
          sw.elapsed_print_and_reset("sink gamma insertion");


          C["t"] = - norm * nyom::g0_sign[g_src] * Sup_bwd["tXYZIJA"] * Sup_fwd["tXYZJIA"];
          path_list = nyom::h5::make_os_meson_2pt_path_list(g_src, g_snk, "u", "u", src_ts);
          nyom::h5::write_dataset(core, filename, path_list, C);
          
          C["t"] = - norm * nyom::g0_sign[g_src] * Sdown_bwd["tXYZIJA"] * Sdown_fwd["tXYZJIA"];
          path_list = nyom::h5::make_os_meson_2pt_path_list(g_src, g_snk, "d", "d", src_ts);
          nyom::h5::write_dataset(core, filename, path_list, C);

          C["t"] = -norm * nyom::g0_sign[g_src] * Sup_bwd["tXYZIJA"] * Sdown_fwd["tXYZJIA"];
          path_list = nyom::h5::make_os_meson_2pt_path_list(g_src, g_snk, "d", "u", src_ts);
          nyom::h5::write_dataset(core, filename, path_list, C);

          C["t"] = -norm * nyom::g0_sign[g_src] * Sdown_bwd["tXYZIJA"] * Sup_fwd["tXYZJIA"];
          path_list = nyom::h5::make_os_meson_2pt_path_list(g_src, g_snk, "u", "d", src_ts);
          nyom::h5::write_dataset(core, filename, path_list, C);

          C_lhh["fgsrt"] = - norm * nyom::g0_sign[g_src] * Sdown_bwd["tXYZIJA"] * S_nd_fwd["tXYZJIAfg"];
          path_list = nyom::h5::make_os_meson_2pt_path_list(g_src, g_snk, "h", "d", src_ts);
          nyom::h5::write_dataset(core, filename, path_list, C_lhh.tensor);
          
          C_ffhh["fghisrt"] = - norm * nyom::g0_sign[g_src] * S_nd_bwd["tXYZIJAfg"] * S_nd_fwd["tXYZJIAhi"];
          path_list = nyom::h5::make_os_meson_2pt_path_list(g_src, g_snk, "h", "h", src_ts);
          nyom::h5::write_dataset(core, filename, path_list, C_ffhh.tensor);
        }
      }
    } // src loop
  } // cid loop

  MPI_Barrier( core.geom.get_nyom_comm() );

  return 0;
}