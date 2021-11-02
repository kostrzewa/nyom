#pragma once

#include "gammas.hpp"
#include "Core.hpp"
#include "PointSourcePropagator.hpp"
#include "Logger.hpp"
#include "Stopwatch.hpp"
#include "ComplexMatrixField.hpp"

#include <complex>
#include <sstream>
#include <iomanip>
#include <utility>
#include <iostream>
#include <map>

template <typename T>
size_t
count_unique_pairs(const std::vector<T> & v1, const std::vector<T> &v2, const nyom::Core & core)
{
  std::vector< std::pair<T,T> > pairs;
  for(auto e1 : v1){
    for(auto e2 : v2){
      bool not_found = true;
      for(auto ep : pairs){
        if( (ep.first == e1 && ep.second == e2) || (ep.first == e2 && ep.second == e1) ){
          not_found = false;
          break;
        }
      }
      if(not_found) pairs.push_back( std::make_pair(e1,e2) );
    }
  }
  return pairs.size();
}

template <int Nf>
inline void 
local_2pt(nyom::PointSourcePropagator<Nf> & S, nyom::PointSourcePropagator<Nf> & Sbar,
          const nyom::Core &core, const int gauge_conf_id, const int scalar_idx)
{
  nyom::Logger logger(core, 0, 0);

  // temporaries
  nyom::PointSourcePropagator<Nf> S1i(core), S1ig(core);
  nyom::PointSourcePropagator<Nf> S2j(core), S2jg(core);

  CTF::Vector< std::complex<double> > C(core.input_node["Nt"].as<int>(), core.geom.get_world());

  // we use the same set of gamma and isospin structures at source and sink
  // if we use different ones, the loops below need to be adjusted
  const std::vector<std::string> g_src = {"5", "I", "0", "1", "2", "3", "05", "01", "02", "03"};
  const std::vector<std::string> g_snk = g_src;

  const std::vector<std::string> t_src = {"1", "2", "3"};
  const std::vector<std::string> t_snk = t_src;

  const size_t n_contr = count_unique_pairs(g_src, g_snk, core) * count_unique_pairs(t_src, t_snk, core);

  int contr = 0;

  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  char fname[100];
  snprintf(fname, 100,
           "local2pt_conf%04d_scalar%04d",
           gauge_conf_id, scalar_idx);

  std::ofstream correls;

  if(core.geom.get_myrank() == 0) correls.open(fname);

  sw.reset();

  for( size_t it_snk = 0; it_snk < t_snk.size(); it_snk++ ){
    // Sbar has already been transposed, all multiplications
    // can proceed in the normal way and we take a regular scalar product
    // in the end
    S1i["txyzijabfg"] = Sbar["txyzijabfH"] * nyom::tau[ t_snk[it_snk] ]["Hg"];

    for( size_t it_src = it_snk; it_src < t_src.size(); it_src++ ){
      S2j["txyzijabfg"] = S["txyzijabfH"] * nyom::tau[ t_src[it_src] ]["Hg"];

      for( size_t ig_snk = 0; ig_snk < g_snk.size(); ig_snk++ ){
        // Sbar has already been tranposed
        S1ig["txyzijabfg"] = S1i["txyziKabfg"] * nyom::g[ g_snk[ig_snk] ]["Kj"];

        for( size_t ig_src = ig_snk; ig_src < g_src.size(); ig_src++ ){
          std::stringstream cname;
          cname << "g_snk" << g_snk[ig_snk] << "-g_src" << g_src[ig_src] <<
            "-t_snk" << t_snk[it_snk] << "-t_src" << t_src[it_src];

          contr++;
          logger << "Contraction " << contr << " / " << n_contr << ": " <<
            cname.str() << std::endl;

          S2jg["txyzijabfg"] = S2j["txyziKabfg"] * nyom::g[ g_src[ig_src] ]["Kj"];

          C["t"] = - 0.25 * nyom::g0_sign[ g_src[ig_src] ] * S1ig["tXYZIJABFG"] * S2jg["tXYZJIBAGF"];
          std::complex<double> * values;
          int64_t nval;
          C.read_all(&nval, &values);

          if(core.geom.get_myrank() == 0){
            for(int64_t i = 0; i < nval; i++){
              correls << cname.str() << '\t' << i << '\t' <<
                std::scientific << std::setprecision(10) << values[i].real() << '\t' << 
                std::scientific << std::setprecision(10) << values[i].imag() << 
                std::endl;
            }
          }

          free(values);
          sw.elapsed_print_and_reset("Done.");
        }
      }
    }
  }
  if(core.geom.get_myrank()) correls.close();
}

template <int Nf>
inline void 
PDP(nyom::PointSourcePropagator<Nf> & S, nyom::PointSourcePropagator<Nf> & Sbar,
    std::map< std::string, nyom::ComplexMatrixField<2,2> > & theta,
    std::map< std::string, nyom::ComplexMatrixField<2,2> > & theta_tilde,
    const nyom::Core &core, const int gauge_conf_id, const int scalar_idx)
{
  nyom::Logger logger(core, 0, 0);

  // temporaries
  nyom::PointSourcePropagator<Nf> S1ir(core), S1il(core);
  nyom::PointSourcePropagator<Nf> S2j(core), S2j_tilde(core);

  CTF::Vector< std::complex<double> > C(core.input_node["Nt"].as<int>(), core.geom.get_world());

  const std::vector<std::string> t_src = {"1", "2", "3"};
  const std::vector<std::string> t_snk = t_src;

  const size_t n_contr = count_unique_pairs(t_src, t_snk, core);

  int contr = 0;

  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  char fname[100];
  snprintf(fname, 100,
           "PDP_conf%04d_scalar%04d",
           gauge_conf_id, scalar_idx);

  std::ofstream correls;

  if(core.geom.get_myrank() == 0) correls.open(fname);

  sw.reset();

  for( size_t it_src = 0; it_src < t_src.size(); it_src++ ){
    // Sbar has already been transposed, all multiplications
    // can proceed in the normal way and we take a regular scalar product
    // in the end
    S1ir["txyzijabfg"] = nyom::g["Ip5"]["iK"] * S["txyzKjabfH"] * nyom::tau[ t_src[it_src] ]["Hg"];
    S1il["txyzijabfg"] = nyom::g["Im5"]["iK"] * S["txyzKjabfH"] * nyom::tau[ t_src[it_src] ]["Hg"];

    for( size_t it_snk = it_src; it_snk < t_snk.size(); it_snk++ ){
      S2j["txyzijabfg"] = S["txyzijabfH"] * theta.at( t_snk[it_snk] )["Hgtxyz"];
      S2j_tilde["txyzijabfg"] = S["txyzijabfH"] * theta_tilde.at( t_snk[it_snk] )["Hgtxyz"];

      std::stringstream cname;
      cname << "t_snk" << t_snk[it_snk] << "-t_src" << t_src[it_src];

      contr++;
      logger << "Contraction " << contr << " / " << n_contr << ": " <<
        cname.str() << std::endl;

      C["t"] = - 0.5 * nyom::g0_sign["5"] * ( S1ir["tXYZIJABFG"] * S2j["tXYZJIBAGF"] -
                                              S1il["tXYZIJABFG"] * S2j_tilde["tXYZJIBAGF"] );
      std::complex<double> * values;
      int64_t nval;
      C.read_all(&nval, &values);

      if(core.geom.get_myrank() == 0){
        for(int64_t i = 0; i < nval; i++){
          correls << cname.str() << '\t' << i << '\t' <<
            std::scientific << std::setprecision(10) << values[i].real() << '\t' << 
            std::scientific << std::setprecision(10) << values[i].imag() << 
            std::endl;
        }
      }
      free(values);
      sw.elapsed_print_and_reset("Done.");
    }
  }
  if(core.geom.get_myrank()) correls.close();
}

