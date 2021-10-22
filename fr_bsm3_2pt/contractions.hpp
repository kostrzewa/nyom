#pragma once

#include "gammas.hpp"
#include "Core.hpp"
#include "PointSourcePropagator.hpp"
#include "Logger.hpp"
#include "Stopwatch.hpp"

#include <complex>

template <int Nf>
inline void 
local_2pt(nyom::PointSourcePropagator<Nf> & S, nyom::PointSourcePropagator<Nf> & Stilde,
          const nyom::Core &core)
{
  nyom::Logger logger(core, 0, 0);

  nyom::PointSourcePropagator<Nf> S1(core);
  nyom::PointSourcePropagator<Nf> S2(core);

  CTF::Vector< std::complex<double> > C(core.input_node["Nt"].as<int>(), core.geom.get_world());

  const std::vector<std::string> gammas = {"5", "0", "1", "2", "3", "05"};

  const int n_contr = nyom::i_tau.size() * nyom::i_tau.size() *
                      gammas.size() * gammas.size();

  int contr = 0;

  nyom::Stopwatch sw( core.geom.get_nyom_comm() );

  for( auto t_src : nyom::i_tau ){
    S1["txyzijabfg"] = nyom::tau[t_src]["fH"] * Stilde["txyzijabgH"];
    for( auto t_snk : nyom::i_tau ){
      S2["txyzijabfg"] = nyom::tau[t_snk]["fH"] * S["txyziKabHg"];
      for( auto g_src : gammas ){
        S2["txyzijabfg"] = S2["txyziKabfg"] * nyom::g[g_src]["Kj"];
        for( auto g_snk : gammas ){
          contr++;
          logger << "Contraction " << contr << " of " << n_contr << std::endl; 
          S1["txyzijabfg"] = S1["txyzKjabfg"] * nyom::g[g_snk]["Ki"];
          
          C["t"] = S1["tXYZJIBAGF"] * S2["tXYZIJABFG"];
          sw.elapsed_print_and_reset("Done.");
        }
      }
    }
  }
}

