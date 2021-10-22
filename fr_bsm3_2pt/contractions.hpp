#pragma once

#include "gammas.hpp"
#include "Core.hpp"
#include "PointSourcePropagator.hpp"
#include <complex>

template <int Nf>
inline void 
local_2pt(nyom::PointSourcePropagator<Nf> & S, nyom::PointSourcePropagator<Nf> & Stilde,
          const nyom::Core &core)
{
  nyom::PointSourcePropagator<Nf> S1(core);
  nyom::PointSourcePropagator<Nf> S2(core);

  CTF::Vector< std::complex<double> > C(core.input_node["Nt"].as<int>(), core.geom.get_world());

  for( auto t_src : nyom::i_tau ){
    S1["txyzijabfg"] = nyom::tau[t_src]["fH"] * Stilde["txyzijabgH"];
    for( auto t_snk : nyom::i_tau ){
      S2["txyzijabfg"] = nyom::tau[t_snk]["fH"] * S["txyziKabHg"];
      for( std::string g_src : {"5", "0", "1", "2", "3", "05"} ){
        S2["txyzijabfg"] = S2["txyziKabfg"] * nyom::g[g_src]["Kj"];
        for( std::string g_snk : {"5", "0", "1", "2", "3", "05"} ){
          S1["txyzijabfg"] = S1["txyzKjabfg"] * nyom::g[g_snk]["Ki"];

          C["t"] = S1["tXYZJIBAGF"] * S2["tXYZIJABFG"];
        }
      }
    }
  }
}

