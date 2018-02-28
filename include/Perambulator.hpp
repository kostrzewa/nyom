/**********************************************************************
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

namespace nyom {

typedef enum Perambulator_dims_t {
  PERAM_DIM_ESNK = 0,
  PERAM_DIM_ESRC,
  PERAM_DIM_DSNK,
  PERAM_DIM_DSRC,
  PERAM_DIM_TSNK,
  PERAM_DIM_TSRC
} Perambulator_dims_t;

typedef CTF::Tensor< complex<double> > Perambulator;

nyom::Perambulator make_Perambulator(const int Nev,
                                     const int Nt_src,
                                     const int Nt_snk,
                                     const nyom::Core & core){
  int shapes[6] = {NS, NS, NS, NS, NS, NS};
  int sizes[6];
  sizes[PERAM_DIM_ESNK] = Nev;
  sizes[PERAM_DIM_ESRC] = Nev;
  sizes[PERAM_DIM_DSNK] = 4;
  sizes[PERAM_DIM_DSRC] = 4;
  sizes[PERAM_DIM_TSNK] = Nt_snk;
  sizes[PERAM_DIM_TSRC] = Nt_src;
  return( CTF::Tensor< complex<double> >(6, sizes, shapes, core.geom.get_world() ) );
}

} // namespace

