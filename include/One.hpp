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

#include <ctf.hpp>

namespace nyom {

typedef CTF::Tensor< complex<double> > One;
typedef CTF::Tensor< complex<double> > BiOne;
typedef CTF::Tensor< complex<double> > TriOne;

/**
 * @brief Unit basis vector. This can be used to grow tensors one entry at a time
 * using an outer product notation.
 *
 * @param Nidx Length of the basis vector
 * @param idx_nonzero Non-zero entry of the basis vector
 * @param world
 *
 * @return rank one sparse tensor with a single non-zero element
 */
nyom::One make_One(const int Nidx,
                   const int idx_nonzero,
                   CTF::World &world){
  const int shapes[1] = {NS};
  int sizes[1];
  sizes[0]  = Nidx;
  nyom::One ret(1, sizes, shapes, world);
  const complex<double> entry = 1.0;
  const int64_t idx = idx_nonzero;
  ret.write((int64_t)1, &idx, &entry);
  //ret.sparsify();
  return ret;
}

nyom::BiOne make_BiOne(const int Nidx,
                       const int idx_nonzero,
                       CTF::World &world){
  const int shapes[2] = {NS,NS};
  int sizes[2];
  sizes[0]  = Nidx;
  sizes[1]  = Nidx;
  nyom::BiOne ret(2, sizes, shapes, world);
  int64_t gbl_idx = idx_nonzero*Nidx + idx_nonzero;
  complex<double> entry = 1.0;
  ret.write((int64_t)1, &gbl_idx, &entry);
  //ret.sparsify();
  return ret;
}

nyom::TriOne make_TriOne(const int Nidx,
                        const int idx_nonzero,
                        CTF::World &world){
  const int shapes[3] = {NS,NS,NS};
  int sizes[3];
  sizes[0]  = Nidx;
  sizes[1]  = Nidx;
  sizes[2]  = Nidx;
  nyom::BiOne ret(3, sizes, shapes, world);
  int64_t gbl_idx = idx_nonzero*Nidx*Nidx + idx_nonzero*Nidx + idx_nonzero;
  complex<double> entry = 1.0;
  ret.write((int64_t)1, &gbl_idx, &entry);
  //ret.sparsify();
  return ret;
}


} // namespace(nyom)
