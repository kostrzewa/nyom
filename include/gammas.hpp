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

#pragma once

#include <map>
#include <numeric>
#include <vector>
#include <complex>
#include <ctf.hpp>

namespace nyom {

extern const int gamma_sizes[2];
extern const int gamma_shapes[2];

// namespace-global storage for gamma matrices
extern std::map < std::string, CTF::Tensor< std::complex<double> > > g;

// namespace-global storage for possible sign change under
// \gamma -> \gamma_0 \gamma^\dagger \gamma_0 = g0_sign[\gamma] \gamma
extern std::map < std::string, double > g0_sign;

extern std::vector < std::string > i_g;

void init_gammas(CTF::World& dw);

} // namespace nyom
