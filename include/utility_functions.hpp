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

#include <stdexcept>

#pragma once

namespace nyom {

  static inline double gcd( const double a, const double b ){
    if( a == 0 || b == 0 ){
      throw std::runtime_error("gcd: neither argument can be zero!\n");
    }

    if( a == b ){
      return a;
    } else if( a > b ){
      return gcd(a-b,b);
    } else {
      return gcd(a,b-a);
    }
  }

}
