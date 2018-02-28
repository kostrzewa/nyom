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

#include <string>

namespace nyom {

typedef enum logtype_t {
  log = 0,
  log_perf,
  log_err
} logtype_t;

typedef enum log_location_t {
  log_master = 0,
  log_all
} log_location_t;

std::string logtype_to_string(const logtype_t type){
  switch(type){
    case log:
      return(std::string("log"));
      break;
    case log_perf:
      return(std::string("perf"));
      break;
    case log_err:
      return(std::string("err"));
      break;
    default:
      throw( std::invalid_argument("logtype_to_string: log type passed is of unknown type, this should not have happened!\n") );
      break;
  }
}

}
