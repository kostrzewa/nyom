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

//#include "utility_functions.hpp"

#include "geometry.hpp"

#include <ctf.hpp>
#include <tmLQCD.h>
#include <yaml-cpp/yaml.h>
#include <boost/program_options.hpp>

#include <stdexcept>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <string>
#include <iostream>

namespace nyom {

typedef enum logtype_t {
  log = 0,
  log_perf,
  log_err
} logtype_t;

typedef struct logfile_t {
  
} logfile_t;

  /**
   * @brief The Core class is responsible for overall management of the program's
   * execution. It takes care of reading the input file and providing a representation
   * of that to the outside world.
   * It is also resposible for diagnostic output
   */
class Core {
  public:
    Core(int argc,
         char ** argv) : geom(argc, argv), desc("cmd_options")
    {
      declare_cmd_options();
      try
      {
        store(boost::program_options::parse_command_line(argc, argv, desc), cmd_options);
      }
      catch ( const error &ex )
      {
        std::cerr << ex.what() << "\n";
      }

      config = YAML:LoadFile( cmd_options["input_file"].as<std::string>() );
    }
      

    ~Core(){
      for( auto && logfile : logfiles ){
        logfile.file.close();
      }
    }

    void Core::Logger(std::string objname,){
    }
                      

  private:
    void declare_cmd_options(void){
      desc.add_options()
        ("help,h,?", "usage information")
        ("input,i",value<std::string>()default_value("config.yaml"), "input file");
    }

    boost::program_options::options_description;
    boost::program_options::variables_map cmd_options;
    nyom::Geometry geom;
    unordered_map<std::string,logfile_t> logfiles;
    YAML::Node input_file;
};

}
