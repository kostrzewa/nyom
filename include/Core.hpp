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

// for some reason, linking errors occur if boost::program_options is not
// included at the very top
#include <boost/program_options.hpp>

#include "Geometry.hpp"

#include <ctf.hpp>
#include <tmLQCD.h>
#include <yaml-cpp/yaml.h>

#include <stdexcept>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <exception>
#include <string>

namespace nyom {

typedef enum logtype_t {
  log = 0,
  log_perf,
  log_err
} logtype_t;

typedef struct logfile_t {
  std::fstream file; 
} logfile_t;

  /**
   * @brief The Core class is responsible for overall management of the program's
   * execution. It takes care of reading the input file and providing a representation
   * of that to the outside world.
   * It is also resposible for diagnostic output
   */
class Core {
  public:
    Core(int argc, char ** argv) : 
      geom(argc, argv), 
      cmd_desc("cmd_options")
    {
      declare_cmd_options();
      try
      {
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, cmd_desc), 
                                      cmd_options);
        notify(cmd_options);
      }
      catch ( const std::exception &ex )
      {
        std::cerr << ex.what() << "\n";
      }
      if( cmd_options.count("help") ){
        print_usage();
        finalise();
      }
      input_node = YAML::LoadFile( cmd_options["input"].as<std::string>() );
    }
      

    ~Core(){
      // the auto constructs a ref to std::pair<std::string, nyom::logfile_t>
      // would prefer to do this here explictly, but the compiler complains
      // for some esoteric reason
      for( auto & logfile : logfiles ){
        if( logfile.second.file.is_open() ){
          logfile.second.file.close();
        }
      }
    }

    void Logger(std::string objname){
    }
    
    void print_usage(void){
      if( geom.get_myrank() == 0 ){
        std::cout << "useful usage information" << std::endl;
      }
    }

    void finalise(void){
      finalise("");
    }

    void finalise(const std::string msg){
      geom.finalise();
      if( msg.length() > 0 ){
        std::cout << msg << std::endl;
      }
    }

  private:
    void declare_cmd_options(void){
      cmd_desc.add_options()
        ("help,h", "usage information")
        ("input,i",
         boost::program_options::value<std::string>()->default_value( std::string("config.yaml") ), 
         "filename of nyom YAML-formatted input file");
    }

    boost::program_options::options_description cmd_desc;
    boost::program_options::variables_map cmd_options;
    nyom::Geometry geom;
    unordered_map<std::string,logfile_t> logfiles;
    YAML::Node input_node;

    bool initialised;
};

}
