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
#include "Logfile.hpp"

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
#include <utility>

namespace nyom {
      
  namespace po=boost::program_options;
  namespace fs=boost::filesystem;

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
        po::store(po::parse_command_line(argc, argv, cmd_desc), 
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
      // would prefer to do this here explicitly, but the compiler complains
      // for some esoteric reason
      for( auto & logfile : logfiles ){
        if( logfile.second.file.is_open() ){
          logfile.second.file.close();
        }
      }
    }

    /**
     * @brief The Logger function manages the logging of different
     * types of logs coming from different parts of the program
     * into a multitude of files, either just on the master rank
     * or on all ranks, as required for each particular logging
     * operations.
     *
     * @param objname Name of the 'thing' for which a log file is
     * being created or updated.
     * @param type Type of log file to be created / updated. This
     * allows separation of status loggin (log), error loggin (log_err)
     * and performance logging (log_perf).
     * @param msg The log message.
     * @param log_location Whether the log is only done on the master
     * rank or if all ranks perform the logging.
     */
    void Logger(const std::string objname,
                const logtype_t type,
                const std::string msg,
                const log_location_t log_location = log_master){
      if( log_location == log_master && geom.get_myrank() != 0 ){
        return;
      } 
      
      std::string key = objname;
      if( log_location != log_master ){
        key = key + "_p" + std::to_string(geom.get_myrank());
      }
      key += "." + logtype_to_string(type);

      if( logfiles.find(key) == logfiles.end() ){
      //  logfiles.emplace( key, logfile_t(key) );
      } else {
      }
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
         po::value<std::string>()->default_value( std::string("config.yaml") ), 
         "filename of nyom YAML-formatted input file");
    }

    po::options_description cmd_desc;
    po::variables_map cmd_options;

    nyom::Geometry geom;

    unordered_map<std::string,logfile_t> logfiles;

    YAML::Node input_node;

    bool initialised;
};

}
