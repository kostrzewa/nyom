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

#include "Core.hpp"
#include "Stopwatch.hpp"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <string>
#include <sstream>
#include <unordered_map>
#include <functional>

namespace nyom {

  namespace fs=boost::filesystem;

  /**
   * @brief Abstract base class for run-time auto-tuning. Can be used as a base class
   * for functors, but also as a stand-alone thing.
   */
class AutoTunable {
  public:
    AutoTunable(const std::string name_in,
                const std::string identifier_in,
                const nyom::Core & core_in) :
      core(core_in),
      name(name_in),
      identifier(identifier_in)
    {
      tunepath = "tune/";
      if( core.input_node["tunepath"] ){
        tunepath = core.input_node["tunepath"].as<std::string>() + "/";
      }
      tunepath += name + "/" + identifier + "/";

      fs::path fs_path(tunepath);
      std::stringstream msg;

      if( core.geom.get_myrank() == 0 ) {
        if( !fs::exists(fs_path) ){
          bool success = fs::create_directories(fs_path);
          if( !success ){
            msg << "Failed to create " << fs_path << std::endl;
          }
        } else {
          if( !fs::is_directory(fs_path) ){
            std::stringstream msg;
            msg << "File '" << fs_path << "' exists, but is not a directory!" << std::endl;
          }
        }
        if( msg.str().size() > 0 ) { throw( std::runtime_error(msg.str()) ); }
      }

      MPI_Barrier(MPI_COMM_WORLD);

      for( auto& entry : boost::make_iterator_range( fs::directory_iterator(fs_path), {} ) ){
        double fom = 0.0;
        std::ifstream ifile( entry.path().string() );
        if(!ifile) {
          msg << entry.path().string() << " could not be openend on process " <<
            core.geom.get_myrank() << std::endl;
          throw( std::runtime_error(msg.str()) );
        }
        ifile >> fom;
        ifile.close();

        add_measurement(entry.path().filename().string(), fom, true);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // cannot be standard constructible because nyom::Core must have been initialised!
    AutoTunable() = delete;

    std::string get_optimal_variant()
    {
      return(optimal_variant);
    }

    size_t get_foms_size()
    {
      return( foms.size() );
    }

    double get_fom( const std::string variant ){
      return( foms[variant] );
    }

    void add_measurement(const std::string variant,
                         const double fom,
                         bool init = false)
    {
      foms[variant] = fom;
      if( optimal_variant.size() == 0 ){
        optimal_variant = variant;
      } else {
        if( foms[optimal_variant] > fom ){
          optimal_variant = variant;
        }
      }
      // when we use add_measurement to populate the foms map,
      // we don't want to write the tuning results to file
      if( !init && core.geom.get_myrank() == 0 ){
        write_measurement(tunepath + variant, fom);
      }
    }
    
  private:

    void write_measurement(const std::string filename,
                           const double fom)
    {
      std::stringstream msg;
      std::ofstream file(filename);
      if(!file) {
        msg << filename << " could not be openend on process " <<
          core.geom.get_myrank() << std::endl;
        throw( std::runtime_error(msg.str()) );
      }
      file << fom;
      file.close();
    }

    const nyom::Core & core;
    const std::string name;
    const std::string identifier;

    std::string tunepath;
    std::unordered_map<std::string, double> foms;
    std::string optimal_variant;

};


} // namespace nyom
