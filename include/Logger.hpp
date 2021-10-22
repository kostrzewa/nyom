#pragma once

#include <iostream>
#include <string>
#include <ios>

#include "Core.hpp"

namespace nyom {

  /**
   * @brief A streambuf which does not produce any output
   */
class NullBuffer : public std::basic_streambuf<char>
{
public:
  int overflow(int c) { return c; }
};

class Logger {
private:
  int output_rank;
  int verbosity_threshold;
  NullBuffer nulbuf;
  std::ostream nul_stream;
  std::ostream out_stream;
  const nyom::Core & core;

public: 
  Logger() = delete; 

  Logger(const nyom::Core & core_in, const int rank_in, const int verbosity_threshold_in) :
    core(core_in),
    out_stream(std::cout.rdbuf()),
    nul_stream(&nulbuf),
    verbosity_threshold(verbosity_threshold_in),
    output_rank(rank_in) {}

  Logger(const nyom::Core & core_in, const int rank_in, const int verbosity_threshold_in, std::ostream & out_in) :
    core(core_in),
    out_stream(out_in.rdbuf()),
    nul_stream(&nulbuf),
    verbosity_threshold(verbosity_threshold_in),
    output_rank(rank_in) {}

  void set_verbosity_threshold(const int level)
  {
    verbosity_threshold = level;
  }
  
  void set_output_rank(const int id)
  {
    output_rank = id;
  }
 
  template<typename T>
  std::ostream & operator<<(T val)
  {
    if( output_rank == core.geom.get_myrank() && verbosity_threshold <= core.input_node["verbosity_level"].as<int>() ){
      return (out_stream << val);
    } else {
      return (nul_stream << val);
    }
  }

  // support for io maniupulators and std::endl
  std::ostream & operator<<( std::ios_base & (*func)(std::ios_base &) )
  {
    if( output_rank == core.geom.get_myrank() && verbosity_threshold <= core.input_node["verbosity_level"].as<int>() ){
      return (out_stream << func);
    } else {
      return (nul_stream << func);
    }
  }

  std::ostream & operator<<( std::ostream& (*func)(std::ostream &) )
  {
    if( output_rank == core.geom.get_myrank() && verbosity_threshold <= core.input_node["verbosity_level"].as<int>() ){
      return (out_stream << func);
    } else {
      return (nul_stream << func);
    }
  }

};

} //namespace(nyom)

