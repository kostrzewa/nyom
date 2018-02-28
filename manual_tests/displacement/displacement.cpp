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

// for some exotic reason within Boost, this header must always be the first one
// to be included.
#include "Core.hpp"
#include "Logfile.hpp"

#include "Geometry.hpp"
#include "LapH_eigsys.hpp"
#include "Shifts.hpp"
#include "Gaugefield.hpp"

#include <yaml-cpp/yaml.h>

#include <sstream>

constexpr int Nc = 3;

int main(int argc, char ** argv){
  nyom::Core core(argc,argv);

  int Lt = -1;
  int Ls = -1;
  int Nev = -1;

  int conf_start = -1;
  int conf_stride = -1;
  int conf_end = -1;

  Lt = core.input_node["Lt"].as<int>();
  Ls = core.input_node["Ls"].as<int>();
  Nev = core.input_node["Nev"].as<int>();
  conf_start = core.input_node["conf_start"].as<int>();
  conf_stride = core.input_node["conf_stride"].as<int>();
  conf_end = core.input_node["conf_end"].as<int>();

  std::cout << "Parameters interpreted from config file:" << std::endl;
  printf("Lt=%d, Ls=%d, Nev=%d\n",
          Lt, Ls, Nev);
  printf("conf_start=%d, conf_stride=%d, conf_end=%d\n",
          conf_start, conf_stride, conf_end);
  //std::cout << "input_dir: " << input_dir << std::endl;
  //std::cout << "output_dir: " << output_dir << std::endl;
  std::cout << std::endl;
  
  nyom::Stopwatch sw;
  nyom::duration duration;
  std::stringstream log;

  std::vector< CTF::Tensor<complex<double> > > shifts = nyom::make_shifts(Ls, Ls, Ls, core.geom.get_world() );
  
  nyom::LapH_eigsys V = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, core.geom.get_world() );
  nyom::LapH_eigsys Vxp1 = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, core.geom.get_world() );
  nyom::LapH_eigsys UxVxp1 = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, core.geom.get_world() );
  nyom::LapH_eigsys UxVx = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, core.geom.get_world() );
  nyom::LapH_eigsys Vback = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, core.geom.get_world() );

  nyom::Gaugefield gf = nyom::make_Gaugefield(Lt, Ls, Ls, Ls, Nc, core );

  nyom::read_Gaugefield_from_file(gf,
                                  conf_start,
                                  core);

  nyom::read_LapH_eigsys_from_files(V,
                                    "ev",
                                    conf_start,
                                    core );
  
  //sw.reset();
  //nyom::LapH_eigsys Vdagger( V );
  //((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Vdagger["cizyxt"]);
  //sw.elapsed_print("Vdagger");

  //V.print();

  
  sw.reset();
  Vxp1["czyxit"] = shifts[nyom::xp1]["xX"] * V["czyXit"];
  Vback["czyxit"] = shifts[nyom::xm1]["xX"] * Vxp1["czyXit"];
  duration = sw.elapsed();

  log.str(std::string());
  log << "Vshift(mean,min,max): " << duration.mean <<
    " " << duration.min << " " << duration.max << std::endl;
  core.Logger("Vshift",
              nyom::log_perf,
              log.str().c_str());


  sw.reset();
  UxVxp1["czyxit"] = gf.U[nyom::GF_DIR_X]["Cczyxt"] * Vxp1["Czyxit"];
  duration = sw.elapsed();
  
  log.str(std::string());
  log << "UxVxp1(mean,min,max): " << duration.mean <<
    " " << duration.min << " " << duration.max << std::endl;
  core.Logger("UV",
              nyom::log_perf,
              log.str().c_str());


  sw.reset();
  UxVx["czyxit"] = gf.U[nyom::GF_DIR_X]["Cczyxt"] * V["Czyxit"];
  duration = sw.elapsed();

  log.str(std::string());
  log << "UxVx(mean,min,max): " << duration.mean <<
    " " << duration.min << " " << duration.max << std::endl;
  core.Logger("UV",
              nyom::log_perf,
              log.str().c_str());


  write_LapH_eigsys_to_files(Vxp1,
                             "Vxp1",
                             conf_start,
                             core );

  write_LapH_eigsys_to_files(Vback,
                             "Vback",
                             conf_start,
                             core );

  write_LapH_eigsys_to_files(V,
                             "V_io_test",
                             conf_start,
                             core);

  write_LapH_eigsys_to_files(UxVx,
                             "UxVx",
                             conf_start,
                             core );

  write_LapH_eigsys_to_files(UxVxp1,
                             "UxVxp1",
                             conf_start,
                             core );


  return 0;
}
