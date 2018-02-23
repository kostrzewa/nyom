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

#include "Geometry.hpp"
#include "LapH_eigsys.hpp"
#include "Shifts.hpp"
#include "Gaugefield.hpp"

#include <yaml-cpp/yaml.h>

constexpr int Nc = 3;

int main(int argc, char ** argv){
  YAML::Node config = YAML::LoadFile("config.yaml");
  std::cout << "Read configuration file from 'config.yaml':" << std::endl;
  std::cout << config << std::endl << std::endl;

  int Lt = -1;
  int Ls = -1;
  int Nev = -1;

  int conf_start = -1;
  int conf_stride = -1;
  int conf_end = -1;

  Lt = config["Lt"].as<int>();
  Ls = config["Ls"].as<int>();
  Nev = config["Nev"].as<int>();
  conf_start = config["conf_start"].as<int>();
  conf_stride = config["conf_stride"].as<int>();
  conf_end = config["conf_end"].as<int>();


  std::cout << "Parameters interpreted from config file:" << std::endl;
  printf("Lt=%d, Ls=%d, Nev=%d\n",
          Lt, Ls, Nev);
  printf("conf_start=%d, conf_stride=%d, conf_end=%d\n",
          conf_start, conf_stride, conf_end);
  //std::cout << "input_dir: " << input_dir << std::endl;
  //std::cout << "output_dir: " << output_dir << std::endl;
  std::cout << std::endl;
  
  nyom::Geometry geom(argc,argv);
  nyom::Stopwatch sw;
  
  std::vector< CTF::Tensor<complex<double> > > shifts = nyom::make_shifts(Ls, Ls, Ls, geom.get_world() );
  
  nyom::LapH_eigsys V = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, geom.get_world() );
  nyom::LapH_eigsys Vxp1 = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, geom.get_world() );
  nyom::LapH_eigsys UxVxp1 = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, geom.get_world() );
  nyom::LapH_eigsys UxVx = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, geom.get_world() );
  nyom::LapH_eigsys Vback = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, geom.get_world() );

  nyom::Gaugefield gf = nyom::make_Gaugefield(Lt, Ls, Ls, Ls, Nc, geom );

  nyom::read_Gaugefield_from_file(gf,
                                  conf_start,
                                  geom);

  nyom::read_LapH_eigsys_from_files(V,
                                    "ev",
                                    conf_start,
                                    geom );
  
  //sw.reset();
  //nyom::LapH_eigsys Vdagger( V );
  //((CTF::Transform< std::complex<double> >)([](std::complex<double> & s){ s = conj(s); }))(Vdagger["cizyxt"]);
  //sw.elapsed_print("Vdagger");

  //V.print();

  
  sw.reset();
  Vxp1["czyxit"] = shifts[nyom::xp1]["xX"] * V["czyXit"];
  Vback["czyxit"] = shifts[nyom::xm1]["xX"] * Vxp1["czyXit"];
  sw.elapsed_print_and_reset("shift V");

  UxVxp1["czyxit"] = gf.U[nyom::GF_DIR_X]["Cczyxt"] * Vxp1["Czyxit"];
  sw.elapsed_print_and_reset("UxVxp1");

  UxVx["czyxit"] = gf.U[nyom::GF_DIR_X]["Cczyxt"] * V["Czyxit"];
  sw.elapsed_print_and_reset("UxVx");

  write_LapH_eigsys_to_files(Vxp1,
                             "Vxp1",
                             conf_start,
                             geom );

  write_LapH_eigsys_to_files(Vback,
                             "Vback",
                             conf_start,
                             geom );

  write_LapH_eigsys_to_files(V,
                             "V_io_test",
                             conf_start,
                             geom);

  write_LapH_eigsys_to_files(UxVx,
                             "UxVx",
                             conf_start,
                             geom );

  write_LapH_eigsys_to_files(UxVxp1,
                             "UxVxp1",
                             conf_start,
                             geom );


  return 0;
}
