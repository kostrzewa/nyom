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
  //nyom::LapH_eigsys Vback = nyom::make_LapH_eigsys(Nev, Lt, Ls, Ls, Ls, Nc, geom.get_world() );

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

  //CTF::Vector< complex<double> > TrVdagV( Lt, geom.get_world() );

  //shifts[nyom::xp1].print();
  
  //std::cout << std::endl << std::endl;
  
  //shifts[nyom::xm1].print();

  //int s[2];
  //for( auto & v : s ) v = Ls;
  //int sym[2] = {NS, NS};
  //CTF::Tensor< double > pmtest(2, s, sym, geom.get_world() );

  //pmtest["xy"] = shifts[nyom::xp1]["xy"] + shifts[nyom::xm1]["xy"];
  //pmtest.print();

  // test shifts 
  //int64_t * indices;
  //int64_t npair;
  //double * pairs;
  //int64_t counter = 0;

  //CTF::Vector< double > v1( Ls, geom.get_world() );
  //v1.read_local(&npair, &indices, &pairs);
  //while( counter < npair ){
  //  pairs[counter] = counter;
  //  counter++;
  //}
  //v1.write(npair, indices, pairs);
  //free(indices); free(pairs);
  //v1.print();
  //std::cout << std::endl << std::endl;

  //CTF::Vector< double > v2( Ls, geom.get_world() );
  //CTF::Vector< double > v3( Ls, geom.get_world() );
  //CTF::Vector< double > v4( Ls, geom.get_world() );

  //v2["x"] = shifts[nyom::xp1]["xX"] * v1["X"];
  ////v3["x"] = shifts[nyom::xm1]["xX"] * v1["X"];

  ////v2.print();
  ////std::cout << std::endl << std::endl;

  ////v3.print();
  ////std::cout << std::endl << std::endl;
  ////
  //v4["x"] = shifts[nyom::xm1]["xX"] * v2["X"];
  //v4.print();
  // test shifts done
  
  sw.reset();
  Vxp1["czyxit"] = shifts[nyom::xp1]["xX"] * V["czyXit"];
  sw.elapsed_print_and_reset("shift V");
  UxVxp1["czyxit"] = gf.U[nyom::GF_DIR_X]["Cczyxt"] * Vxp1["Czyxit"];
  sw.elapsed_print("UxVxp1");

  UxVx["czyxit"] = gf.U[nyom::GF_DIR_X]["Cczyxt"] * V["Czyxit"];

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

  //sw.reset();
  //Vback["cizyxt"] =  shifts[nyom::xm1]["xX"] * Vxp1["cizyXt"];
  //sw.elapsed_print("shift Vp1 back");

  //Vtest["cizyxt"] = Vback["cizyxt"] - V["cizyxt"];

  //Vtest.print();
  //gf.U[1].print();

  //Vxp1["cizyxt"] = gf.U[nyom::GF_DIR_X]["aBzyxt"] * Vxp1["Bizyxt"];
  

  //TrVdagV["t"] = Vdagger["CIZYXt"]*V["CIZYXt"];

  //TrVdagV.print();

  // shift V one lattice site forward
  //Vp1["tixyzc"] = shifts[nyom::xp1]["xX"]*V["tiXyzc"];

  //Vp1.print();

  return 0;
}