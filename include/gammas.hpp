/***********************************************************************
 * Copyright (C) 2016 Bartosz Kostrzewa
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

#include <map>
#include <numeric>
#include <vector>
#include <complex>
#include <ctf.hpp>

namespace nyom {

const int gamma_sizes[2] = { 4, 4 };
const int gamma_shapes[2] = { NS, NS };

std::map < std::string, CTF::Tensor< std::complex<double> > > g;

std::vector < std::string > i_g;

void init_gammas(CTF::World& dw){
  for( std::string g1 : { "I", "0", "1", "2", "3", "5" } ){
    i_g.push_back(g1);
  }
  
  g["I"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gI");
  g["0"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g0");
  g["1"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g1");
  g["2"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g2");
  g["3"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g3");
  g["5"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g5");
  g["Im5"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIm5");
  g["Ip5"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIp5");
  g["Ip0"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIp0");
  g["Im0"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIm0");

  for( std::string g1 : { "0", "1", "2", "3", "5" } ){
    for( std::string g2 : { "0", "1", "2", "3", "5" } ){
      if( g1 == g2 ) continue;
      i_g.push_back(g1+g2);
      g[g1+g2] = CTF::Tensor<std::complex<double> >
        (2, 
         gamma_sizes, 
         gamma_shapes, 
         dw, 
         std::string( std::string("g")+g1+g2 ).c_str()
        );
    }
  } 

  // remember that in CTF the left-most index runs fastest, for a rank 2 tensor, this means
  // that the "row" index runs fastest such that the conditionals below should be thought of
  // as iterating over the columns of the matrix, top to bottom, left to right
  int64_t npair;
  int64_t* idx;
  std::complex<double>* pairs;

  g["I"].read_local(&npair,&idx,&pairs);
  for(int64_t i = 0; i < npair; ++i){
    if( idx[i]%4==idx[i]/4 ) pairs[i] = std::complex<double>(1,0);
  }
  g["I"].write(npair,idx,pairs); 
  free(idx); free(pairs); 

  g["0"].read_local(&npair,&idx,&pairs);
  for(int64_t i = 0; i < npair; ++i){
    if(idx[i]==2 || idx[i]==7 || idx[i]==8 || idx[i]==13) pairs[i] = std::complex<double>(-1,0);
  }
  g["0"].write(npair,idx,pairs);
  free(idx); free(pairs); 
   
  g["1"].read_local(&npair,&idx,&pairs);
  for(int64_t i = 0; i < npair; ++i){
    if( idx[i]==3 || idx[i]==6 ) pairs[i] = std::complex<double>(0,1);
    if( idx[i]==9 || idx[i]==12) pairs[i] = std::complex<double>(0,-1);
  }
  g["1"].write(npair,idx,pairs);
  free(idx); free(pairs); 

  g["2"].read_local(&npair,&idx,&pairs);
  for(int64_t i = 0; i < npair; ++i){
    if( idx[i]==3 ) pairs[i] = std::complex<double>(-1,0);
    if( idx[i]==6 ) pairs[i] = std::complex<double>(1,0);
    if( idx[i]==9 ) pairs[i] = std::complex<double>(1,0);
    if( idx[i]==12) pairs[i] = std::complex<double>(-1,0);
  }
  g["2"].write(npair,idx,pairs);
  free(idx); free(pairs); 

  g["3"].read_local(&npair,&idx,&pairs);
  for(int64_t i = 0; i < npair; ++i){
    if( idx[i]==2 ) pairs[i] = std::complex<double>(0,1);
    if( idx[i]==7 ) pairs[i] = std::complex<double>(0,-1);
    if( idx[i]==8 ) pairs[i] = std::complex<double>(0,-1);
    if( idx[i]==13) pairs[i] = std::complex<double>(0,1);
  }
  g["3"].write(npair,idx,pairs);
  free(idx); free(pairs);

  g["5"]["ab"] = (g["0"])["aI"] * (g["1"])["IJ"] * (g["2"])["JK"] * (g["3"])["Kb"];

  for( std::string g1 : { "0", "1", "2", "3", "5" } ){
    for( std::string g2 : { "0", "1", "2", "3", "5" } ){
      if( g1 == g2 ) continue;
      g[g1+g2]["ab"] = (g[g1])["aI"] * (g[g2])["Ib"];
    }
  }

  g["Im5"]["ab"] = 0.5*( (g["I"])["ab"] - (g["5"])["ab"] ); 
  g["Ip5"]["ab"] = 0.5*( (g["I"])["ab"] + (g["5"])["ab"] ); 
  g["Im0"]["ab"] = 0.5*( (g["I"])["ab"] - (g["0"])["ab"] );
  g["Ip0"]["ab"] = 0.5*( (g["I"])["ab"] + (g["0"])["ab"] ); 
}

} // namespace nyom
