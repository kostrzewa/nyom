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

// namespace-global storage for gamma matrices
std::map < std::string, CTF::Tensor< std::complex<double> > > g;

// namespace-global storage for possible sign change under
// \gamma -> \gamma_0 \gamma^\dagger \gamma_0 = g0_sign[\gamma] \gamma
std::map < std::string, double > g0_sign;

std::vector < std::string > i_g;

void init_gammas(CTF::World& dw){
  for( std::string g1 : { "I", "iI", "0", "1", "2", "3", "5", "i05" } ){
    i_g.push_back(g1);
  }
  
  g["I"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gI");
  // imaginary unit
  g["iI"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gI");
  g["0"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g0");
  g["1"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g1");
  g["2"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g2");
  g["3"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g3");
  g["5"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "g5");
  g["i05"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gi05");
  g["Im5"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIm5");
  g["Ip5"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIp5");
  g["Ip0"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIp0");
  g["Im0"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gIm0");
  g["TwistPlus"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gTwistPlus");
  g["TwistMinus"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gTwistMinus");
  g["C"] = CTF::Tensor<std::complex<double> >(2, gamma_sizes, gamma_shapes, dw, "gC");

  g0_sign["I"] = 1.0;
  g0_sign["iI"] = 1.0;
  g0_sign["0"] = 1.0;
  g0_sign["1"] = -1.0;
  g0_sign["2"] = -1.0;
  g0_sign["3"] = -1.0;
  g0_sign["5"] = -1.0;

  for( std::string g1 : { "0", "1", "2", "3", "5" } ){
    for( std::string g2 : { "0", "1", "2", "3", "5" } ){
      // want to make: 01 02 03 05 15 25 35 12 13 23 21 31 
      if( g1 == g2 || g1 == std::string("5") || g2 == std::string("0") ){
        continue;
      }
      i_g.push_back(g1+g2);
      g[g1+g2] = CTF::Tensor<std::complex<double> >
        (2, 
         gamma_sizes, 
         gamma_shapes, 
         dw, 
         std::string( std::string("g")+g1+g2 ).c_str()
        );
      // everything except for \gamma_0 \gamma_{1,2,3,5} is negative under
      // \Gamma -> \gamma_0 \Gamma^\dagger \gamma_0
      g0_sign[g1+g2] = ( g1 == std::string("0") ) ? 1.0 : -1.0;
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
  
  g["iI"].read_local(&npair,&idx,&pairs);
  for(int64_t i = 0; i < npair; ++i){
    if( idx[i]%4==idx[i]/4 ) pairs[i] = std::complex<double>(0,-1);
  }
  g["iI"].write(npair,idx,pairs); 
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

  g["05"]["ab"] = (g["0"])["aI"] * (g["5"])["Jb"];
  // sign has been set in the loop above

  g["i05"]["ab"] = (g["iI"])["aK"] * (g["05"])["Kb"];
  g0_sign["i05"] = -1.0;

  // charge conjugation
  // no support for complex number scaling during contraction...
  g["C"]["ab"] = g["iI"]["aK"] * g["0"]["KJ"] * g["2"]["Jb"];
  g0_sign["C"] = -1.0;

  // and products with that
  for( std::string g1 : { "0", "1", "2", "3", "5", "05", "i05" } ){
    std::string name = std::string("C") + g1;
    i_g.push_back(name);
    g[name] = CTF::Tensor<std::complex<double> >
      (2, 
       gamma_sizes, 
       gamma_shapes, 
       dw, 
       name.c_str()
      );
    g[name]["ab"] = g["C"]["aI"] * g[g1]["Ib"];
    
    g0_sign[name] = 1.0;
    if( g1 == std::string("2") || g1 == std::string("i05") ){
      g0_sign[name] = -1.0;
    }
  }

  // parity and handedness projectors
  // here we don't do any signs
  g["Im5"]["ab"] = (1.0/sqrt(2.0)) * ( (g["I"])["ab"] - (g["5"])["ab"] ); 
  g["Ip5"]["ab"] = (1.0/sqrt(2.0)) * ( (g["I"])["ab"] + (g["5"])["ab"] ); 
  g["Im0"]["ab"] = (1.0/sqrt(2.0)) * ( (g["I"])["ab"] - (g["0"])["ab"] );
  g["Ip0"]["ab"] = (1.0/sqrt(2.0)) * ( (g["I"])["ab"] + (g["0"])["ab"] ); 

  // \omega = \pi / 2  twist rotations
  g["TwistPlus"]["ab"] = (1.0/sqrt(2.0)) * ( (g["I"])["ab"] + ( (g["iI"])["aK"] * (g["5"])["Kb"] ) );
  g["TwistMinus"]["ab"] = (1.0/sqrt(2.0)) * ( (g["I"])["ab"] - ( (g["iI"])["aK"] * (g["5"])["Kb"] ) );
}

} // namespace nyom
