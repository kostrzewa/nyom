
#include <cstdio>

#include <vector>
#include <iostream>

#include <ctf.hpp>

#include "../../external/ranlxd.h"

#define Ds 4
#define Cs 3
#define Fs 2

#define Ts 16
#define Ls 8

using namespace CTF;
using namespace std;

int main(int argc, char ** argv) {
  int rank, np, d;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  World dw(argc, argv);

  int prop_sizes[10] = { Ts, Ls, Ls, Ls, Fs, Fs, Ds, Ds, Cs, Cs };
  int prop_shapes[10] = { NS, NS, NS, NS, NS, NS, NS, NS, NS, NS };
 
  int tau_sizes[2] = {Fs, Fs};
  int tau_shapes[2] = {NS, NS};
  
  int phi_sizes[6] = { Ts, Ls, Ls, Ls, Fs, Fs};
  int phi_shapes[6] = { NS, NS, NS, NS, NS, NS};
  
  int gamma_sizes[2] = {Ds, Ds};
  int gamma_shapes[2] = {NS, NS};
  
  // we only need the gauge links in time
  int su3_sizes[6] = { Ts, Ls, Ls, Ls, Cs, Cs };
  int su3_shapes[6] = { NS, NS, NS, NS, NS, NS };

  int corr_sizes[1] = { Ts };
  int corr_shapes[1] = { NS };

  int Tshift_sizes[2] = {Ts, Ts};
  int Tshift_shapes[2] = {NS, NS};
 
  Tensor< std::complex<double> > S1(10, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S2(10, prop_sizes, prop_shapes, dw);
  
  Tensor< std::complex<double> > S1_m1(10, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S2_m1(10, prop_sizes, prop_shapes, dw);
 
  Tensor< std::complex<double> > S_t1(10, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S_t2(10, prop_sizes, prop_shapes, dw);

  Tensor< std::complex<double> > S_f1(10, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S_f2(10, prop_sizes, prop_shapes, dw);

  Tensor< std::complex<double> > S1_m2(10, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S2_m2(10, prop_sizes, prop_shapes, dw);
  
  Tensor< std::complex<double> > Phi(6, phi_sizes, phi_shapes, dw);
  Tensor< std::complex<double> > Phi_m1(6, phi_sizes, phi_shapes, dw);
  Tensor< std::complex<double> > Phi_m2(6, phi_sizes, phi_shapes, dw);
  
  Tensor< std::complex<double> > Phi_t1(6, phi_sizes, phi_shapes, dw);
  Tensor< std::complex<double> > Phi_t2(6, phi_sizes, phi_shapes, dw);
  
  Tensor< std::complex<double> > U( 6, su3_sizes, su3_shapes, dw);
  Tensor< std::complex<double> > U_t1( 6, su3_sizes, su3_shapes, dw);
  Tensor< std::complex<double> > U_t2( 6, su3_sizes, su3_shapes, dw);
  Tensor< std::complex<double> > U_m1( 6, su3_sizes, su3_shapes, dw);
  Tensor< std::complex<double> > U_m2( 6, su3_sizes, su3_shapes, dw);

  Tensor< std::complex<double> > g0(2, gamma_sizes, gamma_shapes, dw);
  Tensor< std::complex<double> > tau3(2, tau_sizes, tau_shapes, dw);

  Tensor< std::complex<double> > Tshift_m1(2, Tshift_sizes, Tshift_shapes, dw);
  Tensor< std::complex<double> > Tshift_m2(2, Tshift_sizes, Tshift_shapes, dw);

  // fill tensors with random data and set up the various matrices
  srand48(rank);

  std::complex<double>* pairs;
  int64_t *indices;
  int64_t  npair;

  U.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    pairs[i]=std::complex<double>(drand48(),drand48());
  }
  U.write(npair, indices, pairs);
  free(pairs); free(indices);
  
  S1.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    pairs[i]=std::complex<double>(drand48(),drand48());
  }
  S1.write(npair, indices, pairs);
  free(pairs); free(indices);

  S2.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    pairs[i]=std::complex<double>(drand48(),drand48());
  }
  S2.write(npair, indices, pairs);
  free(pairs); free(indices);

  Phi.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    pairs[i]=std::complex<double>(drand48(),drand48());
  }
  Phi.write(npair, indices, pairs);
  free(pairs); free(indices);

  // construct time-shift marix (minus one unit)
  Tshift_m1.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    // here, the "row" index runs fastest
    int r = indices[i] % Ts;
    int c = indices[i] / Ts;

    if( c == ((r-1)+Ts) % Ts ){
      pairs[i] = 1.0;
    } else {
      pairs[i] = 0.0;
    }
  }
  Tshift_m1.write(npair, indices, pairs);
  free(pairs); free(indices);

  // construct time-shift matrix (minus two units)
  Tshift_m2.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    int r = indices[i] % Ts;
    int c = indices[i] / Ts;
    if( c == ((r-2)+Ts) % Ts ){
      pairs[i] = 1;
    } else {
      pairs[i] = 0;
    }
  }
  Tshift_m2.write(npair, indices, pairs);
  free(pairs); free(indices);
  
  // construct time shifted fields
  U_m1["txyzab"] = Tshift_m1["tT"]*U["Txyzab"];
  U_m2["txyzab"] = Tshift_m2["tT"]*U["Txyzab"];

  Phi_m1["txyzfg"] = Tshift_m1["tT"]*Phi["Txyzfg"];
  Phi_m2["txyzfg"] = Tshift_m2["tT"]*Phi["Txyzfg"];

  S1_m1["txyzfgjkab"] = Tshift_m1["tT"]*S1["Txyzfgjkab"];
  S2_m1["txyzfgjkab"] = Tshift_m1["tT"]*S2["Txyzfgjkab"];

  S1_m2["txyzfgjkab"] = Tshift_m2["tT"]*S1["Txyzfgjkab"];
  S2_m2["txyzfgjkab"] = Tshift_m2["tT"]*S2["Txyzfgjkab"];

  // now construct gamma^0
  g0.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    if( i == 2 || i == 7 || i == 8 || i == 13 ){
      pairs[i] = 1.0;
    } else {
      pairs[i] = 0.0;
    }
  }
  g0.write(npair, indices, pairs);
  free(pairs); free(indices);
  // and tau^3
  tau3.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    if( i == 0 ){
      pairs[i] = 1.0;
    } else if ( i == 3){
      pairs[i] = -1.0;
    } else {
      pairs[i] = 0.0;
    }
  }
  tau3.write(npair, indices, pairs);
  free(pairs); free(indices);
 
  U_t1["txyzab"] = U_m2["txyzaC"]*U_m1["txyzCb"];
  Phi_t1["txyzfg"] = 0.5*Phi_m1["txyzfX"]*tau3["Xg"];

  S_t1["txyzfgijab"] = 0.5*S2_m2["txyzfgXjab"]*g0["Xi"];

  S_t2["txyzfgijab"] = S_t1["txyzfgijaX"]*U_t1["txyzXb"];
  S_f1["txyzfgijab"] = S_t2["txyzXgijab"]*Phi_t1["txyzXf"];
  
  Phi_t2["txyzfg"] = 0.5*Phi["txyzfX"]*tau3["Xg"];
  S_t1["txyzfgijab"] = S1["txyzfgiXab"]*g0["Xj"];
  S_f2["txyzfgijab"] = S_t1["txyzfXijab"]*Phi_t2["txyzXg"];

  Vector< std::complex<double> > C(Ts,dw);

  C["t"] = S_f1["tXYZFGIJAB"]*S_f2["tXYZFGIJAB"];
  C.print();

  return 0;
}
