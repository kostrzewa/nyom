
#include <cstdio>

#include <vector>
#include <iostream>

#include <ctf.hpp>

#include "../../external/ranlxd.h"

#define Ds 4
#define Cs 3
#define Fs 2
#define Ss 4

using namespace CTF;
using namespace std;

int main(int argc, char ** argv) {
  int rank, np, d;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  World dw(argc, argv);

  int prop_sizes[6] = { Fs, Fs, Ds, Ds, Cs, Cs };
  int prop_shapes[6] = { NS, NS, NS, NS, NS, NS };
 
  int tau_sizes[2] = {Fs, Fs};
  int tau_shapes[2] = {NS, NS};
  
  int gamma_sizes[2] = {Ds, Ds};
  int gamma_shapes[2] = {NS, NS};

  int su3_sizes[2] = {Cs, Cs};
  int su3_shapes[2] = {NS, NS};

  Tensor< std::complex<double> > S1(6, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S2(6, prop_sizes, prop_shapes, dw);
  Tensor< std::complex<double> > S3(6, prop_sizes, prop_shapes, dw);
  
  Tensor< std::complex<double> > tau3(2, tau_sizes, tau_shapes, dw);
  
  Tensor< std::complex<double> > Phi(2, tau_sizes, tau_shapes, dw);

  Tensor< std::complex<double> > g0(2, gamma_sizes, gamma_shapes, dw);

  Tensor< std::complex<double> > U(2, su3_sizes, su3_shapes, dw);
  srand48(rank);
  ((Transform< std::complex<double> >)([](std::complex<double> & u){ u= std::complex<double>(drand48(),drand48()); }))(U["ij"]);

  rlxd_init(1,123213);
  double* r = (double*)malloc(2*2*sizeof(double)*Ds*Ds*Cs*Cs*Fs*Fs);
  ranlxd(r, 2*2*Ds*Ds*Cs*Cs*Fs*Fs);

  unsigned long int j = 0;
  std::complex<double>* pairs;
  int64_t *indices; 
  int64_t  npair;

  // fill "propagators" with random values
  S1.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    pairs[i] = std::complex<double>(r[j],r[j+1]);
    j+=2;
  }
  S1.write(npair, indices, pairs);
  free(pairs); free(indices);
  
  S2.read_local(&npair, &indices, &pairs);
  for(int i=0; i<npair; ++i){
    pairs[i] = std::complex<double>(r[j],r[j+1]);
    j+=2;
  }
  S2.write(npair, indices, pairs);
  free(pairs); free(indices);
  
  free(r);
  //
  // as well as the scalar field
  Phi.read_local(&npair, &indices, &pairs);
  r = (double*)malloc(2*sizeof(double)*Fs*Fs);
  ranlxd(r, 2*Fs*Fs);
  j=0;
  for(int i=0; i<npair; ++i){
    pairs[i] = std::complex<double>(r[j],r[j+1]);
    j+=2;
  }
  Phi.write(npair, indices, pairs);
  free(pairs); free(indices);
  free(r);
  
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
  
  Tensor< std::complex<double> > Cc(2,su3_sizes,su3_shapes,dw);
  
  // Segmentation fult here...  
  Cc["xy"] = S1["abcdxf"]*Phi["bg"]*tau3["gh"]*g0["di"]*S2["jhkilf"]*Phi["jm"]*tau3["ma"]*g0["kc"]*U["ly"];
  
  Tensor< std::complex<double> > Cc2(2,su3_sizes,su3_shapes,dw);
  S1["agcdxf"] = S1["abcdxf"]*Phi["bg"];
  S1["ahcdxf"] = S1["agcdxf"]*tau3["gh"];
  S1["ahcixf"] = S1["ahcdxf"]*g0["di"];
  S2["mhkilf"] = S2["jhkilf"]*Phi["jm"];
  S2["ahkilf"] = S2["mhkilf"]*tau3["ma"];
  S2["ahcilf"] = S2["ahkilf"]*g0["kc"];
  S2["ahciyf"] = S2["ahcilf"]*U["ly"];
  Cc2["xy"] = S1["ahcixf"]*S2["ahciyf"];
  //Cc.print();
  //Cc2.print();
  Cc2["xy"] -= Cc["xy"];
  Tensor<double> err(2,su3_sizes,su3_shapes,dw);
  err["xy"] = ((Function< std::complex<double>, double >)([](std::complex<double> c){ return c.real()*c.real() + c.imag()*c.imag(); }))(Cc2["xy"]);
  double nrm_err_Cc2 = err["xy"];
  nrm_err_Cc2 = sqrt(nrm_err_Cc2);
  if (rank == 0) printf("error norm = %E\n", nrm_err_Cc2);


  return 0;
}
