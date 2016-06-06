
#include <cstdio>

#include <vector>
#include <iostream>

#include <ctf.hpp>

#include "../../external/ranlxd.h"

#define Ds 4
#define Cs 3
#define Fs 2
#define Ss 4
#define Xs 256

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
  
  int phi_sizes[2] = { Fs, Fs};
  int phi_shapes[2] = { NS, NS};
  
  int gamma_sizes[2] = {Ds, Ds};
  int gamma_shapes[2] = {NS, NS};

  int su3_sizes[2] = { Cs, Cs};
  int su3_shapes[2] = { NS, NS};

 
  std::vector< Tensor< std::complex<double> > > S1v( Xs, Tensor< std::complex<double> >(6, prop_sizes, prop_shapes, dw) );
  std::vector< Tensor< std::complex<double> > > S2v( Xs, Tensor< std::complex<double> >(6, prop_sizes, prop_shapes, dw) );
  //Tensor< std::complex<double> > S2(6, prop_sizes, prop_shapes, dw);
  
  std::vector< Tensor< std::complex<double> > > Phiv( Xs, Tensor< std::complex<double> >(2, tau_sizes, tau_shapes, dw) );
  std::vector< Tensor< std::complex<double> > > Uv( Xs, Tensor< std::complex<double> >(2, su3_sizes, su3_shapes, dw) );

  //Tensor< std::complex<double> > Phi(2, tau_sizes, tau_shapes, dw);
  //Tensor< std::complex<double> > U(2, su3_sizes, su3_shapes, dw);

  Tensor< std::complex<double> > g0(2, gamma_sizes, gamma_shapes, dw);
  Tensor< std::complex<double> > tau3(2, tau_sizes, tau_shapes, dw);

  if(rank==0) std::cout << S1v.size() << std::endl;

  srand48(rank);
  for(auto &&U : Uv){
    ((Transform< std::complex<double> >)([](std::complex<double> & u){ u= std::complex<double>(drand48(),drand48()); }))(U["ij"]);
  }

  rlxd_init(1,123213);
  double* r = (double*)malloc(2*2*sizeof(double)*Xs*Ds*Ds*Cs*Cs*Fs*Fs);
  ranlxd(r, 2*2*Xs*Ds*Ds*Cs*Cs*Fs*Fs);

  unsigned long int j = 0;
  std::complex<double>* pairs;
  int64_t *indices; 
  int64_t  npair;

  // fill "propagators" with random values
  for(auto &&S1 : S1v){
    S1.read_local(&npair, &indices, &pairs);
    for(int i=0; i<npair; ++i){
      pairs[i] = std::complex<double>(r[j],r[j+1]);
      j+=2;
    }
    S1.write(npair, indices, pairs);
    free(pairs); free(indices);
  }
 
  for(auto &&S2 : S2v){ 
    S2.read_local(&npair, &indices, &pairs);
    for(int i=0; i<npair; ++i){
      pairs[i] = std::complex<double>(r[j],r[j+1]);
      j+=2;
    }
    S2.write(npair, indices, pairs);
    free(pairs); free(indices);
  }
  
  free(r);
  //
  // as well as the scalar field
  r = (double*)malloc(2*sizeof(double)*Xs*Fs*Fs);
  ranlxd(r, 2*Xs*Fs*Fs);
  
  r = (double*)malloc(2*sizeof(double)*Xs*Fs*Fs);
  ranlxd(r, 2*Xs*Fs*Fs);
  j=0;
  
  for(auto &&Phi : Phiv){ 
    Phi.read_local(&npair, &indices, &pairs);
    for(int i=0; i<npair; ++i){
      pairs[i] = std::complex<double>(r[j],r[j+1]);
      j+=2;
    }
    Phi.write(npair, indices, pairs);
    free(pairs); free(indices);
  }
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
  
  // this is very slow because of temporaries
  //Cc["xy"] = S1["abcdxf"]*Phi["bg"]*tau3["gh"]*g0["di"]*S2["jhkilf"]*Phi["jm"]*tau3["ma"]*g0["kc"]*U["ly"];
  
  Tensor< std::complex<double> > Cc2(2,su3_sizes,su3_shapes,dw);
  S1v[0]["agcdxf"] = S1v[0]["abcdxf"]*Phiv[0]["bg"];
  S1v[0]["ahcdxf"] = S1v[0]["agcdxf"]*tau3["gh"];
  S1v[0]["ahcixf"] = S1v[0]["ahcdxf"]*g0["di"];
  S2v[0]["mhkilf"] = S2v[0]["jhkilf"]*Phiv[0]["jm"];
  S2v[0]["ahkilf"] = S2v[0]["mhkilf"]*tau3["ma"];
  S2v[0]["ahcilf"] = S2v[0]["ahkilf"]*g0["kc"];
  S2v[0]["ahciyf"] = S2v[0]["ahcilf"]*Uv[0]["ly"];
  Cc2["xy"] = S1v[0]["ahcixf"]*S2v[0]["ahciyf"];

  for(size_t x = 1; x < Xs; ++x){
    S1v[x]["agcdxf"] = S1v[x]["abcdxf"]*Phiv[0]["bg"];
    S1v[x]["ahcdxf"] = S1v[x]["agcdxf"]*tau3["gh"];
    S1v[x]["ahcixf"] = S1v[x]["ahcdxf"]*g0["di"];
    S2v[x]["mhkilf"] = S2v[x]["jhkilf"]*Phiv[0]["jm"];
    S2v[x]["ahkilf"] = S2v[x]["mhkilf"]*tau3["ma"];
    S2v[x]["ahcilf"] = S2v[x]["ahkilf"]*g0["kc"];
    S2v[x]["ahciyf"] = S2v[x]["ahcilf"]*Uv[0]["ly"];
    Cc2["xy"] += S1v[x]["ahcixf"]*S2v[x]["ahciyf"];
  }

  Cc2.print();

  //Cc2.print();

  //Cc.print();
  //Cc2.print();
  //Cc2["xy"] -= Cc["xy"];
  //Tensor<double> err(2,su3_sizes,su3_shapes,dw);
  //err["xy"] = ((Function< std::complex<double>, double >)([](std::complex<double> c){ return c.real()*c.real() + c.imag()*c.imag(); }))(Cc2["xy"]);
  //double nrm_err_Cc2 = err["xy"];
  //nrm_err_Cc2 = sqrt(nrm_err_Cc2);
  //if (rank == 0) printf("error norm = %E\n", nrm_err_Cc2);


  return 0;
}
