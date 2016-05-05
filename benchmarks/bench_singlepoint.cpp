
#include <cstdio>

#include <vector>
#include <iostream>

#include <blitz/array.h>

#include "../external/ranlxd.h"

#define T 16
#define L 8
#define Ds 4
#define Cs 3
#define Fs 2
#define Ss 4

using namespace blitz;
using namespace std;

int main(void) {
  int Vs = L*L*L;

  //blitz::Array<std::complex<double>, 8> S1(T, L*L*L, F, F, D, D, C, C);
  //blitz::Array<std::complex<double>, 8> S2(T, L*L*L, F, F, D, D, C, C);
  //blitz::Array<std::complex<double>, 4> Phi(T, L*L*L, F, F);
  //blitz::Array<std::complex<double>, 4> Phi_m0(T, L*L*L, F, F);
  //blitz::Array<std::complex<double>, 4> U(T, L*L*L, S, C, C);
  //blitz::Array<std::complex<double>, 4> U_m0(T, L*L*L, S, C, C);

  //Range rFs(0,Fs-1), rDs(0,Ds-1), rCs(0,Cs-1);

  blitz::Array<std::complex<double>, 6> S1(Fs, Fs, Ds, Ds, Cs, Cs);
  blitz::Array<std::complex<double>, 6> S2(Fs, Fs, Ds, Ds, Cs, Cs);
  blitz::Array<std::complex<double>, 2> Phi(Fs, Fs);
  blitz::Array<std::complex<double>, 2> Phi_m0(Fs, Fs);
  blitz::Array<std::complex<double>, 2> U(Cs, Cs);
  blitz::Array<std::complex<double>, 2> U_m0(Cs, Cs);
  
  blitz::Array<std::complex<double>, 3> tau(3, Fs, Fs);
  blitz::Array<std::complex<double>, 3> gamma(1, Ds, Ds);

  rlxd_init(1,123213);
  double* r = (double*)malloc(2*2*sizeof(double)*T*L*L*L*Ds*Ds*Cs*Cs*Fs*Fs);
  ranlxd(r, 2*2*T*L*L*L*Ds*Ds*Cs*Cs*Fs*Fs);

  unsigned long int i = 0;
//  for(int t = 0; t <= T; ++t){
//    for(int x = 0; x <= Vs; ++x){
      for(int sif = 0; sif <= Fs; ++sif){
        for(int sof = 0; sof <= Fs; ++sof){
          for(int sid = 0; sid <= Ds; ++sid){
            for(int sod = 0; sod <= Ds; ++sod){
              for(int sic = 0; sic <= Cs; ++sic){
                for(int soc = 0; soc <= Cs; ++soc){
                  S1(sif,sof,sid,sod,sic,soc) = std::complex<double>(r[i],r[i+1]);
                  //S1(t,x,sif,sof,sid,sod,sic,soc) = (r[i],r[i+1]);
                  i=i+2; 
                  S2(sif,sof,sid,sod,sic,soc) = std::complex<double>(r[i],r[i+1]);
                  //S2(t,x,sif,sof,sid,sod,sic,soc) = (r[i],r[i+1]);
                  i=i+2; 
                  //std::cout << S1(sif,sof,sid,sod,sic,soc) << " ";
                } // soc
              //std::cout << std::endl;
              } // sic
            } // sod
          } // sid
        } // sof
      } // sif
//    } // x
//  } // t
  free(r);

  r = (double*)malloc(2*sizeof(double)*T*L*L*L*Ss*Cs*Cs);
  ranlxd(r, 2*T*L*L*L*Ss*Cs*Cs);
  i = 0;
//  for(int t = 0; t <= T; ++t){
//    for(int x = 0; x <= Vs; ++x){
      for(int mu = 0; mu <= Ss; ++mu){
        for(int sic = 0; sic <= Cs; ++sic){
          for(int soc = 0; soc <= Cs; ++soc){
            U(sic,soc) = std::complex<double>(r[i],r[i+1]);
            //U(t,x,mu,sic,soc) = (r[i],r[i+1]);
            i=i+2; 
          } // soc
        } // sic
      } // mu
//    } // x
//  } // t
  free(r);

//  for(int t = 0; t <= T; ++t){
//    for(int x = 0; x <= Vs; ++x){
//      for(int mu = 0; mu <= S; ++mu){
//        for(int sic = 0; sic <= C; ++sic){
//          for(int soc = 0; soc <= C; ++soc){
//            int tm1 = t-1 < 0 ? T-1 : t-1;
//            U_m0(t,x,mu,sic,soc) = U(tm1,x,mu,sic,soc);
//          } // soc
//        } // sic
//      } // mu
//    } // x
//  } // t

  r = (double*)malloc(2*sizeof(double)*T*L*L*L*Fs*Fs);
  ranlxd(r,2*T*L*L*L*Fs*Fs);
  i=0;
//  for(int t = 0; t <= T; ++t){
//    for(int x = 0; x <= Vs; ++x){
      for(int sif = 0; sif <= Fs; ++sif){
        for(int sof = 0; sof <= Fs; ++sof){
          //Phi(t,x,sif,sof) = (r[i],r[i+1]);
          Phi(sif,sof) = std::complex<double>(r[i],r[i+1]);
          i=i+2;
        }
      }
//    }
//  }
  free(r);
  
//  for(int t = 0; t <= T; ++t){
//    for(int x = 0; x <= Vs; ++x){
//      for(int sif = 0; sif <= F; ++sif){
//        for(int sof = 0; sof <= F; ++sof){
//          int tm1 = t-1 < 0 ? T-1 : t-1;
//          Phi_m0(t,x,sif,sof) = Phi(tm1,x,sif,sof);
//        }
//      }
//    }
//  }

  //firstIndex i1; secondIndex i2; thirdIndex   i3; fourthIndex i4;
  //fifthIndex i5; sixthIndex  i6; seventhIndex i7; eighthIndex i8;
  //ninthIndex i9; tenthIndex i10; eleventhIndex i11;

  // contract S1 and S2 colour indices in natural order
  // Array<std::comple<double>, 10> smatrix(Fs, Fs, Ds, Ds, Cs, Fs, Fs, Ds, Ds, Cs);
  // smatrix = sum( S1(i1, i2, i3, i4, i5, i11) * S2(i6, i7, i8, i9, i10, i11), i11);
  
  //Array<std::complex<double>, 2> test;
  //test = sum( Phi(i1,i3) * Phi(i3,i2), i3);



  //std::cout << test.extent(firstDim) << std::cout;

  return 0;
}
