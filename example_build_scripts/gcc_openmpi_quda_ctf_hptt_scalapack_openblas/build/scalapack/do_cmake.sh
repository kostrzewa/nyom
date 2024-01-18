source ../../load_modules.sh

optflags="-mtune=znver2 -march=znver2 -mavx2 -mfma -fopenmp -O3 -fPIC"

cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install_dir \
  -DCMAKE_CXX_FLAGS="${optflags}" \
  -DCMAKE_C_FLAGS="${optflags}" \
  -DCMAKE_Fortran_FLAGS="${optflags}" \
  ../../sources/scalapack

