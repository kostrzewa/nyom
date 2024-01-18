source ../../load_modules.sh
cmake \
  -DCMAKE_INSTALL_PREFIX=$(pwd)/install_dir \
  -DCMAKE_CXX_FLAGS="-O3 -mtune=znver2 -march=znver2 -mavx2 -mfma -fopenmp -fPIC" \
  -DENABLE_AVX=1 \
  ../../sources/hptt

