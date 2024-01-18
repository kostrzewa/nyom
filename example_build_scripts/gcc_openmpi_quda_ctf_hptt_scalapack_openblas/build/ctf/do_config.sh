source ../../load_modules.sh

hptt_path=$(pwd)/../hptt/install_dir
scalapack_path=$(pwd)/../scalapack/install_dir

../../sources/ctf/configure \
  --build-dir=$(pwd) \
  --install-dir=$(pwd)/install_dir \
  --with-hptt \
  --with-scalapack \
  --verbose \
  CXXFLAGS="-mtune=znver2 -march=znver2 -mavx2 -mfma -O3 -fopenmp -fPIC" \
  INCLUDES="-I${hptt_path}/include" \
  LIB_PATH="-L${hptt_path}/lib -L${scalapack_path}/lib" \
  LIBS="-lopenblas"
