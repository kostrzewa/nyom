source ../../load_modules.sh
quda_dir=$(pwd)/../quda/install_dir
CC=mpicc CXX=mpicxx F77=f77 \
CFLAGS="-mtune=znver2 -march=znver2 -O3 -mavx2 -mfma -fopenmp" \
CXXFLAGS="-mtune=znver2 -march=znver2 -O3 -mavx2 -mfma -fopenmp" \
LDFLAGS="-fopenmp" \
../../sources/tmLQCD/configure \
  --enable-quda_experimental \
  --enable-mpi \
  --enable-omp \
  --with-mpidimension=4 \
  --disable-sse2 --disable-sse3 \
  --with-cudadir=${EBROOTCUDA}/lib64 \
  --with-qudadir=${quda_dir} \
  --with-limedir=$(pwd)/../lime/install_dir \
  --with-lemondir=$(pwd)/../lemon/install_dir \
  --with-lapack="-lopenblas"
