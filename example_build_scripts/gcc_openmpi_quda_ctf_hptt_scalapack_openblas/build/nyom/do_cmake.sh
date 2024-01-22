source ../load_modules.sh

cmake \
  -DCMAKE_C_FLAGS="-mtune=znver2 -march=znver2 -O3 -mavx2 -mfma -fopenmp" \
  -DCMAKE_CXX_FLAGS="-mtune=znver2 -march=znver2 -O3 -mavx2 -mfma -fopenmp" \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp" \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DTMLQCD_SRC=$(pwd)/../../sources/tmLQCD \
  -DTMLQCD_BUILD=$(pwd)/../tmLQCD \
  -DCTF_HOME=$(pwd)/../ctf/install_dir \
  -DLIME_HOME=$(pwd)/../lime/install_dir \
  -DLEMON_HOME=$(pwd)/../lemon/install_dir \
  -DQUDA_HOME=$(pwd)/../quda/install_dir \
  ../../sources/nyom

