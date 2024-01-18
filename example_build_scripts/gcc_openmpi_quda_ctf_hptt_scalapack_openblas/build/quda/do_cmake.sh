#!/bin/bash
source ../../load_modules.sh
cmake \
-DCMAKE_INSTALL_PREFIX="$(pwd)/install_dir" \
-DCMAKE_BUILD_TYPE=RELEASE \
-DQUDA_BUILD_ALL_TESTS=OFF \
-DQUDA_GPU_ARCH=sm_80 \
-DQUDA_INTERFACE_QDP=ON \
-DQUDA_INTERFACE_MILC=OFF \
-DQUDA_DIRAC_WILSON=ON \
-DQUDA_DIRAC_TWISTED_MASS=ON \
-DQUDA_DIRAC_TWISTED_CLOVER=ON \
-DQUDA_DIRAC_NDEG_TWISTED_CLOVER=ON \
-DQUDA_DIRAC_NDEG_TWISTED_MASS=ON \
-DQUDA_DIRAC_CLOVER=ON \
-DQUDA_DIRAC_STAGGERED=OFF \
-DQUDA_MULTIGRID=ON \
-DQUDA_MPI=ON \
-DQUDA_QMP=OFF \
-DQUDA_QIO=OFF \
-DQUDA_DOWNLOAD_USQCD=OFF \
../../sources/quda
