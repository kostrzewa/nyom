source ../../load_modules.sh
CC=mpicc \
CFLAGS="-march=znver2 -mtune=znver2 -O2" \
../../sources/lemon/configure \
  --prefix=$(pwd)/install_dir

