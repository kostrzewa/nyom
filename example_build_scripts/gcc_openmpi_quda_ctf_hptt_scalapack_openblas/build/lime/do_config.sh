source ../../load_modules.sh
CC=gcc \
CFLAGS="-mtune=znver2 -march=znver2 -mavx2 -mfma -O2" \
../../sources/lime/configure \
  --prefix=$(pwd)/install_dir
