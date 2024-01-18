# number of processes to pass to 'make -jN'
make_nprocs=32

exitcode=0

source load_modules.sh

pushd sources
./fetch_ctf.sh
./fetch_hptt.sh
./fetch_scalapack.sh
./fetch_lime.sh
./fetch_lemon.sh
./fetch_quda.sh
./fetch_tmlqcd.sh
popd

# prepare the build systems of the libraries that we use
pushd sources/lime
./autogen.sh
exitcode=$?
if [ $exitcode -ne 0 ]; then
  echo Failed to run ./autogen.sh for lime library
fi
popd

if [ $exitcode -eq 0 ]; then
  pushd sources/lemon
  
  autoreconf --install
  exitcode=$?
  if [ $exitcode -ne 0 ]; then
    echo Failed to run autoreconf --install for the lemon library
  fi
  
  popd
fi

if [ $exitcode -eq 0 ]; then
  pushd sources/tmLQCD
  
  autoconf
  exitcode=$?
  if [ $exitcode -ne 0 ]; then
    echo Failed to run autoconf for tmLQCD
  fi
  
  popd
fi

# begin compilation
if [ $exitcode -eq 0 ]; then
  pushd build/lime
  
  ./do_config.sh
  exitcode=$?
  if [ $exitcode -ne 0 ]; then
    echo Failed to configure lime library
  else 
    make -j${make_nprocs}
    exitcode=$?
    if [ $exitcode -ne 0 ]; then
      echo Failed to compile lime library
    else
      make install
      exitcode=$?
      if [ $exitcode -ne 0 ]; then
        echo Failed to install lime library
      fi
    fi
  fi
  
  popd
fi

if [ $exitcode -eq 0 ]; then
  pushd build/lemon
  
  ./do_config.sh
  exitcode=$?
  if [ $exitcode -ne 0 ]; then
    echo Failed to configure lemon library
  else 
    make -j${make_nprocs}
    exitcode=$?
    if [ $exitcode -ne 0 ]; then
      echo Failed to compile lemon library
    else
      make install
      exitcode=$?
      if [ $exitcode -ne 0 ]; then
        echo Failed to install lemon library
      fi
    fi
  fi
  
  popd
fi

if [ $exitcode -eq 0 ]; then
  pushd build/quda

  ./do_cmake.sh
  exitcode=$?
  if [ $exitcode -ne 0 ]; then
    echo Failed to configure QUDA library
  else
    make -j${make_nprocs}
    exitcode=$?
    if [ $exitcode -ne 0 ]; then
      echo Failed to compile QUDA library
    else
      make install
      exitcode=$?
      if [ $exitcode -ne 0 ]; then
        echo Failed to install QUDA library
      fi
    fi
  fi

  popd
fi

if [ $exitcode -eq 0 ]; then
  pushd build/tmLQCD
  
  ./do_config.sh
  exitcode=$?
  if [ $exitcode -ne 0 ]; then
    echo Failed to configure tmLQCD
  else
    make -j${make_nprocs}
    if [ $exitcode -ne 0 ]; then
      echo Failed to compile tmLQCD
    fi
  fi
  
  popd
fi
