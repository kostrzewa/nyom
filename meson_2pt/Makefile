CTF_DIR=$(HOME)/code/ctf/include/
TMLQCD_BUILDDIR=/home/bartek/code/build/tmLQCD.etmc/build_hybrid_4D
TMLQCD_SRCDIR=/home/bartek/code/tmLQCD.etmc2
TMLQCD_LIBS=-lwrapper -lhmc -lmonomial -loperator -lmeas -lsolver -linit -llinalg -lhmc -lxchange -lrational -lio -lblas -llapack -lm -llime -gfortran
LIMEDIR=$(HOME)/local/lime_gcc

INCL=-I$(CTF_DIR) -I${TMLQCD_SRCDIR}/ -I${TMLQCD_SRCDIR}/include/ -I${TMLQCD_BUILDDIR} -I${LIMEDIR}/include/
LIBS=-L$(HOME)/code/ctf/lib -L$(LIMEDIR)/lib -L${TMLQCD_BUILDDIR}/lib -lctf -lblas -llapack ${TMLQCD_LIBS}

CXX = mpicxx.mpich
CXXFLAGS = -std=c++11 -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -fopenmp -O3 -mtune=corei7 -march=corei7
LDFLAGS = -fopenmp

LINK=${CXX} -o $@ ${LDFLAGS}

include make.inc
