top_srcdir = ../..

CTF_DIR=$(HOME)/code/ctf/include/
TMLQCD_BUILDDIR=/home/bartek/code/build/tmLQCD.etmc/build_hybrid_4D
TMLQCD_SRCDIR=/home/bartek/code/tmLQCD.etmc2
TMLQCD_LIBS=-lwrapper -lhmc -lmonomial -loperator -lmeas -lsolver -linit -llinalg -lhmc -lxchange -lrational -lio -lblas -llapack -lm -llime -gfortran
LIMEDIR=$(HOME)/local/lime_gcc

INCL=-I$(CTF_DIR) -I${TMLQCD_SRCDIR}/ -I${TMLQCD_SRCDIR}/include/ -I${TMLQCD_BUILDDIR} -I${LIMEDIR}/include/
LIBS=-L$(HOME)/code/ctf/lib -L$(LIMEDIR)/lib -L${TMLQCD_BUILDDIR}/lib -lctf -lblas -llapack ${TMLQCD_LIBS} #${TMLQCD_BUILDDIR}/read_input.o 

CXX = mpicxx.mpich
CXXFLAGS = -std=c++11 -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -fopenmp
LDFLAGS = -fopenmp

LINK=${CXX} -o $@ ${LDFLAGS}

EXT_C_MODULES=ranlxd
MODULES=
PROGRAMS=bench_ctf pion_2pt_ctf

all: ${PROGRAMS}

$(addsuffix .o, ${PROGRAMS}): %.o: %.cpp Makefile
	${CXX} ${CXXFLAGS} ${INCL} -c $<

$(addsuffix .o, ${MODULES}): %.o: %.cpp Makefile
	${CXX} ${CXXFLAGS} ${INCL} -c $<

${PROGRAMS}: %: %.o $(addsuffix .o, ${EXT_C_MODULES}) Makefile
	${LINK} $(addsuffix .o,${MODULES}) $(addsuffix .o ,${EXT_C_MODULES}) $@.o ${LIBS}

$(addsuffix .o,${EXT_C_MODULES}): %.o: ${top_srcdir}/external/%.c ${top_srcdir}/external/%.h Makefile
	${CXX} ${CXXFLAGS} ${INCL} -c $<

clean:
	rm bench_ctf
	rm *.o

.PHONY: all clean
