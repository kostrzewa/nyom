top_srcdir = ../..

INCL=-I$(HOME)/code/ctf/include/
LIBS=-L$(HOME)/code/ctf/lib -lctf -lblas -llapack

CXX = mpicxx.mpich
#CXXFLAGS = -funroll-loops -fopenmp -O3 -mtune=core2 -march=core2 -m64 -msse4.2 -g -ggdb -std=gnu++11 -ftree-vectorize -fopt-info-vec-missed-optimized=output.opt
CXXFLAGS = -std=c++11 -D_POSIX_C_SOURCE=200112L -D__STDC_LIMIT_MACROS -DFTN_UNDERSCORE=1 -fopenmp 
LDFLAGS = -fopenmp 

OBJS = bench_ctf.o ranlxd.o

all: bench_ctf

bench_ctf: $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) $(LIBS) -o bench_ctf

bench_ctf.o: bench_ctf.cpp
	$(CXX) $(CXXFLAGS) $(INCL) -c bench_ctf.cpp

ranlxd.o: $(top_srcdir)/external/ranlxd.c $(top_srcdir)/external/ranlxd.h
	$(CXX) $(CXXFLAGS) $(INCL) -c $(top_srcdir)/external/ranlxd.c

clean:
	rm bench_ctf
	rm *.o

.PHONY: all clean
