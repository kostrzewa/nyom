top_srcdir = ../..

INCL=-I$(HOME)/code/blitz-0.10/
LIBS=-L$(HOME)/code/blitz-0.10/lib/libblitz.la -lm

CXX = g++
#CXXFLAGS = -funroll-loops -fopenmp -O3 -mtune=core2 -march=core2 -m64 -msse4.2 -g -ggdb -std=gnu++11 -ftree-vectorize -fopt-info-vec-missed-optimized=output.opt
CXXFLAGS =
LDFLAGS = 

OBJS = bench_singlepoint.o ranlxd.o

all: bench_singlepoint matmul

bench_singlepoint: $(OBJS)
	$(CXX) $(LDFLAGS) $(LIBS) $(OBJS) -o bench_singlepoint

bench_singlepoint.o: bench_singlepoint.cpp
	$(CXX) $(CXXFLAGS) $(INCL) -c bench_singlepoint.cpp

ranlxd.o: ../external/ranlxd.c ../external/ranlxd.h
	$(CXX) $(CXXFLAGS) $(INCL) -c ../external/ranlxd.c

matmul.o: matmul.cpp
	$(CXX) $(CXXFLAGS) $(INCL) -c matmul.cpp

matmul: matmul.o
	$(CXX) $(LDFLAGS) $(LIBS) matmul.o -o matmul

clean:
	rm bench_singlepoint
	rm *.o

.PHONY: all clean
