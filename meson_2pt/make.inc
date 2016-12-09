top_srcdir = ..
INCL+=-I${top_srcdir}/include/
EXT_C_MODULES=
MODULES=
PROGRAMS=meson_2pt

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
