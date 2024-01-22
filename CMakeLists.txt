cmake_minimum_required (VERSION 3.0)
project (nyom)

enable_language(CXX)
enable_language(C)
enable_language(Fortran)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

Include(FetchContent)

# we will collect any libraries that we want to link against to compile
# main programs in this list which we grow below with list(APPEND nyom_libs [...])
set(nyom_libs)
# any link directories hiere
set(nyom_link_dirs)
# and any includes directories here
set(nyom_include_dirs)

set( MPI_HOME CACHE PATH "path to MPI library installation directory" )
if( MPI_HOME )
  list(APPEND nyom_libs "-lmpi")
  list(APPEND nyom_link_dirs ${MPI_HOME}/lib)
  list(APPEND nyom_include_dirs ${MPI_HOME}/include)
else()
  find_package(MPI)
  if( MPI_C_FOUND )
    list(APPEND nyom_libs ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    list(APPEND nyom_include_dirs ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
  else()
    message( FATAL_ERROR "MPI could not be found automatically and MPI_HOME has not been specified!" )
  endif()
endif()

set(CTF_HOME CACHE PATH "path to configured CTF source or build directory")
if( CTF_HOME )
  list(APPEND nyom_libs "-lctf")
  list(APPEND nyom_link_dirs ${CTF_HOME}/lib)
  list(APPEND nyom_include_dirs ${CTF_HOME}/include) 
else()
  message( FATAL_ERROR "CTF_HOME must be defined!" )
endif()

#set( SCALAPACK_HOME CACHE PATH "path to SCALAPACK installation" )
#if( SCALAPACK_HOME )
#  link_directories(${SCALAPACK_HOME}/lib)
#  ## FIXME: we link -lgfortran but really should have some general solution for any fortran std lib...
#  link_libraries(-lscalapack -lgfortran)
#else()
#  message( FATAL_ERROR "SCALAPACK_HOME must be specified!" )
#endif()
#
#set( HPTT_HOME CACHE PATH "path to HPTT library installation directory" )
#if( HPTT_HOME )
#  include_directories(${HPTT_HOME}/include)
#  link_directories(${HPTT_HOME}/lib)
#  link_libraries(-lhptt)
#else()
#  message( FATAL_ERROR "HPTT_HOME must be defined!" )
#endif()

set( TMLQCD_SRC CACHE PATH "path to tmLQCD source directory" )
if( TMLQCD_SRC )
  list(APPEND nyom_include_dirs ${TMLQCD_SRC})
  list(APPEND nyom_include_dirs ${TMLQCD_SRC}/include )
else()
  message(FATAL_ERROR "TMLQCD_SRC must be defined!" )
endif()

set( TMLQCD_BUILD CACHE PATH "path to tmLQCD build directory (must be build directory for source in TMLQCD_SRC!)" )
if( TMLQCD_BUILD )
  ## # explicit full paths to generate dependenciec such that targets are rebuilt
  ## # if something changes on the tmLQCD side
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libwrapper.a)
  # circular dependencies, need to specify libmonomial twice... 
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libmonomial.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/librational.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libhmc.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libmonomial.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libmeas.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/liboperator.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libinit.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libxchange.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libio.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libbuffers.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/libsolver.a)
  list(APPEND nyom_libs ${TMLQCD_BUILD}/lib/liblinalg.a)
  list(APPEND nyom_libs -lm)
  list(APPEND nyom_link_dirs ${TMLQCD_BUILD}/lib)
  list(APPEND nyom_include_dirs ${TMLQCD_BUILD})
  list(APPEND nyom_include_dirs ${TMLQCD_BUILD}/include)
else()
  message(FATAL_ERROR "TMLQCD_BUILD must be defined!" )
endif()


set( LIME_HOME CACHE PATH "path to the c-lime library installation" )
if( LIME_HOME )
  list(APPEND nyom_libs "-llime")
  list(APPEND nyom_link_dirs ${LIME_HOME}/lib)
  list(APPEND nyom_include_dirs ${LIME_HOME}/include)
else()
  message(FATAL_ERROR "LIME_HOME must be defined!" )
endif()

set( LEMON_HOME CACHE PATH "path to the LEMON library installation (optional, depends on how tmLQCD was configured)" )
if( LEMON_HOME )
  list(APPEND nyom_libs "-llemon")
  list(APPEND nyom_link_dirs ${LEMON_HOME}/lib)
  list(APPEND nyom_include_dirs ${LEMON_HOME}/include)
endif()

set( QMP_HOME CACHE PATH "path to the QMP library required by QPhiX (optional, depends on how tmLQCD was configured)" )
if( QMP_HOME )
  list(APPEND nyom_libs "-lqmp")
  list(APPEND nyom_link_dirs ${QMP_HOME}/lib)
  list(APPEND nyom_include_dirs ${QMP_HOME}/include)
endif()

set( QPHIX_HOME CACHE PATH "path to the QPhiX library installation (optional, depends on how tmLQCD was configured)" )
if( QPHIX_HOME )
  ## this is a really ugly hack to make sure that all symbols are found, bit it works!
  set( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${TMLQCD_BUILD}/qphix_interface.o" )
  
  list(APPEND nyom_libs "-lqphix_codegen -lqphix_solver")
  list(APPEND nyom_include_dirs ${QPHIX_HOME}/include)
  list(APPEND nyom_link_dirs ${QPHIX_HOME}/lib) 
endif()

set( DD_ALPHA_AMG_HOME CACHE PATH "path to the DDalphaAMG library installation (optional, depends on how tmLQCD was configured)" )
if( DD_ALPHA_AMG_HOME )
  list(APPEND nyom_libs "-lDDalphaAMG")
  list(APPEND nyom_include_dirs ${DD_ALPHA_AMG_HOME}/include)
  list(APPEND nyom_link_dirs ${DD_ALPHA_AMG_HOME}/lib)
endif()

set( QUDA_HOME CACHE PATH "path to QUDA installation (optional, depends on whether tmLQCD uses QUDA sovers)" ) 
if( QUDA_HOME )
  list(APPEND nyom_libs "-lquda")
  list(APPEND nyom_link_dirs ${QUDA_HOME}/lib)
  list(APPEND nyom_include_dirs ${QUDA_HOME}/include)

  find_package(CUDA)
  if( CUDA_FOUND )
    # QUDA required -lcuda for cuMemAlloc in addition to -lcudart
    list(APPEND nyom_libs ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} -lcuda )
    list(APPEND nyom_include_dirs ${CUDA_INCLUDE_DIRS} )
    list(APPEND nyom_link_dirs ${CUDA_LIBRARY_DIRS} )
  else()
    message( FATAL_ERROR "Unable to find CUDA!")
  endif()

endif()

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        yaml-cpp-0.7.0
)
# MakeAvailable does not work for yaml-cpp
FetchContent_MakeAvailable(yaml-cpp)
FetchContent_GetProperties(yaml-cpp)
if(NOT yaml-cpp_POPULATED)
  FetchContent_Populate(yaml-cpp)
  add_subdirectory(${yaml-cpp_SOURCE_DIR} ${yaml-cpp_BINARY_DIR})
endif()
list(APPEND nyom_libs yaml-cpp::yaml-cpp)

find_package(LAPACK REQUIRED)
if( LAPACK_FOUND )
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LAPACK_LINKER_FLAGS}")
  list(APPEND nyom_libs "${LAPACK_LIBRARIES}")
else()
  message( FATAL_ERROR "Unable to find LAPACK!" )
endif()

find_package(BLAS REQUIRED)
if( BLAS_FOUND )
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${BLAS_LINKER_FLAGS}")
  list(APPEND nyom_libs "${BLAS_LIBRARIES}")
else()
  message( FATAL_ERROR "Unable to find BLAS!" )
endif()

find_package(OpenMP REQUIRED)
if ( OPENMP_FOUND )
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  list(APPEND nyom_libs "${OPENMP_LIBRARIES}")
else()
  message( FATAL_ERROR "Unable to find OpenMP!" )
endif()

find_package(Boost 
             REQUIRED
             program_options
             filesystem)
if( Boost_FOUND )
  list(APPEND nyom_libs "${Boost_LIBRARIES}")
  list(APPEND nyom_include_dirs ${Boost_INCLUDE_DIRS} )
  list(APPEND nyom_link_dirs ${Boost_LIBRARY_DIRS} )
else()
  message( FATAL_ERROR "Unable to find Boost (or one of the requested components, see CMakeLists.txt!")
endif()

set(HDF5_USE_STATIC_LIBRARIES)    
find_package(HDF5)    
if(HDF5_FOUND)    
  add_definitions(-DHAVE_HDF5)    
  list(APPEND nyom_libs ${HDF5_LIBRARIES})
  list(APPEND nyom_include_dirs ${HDF5_INCLUDE_DIRS} )
  message(STATUS "HDF5 found")
  message(STATUS "  HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS}")
  message(STATUS "  HDF5_LIBRARIES=${HDF5_LIBRARIES}")
else()    
  message(FATAL_ERROR "HDF5 could not be found!")    
endif()    


FetchContent_Declare(
  HighFive
  GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
  GIT_TAG        v2.8.0
  OVERRIDE_FIND_PACKAGE
)
FetchContent_MakeAvailable(HighFive)
FetchContent_GetProperties(HighFive)
if(NOT highfive_POPULATED)
  FetchContent_Populate(HighFive)
endif()
#include(${highfive_SOURCE_DIR}/CMake/HighFiveConfig.cmake)

message(STATUS "nyom_libs: ${nyom_libs}")

# base package includes and libraries
include_directories(include)

add_subdirectory(lib)
add_subdirectory(peram_gen)
add_subdirectory(meson_2pt)
add_subdirectory(baryon_2pt)
add_subdirectory(heavy_2pt)
add_subdirectory(tests)
add_subdirectory(manual_tests)

## disable by default, belongs to a currently defunct project
#add_subdirectory(fr_bsm3_2pt)

