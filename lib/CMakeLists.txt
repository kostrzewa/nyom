add_library(nyom
  gammas.cpp)

target_include_directories(nyom PUBLIC ${yaml-cpp_SOURCE_DIR}/include ${nyom_include_dirs})
