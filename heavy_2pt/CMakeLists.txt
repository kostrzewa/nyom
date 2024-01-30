add_executable(heavy_2pt
  heavy_2pt.cpp)
target_link_libraries(heavy_2pt PUBLIC nyom)
target_link_libraries(heavy_2pt PUBLIC ${nyom_libs})
target_link_directories(heavy_2pt PUBLIC ${nyom_link_dirs})
target_include_directories(heavy_2pt PUBLIC ${yaml-cpp_SOURCE_DIR}/include ${nyom_include_dirs})
target_link_libraries(heavy_2pt PUBLIC HighFive)

