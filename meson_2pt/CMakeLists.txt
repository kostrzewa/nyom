add_executable(meson_2pt
  meson_2pt.cpp)
target_link_libraries(meson_2pt PUBLIC nyom)
target_link_libraries(meson_2pt PUBLIC ${nyom_libs})
target_link_directories(meson_2pt PUBLIC ${nyom_link_dirs})
target_include_directories(meson_2pt PUBLIC ${yaml-cpp_SOURCE_DIR}/include ${nyom_include_dirs})
