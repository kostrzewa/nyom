add_executable(lib_includes
  lib_includes/lib_includes.cpp)

add_executable(displacement
  displacement/displacement.cpp)

target_link_libraries(lib_includes PUBLIC nyom)
target_link_libraries(displacement PUBLIC nyom)
