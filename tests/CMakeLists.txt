## this setup comes from
## https://stackoverflow.com/a/27160614

include(ExternalProject)
ExternalProject_Add(gtest
  GIT_REPOSITORY https://github.com/google/googletest

  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
  INSTALL_COMMAND ""
)
ExternalProject_Get_Property(gtest source_dir binary_dir)

message( STATUS "GTEST binary_dir: " ${binary_dir})
message( STATUS "GETST source_Dir: " ${source_dir})

add_executable(unitTests
  unitTests.cpp)

add_dependencies(unitTests gtest)
include_directories(${source_dir}/include)
target_link_libraries(unitTests ${binary_dir}/googlemock/gtest/libgtest.a)
target_link_libraries(unitTests ${binary_dir}/googlemock/gtest/libgtest_main.a)

add_test(
  NAME unitTests
  COMMAND unitTests
)
