## this setup comes from
## https://stackoverflow.com/a/27160614

Include(FetchContent)                                                                                                                      
FetchContent_Declare(    
  googletest    
  GIT_REPOSITORY https://github.com/google/googletest.git    
  GIT_TAG        v1.13.0    
)    
FetchContent_MakeAvailable(googletest)    

enable_testing()

add_executable(unitTests
  unitTests.cpp)

#add_dependencies(unitTests googletest)

target_link_libraries(unitTests PUBLIC
  GTest::gtest_main
  nyom)

add_test(
  NAME unitTests
  COMMAND unitTests
)
