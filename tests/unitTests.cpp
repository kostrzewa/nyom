
#include <gtest/gtest.h>
#include "Core.hpp"
#include "Logfile.hpp"

TEST(Core, Instantiation)
{
  EXPECT_EQ(0,0);
}

int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);
  nyom::Core core(argc,argv);

  core.Logger("hatchepsut",
              nyom::log_perf,
              "This thing is really fast!!!");

  return RUN_ALL_TESTS();
}
