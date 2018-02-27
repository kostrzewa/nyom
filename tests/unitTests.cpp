
#include <gtest/gtest.h>
#include "Core.hpp"
#include "Logfile.hpp"

TEST(InitCore, Test1)
{
  EXPECT_EQ(0,0);
}

TEST(Logfile, instantiation)
{
  nyom::logfile_t test("testfile");
  EXPECT_EQ(0,0);
}

int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
