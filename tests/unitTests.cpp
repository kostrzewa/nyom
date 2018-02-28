
#include <gtest/gtest.h>
#include "Core.hpp"
#include "Perambulator.hpp"
#include "LapH_eigsys.hpp"

// test fixture which sets up nyom::Core
class nyomTest : public ::testing::Test {
  protected:
    static void SetUpTestCase() {
      core = new nyom::Core(1,NULL);
    }

    static void TearDownTestCase() {
      delete core;
      core = NULL;
    }

  public:
    static nyom::Core* core;
};

nyom::Core* nyomTest::core = NULL;

TEST_F(nyomTest, Perambulator_Instantiation){
  nyom::Perambulator peram = make_Perambulator(100, 10, 10, *(nyomTest::core) );
}

TEST_F(nyomTest, LapH_eigsys_Instantiation){
  nyom::LapH_eigsys eigsys = nyom::make_LapH_eigsys(/*Nev*/ 100, 
                                                    /*Nt*/ 10,
                                                    /*Nx*/ 10,
                                                    /*Ny*/ 10,
                                                    /*Nz*/ 10,
                                                    /*Nc*/ 3,
                                                    (*core).geom.get_world());
}

TEST_F(nyomTest, LapH_eigsys_reading){
  nyom::LapH_eigsys eigsys = nyom::make_LapH_eigsys(/*Nev*/ 48, 
                                                    /*Nt*/ 24,
                                                    /*Nx*/ 12,
                                                    /*Ny*/ 12,
                                                    /*Nz*/ 12,
                                                    /*Nc*/ 3,
                                                    (*core).geom.get_world());
  nyom::read_LapH_eigsys_from_files(eigsys,
                                    "ev",
                                    400,
                                    *core);

}

int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
