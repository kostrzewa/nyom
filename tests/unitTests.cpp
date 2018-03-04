
#include <gtest/gtest.h>
#include "Core.hpp"
#include "Perambulator.hpp"
#include "LapH_eigsys.hpp"
#include "One.hpp"

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

TEST_F(nyomTest, GrowTensor){
  const int r2sizes[2] = {4, 5};
  const int r2shapes[2] = {NS, NS};
  CTF::Tensor< complex<double> > rank2(2, r2sizes, r2shapes, core->geom.get_world());

  const int r3sizes[3] = {7, 4, 5};
  const int r3shapes[3] = {NS, NS, NS};
  CTF::Tensor< complex<double> > rank3(3, r3sizes, r3shapes, core->geom.get_world());

  int64_t npair;
  int64_t *indices;
  complex<double> *pairs;
  rank2.read_local(&npair, &indices, &pairs);
  for(int i = 0; i < 7; ++i){
    for(int64_t p = 0; p < npair; ++p){
      pairs[p] = complex<double>(drand48(), drand48());
    }
    rank2.write(npair, indices, pairs);

    nyom::One one = nyom::make_One(/* Nidx */        7,
                                   /* idx_nonzero */ i,
                                   core->geom.get_world());

    rank3["ijk"] += one["i"] * rank2["jk"];

    rank3.print();
  }
  free(indices);
  free(pairs);
}


int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
