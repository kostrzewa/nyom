/**********************************************************************
 * Copyright (C) 2018 Bartosz Kostrzewa
 *
 * This file is part of nyom.
 *
 * nyom is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * nyom is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with nyom.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/

#pragma once

#include "Core.hpp"

#include <ctf.hpp>
#include <vector>
#include <sstream>

namespace nyom {

class SimpleShift
{
public:
  SimpleShift() = delete;

  SimpleShift(const nyom::Core &core_in,
              const int dim_size_in,
              const int shift_in = 0,
              const complex<double> phase_angle_in = 0.0):
    core(core_in), dim_size(dim_size_in)
  {
    init();
    set_shift(shift_in, phase_angle_in);
  }

  void set_shift(const int shift_in, const complex<double> phase_angle_in = 0.0){
    shift = shift_in;
    phase_angle = phase_angle_in;
   
    int64_t nval;
    int64_t* indices;
    complex<double>* values;
    tensor.read_local(&nval, &indices, &values);
    for(int64_t i = 0; i < nval; ++i ){
      // In Fortran fashion, the row-index runs fastest as anywhere in CTF
      int64_t c = indices[i] / dim_size;
      int64_t r = indices[i] % dim_size;
      // the resulting matrix has non-zero entries such that the translation
      //
      //  x'_i = M_ij x_j
      //
      //  results in 
      //
      //  x'_i = x_i + shift
      //
      // respecting periodic boundary conditions
      //
      // via phase_angle, we also directly apply a possible phase factor which might be
      // necessary when working with twisted boundary conditions, for example
      if( r == ( ((c+shift)+dim_size) % dim_size) ){
        values[i] = std::exp( nyom::imag_unit*phase_angle*static_cast<double>(r) );
      } else {
        values[i] = 0.0;
      }
    }
    tensor.write(nval, indices, values);
    free(values); free(indices);
  }

  CTF::Idx_Tensor operator[](const char * idx_map)
  {
    return tensor[idx_map];
  }

  CTF::Tensor< complex<double> > tensor;

private:
  const nyom::Core & core;
  const int dim_size;

  int shapes[2];
  int sizes[2];
  int shift;
  complex<double> phase_angle;

  void init()
  {
    for(int d = 0; d < 2; ++d){
      shapes[d] = NS;
      sizes[d] = dim_size;
    }
    std::stringstream name;
    name << "SimpleShift" << dim_size;
    tensor = CTF::Tensor< complex<double> >(2, sizes, shapes, core.geom.get_world(), name.str().c_str() );
  }
};

  typedef enum shift_dimension_t {
    xm1 = 0,
    xp1,
    ym1,
    yp1,
    zm1,
    zp1
  } shift_dimension_t;


  /**
   * @brief build a vector of shift matrices (2-index tensors). For shifts
   * by a single lattice site in negative or positive direction in each
   * of the three spatial dimensions as given in shift_dimension_t ordering.
   *
   * @param Nx global lattice extent in X direction
   * @param Ny global lattice extent in Y direciton
   * @param Nz global lattice extent in Z direction
   * @param world CTF::World from the currently defined geometry
   *
   * @return vector of shift tensors initialised to produce displacements
   * by a single lattice spacing 
   */
  std::vector< CTF::Tensor< complex<double> > > make_shifts(const int Nx,
                                                 const int Ny,
                                                 const int Nz,
                                                 CTF::World &world){
    const int shapes[2] = {NS, NS};
    int sizes[2];
    int dim_sizes[3] = { Nx, Ny, Nz };

    std::vector< CTF::Tensor< complex<double> > > shifts;

    sizes[0] = sizes[1] = Nx;
    shifts.emplace_back( CTF::Tensor< complex<double> >(2, sizes, shapes, world, "shift_mx") );
    shifts.emplace_back( CTF::Tensor< complex<double> >(2, sizes, shapes, world, "shift_px") );
    
    sizes[0] = sizes[1] = Ny;
    shifts.emplace_back( CTF::Tensor< complex<double> >(2, sizes, shapes, world, "shift_my") );
    shifts.emplace_back( CTF::Tensor< complex<double> >(2, sizes, shapes, world, "shift_py") );
    
    sizes[0] = sizes[1] = Nz;
    shifts.emplace_back( CTF::Tensor< complex<double> >(2, sizes, shapes, world, "shift_mz") );
    shifts.emplace_back( CTF::Tensor< complex<double> >(2, sizes, shapes, world, "shift_pz") );

    // for each direction (-+x , -+y, -+z), construct the shift
    // matrix (for shifts by a single lattice site)
    for( int dir : {0, 1, 2, 3, 4, 5} ){
      int64_t nval;
      int64_t* indices;
      complex<double>* values;
      int dim = dir / 2;
      int offset = 1;
      if( dir % 2 == 0 ) offset = -1;
      shifts[dir].read_local(&nval, &indices, &values);
      for(int64_t i = 0; i < nval; ++i ){
        // In Fortran fashion, the row-index runs fastest as anywhere in CTF
        int64_t c = indices[i] / dim_sizes[dim];
        int64_t r = indices[i] % dim_sizes[dim];
        // the resulting matrix has non-zero entries on the first upper or lower
        // off-diagonal plus entries which implement periodic boundary conditions
        if( c == ( ((r+offset)+dim_sizes[dim]) % dim_sizes[dim]) ){
          values[i] = 1.0;
        } else {
          values[i] = 0.0;
        }
      }
      shifts[dir].write(nval, indices, values);
      free(values); free(indices);
    }
    return( shifts );
  }


} // namespace(nyom)
