#pragma once

#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>

#include <hdf5.h>
#include <cstring>
#include <cstdio>
#include <string>
#include <iostream>
#include <string>
#include <list>

#include <Core.hpp>
#include <Logger.hpp>

namespace nyom {
namespace h5 {

  static inline std::string path_list_to_key(const std::list<std::string> & path_list)
  {
    std::string key;
    for( auto const & subpath : path_list ){
      key += "/" + subpath;
    }
    return(key);
  }

  static inline std::string recursive_path_create(
      const nyom::Core & core,
      HighFive::File & file,
      const std::list<std::string> & path_list,
      const bool is_dataset_path = true)
  {
    const int rank = core.geom.get_myrank();
    
    if( rank != 0 ){
      std::stringstream errmsg;
      errmsg << __func__ << " should only be called from rank 0!" << std::endl;
      throw std::runtime_error( errmsg.str() );
    }
    
    MPI_Comm comm = core.geom.get_nyom_comm();
    nyom::Logger verbose4_logger0(core, 0, 4);

    std::string path;
    for( auto const & subpath : path_list ){
      path += "/" + subpath;
      // up until the last element, these are groups
      // the final element is instead the name of the dataset
      // and we will not create a group with this name
      auto end_elem = path_list.end();
      if( is_dataset_path ){
        end_elem = (--path_list.end());
      }
      if( !file.exist(path) && subpath != *(end_elem) ){
        verbose4_logger0 << "# [nyom::h5::recursive_path_create] Creating H5 path" << path << std::endl;
        file.createGroup(path);
      }
    }
    return path;
  }

  /**
   * @brief Write a dataset to file from rank 0
   *
   * @param file
   * @param path_list
   * @param local_data
   */
  static inline void write_dataset(
      const nyom::Core & core,
      const std::string & filename, 
      const std::list<std::string> & path_list,
      CTF::Tensor< std::complex<double> > & tensor)
  {
    const int rank = core.geom.get_myrank();
    MPI_Comm comm = core.geom.get_nyom_comm();
 
    std::complex<double>* values;
    int64_t nval;
    tensor.read_all(&nval, &values);
    std::vector<std::complex<double>> vec(values, values+nval);
    free(values);

    if( rank == 0 )
    {
      HighFive::File file(filename, HighFive::File::OpenOrCreate); 
    
      // create the path hierarchy all the way up until the dataset name
      std::string path = recursive_path_create(core, file, path_list, true);

      // we check if the dataset path exists and if it does not, we create it
      // and we attempt to write the dataset, if it already exsists it should
      // just be overwritten, as long as the buffer has the same size
      if( !file.exist(path_list_to_key(path_list)) ){
        HighFive::DataSet dataset = file.createDataSet<std::complex<double>>(path, HighFive::DataSpace::From(vec));
        dataset.write(vec);
      } else {
        HighFive::DataSet dataset = file.getDataSet(path);
        dataset.write(vec);
      }
      file.flush();
    }
  }

  #define H5UTILS_MAX_KEY_LENGTH 500
  
  /**
   * @brief check if a particular group / dataset exists
   * Traverses the h5 tree and checks if the requested group
   * hierarchy exists.
   *
   * @param loc_id HDF5 File or group id below which the hierarchy
   * should be traversed
   * @param key Group / Dataset key of the form "/grp1/grp2/grp3/dataset1"
   * @param fail_path First link path of searched hierarchy which
   *                  could not be found. Note pass as reference.
   * @param full_path Pass 'true' if the key refers to a complete
   *                  path with a leading forward slash.
   *
   * @return true if path was found, false if it was not. Path of
   *         failure is returned vial fail_path argument.
   */
  static inline bool check_key_exists(hid_t loc_id,
                                      char const * const key,
                                      std::string & fail_path,
                                      bool full_path = true)
  {
    if( full_path ){
      char key_copy[H5UTILS_MAX_KEY_LENGTH];
      if( strlen(key) < H5UTILS_MAX_KEY_LENGTH ){
        strcpy( key_copy, key );
      } else {
        std::stringstream msg;
        msg << "[nyom::h5::check_key_exists] Length of key exceeds " << H5UTILS_MAX_KEY_LENGTH 
          << " characters. File: " << __FILE__ << " Line: " << __LINE__ << "\n";
        throw std::runtime_error(msg.str());
      }
  
      htri_t status;
      std::string curr_path("");
  
      curr_path = "/";
      status = H5Lexists(loc_id, curr_path.c_str(), H5P_DEFAULT);
      if( status <= 0 ){
        fail_path = "/";
        return false;
      }
  
      const char grp_sep[] = "/";
      char * curr_grp_substr = strtok( key_copy, grp_sep );
      while( curr_grp_substr != NULL ){
        curr_path += std::string(curr_grp_substr);
        status = H5Lexists(loc_id, curr_path.c_str(), H5P_DEFAULT);
        if( status <= 0 ){
          fail_path = curr_path;
          return false;
        }
        curr_grp_substr = strtok( NULL, grp_sep );
        curr_path += std::string("/");
      }
      // traversal was successful
      fail_path = "";
      return true;
    } else {
      if( H5Lexists(loc_id, key, H5P_DEFAULT) <= 0 ){
        fail_path = std::string(key);
        return false;
      } else {
        fail_path = "";
        return true;
      }
    } // if(full_path)
  }

  std::list<std::string> make_os_meson_2pt_path_list(
      const std::string g_src, 
      const std::string g_snk,
      const std::string fwd_flav,
      const std::string bwd_flav,
      const int src_ts,
      const std::array<int, 3> src_p = {0,0,0},
      const std::array<int, 3> snk_p = {0,0,0})
  {
    std::list<std::string> path_list;
    char subpath[100];

    snprintf(subpath, 100, "%s+-g-%s-g", bwd_flav.c_str(), fwd_flav.c_str());
    path_list.push_back(subpath);
    
    snprintf(subpath, 100, "t%d", src_ts);
    path_list.push_back(subpath);

    snprintf(subpath, 100, "gf%s", g_snk.c_str());
    path_list.push_back(subpath);

    snprintf(subpath, 100, "pfx%dpfy%dpfz%d", snk_p[0], snk_p[1], snk_p[2]);
    path_list.push_back(subpath);
    
    snprintf(subpath, 100, "gi%s", g_src.c_str());
    path_list.push_back(subpath);

    snprintf(subpath, 100, "pix%dpiy%dpiz%d", src_p[0], src_p[1], src_p[2]);
    path_list.push_back(subpath);

    return path_list;
  }

} // namespace(h5)
} // namespace(nyom)
