/**
  OTL Helper functions.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  This class implements a few functions that assist in the main workings.
  Right now, implements saving and loading Eigen VectorXd and MatrixXd objects
  to/from streams.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_HELPERS_H_32718964589163489217398127489164891734
#define OTL_HELPERS_H_32718964589163489217398127489164891734


#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include <exception>
#include "otl_kernel_factory.h"

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
  \brief Writes a MatrixXd object to a stream
  \param out a valid output ostream
  \param A the Matrix you want to write
  */
void saveMatrixToStream(std::ostream &out, MatrixXd &A);

/**
  \brief Reads a MatrixXd object from a stream
  \param in a valid input istream
  \param A where to put the loaded information
  */
void readMatrixFromStream(std::istream &in, MatrixXd &A);

/**
  \brief Writes a VectorXd object to a stream
  \param out a valid output ostream
  \param A the VectorXd you want to write
  */
void saveVectorToStream(std::ostream &out, VectorXd &A) ;

/**
  \brief Reads a VectorXd object from a stream
  \param in a valid input istream
  \param A where to put the loaded information
  */
void readVectorFromStream(std::istream &in, VectorXd &A);

/**
  \brief initialises the kernel factory with default kernels
  **/
void initKernelFactory(KernelFactory &kfact);

/**
  \brief computes the standard normal cumulative distribution function at z
  **/
double stdnormcdf(double x);

}

#endif
