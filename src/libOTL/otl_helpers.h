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

#ifdef __GNUG__
#  include <cxxabi.h>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

/**
  \brief Writes a MatrixXd object to a stream
  \param out a valid output ostream
  \param A the Matrix you want to write
  */
void saveMatrixToStream(std::ostream &out, MatrixXd &A) {

    if (!out) {
        throw OTLException("Whoops. Your stream seems to be wrong");
    }
    out.precision(15);
    out << A.rows() << std::endl << A.cols() << std::endl;
    for (unsigned int i=0; i<A.rows(); i++) {
        for (unsigned int j=0; j<A.cols(); j++) {
            try {
                out << A(i,j) << " ";
            } catch (std::exception &e) {
                throw OTLException("Error during read of stream");
            }
        }
        out << "\n";
    }

}

/**
  \brief Reads a MatrixXd object from a stream
  \param in a valid input istream
  \param A where to put the loaded information
  */
void readMatrixFromStream(std::istream &in, MatrixXd &A) {
    if (!in) {
        throw OTLException("Whoops. Your in stream seems to be wrong");
    }

    try {
        unsigned int m,n;
        in >> m >> n;
        A = MatrixXd(m,n);

        for (unsigned int i=0; i<A.rows(); i++) {
            for (unsigned int j=0; j<A.cols(); j++) {
                double temp;
                in >> temp;
                A(i,j) = temp;
            }
        }
    } catch (std::exception &e) {
        throw OTLException("Error during read of stream");
    }
}


/**
  \brief Writes a VectorXd object to a stream
  \param out a valid output ostream
  \param A the VectorXd you want to write
  */
void saveVectorToStream(std::ostream &out, VectorXd &A) {
    if (!out) {
        throw OTLException("Whoops. Your stream seems to be wrong");
    }
    out.precision(15);
    out << A.rows() << std::endl;
    for (unsigned int i=0; i<A.rows(); i++) {
        try {
            out << A(i) << std::endl;
        } catch (std::exception &e) {
            throw OTLException("Error during read of stream");
        }
    }
    out << "\n";
}


/**
  \brief Reads a VectorXd object from a stream
  \param in a valid input istream
  \param A where to put the loaded information
  */
void readVectorFromStream(std::istream &in, VectorXd &A) {
    if (!in) {
        throw OTLException("Whoops. Your in stream seems to be wrong");
    }

    unsigned int m;
    in >> m;
    A.resize(m);
    for (unsigned int i=0; i<m; i++) {
        double temp;
        try {
            in >> temp;
        } catch (std::exception &e) {
            throw OTLException("Error during read of stream");
        }

        A(i) = temp;
    }
}

// Based on Stack Overflow answer by Ali
// http://stackoverflow.com/a/4541470/935415
std::string demangle(const char* name)
{
   int status = -4;
   char * demangledName = NULL;

#ifdef __GNUC__
   demangledName = abi::__cxa_demangle(name, NULL, NULL, &status);
#endif

   std::string result = (status == 0) ? demangledName : name;

   if ( demangledName != NULL)
       free(demangledName);

   return result;
}


}

#endif
