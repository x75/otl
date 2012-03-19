#ifndef OTL_HELPERS_H_32718964589163489217398127489164891734
#define OTL_HELPERS_H_32718964589163489217398127489164891734


#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <iostream>
#include <exception>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

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



void readVectorFromStream(std::istream &in, VectorXd &A) {
    if (!in) {
        throw OTLException("Whoops. Your in stream seems to be wrong");
    }

    unsigned int m;
    in >> m;
    A = VectorXd(m);

    for (unsigned int i=0; i<A.rows(); i++) {
        double temp;
        try {
            in >> temp;
        } catch (std::exception &e) {
            throw OTLException("Error during read of stream");
        }

        A(i) = temp;
    }
}


}

#endif
