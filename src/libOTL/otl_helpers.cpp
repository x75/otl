#include "otl_helpers.h"

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

/**
  \brief initialises the kernel factory.
  Change this function if you want to add more kernel types to the default
  initialisation
  **/
void initKernelFactory(KernelFactory &kfact) {
    GaussianKernel gk;
    RecursiveGaussianKernel rgk;

    kfact.set(gk.getName(), &gk);
    kfact.set(rgk.getName(), &rgk);
}
}
