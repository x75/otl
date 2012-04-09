/**
  Spatial-temporal online recursive kernel gaussian progress.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the STORKGP algorithm.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_STORKGP_321809753827490718329658392704893241
#define OTL_STORKGP_321809753827490718329658392704893241

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <istream>
#include <ostream>
#include <vector>
#include <fstream>

#include "otl_window.h"
#include "otl_sogp.h"
#include "otl_kernel_recursive_equality_gaussian.h"
#include "otl_kernel_recursive_gaussian.h"

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

class STORKGP {
public:
    /**
        types of kernels
       **/
    enum {
        RECURSIVE_GAUSSIAN = 0,
        RECURSIVE_EQUALITY_GAUSSIAN = 1
    };

    /**
      \brief Sets up the reservoir
      \param input_dim the input dimension
      \param output_dim the output dimension
      \param tau window size for approximation
      \param kernel_type the type of kernel (enum)
      \param kernel_parameters kernel parameters for the recursive kernel
      \param noise noise for SOGP
      \param epsilon threshold for SOGP
      \param capacity the capacity for the SOGP
      **/
    void init(
            unsigned int input_dim,
            unsigned int output_dim,
            unsigned int tau,
            int kernel_type,
            VectorXd &kernel_parameters,
            double noise,
            double epsilon,
            unsigned int capacity
            );
    /**
      \brief updates STORKGP window with input
      \param input the input (a VectorXd)
      */
    virtual void update(const VectorXd &input);

    /**
      \brief trains the STORKGP given the current state and the output
      \param output (a VectorXd)
      */
    virtual void train(const VectorXd &output);

    /**
      \brief make a prediction and prediction variance.
      \param prediction the output prediction (a VectorXd)
      \param prediction_variance the output prediction variance (a VectorXd)
      */
    virtual void predict(VectorXd &prediction, VectorXd &prediction_variance);

    /**
      \brief resets the state of the reservoir
      */
    virtual void resetState();

    /**
      \brief resets the SOGP algorthm
      */
    virtual void resetSOGP();

    /**
      \brief gets the state of the reservoir
      */
    virtual void getState(VectorXd &state);

    /**
      \brief sets the state of the reservoir
      */
    virtual void setState(const VectorXd &state);

    /**
      \brief resets the reservoir
      */
    virtual void reset(void);

    /**
        \brief Saves the reservoir to a file
        \param filename a string specifying the filename to save to
    **/
    virtual void save(std::string filename);

    /**
        \brief loads the model from disk
        \param filename where to load from?
    **/
    virtual void load(std::string filename);

    /**
      \brief returns the reservoir size
      **/
    virtual unsigned int getStateSize(void);

    /**
      \brief gets the current number of basis vectors
      \returns unsigned int (the number of basis vectors)
      */
    unsigned int getCurrentSize(void);

private:
    Window window;
    RecursiveGaussianKernel rec_gaussian_kernel;
    RecursiveEqualityGaussianKernel rec_equality_gaussian_kernel;
    SOGP sogp;
};


}

#endif
