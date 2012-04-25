/**
  Online Echo state gaussian progress.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the online echo-state gaussian process.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_OESGP_78342917490281903789643782889423719
#define OTL_OESGP_78342917490281903789643782889423719

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <istream>
#include <ostream>
#include <vector>
#include <fstream>

#include "otl_reservoir.h"
#include "otl_sogp.h"

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

class OESGP {
public:
    /**
      \brief Sets up the reservoir
      \param input_dim the input dimension
      \param output_dim the output dimension
      \param reservoir_size how big is your reservoir?
      \param input_weight How much to weight your inputs (typically 1)
      \param output_feedback_weight Set this >0 if you want output feedback
                with the specified weighting
      \param activation function 0 for TANH, 1 for Linear
      \param leak_rate the leak rate (between 0 and 1.0) of the reservoir
                (depends on your application)
      \param connectivity connectivity of the reservoir (between 0 and 1.0).
                Typically small e.g., 0.01 or 0.1
      \param spectral_radius the spectral radius of the reservoir. This should
                be < 1.0 but you can set it higher if you want.
      \param use_inputs_in_state do we want to use the inputs directly in
                the state vector?
      \param kernel_parameters kernel parameters for the gaussian kernel
      \param noise noise for SOGP
      \param epsilon threshold for SOGP
      \param capacity the capacity for the SOGP
      \param random_seed the random seed to initialise the reservoir. The same
                random seed will generate the same reservoir.
      **/
    void init(
            unsigned int input_dim,
            unsigned int output_dim,
            unsigned int reservoir_size,
            double input_weight,
            double output_feedback_weight,
            int activation_function,
            double leak_rate,
            double connectivity,
            double spectral_radius,
            bool use_inputs_in_state,
            VectorXd &kernel_parameters,
            double noise,
            double epsilon,
            unsigned int capacity,
            unsigned int random_seed
            );
    /**
      \brief updates the reservoir with input
      \param input the input (a VectorXd)
      */
    virtual void update(const VectorXd &input);

    /**
      \brief trains the OESGP given the current state and the output
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
      \brief updates the reservoir with the given input and output_feedback
      \param input the input
      \param output_feedback the output to feedback into the reservoir
      */
    virtual void update(const VectorXd &input, const VectorXd &output_feedback);

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
      \brief computes the spectral radius of the reservoir (NOT necessarily
      the DESIRED spectral radius)
      **/
    double getActualSpectralRadius(void);

    /**
      \brief sets the reservoirs input weights
      \param input_weights the input weights. size should be
      reservoir size x input dimension
      */
    void setInputWeights(MatrixXd &input_weights);

    /**
      \brief sets the reservoirs output feedback weights
      \param output_feedback_weights the output feedback weights. size should be
      reservoir size x output dimension
      */
    void setOutputFeedbackWeights(MatrixXd &output_feedback_weights);

    /**
      \brief sets the reservoirs weights
      \param reservoir_weights the reservoir weights. size should be
      reservoir size x output dimension
      */
    void setReservoirWeights(MatrixXd &reservoir_weights);

    /**
      \brief gets the current number of basis vectors
      \returns unsigned int (the number of basis vectors)
      */
    unsigned int getCurrentSize(void);

private:
    Reservoir res;
    GaussianKernel gaussian_kernel;
    SOGP sogp;
};


}

#endif
