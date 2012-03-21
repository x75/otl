/**
  OTL Reservoir Class
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the basic reservoir proposed by Jaeger.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_RESERVOIR_H_893201789576921893812908490127348470398
#define OTL_RESERVOIR_H_893201789576921893812908490127348470398

#include "otl_aug_state.h"
#include "otl_helpers.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>


namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Reservoir : public AugmentedState {
public:
    /**
        types of activation functions
        we only support TANH and LINEAR for now
       **/
    enum {
        TANH = 0,
        LINEAR = 1
    };

    Reservoir(void);


    /**
      \brief Copy constructor
      */
    Reservoir(Reservoir &rhs);

    /**
      \brief updates the reservoir with input
      \param input the input (a VectorXd)
      */
    virtual void update(const VectorXd &input);

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
        \brief loads the reservoir from a file
        \param filename where to load from?
    **/
    virtual void load(std::string filename);

    /**
      \brief returns the reservoir size
      **/
    virtual unsigned int getStateSize(void);

    /**
      \brief Sets up the reservoir
      \param input_dim the input dimension
      \param output_dim the output dimension
      \param reservoir_size how big is your reservoir?
      \param input_weight How much to weight your inputs (typically 1)
      \param output_feedback_weight Set this >0 if you want output feedback
                with the specified weighting
      \param activation function only OTLParams::TANH is supported for now.
      \param leak_rate the leak rate (between 0 and 1.0) of the reservoir
                (depends on your application)
      \param connectivity connectivity of the reservoir (between 0 and 1.0).
                Typically small e.g., 0.01 or 0.1
      \param spectral_radius the spectral radius of the reservoir. This should
                be < 1.0 but you can set it higher if you want.
      \param use_inputs_in_state do we want to use the inputs directly in
                the state vector?
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
            unsigned int random_seed
            );


    /*---------------------------------------------------------------------
            ADDITIONAL FUNCTIONS SPECIFIC TO RESERVOIRS
      ---------------------------------------------------------------------*/
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

private:
    bool initialized;

    //common parameters
    unsigned int input_dim;
    unsigned int output_dim;

    //Reservoir parameters
    unsigned int reservoir_size;
    double input_weight;
    double output_feedback_weight;
    unsigned int activation_function;
    double leak_rate;
    double connectivity;
    double spectral_radius;
    unsigned int random_seed;
    bool use_inputs_in_state;

    //Reservoir
    /** the input weights **/
    MatrixXd input_weights;

    /** the output feedback weights **/
    MatrixXd output_feedback_weights;

    /** the neuron reservoir weights **/
    MatrixXd reservoir_weights;

    /** the current reservoir state  **/
    VectorXd curr_reservoir_state;

    /** augmented reservoir state **/
    VectorXd state;
    unsigned int state_size;

    void activation(VectorXd &inputs, VectorXd &results);
    void linear_activation(VectorXd &inputs, VectorXd &results);
    void tanh_activation(VectorXd &inputs, VectorXd &results);
};


}
#endif
