#include "otl_oesgp.h"


namespace OTL {

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
  \param kernel_parameters
  \param random_seed the random seed to initialise the reservoir. The same
            random seed will generate the same reservoir.
  **/
void OESGP::init(
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
        )
{
    res.init(input_dim, output_dim, reservoir_size,
             input_weight, output_feedback_weight,
             activation_function, leak_rate,
             connectivity, spectral_radius,
             use_inputs_in_state, random_seed);

    gaussian_kernel.init(res.getStateSize(), kernel_parameters);

    sogp.init(res.getStateSize(), output_dim, gaussian_kernel,
              noise, epsilon, capacity);
}

/**
  \brief updates the reservoir with input
  \param input the input (a VectorXd)
  */
void OESGP::update(const VectorXd &input) {
    res.update(input);
}

/**
  \brief updates the reservoir with the given input and output_feedback
  \param input the input
  \param output_feedback the output to feedback into the reservoir
  */
void OESGP::update(const VectorXd &input, const VectorXd &output_feedback) {
    res.update(input, output_feedback);
}

/**
  \brief trains the OESGP given the current state and the output
  \param output (a VectorXd)
  */
void OESGP::train(const VectorXd &output) {
    VectorXd state;
    res.getState(state);
    sogp.train(state, output);
}

/**
  \brief make a prediction and prediction variance.
  \param prediction the output prediction (a VectorXd)
  \param prediction_variance the output prediction variance (a VectorXd)
  */
void OESGP::predict(VectorXd &prediction, VectorXd &prediction_variance){
    VectorXd state;
    res.getState(state);
    sogp.predict(state, prediction, prediction_variance);
}

/**
  \brief resets the state of the reservoir
  */
void OESGP::resetState() {
    res.reset();
}

/**
  \brief resets the SOGP algorthm
  */
void OESGP::resetSOGP() {
    sogp.reset();
}


/**
  \brief gets the state of the reservoir
  */
void OESGP::getState(VectorXd &state) {
    res.getState(state);
}

/**
  \brief sets the state of the reservoir
  */
void OESGP::setState(const VectorXd &state) {
    res.setState(state);
}

/**
  \brief resets the reservoir
  */
void OESGP::reset(void) {
    res.reset();
    sogp.reset();
}

/**
    \brief Saves the OESGP. this saves two files. a [filename].model file (for the sogp)
    and a [filename].features file to save the reservoir
    \param filename a string specifying the filename to save to
**/
void OESGP::save(std::string filename) {
    std::string model_filename = filename + ".model";
    std::string feature_filename = filename + ".features";
    sogp.save(model_filename);
    res.save(feature_filename);
}

/**
    \brief loads the model from disk
    \param filename where to load from?
**/
void OESGP::load(std::string filename)
{
    std::string model_filename = filename + ".model";
    std::string feature_filename = filename + ".features";
    sogp.load(model_filename);
    res.load(feature_filename);
}

/**
  \brief returns the reservoir size
  **/
unsigned int OESGP::getStateSize(void) {
    return res.getStateSize();
}


/**
  \brief computes the spectral radius of the reservoir (NOT necessarily
  the DESIRED spectral radius)
  **/
double OESGP::getActualSpectralRadius(void) {
    return res.getActualSpectralRadius();
}

/**
  \brief sets the reservoirs input weights
  \param input_weights the input weights. size should be
  reservoir size x input dimension
  */
void OESGP::setInputWeights(MatrixXd &input_weights) {
    res.setInputWeights(input_weights);
}

/**
  \brief sets the reservoirs output feedback weights
  \param output_feedback_weights the output feedback weights. size should be
  reservoir size x output dimension
  */
void OESGP::setOutputFeedbackWeights(MatrixXd &output_feedback_weights) {
    res.setOutputFeedbackWeights(output_feedback_weights);
}

/**
  \brief sets the reservoirs weights
  \param reservoir_weights the reservoir weights. size should be
  reservoir size x output dimension
  */
void OESGP::setReservoirWeights(MatrixXd &reservoir_weights) {
    res.setReservoirWeights(reservoir_weights);
}

unsigned int OESGP::getCurrentSize(void) {
    return sogp.getCurrentSize();
}

}
