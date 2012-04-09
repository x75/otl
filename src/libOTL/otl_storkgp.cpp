#include "otl_storkgp.h"


namespace OTL {

/**
  \brief Sets up the STORKGP algorithm
  \param input_dim the input dimension
  \param output_dim the output dimension
  \param tau window size for approximation
  \param kernel_type the type of kernel (enum)
  \param kernel_parameters kernel parameters for the recursive kernel
  \param noise noise for SOGP
  \param epsilon thwindowhold for SOGP
  \param capacity the capacity for the SOGP
  **/
void STORKGP::init(
        unsigned int input_dim,
        unsigned int output_dim,
        unsigned int tau,
        int kernel_type,
        VectorXd &kernel_parameters,
        double noise,
        double epsilon,
        unsigned int capacity
        )
{

    window.init(input_dim, output_dim, tau);

    switch (kernel_type) {
    case RECURSIVE_GAUSSIAN:
        rec_gaussian_kernel.init(window.getStateSize(), kernel_parameters);
        sogp.init(window.getStateSize(), output_dim, rec_gaussian_kernel,
                  noise, epsilon, capacity);
        break;
    case RECURSIVE_EQUALITY_GAUSSIAN:
        rec_equality_gaussian_kernel.init(window.getStateSize(), kernel_parameters);
        sogp.init(window.getStateSize(), output_dim, rec_equality_gaussian_kernel,
                  noise, epsilon, capacity);
        break;
    }

}

/**
  \brief updates the window with input
  \param input the input (a VectorXd)
  */
void STORKGP::update(const VectorXd &input) {
    window.update(input);
}


/**
  \brief trains the STORKGP given the current state and the output
  \param output (a VectorXd)
  */
void STORKGP::train(const VectorXd &output) {
    VectorXd state;
    window.getState(state);
    sogp.train(state, output);
}

/**
  \brief make a prediction and prediction variance.
  \param prediction the output prediction (a VectorXd)
  \param prediction_variance the output prediction variance (a VectorXd)
  */
void STORKGP::predict(VectorXd &prediction, VectorXd &prediction_variance){
    VectorXd state;
    window.getState(state);
    sogp.predict(state, prediction, prediction_variance);
}

/**
  \brief windowets the state of the window
  */
void STORKGP::resetState() {
    window.reset();
}

/**
  \brief windowets the SOGP algorthm
  */
void STORKGP::resetSOGP() {
    sogp.reset();
}


/**
  \brief gets the state of the window
  */
void STORKGP::getState(VectorXd &state) {
    window.getState(state);
}

/**
  \brief sets the state of the window
  */
void STORKGP::setState(const VectorXd &state) {
    window.setState(state);
}

/**
  \brief windowets the window
  */
void STORKGP::reset(void) {
    window.reset();
    sogp.reset();
}

/**
    \brief Saves the STORKGP. this saves two files. a [filename].model file (for the sogp)
    and a [filename].featuwindow file to save the window
    \param filename a string specifying the filename to save to
**/
void STORKGP::save(std::string filename) {
    std::string model_filename = filename + ".model";
    std::string feature_filename = filename + ".features";
    sogp.save(model_filename);
    window.save(feature_filename);
}

/**
    \brief loads the model from disk
    \param filename where to load from?
**/
void STORKGP::load(std::string filename)
{
    std::string model_filename = filename + ".model";
    std::string feature_filename = filename + ".features";
    sogp.load(model_filename);
    window.load(feature_filename);
}

/**
  \brief returns the window size
  **/
unsigned int STORKGP::getStateSize(void) {
    return window.getStateSize();
}

unsigned int STORKGP::getCurrentSize(void) {
    return sogp.getCurrentSize();
}

}
