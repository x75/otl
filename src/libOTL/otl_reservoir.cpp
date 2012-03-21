#include "otl_reservoir.h"

namespace OTL {

Reservoir::Reservoir(void) {
    //this reservoir is not yet initialized
    this->initialized = false;
    return;
}

/**
  \brief Copy constructor
  */
Reservoir::Reservoir(Reservoir &rhs) : AugmentedState() {

    this->input_dim = rhs.input_dim;
    this->output_dim = rhs.output_dim;

    this->reservoir_size = rhs.reservoir_size;
    this->input_weight = rhs.input_weight;
    this->output_feedback_weight = rhs.output_feedback_weight;
    this->activation_function = rhs.activation_function;
    this->leak_rate = rhs.leak_rate;
    this->connectivity = rhs.connectivity;
    this->spectral_radius = rhs.spectral_radius;
    this->random_seed = rhs.random_seed;
    this->use_inputs_in_state = rhs.use_inputs_in_state;

    this->input_weights = rhs.input_weights;

    this->output_feedback_weights = rhs.output_feedback_weights;

    this->reservoir_weights = rhs.reservoir_weights;

    this->curr_reservoir_state = rhs.curr_reservoir_state;

    this->state = rhs.state;
    this->state_size = rhs.state_size;

    this->initialized = rhs.initialized;
}

/**
  \brief updates the reservoir with input
  \param input the input (a VectorXd)
  */
 void Reservoir::update(const VectorXd &input) {
    VectorXd state_before_activation(this->reservoir_size);

    //update function
    state_before_activation = this->reservoir_weights*this->curr_reservoir_state +
            this->input_weights*input;

    //use the activation function
    VectorXd new_state;
    this->activation(state_before_activation, new_state);

    //use the leak rate if needed

    if (this->leak_rate > 0.0) {
        new_state = (1.0 - this->leak_rate)*new_state +
                this->leak_rate*this->curr_reservoir_state;
    }

    //construct augmented state vector
    if (this->use_inputs_in_state) {
        this->state.segment(0, this->reservoir_size) =
                new_state;
        this->state.segment(this->reservoir_size, this->input_dim) =
                input;
    } else {
        this->state = new_state;
    }

    //set the internal state
    this->curr_reservoir_state = new_state;

    return;
}

/**
  \brief updates the reservoir with the given input and output_feedback
  \param input the input
  \param output_feedback the output to feedback into the reservoir
  */
 void Reservoir::update(const VectorXd &input, const VectorXd &output_feedback) {
    VectorXd state_before_activation(this->reservoir_size);

    //update function
    state_before_activation = this->reservoir_weights*this->curr_reservoir_state +
            this->input_weights*input +
            this->output_feedback_weights*output_feedback;

    //use the activation function
    VectorXd new_state;
    this->activation(state_before_activation, new_state);

    //use the leak rate if needed

    if (this->leak_rate > 0.0) {
        new_state = (1.0 - this->leak_rate)*new_state +
                this->leak_rate*this->curr_reservoir_state;
    }

    //construct augmented state vector
    if (this->use_inputs_in_state) {
        this->state.segment(0, this->reservoir_size) =
                state;
        this->state.segment(this->reservoir_size, this->input_dim) =
                input;
    } else {
        this->state = new_state;
    }

    //set the internal state
    this->curr_reservoir_state = new_state;

    return;
}

/**
  \brief gets the state of the reservoir
  */
 void Reservoir::getState(VectorXd &state) {
    state = this->state;
}

/**
  \brief sets the state of the reservoir
  */
 void Reservoir::setState(const VectorXd &state) {
    this->state = state;
}

/**
  \brief resets the reservoir
  */
 void Reservoir::reset(void) {
    this->state = VectorXd::Zero(this->state_size);
}

/**
    \brief Saves the reservoir to a file
    \param filename a string specifying the filename to save to
**/
 void Reservoir::save(std::string filename) {

    std::ofstream out;
    try {
        out.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    out << this->input_dim << std::endl;
    out << this->output_dim << std::endl;

    out << this->reservoir_size << std::endl;
    out << this->input_weight << std::endl;
    out << this->output_feedback_weight << std::endl;
    out << this->activation_function << std::endl;
    out << this->leak_rate << std::endl;
    out << this->connectivity << std::endl;
    out << this->spectral_radius << std::endl;
    out << this->random_seed << std::endl;
    out << this->use_inputs_in_state << std::endl;

    saveMatrixToStream(out, this->input_weights);
    saveMatrixToStream(out, this->output_feedback_weights);
    saveMatrixToStream(out, this->reservoir_weights);
    saveVectorToStream(out, this->curr_reservoir_state);
    saveVectorToStream(out, this->state);

    out << this->state_size  << std::endl;
    out << this->initialized  << std::endl;
}

/**
    \brief loads the reservoir from a file
    \param filename where to load from?
**/
void Reservoir::load(std::string filename) {
    std::ifstream in;
    try {
        in.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    in >> this->input_dim;
    in >> this->output_dim;

    in >> this->reservoir_size;
    in >> this->input_weight;
    in >> this->output_feedback_weight;
    in >> this->activation_function;
    in >> this->leak_rate;
    in >> this->connectivity;
    in >> this->spectral_radius;
    in >> this->random_seed;
    in >> this->use_inputs_in_state;

    readMatrixFromStream(in, this->input_weights);
    readMatrixFromStream(in, this->output_feedback_weights);
    readMatrixFromStream(in, this->reservoir_weights);
    readVectorFromStream(in, this->curr_reservoir_state);
    readVectorFromStream(in, this->state);

    in >> this->state_size;
    in >> this->initialized;
}

/**
  \brief returns the reservoir size
  **/
unsigned int Reservoir::getStateSize(void) {
    return this->state_size;
}

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
void Reservoir::init(
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
        )
{

    //set all the parameters
    this->input_dim = input_dim;
    this->output_dim = output_dim;
    this->reservoir_size = reservoir_size;
    this->input_weight = input_weight;
    this->output_feedback_weight = output_feedback_weight;
    this->activation_function = activation_function;
    this->leak_rate = leak_rate;
    this->connectivity = connectivity;
    this->spectral_radius = spectral_radius;
    this->use_inputs_in_state = use_inputs_in_state;
    this->random_seed = random_seed;


    //set the random seed
    srand(random_seed);

    // initialize the input weights
    this->input_weights = MatrixXd::Ones(reservoir_size, input_dim)*this->input_weight;

    // intitialize the feedback weights
    if (output_feedback_weight > 0.0) {
        this->output_feedback_weights = MatrixXd::Ones(reservoir_size, output_dim) * this->output_feedback_weight;
    } else {
        this->output_feedback_weights = MatrixXd::Zero(reservoir_size, output_dim);
    }

    // initialize the reservoir weights
    this->reservoir_weights = MatrixXd::Random(reservoir_size, reservoir_size);
    for (unsigned int i=0; i<reservoir_size; i++) {
        for (unsigned int j=0; j<reservoir_size; j++) {
            //we constrain the network to have the desired connectivity
            if (fabs((MatrixXd::Random(1,1))(0,0)) > connectivity) {
                this->reservoir_weights(i,j) = 0.0;
            }  else {
                //this is an inefficient hack -- need to find out how to call a single random value usiong the eigen library
                //we are using the box muller transform here
                double u1 = fabs((MatrixXd::Random(1,1))(0,0));
                double u2 = fabs((MatrixXd::Random(1,1))(0,0));
                double r = sqrt(-2.0*log(u1))*cos(2*M_PI*u2);
                this->reservoir_weights(i,j) = r;
            }
        }
    }

    //std::cout << reservoir_weights << std::endl;
    // use the spectral radius to constrain the reservoir weights
    // first we calculate the spectral radius of the current reservoir
    double denom = this->getActualSpectralRadius();

    // then, we rescale the reservoir approrpiately
    this->reservoir_weights = this->reservoir_weights*this->spectral_radius/denom;

    //set the current reservoir state to zero
    //this->curr_reservoir_state_before_act = VectorXd::Zero(reservoir_size);
    this->curr_reservoir_state = VectorXd::Zero(this->reservoir_size);

    //initialise the state
    if (this->use_inputs_in_state) {
        this->state_size = this->reservoir_size + this->input_dim;
    } else {
        this->state_size = this->reservoir_size;
    }
    this->state = VectorXd::Zero(this->state_size);

    //set initialized flag to true
    this->initialized = true;


    return;
}


/*---------------------------------------------------------------------
        ADDITIONAL FUNCTIONS SPECIFIC TO RESERVOIRS
  ---------------------------------------------------------------------*/
/**
  \brief computes the spectral radius of the reservoir (NOT necessarily
  the DESIRED spectral radius)
  **/
double Reservoir::getActualSpectralRadius(void) {
    //just compute the largest eigenvalue (absolute)
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigen_solver(this->reservoir_weights);
    if (eigen_solver.info() != Eigen::Success) {
        std::cout << "Whoops! can't seem to compute the eigenvalues" << std::endl;
        return 0.0;
    }

    double max_value = fabs(eigen_solver.eigenvalues().maxCoeff());
    double min_value = fabs(eigen_solver.eigenvalues().minCoeff());
    return (max_value > min_value) ? max_value : min_value;
}

/**
  \brief sets the reservoirs input weights
  \param input_weights the input weights. size should be
  reservoir size x input dimension
  */
void Reservoir::setInputWeights(MatrixXd &input_weights) {
    if (!this->initialized) {
        throw OTLException("Please initialize the Reservoir first.");
    }

    if (input_weights.cols() != this->input_weights.cols()) {
        throw OTLException("Columns do not match");
    }
    if (input_weights.rows() != this->input_weights.rows()) {
        throw OTLException("Rows do not match");
    }

    this->input_weights = input_weights;
}

/**
  \brief sets the reservoirs output feedback weights
  \param output_feedback_weights the output feedback weights. size should be
  reservoir size x output dimension
  */
void Reservoir::setOutputFeedbackWeights(MatrixXd &output_feedback_weights) {
    if (!this->initialized) {
        throw OTLException("Please initialize the Reservoir first.");
    }
    if (output_feedback_weights.cols() != this->output_feedback_weights.cols()) {
        throw OTLException("Columns do not match");
    }
    if (output_feedback_weights.rows() != this->output_feedback_weights.rows()) {
        throw OTLException("Rows do not match");
    }

    this->output_feedback_weights = output_feedback_weights;
}

/**
  \brief sets the reservoirs weights
  \param reservoir_weights the reservoir weights. size should be
  reservoir size x output dimension
  */
void Reservoir::setReservoirWeights(MatrixXd &reservoir_weights) {
    if (!this->initialized) {
        throw OTLException("Please initialize the Reservoir first.");
    }
    if (reservoir_weights.cols() != this->reservoir_weights.cols()) {
        throw OTLException("Columns do not match");
    }
    if (reservoir_weights.rows() != this->reservoir_weights.rows()) {
        throw OTLException("Rows do not match");
    }

    this->reservoir_weights = reservoir_weights;
}


void Reservoir::activation(VectorXd &inputs, VectorXd &results) {
    switch (this->activation_function) {
        case TANH:
            return tanh_activation(inputs, results);
            break;
        case LINEAR:
            return linear_activation(inputs,results);
            break;
        default:
            //should throw an exception but we will put this off till later
            //output something
            throw OTLException("Wrong activation function specified");
    }
}


void Reservoir::linear_activation(VectorXd &inputs, VectorXd &results) {
    unsigned int num_elems = (inputs.rows() > inputs.cols()) ? inputs.rows() : inputs.cols();
    results.resize(num_elems, 1);
    for (unsigned int i=0; i<num_elems; i++) {
        results(i) = inputs(i);
    }
}


void Reservoir::tanh_activation(VectorXd &inputs, VectorXd &results) {
    unsigned int num_elems = (inputs.rows() > inputs.cols()) ? inputs.rows() : inputs.cols();

    results.resize(num_elems, 1);
    for (unsigned int i=0; i<num_elems; i++) {
        results(i) = tanh(inputs(i));
    }
}



}
