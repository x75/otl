#include "otl_rls.h"

namespace OTL {


RLS::RLS(void) {
    this->initialized = false;
}

RLS::RLS(RLS &rhs) : LearningAlgorithm() {
    this->initialized = true; //initialised

    this->delta = rhs.delta;
    this->lambda = rhs.lambda;
    this->noise = rhs.noise;

    this->state_dim = rhs.state_dim;
    this->output_dim = rhs.output_dim;

    for (unsigned int i=0; i<rhs.output_dim; i++) {
        this->P_rls.push_back(rhs.P_rls[i]);
        this->w_rls.push_back(rhs.w_rls[i]);
    }

}

void RLS::train(const VectorXd &state, const VectorXd &output) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("RLS not yet initialised");
    }

    //create the noisy input state with a bias
    VectorXd noisy_input = state + VectorXd::Random(this->state_dim)*this->noise;
    VectorXd noisy_input_w_bias(this->state_dim + 1);
    for (unsigned int i=0; i<noisy_input.rows(); i++) noisy_input_w_bias[i] = noisy_input[i];
    noisy_input_w_bias[this->state_dim] = 1.0; //bias term
    VectorXd noisy_input_t = noisy_input_w_bias.transpose();

    //RLS algorithm
    double T_inv = 1.0/(noisy_input_t.dot(P_rls[0]*noisy_input_w_bias) + this->lambda);

    VectorXd g = this->P_rls[0]*noisy_input_w_bias*T_inv;
    MatrixXd P_mod = this->P_rls[0]*1.0/this->lambda;

    this->P_rls[0] =  P_mod - g*(noisy_input_w_bias.transpose()*P_mod);

    for (unsigned int o=0; o<this->output_dim; o++) {
        double a = output(o) - this->w_rls[o].dot(noisy_input_w_bias);
        this->w_rls[o] = this->w_rls[o] + a*g;
    }

    return;
}

void RLS::predict(const VectorXd &state, VectorXd &prediction,
                  VectorXd &prediction_variance) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("RLS not yet initialised");
    }

    //always need a bias term for RLS
    VectorXd aug_input = VectorXd::Ones(this->state_dim + 1);
    aug_input.segment(0,this->state_dim) = state;

    //allocate memory
    prediction = VectorXd(this->output_dim);
    prediction_variance = VectorXd::Zero(this->output_dim);

    //make our predictions
    for (unsigned int i=0; i<this->output_dim; i++) {
        prediction(i) = this->w_rls[i].dot(aug_input);
    }
    return;
}

void RLS::reset() {
    P_rls.clear();
    w_rls.clear();

    for (unsigned int i=0; i<this->output_dim; i++) {
        MatrixXd P(this->state_dim + 1, this->state_dim +1);
        P.setIdentity();
        P = P*1.0/this->delta;
        this->P_rls.push_back(P);

        VectorXd w = MatrixXd::Zero(this->state_dim + 1,1);
        this->w_rls.push_back(w);
    }

    return;
}

void RLS::save(std::string filename) {
    std::ofstream out(filename.c_str());
    out << this->delta << std::endl;
    out << this->lambda << std::endl;
    out << this->noise << std::endl;
    out << this->state_dim << std::endl;
    out << this->output_dim << std::endl;
    out << this->initialized << std::endl;
    for (unsigned int i=0; i<this->output_dim; i++) {
        OTL::saveMatrixToStream(out, P_rls[i]);
        OTL::saveVectorToStream(out, w_rls[i]);
    }

    out.close();
}

void RLS::load(std::string filename) {
    std::ifstream in(filename.c_str());
    in >> this->delta;
    in >> this->lambda;
    in >> this->noise;
    in >> this->state_dim;
    in >> this->output_dim;
    in >> this->initialized;

    this->P_rls.clear();
    this->w_rls.clear();

    MatrixXd P;
    VectorXd w;
    for (unsigned int i=0; i<this->output_dim; i++) {
        OTL::readMatrixFromStream(in, P);
        OTL::readVectorFromStream(in, w);

        this->P_rls.push_back(P);
        this->w_rls.push_back(w);
    }

    in.close();
}


void RLS::init(unsigned int state_dim, unsigned int output_dim,
               double delta, double lambda, double noise) {

    this->delta = delta;
    this->lambda = lambda;
    this->noise = noise;
    this->state_dim = state_dim;
    this->output_dim = output_dim;

    for (unsigned int i=0; i<this->output_dim; i++) {
        MatrixXd P(this->state_dim + 1, this->state_dim +1);
        P.setIdentity();
        P = P*1.0/this->delta;
        this->P_rls.push_back(P);

        VectorXd w = MatrixXd::Zero(this->state_dim + 1,1);
        this->w_rls.push_back(w);
    }

    this->initialized = true;
}

}
