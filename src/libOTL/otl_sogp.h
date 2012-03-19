#ifndef OTL_SOGP_H_750279082490178956479784073190
#define OTL_SOGP_H_750279082490178956479784073190

#include "otl_exception.h"
#include "otl_learning_algs.h"
#include "otl_helpers.h"
#include "otl_kernels.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class SOGP : public LearningAlgorithm {
public:
    SOGP();
    SOGP(SOGP &rhs);

    virtual void train(const VectorXd &state, const VectorXd &output);
    virtual void predict(const VectorXd &state, VectorXd &prediction, VectorXd &prediction_variance);
    virtual void reset();
    virtual void save(std::string filename);
    virtual void load(std::string filename);

    /**
      \brief Initialises the SOGP
      \param state_dim how big is the state that we want to regress from
      \param output_dim how big is the output state
      \param kernel_params a VectorXd with the kernel parameters
      \param noise the noise parameter (application dependent)
      \param epsilon threshold parameter (typically small 1e-4)
      \param capacity the capacity of the SOGP (application dependent)
      **/
    virtual void init(unsigned int state_dim, unsigned int output_dim,
                      VectorXd &kernel_params,
                      double noise,
                      double epsilon,
                      unsigned int capacity);

private:
    bool initialized; //initialised?

    unsigned int state_dim;
    unsigned int output_dim;

    VectorXd kernel_params;
    double epsilon;
    double noise;
    unsigned int capacity;

    MatrixXd alpha;
    MatrixXd C;
    MatrixXd Q;
};

SOGP::SOGP(void) {
    this->initialized = false;
}

SOGP::SOGP(SOGP &rhs) {
    this->initialized = true; //initialised

    this->delta = rhs.delta;
    this->lambda = rhs.lambda;
    this->noise = rhs.noise;

    this->state_dim = rhs.state_dim;
    this->output_dim = rhs.output_dim;

    for (unsigned int i=0; i<rhs.output_dim; i++) {
        this->P_SOGP.push_back(rhs.P_SOGP[i]);
        this->w_SOGP.push_back(rhs.w_SOGP[i]);
    }

}

void SOGP::train(const VectorXd &state, const VectorXd &output) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("SOGP not yet initialised");
    }

    //create the noisy input state with a bias
    VectorXd noisy_input = state + VectorXd::Random(this->state_dim)*this->noise;
    VectorXd noisy_input_w_bias(this->state_dim + 1);
    for (unsigned int i=0; i<noisy_input.rows(); i++) noisy_input_w_bias[i] = noisy_input[i];
    noisy_input_w_bias[this->state_dim] = 1.0; //bias term
    VectorXd noisy_input_t = noisy_input_w_bias.transpose();

    //SOGP algorithm
    double T_inv = 1.0/(noisy_input_t.dot(P_SOGP[0]*noisy_input_w_bias) + this->lambda);

    VectorXd g = this->P_SOGP[0]*noisy_input_w_bias*T_inv;
    MatrixXd P_mod = this->P_SOGP[0]*1.0/this->lambda;

    this->P_SOGP[0] =  P_mod - g*(noisy_input_w_bias.transpose()*P_mod);

    for (unsigned int o=0; o<this->output_dim; o++) {
        double a = output(o) - this->w_SOGP[o].dot(noisy_input_w_bias);
        this->w_SOGP[o] = this->w_SOGP[o] + a*g;
    }

    return;
}

void SOGP::predict(const VectorXd &state, VectorXd &prediction,
                  VectorXd &prediction_variance) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("SOGP not yet initialised");
    }

    //always need a bias term for SOGP
    VectorXd aug_input = VectorXd::Ones(this->state_dim + 1);
    aug_input.segment(0,this->state_dim) = state;

    //allocate memory
    prediction = VectorXd(this->output_dim);
    prediction_variance = VectorXd::Zero(this->output_dim);

    //make our predictions
    for (unsigned int i=0; i<this->output_dim; i++) {
        prediction(i) = this->w_SOGP[i].dot(aug_input);
    }
    return;
}

void SOGP::reset() {
    P_SOGP.clear();
    w_SOGP.clear();

    for (unsigned int i=0; i<this->output_dim; i++) {
        MatrixXd P(this->state_dim + 1, this->state_dim +1);
        P.setIdentity();
        P = P*1.0/this->delta;
        this->P_SOGP.push_back(P);

        VectorXd w = MatrixXd::Zero(this->state_dim + 1,1);
        this->w_SOGP.push_back(w);
    }

    return;
}

void SOGP::save(std::string filename) {
    std::ofstream out(filename.c_str());
    out << this->delta << std::endl;
    out << this->lambda << std::endl;
    out << this->noise << std::endl;
    out << this->state_dim << std::endl;
    out << this->output_dim << std::endl;
    out << this->initialized << std::endl;
    for (unsigned int i=0; i<this->output_dim; i++) {
        OTL::saveMatrixToStream(out, P_SOGP[i]);
        OTL::saveVectorToStream(out, w_SOGP[i]);
    }

    out.close();
}

void SOGP::load(std::string filename) {
    std::ifstream in(filename.c_str());
    in >> this->delta;
    in >> this->lambda;
    in >> this->noise;
    in >> this->state_dim;
    in >> this->output_dim;
    in >> this->initialized;

    this->P_SOGP.clear();
    this->w_SOGP.clear();

    MatrixXd P;
    VectorXd w;
    for (unsigned int i=0; i<this->output_dim; i++) {
        OTL::readMatrixFromStream(in, P);
        OTL::readVectorFromStream(in, w);

        this->P_SOGP.push_back(P);
        this->w_SOGP.push_back(w);
    }

    in.close();
}


void SOGP::init(unsigned int state_dim, unsigned int output_dim,
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
        this->P_SOGP.push_back(P);

        VectorXd w = MatrixXd::Zero(this->state_dim + 1,1);
        this->w_SOGP.push_back(w);
    }

    this->initialized = true;
}


}
#endif
