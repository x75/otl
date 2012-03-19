#ifndef OTL_SOGP_H_750279082490178956479784073190
#define OTL_SOGP_H_750279082490178956479784073190

#include "otl_exception.h"
#include "otl_learning_algs.h"
#include "otl_helpers.h"
#include "otl_kernel.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class SOGP : public LearningAlgorithm {
public:
    SOGP();
    SOGP(SOGP &rhs);
    ~SOGP();

    virtual void train(const VectorXd &state, const VectorXd &output);
    virtual void predict(const VectorXd &state, VectorXd &prediction, VectorXd &prediction_variance);
    virtual void reset();
    virtual void save(std::string filename);
    virtual void load(std::string filename);

    /**
      \brief Initialises the SOGP
      \param state_dim how big is the state that we want to regress from
      \param output_dim how big is the output state
      \param Kernel an OTL::Kernel object with the kernel you want to use
      \param noise the noise parameter (application dependent)
      \param epsilon threshold parameter (typically small 1e-4)
      \param capacity the capacity of the SOGP (application dependent)
      **/
    virtual void init(unsigned int state_dim, unsigned int output_dim,
                      Kernel &kernel,
                      double noise,
                      double epsilon,
                      unsigned int capacity);

private:
    bool initialized; //initialised?

    unsigned int state_dim;
    unsigned int output_dim;
    unsigned int current_size;

    Kernel *kernel;
    double epsilon;
    double noise;
    unsigned int capacity;

    MatrixXd alpha;
    MatrixXd C;
    MatrixXd Q;

    std::vector<VectorXd> basis_vectors;

};

SOGP::SOGP(void) {
    this->initialized = false;
}

SOGP::SOGP(SOGP &rhs) {
    //TODO
}

SOGP::~SOGP() {
    //TODO
    delete this->kernel;
}

void SOGP::train(const VectorXd &state, const VectorXd &output) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("SOGP not yet initialised");
    }

    double kstar = this->kernel->eval(state);

    //we are just starting.
    if (this->current_size == 0) {

        this->alpha.block(0,0,1, this->output_dim) = output.array() / (kstar + this->noise);
        this->C.block(0,0,1,1) = VectorXd::Ones(1)*-1/(kstar + this->noise);
        this->Q.block(0,0,1,1) = VectorXd::Ones(1)*1/(kstar);
        this->basis_vectors.push_back(state);
        this->current_size++;
        return;
    }

    //Test if this is a "novel" state
    VectorXd k;
    this->kernel->eval(state, this->basis_vectors, k);
    //cache Ck
    VectorXd Ck = this->C.block(0,0, this->current_size, this->current_size)*k;

    VectorXd m = this->alpha.block(0,0,this->current_size, this->output_dim).transpose() * k;
    double s2 = kstar + (k.dot(Ck));

    if (s2 < 1e-12) {
        s2 = 1e-12;
    }

    double r = -1.0/(s2 + this->noise);
    VectorXd q = (output - m)*(-r);
    VectorXd ehat = this->Q.block(0,0, this->current_size, this->current_size)*k;

    double gamma = kstar - k.dot(ehat);
    double eta = 1.0/(1.0 + gamma*r);

    if (gamma < 1e-12) {
        gamma = 0.0;
    }

    if (gamma >= this->epsilon*kstar) {
        //perform a full update
        VectorXd s = Ck;
        s.conservativeResize(this->current_size + 1);
        s(this->current_size) = 1;

        //add to basis vectors
        this->basis_vectors.push_back(state);

        //update Q (inverse of C)
        ehat.conservativeResize(this->current_size+1);
        ehat(this->current_size) = -1;

        MatrixXd diffQ = Q.block(0,0,this->current_size+1, this->current_size+1)
                + (ehat*ehat.transpose())*(1.0/gamma);
        Q.block(0,0,this->current_size+1, this->current_size+1) = diffQ;


        //update alpha
        MatrixXd diffAlpha = alpha.block(0,0, this->current_size+1, this->output_dim)
                + (s*q);
        alpha.block(0,0, this->current_size+1, this->output_dim) = diffAlpha;

        //update C
        MatrixXd diffC = C.block(0,0, this->current_size+1, this->current_size+1)
                + (s*s.transpose());
        C.block(0,0, this->current_size+1, this->current_size+1) = diffC;

        //increment current size
        this->current_size++;

    } else {
        //perform a sparse update
        VectorXd s = Ck + ehat;

        //update alpha
        MatrixXd diffAlpha = alpha.block(0,0, this->current_size, this->output_dim)
                + s*(q*eta).transpose();
        alpha.block(0,0, this->current_size, this->output_dim) = diffAlpha;

        //update C
        MatrixXd diffC = C.block(0,0, this->current_size, this->current_size) +
                r*eta*(s*s.transpose());
        C.block(0,0, this->current_size, this->current_size) = diffC;
    }

    return;
}

void SOGP::predict(const VectorXd &state, VectorXd &prediction,
                  VectorXd &prediction_variance) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("SOGP not yet initialised");
    }

    double kstar = kernel->eval(state,state);

    //check if we not been trained
    if (this->current_size == 0) {
        prediction = VectorXd::Zero(this->output_dim);
        prediction_variance = VectorXd::Ones(this->output_dim)*
                (kstar + this->noise);
        return;
    }

    VectorXd k;
    kernel->eval(state, this->basis_vectors, k);
    //std::cout << "K: \n" << k << std::endl;
    //std::cout << "alpha: \n" << this->alpha.block(0,0,this->current_size, this->output_dim) << std::endl;

    prediction = this->alpha.block(0,0,this->current_size, this->output_dim).transpose() * k;
    prediction_variance = VectorXd::Ones(this->output_dim)*
            (k.dot(this->C.block(0,0, this->current_size, this->current_size)*k)
             + kstar + this->noise);

    return;
}

void SOGP::reset() {
    //TODO

    return;
}

void SOGP::save(std::string filename) {
    std::ofstream out(filename.c_str());
    //TODO
    out.close();
}

void SOGP::load(std::string filename) {
    std::ifstream in(filename.c_str());

        //TODO
    in.close();
}


void SOGP::init(unsigned int state_dim, unsigned int output_dim,
                Kernel &kernel,
                double noise,
                double epsilon,
                unsigned int capacity) {

    this->kernel = kernel.createCopy();
    this->state_dim = state_dim;
    this->output_dim = output_dim;
    this->noise = noise;
    this->epsilon = epsilon;
    this->capacity = capacity;
    this->current_size = 0;

    this->alpha = MatrixXd::Zero(this->capacity, this->output_dim);
    this->C = MatrixXd::Zero(this->capacity, this->capacity);
    this->Q = MatrixXd::Zero(this->capacity, this->capacity);

    this->initialized = true;
}


}
#endif
