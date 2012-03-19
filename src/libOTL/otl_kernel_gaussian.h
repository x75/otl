#ifndef OTL_KERNEL_GAUSSIAN_237017905890183921890489018490123213
#define OTL_KERNEL_GAUSSIAN_237017905890183921890489018490123213

#include "otl_exception.h"
#include "otl_kernel.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class GaussianKernel : public Kernel {
public:
    /**
      \brief initialises the gaussian kernel. Note that this kernel can perform
      ARD.
      \param state_dim the input state dimension
      \param parameters a VectorXd containing:
        [l(1) l(2) ... l(state_dim-1) alpha]
        where l is the characteristic length scale
        and alpha is the magnitude multiplier (leave at 1 if unsure).

        Also permitted is:
        [l alpha]
        if all the characteristic length scales are equal

      **/

    virtual void init(unsigned int state_dim, VectorXd &parameters);
    virtual void getParameters(VectorXd &parameters);
    virtual void save(std::string filename);
    virtual void load(std::string filename);

    virtual double eval(VectorXd &x);
    virtual double eval(VectorXd &x, VectorXd &y);
    virtual void eval(VectorXd &x, std::vector<VectorXd> &Y,
                      VectorXd &kern_vals);

private:
    VectorXd parameters;
    VectorXd b;
    double alpha;

    unsigned int state_dim;

    bool initialised;
};

void GaussianKernel::init(unsigned int state_dim,VectorXd &parameters) {
    if (state_dim == 0) {
        throw OTLException("State dimension must be larger than 0");
    }

    this->state_dim = state_dim;

    //set up the parameters
    if (parameters.rows() == state_dim + 1 ) {
        this->b = parameters.segment(0, state_dim);
        this->alpha = parameters(state_dim);
    } if (parameters.rows() == 2) {
        this->b = VectorXd::Ones(state_dim)*parameters(0);
        this->alpha = parameters(1);
    } else {
        throw OTLException("parameters vector is the wrong size!");
    }

    //transform the length scales
    for (unsigned int i=0; i<state_dim; i++) {
        this->b(i) = 1.0/(2*(this->b(i)*this->b(i)));
        if (std::isnan(this->b(i)) || std::isinf(this->b(i))) {
            throw OTLException("Whoops. The lengthscale resulted"
                               "in a nan or an inf");
        }
    }

    //save the parameters
    this->parameters = parameters;
    this->initialised = true;
}

void GaussianKernel::getParameters(VectorXd &parameters) {
    if (!this->initialised) {
        throw OTLException("You can't get parameters from the Gaussian Kernel"
                           "when it has not been initialised yet.");
    }

    parameters = this->parameters;
}

void GaussianKernel::save(std::string filename) {
    //TODO
}

void GaussianKernel::load(std::string filename) {
    //TODO
}

double GaussianKernel::eval(VectorXd &x) {
    return this->eval(x,x);
}

double GaussianKernel::eval(VectorXd &x, VectorXd &y) {
    if (x.rows() != y.rows()) {
        throw OTLException("Hey, GaussianKernel cannot evaluate"
                           "vectors of different sizes");
    }

    if (x.rows() != this->state_dim) {
        throw OTLException("Hey, GaussianKernel says the vector to evaluate"
                           "has a different size than the initialisation.");
    }

    double kval = 0.0;
    VectorXd diffxy = x-y;
    for (unsigned int i=0; i<diffxy.rows(); i++) {
        kval += diffxy(i)*diffxy(i)*b(i);
    }

    kval = this->alpha*exp(-kval);
    return kval;
}

void GaussianKernel::eval(VectorXd &x, std::vector<VectorXd> &Y,
                          VectorXd &kern_vals)
{
    if (Y.size() == 0) {
        throw OTLException("The vector Y should have nonzero size.");
    }
    kern_vals.resize(Y.size());
    for (unsigned int i=0; i<Y.size(); i++) {
        kern_vals(i) = this->eval(x, Y[i]);
    }

}





}

#endif
