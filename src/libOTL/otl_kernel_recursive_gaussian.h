#ifndef OTL_KERNEL_RecursiveGaussian_237017905890183921890489018490123213
#define OTL_KERNEL_RecursiveGaussian_237017905890183921890489018490123213

#include "otl_exception.h"
#include "otl_kernel.h"
#include "otl_helpers.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class RecursiveGaussianKernel : public Kernel {
public:

    RecursiveGaussianKernel();
    RecursiveGaussianKernel(RecursiveGaussianKernel &rhs);

    /**
      \brief initialises the RecursiveGaussian kernel. Note that this kernel can perform
      ARD.
      \param state_dim the input state dimension
      \param parameters a VectorXd containing:
        [l(1) l(2) ... l(input_dim) rho alpha input_dim]
        where l is the characteristic length scale
            rho is the spectral radius (typically 0.99)
            alpha is the magnitude multiplier (leave at 1 if unsure).
            input_dim is the dimension of the input (since this is a VectorXd,
                this value is rounded to the closest integer)

        Also permitted is:
        [l rho alpha input_dim]
        if all the characteristic length scales are equal

      **/
    virtual void init(const unsigned int state_dim, const VectorXd &parameters);
    virtual void getParameters(VectorXd &parameters);
    virtual void save(const std::string filename);
    virtual void load(const std::string filename);
    virtual void save(std::ostream &out);
    virtual void load(std::istream &in);

    virtual double eval(const VectorXd &x);
    virtual double eval(const VectorXd &x, const VectorXd &y);
    virtual void eval(const VectorXd &x, const std::vector<VectorXd> &Y,
                      VectorXd &kern_vals);
    virtual Kernel* createCopy(void);

private:
    VectorXd parameters;
    VectorXd b;
    double alpha;
    double rho;

    unsigned int state_dim;
    unsigned int input_dim;

    bool initialised;
};


RecursiveGaussianKernel::RecursiveGaussianKernel() : Kernel("RecursiveGaussian"){
    this->initialised = false;
}

RecursiveGaussianKernel::RecursiveGaussianKernel(RecursiveGaussianKernel &rhs) : Kernel("RecursiveGaussian") {
    this->parameters = rhs.parameters;
    this->b = rhs.b;
    this->rho = rhs.rho;
    this->alpha = rhs.alpha;
    this->state_dim = rhs.state_dim;
    this->input_dim = rhs.input_dim;
    this->initialised = rhs.initialised;
}

void RecursiveGaussianKernel::init(const unsigned int state_dim, const VectorXd &parameters) {
    if (state_dim == 0) {
        throw OTLException("State dimension must be larger than 0");
    }

    this->state_dim = state_dim;
    unsigned int end = parameters.rows() - 1;
    this->input_dim = floor(parameters(end) + 0.5);

    //set up the parameters
    if (parameters.rows() == input_dim + 3 ) {
        this->b = parameters.segment(0, input_dim);
        this->rho = parameters(input_dim);
        this->alpha = parameters(input_dim+1);
    } if (parameters.rows() == 4) {
        this->b = VectorXd::Ones(input_dim)*parameters(0);
        this->rho = parameters(1);
        this->alpha = parameters(2);
    } else {
        throw OTLException("parameters vector is the wrong size!");
    }

    //transform the length scales
    for (unsigned int i=0; i<input_dim; i++) {
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

void RecursiveGaussianKernel::getParameters(VectorXd &parameters) {
    if (!this->initialised) {
        throw OTLException("You can't get parameters from the RecursiveGaussian Kernel"
                           "when it has not been initialised yet.");
    }

    parameters = this->parameters;
}

void RecursiveGaussianKernel::save(const std::string filename) {
    std::ofstream out;
    try {
        out.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    this->save(out);

    out.close();
}

void RecursiveGaussianKernel::load(const std::string filename) {
    std::ifstream in;
    try {
        in.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }
    this->load(in);

    in.close();
}

void RecursiveGaussianKernel::save(std::ostream &out) {

    saveVectorToStream(out, this->parameters);
    saveVectorToStream(out, this->b);
    out << this->alpha << std::endl;
    out << this->rho << std::endl;
    out << this->input_dim << std::endl;
    out << this->state_dim << std::endl;
    out << this->initialised << std::endl;

}

void RecursiveGaussianKernel::load(std::istream &in) {

    readVectorFromStream(in, this->parameters);
    readVectorFromStream(in, this->b);
    in >> this->alpha;
    in >> this->rho;
    in >> this->input_dim;
    in >> this->state_dim;
    in >> this->initialised;
}


Kernel* RecursiveGaussianKernel::createCopy(void) {
    return (Kernel*) new RecursiveGaussianKernel(*this);
}

double RecursiveGaussianKernel::eval(const VectorXd &x) {
    return this->eval(x,x);
}

double RecursiveGaussianKernel::eval(const VectorXd &x, const VectorXd &y) {
    if (x.rows() != y.rows()) {
        throw OTLException("Hey, RecursiveGaussianKernel cannot evaluate"
                           "vectors of different sizes");
    }

    if (x.rows() != this->state_dim) {
        throw OTLException("Hey, RecursiveGaussianKernel says the vector to evaluate "
                           "has a different size than the initialisation.");
    }

    double kval = 1.0;
    unsigned int n_segments = this->state_dim/this->input_dim;
    for (unsigned int i=0; i<n_segments; i++) {
        VectorXd diffxy = x.segment(i*this->input_dim, this->input_dim) -
                y.segment(i*this->input_dim, this->input_dim);
        double val = 0.0;
        for (unsigned int j=0; j<this->input_dim; j++) {
            val += diffxy(j)*diffxy(j)*b(j);
        }
        kval = this->alpha*exp(-val)*exp( (kval - 1)/this->rho );
    }
    return kval;
}

void RecursiveGaussianKernel::eval(const VectorXd &x, const std::vector<VectorXd> &Y,
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
