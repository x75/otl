/**
  OTL Gaussian Class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the typical Gaussian kernel (squared exponential) with automatic
  relevance determination (ARD).

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_KERNEL_GAUSSIAN_237017905890183921890489018490123213
#define OTL_KERNEL_GAUSSIAN_237017905890183921890489018490123213

#include "otl_exception.h"
#include "otl_kernel.h"
#include "otl_helpers.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <cmath>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class GaussianKernel : public Kernel {
public:
    static const std::string name;

    GaussianKernel();
    GaussianKernel(const GaussianKernel &rhs);
    ~GaussianKernel();

    /**
      \brief initialises the gaussian kernel. Note that this kernel can perform
      ARD.
      \param state_dim the input state dimension
      \param parameters a VectorXd containing:
        [l(1) l(2) ... l(state_dim) alpha]
        where l is the characteristic length scale
        and alpha is the magnitude multiplier (leave at 1 if unsure).

        Also permitted is:
        [l alpha]
        if all the characteristic length scales are equal

      **/
    virtual void init(const unsigned int state_dim, const VectorXd &parameters);


    /**
      \brief puts the current parameters into parameters
      \param parameters a VectorXd to put the parameters in
      **/
    virtual void getParameters(VectorXd &parameters);

    /**
      \brief saves this kernel to disk
      \param filename the name of the file
      **/
    virtual void save(const std::string filename);

    /**
      \brief loads kernel from disk
      \param filename the name of the file
      **/
    virtual void load(const std::string filename);

    /**
      \brief saves this kernel to an output stream
      \param out the output ostream
      **/
    virtual void save(std::ostream &out);

    /**
      \brief reads the kernel from an input stream
      \param in the input istream
      **/
    virtual void load(std::istream &in);

    /**
      \brief evaluates the kernel k(x,x)
      \param x a VectorXd
      **/
    virtual double eval(const VectorXd &x);

    /**
      \brief evaluates the kernel k(x,y)
      \param x a VectorXd
      \param y a VectorXd
      **/
    virtual double eval(const VectorXd &x, const VectorXd &y);

    /**
      \brief evaluates the kernel k(x,y) for each y in Y
      \param x a VectorXd
      \param Y a std::vector of VectorXds
      \param kern_vals where to put the evaluted kernel values
      **/
    virtual void eval(const VectorXd &x, const std::vector<VectorXd> &Y,
                      VectorXd &kern_vals);


protected:
    /**
      \brief creates a copy of the kernel and returns a Kernel pointer to it.
      **/
    virtual Kernel* internalCreateCopy(void) const;

private:
    VectorXd parameters;
    VectorXd b;
    double alpha;

    unsigned int state_dim;

    bool initialised;



};


GaussianKernel::GaussianKernel() : Kernel("Gaussian") {
    this->initialised = false;
}

GaussianKernel::~GaussianKernel() {
    return;
}

GaussianKernel::GaussianKernel(const GaussianKernel &rhs) :Kernel("Gaussian") {
    this->parameters = rhs.parameters;
    this->b = rhs.b;
    this->alpha = rhs.alpha;
    this->state_dim = rhs.state_dim;
    this->initialised = rhs.initialised;
}

void GaussianKernel::init(const unsigned int state_dim, const VectorXd &parameters) {
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

void GaussianKernel::save(const std::string filename) {
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

void GaussianKernel::load(const std::string filename) {
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

void GaussianKernel::save(std::ostream &out) {

    saveVectorToStream(out, this->parameters);
    saveVectorToStream(out, this->b);
    out << this->alpha << std::endl;
    out << this->state_dim << std::endl;
    out << this->initialised << std::endl;

}

void GaussianKernel::load(std::istream &in) {
    std::cout << "Gaussian Kernel load" << std::endl;
    readVectorFromStream(in, this->parameters);
    readVectorFromStream(in, this->b);
    in >> this->alpha;
    in >> this->state_dim;
    in >> this->initialised;
}


Kernel* GaussianKernel::internalCreateCopy(void) const {
    GaussianKernel* gk = new GaussianKernel(*this);
    return gk;
}

double GaussianKernel::eval(const VectorXd &x) {
    return this->eval(x,x);
}

double GaussianKernel::eval(const VectorXd &x, const VectorXd &y) {
    if (x.rows() != y.rows()) {
        throw OTLException("Hey, GaussianKernel cannot evaluate"
                           "vectors of different sizes");
    }

    if (x.rows() != this->state_dim) {
        throw OTLException("Hey, GaussianKernel says the vector to evaluate "
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

void GaussianKernel::eval(const VectorXd &x, const std::vector<VectorXd> &Y,
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
