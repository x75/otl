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
#include <istream>
#include <ostream>
#include <fstream>


namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;


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


}

#endif
