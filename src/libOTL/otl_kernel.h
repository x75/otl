/**
  OTL Kernel virtual class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Use this class if you want to derive your own Kernels.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_KERNELS_473298757871098392107489372948904821
#define OTL_KERNELS_473298757871098392107489372948904821

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <istream>
#include <ostream>
#include <vector>
#include <fstream>
#include <typeinfo>

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;


class Kernel {
public:

    Kernel(std::string kernel_name) : name(kernel_name) { }
    virtual ~Kernel() { };

    virtual std::string getName() { return name; }

    virtual void init(const unsigned int state_dim, const VectorXd &parameters) = 0;
    virtual void getParameters(VectorXd &parameters) = 0;
    virtual void save(const std::string filename) = 0;
    virtual void load(const std::string filename) = 0;
    virtual void save(std::ostream &out) = 0;
    virtual void load(std::istream &in) = 0;


    virtual double eval(const VectorXd &x) = 0;
    virtual double eval(const VectorXd &x, const VectorXd &y) = 0;
    virtual void eval(const VectorXd &x, const std::vector<VectorXd> &Y,
                      VectorXd &kern_vals) = 0;

    Kernel* createCopy() {
        Kernel *result = internalCreateCopy();

        if (typeid(*result) != typeid(*this)) {
            throw OTLException("internalCreateCopy not properly overidden.");
        }
        return result;
    }

private:
    const std::string name;
    Kernel(const Kernel &rhs);
    Kernel &operator=(const Kernel &rhs);

protected:
    virtual Kernel* internalCreateCopy() const = 0;

};



}

#endif
