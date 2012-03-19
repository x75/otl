#ifndef OTL_KERNELS_473298757871098392107489372948904821
#define OTL_KERNELS_473298757871098392107489372948904821

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class Kernel {
public:
    virtual void init(VectorXd &parameters) = 0;
    virtual void getParameters(VectorXd &parameters) = 0;
    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename) = 0;

    virtual double eval(VectorXd &x) = 0;
    virtual double eval(VectorXd &x, VectorXd &y) = 0;
    virtual void eval(VectorXd &x, MatrixXd &Y, VectorXd &kern_vals) = 0;
};



}

#endif
