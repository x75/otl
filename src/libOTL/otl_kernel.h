#ifndef OTL_KERNELS_473298757871098392107489372948904821
#define OTL_KERNELS_473298757871098392107489372948904821

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <istream>
#include <ostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class Kernel {
public:
    Kernel(std::string kernel_name) : name(kernel_name) { }

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

    virtual Kernel* createCopy() = 0;

    const std::string name;

};



}

#endif
