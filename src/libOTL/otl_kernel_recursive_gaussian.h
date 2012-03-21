#ifndef OTL_KERNEL_RecursiveGaussian_237017905890183921890489018490123213
#define OTL_KERNEL_RecursiveGaussian_237017905890183921890489018490123213

#include "otl_exception.h"
#include "otl_kernel.h"
#include "otl_helpers.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <cmath>
#include <istream>
#include <ostream>

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

class RecursiveGaussianKernel : public Kernel {
public:

    RecursiveGaussianKernel();
    RecursiveGaussianKernel(const RecursiveGaussianKernel &rhs);
    ~RecursiveGaussianKernel();
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

protected:
    virtual Kernel* internalCreateCopy(void) const;

private:
    VectorXd parameters;
    VectorXd b;
    double alpha;
    double rho;

    unsigned int state_dim;
    unsigned int input_dim;

    bool initialised;
};





}

#endif
