#include "otl_kernel_gaussian.h"

namespace OTL {

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
    } else if (parameters.rows() == 2) {
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
