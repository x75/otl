#include "otl_kernel_recursive_equality_gaussian.h"

namespace OTL {

RecursiveEqualityGaussianKernel::RecursiveEqualityGaussianKernel() : Kernel("RecursiveEqualityGaussian"){
    this->initialised = false;
}

RecursiveEqualityGaussianKernel::~RecursiveEqualityGaussianKernel() {
    return;
}

RecursiveEqualityGaussianKernel::RecursiveEqualityGaussianKernel(const RecursiveEqualityGaussianKernel &rhs) : Kernel("RecursiveGaussian") {
    this->parameters = rhs.parameters;
    this->b = rhs.b;
    this->rho = rhs.rho;
    this->alpha = rhs.alpha;
    this->state_dim = rhs.state_dim;
    this->input_dim = rhs.input_dim;
    this->initialised = rhs.initialised;
}

void RecursiveEqualityGaussianKernel::init(const unsigned int state_dim, const VectorXd &parameters) {
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
    } else if (parameters.rows() == 4) {
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

void RecursiveEqualityGaussianKernel::getParameters(VectorXd &parameters) {
    if (!this->initialised) {
        throw OTLException("You can't get parameters from the RecursiveGaussian Kernel"
                           "when it has not been initialised yet.");
    }

    parameters = this->parameters;
}

void RecursiveEqualityGaussianKernel::save(const std::string filename) {
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

void RecursiveEqualityGaussianKernel::load(const std::string filename) {
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

void RecursiveEqualityGaussianKernel::save(std::ostream &out) {

    saveVectorToStream(out, this->parameters);
    saveVectorToStream(out, this->b);
    out << this->alpha << std::endl;
    out << this->rho << std::endl;
    out << this->input_dim << std::endl;
    out << this->state_dim << std::endl;
    out << this->initialised << std::endl;

}

void RecursiveEqualityGaussianKernel::load(std::istream &in) {
    readVectorFromStream(in, this->parameters);
    readVectorFromStream(in, this->b);
    in >> this->alpha;
    in >> this->rho;
    in >> this->input_dim;
    in >> this->state_dim;
    in >> this->initialised;
}


Kernel* RecursiveEqualityGaussianKernel::internalCreateCopy(void) const {
    RecursiveEqualityGaussianKernel* gk = new RecursiveEqualityGaussianKernel(*this);
    return gk;
}

double RecursiveEqualityGaussianKernel::eval(const VectorXd &x) {
    return this->eval(x,x);
}

double RecursiveEqualityGaussianKernel::eval(const VectorXd &x, const VectorXd &y) {
    if (x.rows() != y.rows()) {
        throw OTLException("Hey, RecursiveEqualityGaussianKernel cannot evaluate"
                           "vectors of different sizes");
    }

    if (x.rows() != this->state_dim) {
        throw OTLException("Hey, RecursiveEqualityGaussianKernel says the vector to evaluate "
                           "has a different size than the initialisation.");
    }

    double kval = 1.0;
    int n_segments = this->state_dim/this->input_dim;
    for (int i=(n_segments-1); i>=0; i--) {
        VectorXd diffxy = x.segment(i*this->input_dim, this->input_dim) -
                y.segment(i*this->input_dim, this->input_dim);

        //check for approximate equality
        for (unsigned int k=0; k<diffxy.rows(); k++) {
            if (fabs(diffxy(k)) < 1e-9) {
                //they are approximately equal or equal
                diffxy(k) = 0.0;
            } else {
                //they are unequal
                diffxy(k) = 1.0;
            }
        }

        double val = 0.0;
        for (unsigned int j=0; j<this->input_dim; j++) {
            val += diffxy(j)*b(j);
        }
        kval = this->alpha*exp(-val)*exp( (kval - 1)/this->rho );
    }
    return kval;
}

void RecursiveEqualityGaussianKernel::eval(const VectorXd &x, const std::vector<VectorXd> &Y,
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
