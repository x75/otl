#include "otl_sogp.h"

namespace OTL {


SOGP::SOGP(void) {
    this->kernel = NULL;
    initKernelFactory(this->kernel_factory);
    this->initialized = false;
}

SOGP::SOGP(SOGP &rhs) : LearningAlgorithm() {
    this->kernel = rhs.kernel->createCopy();
    this->state_dim = rhs.state_dim;
    this->output_dim = rhs.output_dim;
    this->kernel_factory = kernel_factory;
    this->noise = rhs.noise;
    this->epsilon = rhs.epsilon;
    this->capacity = rhs.capacity;
    this->current_size = rhs.current_size;

    //initialise the matrices : not that the capacity is +1 since
    //we allow it to grow one more before reducing.
    this->alpha = rhs.alpha;
    this->C = rhs.C;
    this->Q = rhs.Q;

    for (unsigned int i=0; i<rhs.basis_vectors.size(); i++) {
        this->basis_vectors.push_back(rhs.basis_vectors[i]);
    }

    this->initialized = rhs.initialized;
}

SOGP::~SOGP() {
    if (this->kernel != NULL) delete this->kernel;
}

void SOGP::train(const VectorXd &state, const VectorXd &output) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("SOGP not yet initialised");
    }

    double kstar = this->kernel->eval(state);

    //we are just starting.
    if (this->current_size == 0) {

        this->alpha.block(0,0,1, this->output_dim) = (output.array() / (kstar + this->noise)).transpose();
        this->C.block(0,0,1,1) = VectorXd::Ones(1)*-1/(kstar + this->noise);
        this->Q.block(0,0,1,1) = VectorXd::Ones(1)*1/(kstar);
        this->basis_vectors.push_back(state);
        this->current_size++;
        return;
    }

    //Test if this is a "novel" state
    VectorXd k;
    this->kernel->eval(state, this->basis_vectors, k);
    //cache Ck
    VectorXd Ck = this->C.block(0,0, this->current_size, this->current_size)*k;

    VectorXd m = k.transpose()*this->alpha.block(0,0,this->current_size, this->output_dim);
    double s2 = kstar + (k.dot(Ck));

    if (s2 < 1e-12) {
        //std::cout << "s2: " << s2 << std::endl;
        s2 = 1e-12;
    }

    double r = -1.0/(s2 + this->noise);
    VectorXd q = (output - m)*(-r);
    VectorXd ehat = this->Q.block(0,0, this->current_size, this->current_size)*k;

    double gamma = kstar - k.dot(ehat);
    double eta = 1.0/(1.0 + gamma*r);

    if (gamma < 1e-12) {
        gamma = 0.0;
    }

    if (gamma >= this->epsilon*kstar) {
        //perform a full update
        VectorXd s = Ck;
        s.conservativeResize(this->current_size + 1);
        s(this->current_size) = 1;


        //update Q (inverse of C)
        ehat.conservativeResize(this->current_size+1);
        ehat(this->current_size) = -1;

        MatrixXd diffQ = Q.block(0,0,this->current_size+1, this->current_size+1)
                + (ehat*ehat.transpose())*(1.0/gamma);
        Q.block(0,0,this->current_size+1, this->current_size+1) = diffQ;


        //update alpha
        MatrixXd diffAlpha = alpha.block(0,0, this->current_size+1, this->output_dim)
                + (s*q.transpose());
        alpha.block(0,0, this->current_size+1, this->output_dim) = diffAlpha;

        //update C
        MatrixXd diffC = C.block(0,0, this->current_size+1, this->current_size+1)
                + r*(s*s.transpose());
        C.block(0,0, this->current_size+1, this->current_size+1) = diffC;

        //add to basis vectors
        this->basis_vectors.push_back(state);

        //increment current size
        this->current_size++;

    } else {
        //perform a sparse update
        VectorXd s = Ck + ehat;

        //update alpha
        MatrixXd diffAlpha = alpha.block(0,0, this->current_size, this->output_dim)
                + s*((q*eta).transpose());
        alpha.block(0,0, this->current_size, this->output_dim) = diffAlpha;

        //update C
        MatrixXd diffC = C.block(0,0, this->current_size, this->current_size) +
                r*eta*(s*s.transpose());
        C.block(0,0, this->current_size, this->current_size) = diffC;
    }

    //check if we need to reduce size
    if (this->basis_vectors.size() > this->capacity) {
        //std::cout << "Reduction!" << std::endl;
        double min_val = (alpha.row(0)).squaredNorm()/(Q(0,0) + C(0,0));
        unsigned int min_index = 0;
        for (unsigned int i=1; i<this->basis_vectors.size(); i++) {
            double scorei = (alpha.row(i)).squaredNorm()/(Q(i,i) + C(i,i));
            if (scorei < min_val) {
                min_val = scorei;
                min_index = i;
            }
        }

        this->reduceBasisVectorSet(min_index);
    }

    return;
}

void SOGP::reduceBasisVectorSet(unsigned int index) {

    unsigned int end = this->current_size-1;
    VectorXd zero_vector = VectorXd::Zero(this->current_size);

    VectorXd alpha_star = this->alpha.row(index);
    VectorXd last_item = this->alpha.row(end);
    alpha.block(index,0,1,this->output_dim) = last_item.transpose();
    alpha.block(end,0,1, this->output_dim) = VectorXd::Zero(this->output_dim).transpose();
    double cstar = this->C(index, index);
    VectorXd Cstar = this->C.col(index);
    Cstar(index) = Cstar(end);
    Cstar.conservativeResize(end);
    VectorXd Crep = C.col(end);
    Crep(index) = Crep(end);
    C.block(index, 0, 1, this->current_size) = Crep.transpose();
    C.block(0, index, this->current_size, 1) = Crep;
    C.block(end, 0, 1, this->current_size) = zero_vector.transpose();
    C.block(0, end, this->current_size,1) = zero_vector;

    double qstar = this->Q(index, index);
    VectorXd Qstar = this->Q.col(index);
    Qstar(index) = Qstar(end);
    Qstar.conservativeResize(end);
    VectorXd Qrep = Q.col(end);
    Qrep(index) = Qrep(end);
    Q.block(index, 0, 1, this->current_size) = Qrep.transpose();
    Q.block(0, index, this->current_size, 1) = Qrep;
    Q.block(end, 0, 1, this->current_size) = zero_vector.transpose();
    Q.block(0, end, this->current_size,1) = zero_vector;

    VectorXd qc = (Qstar + Cstar)/(qstar + cstar);
    for (unsigned int i=0; i<this->output_dim; i++) {
        VectorXd diffAlpha = alpha.block(0,i,end,1) - alpha_star(i)*qc;
        alpha.block(0,i,end,1) = diffAlpha;
    }

    MatrixXd oldC = C.block(0,0, end, end);
    C.block(0,0, end,end) = oldC + (Qstar*Qstar.transpose())/qstar -
            ((Qstar + Cstar)*((Qstar + Cstar).transpose()))/(qstar+cstar);

    MatrixXd oldQ = Q.block(0,0,end,end);
    Q.block(0,0, end, end) = oldQ - (Qstar*Qstar.transpose())/qstar;

    this->basis_vectors[index] = this->basis_vectors[end];
    this->basis_vectors.pop_back();

    this->current_size = end;
}

void SOGP::predict(const VectorXd &state, VectorXd &prediction,
                  VectorXd &prediction_variance) {
    //check if we have initialised the system
    if (!this->initialized) {
        throw OTLException("SOGP not yet initialised");
    }

    double kstar = kernel->eval(state,state);

    //check if we not been trained
    if (this->current_size == 0) {
        prediction = VectorXd::Zero(this->output_dim);
        prediction_variance = VectorXd::Ones(this->output_dim)*
                (kstar + this->noise);
        return;
    }

    VectorXd k;
    kernel->eval(state, this->basis_vectors, k);
    //std::cout << "K: \n" << k << std::endl;
    //std::cout << "alpha: \n" << this->alpha.block(0,0,this->current_size, this->output_dim) << std::endl;

    prediction = k.transpose() *this->alpha.block(0,0,this->current_size, this->output_dim);
    prediction_variance = VectorXd::Ones(this->output_dim)*
            (k.dot(this->C.block(0,0, this->current_size, this->current_size)*k)
             + kstar + this->noise);

    return;
}

void SOGP::reset() {
    this->alpha = MatrixXd::Zero(this->capacity+1, this->output_dim);
    this->C = MatrixXd::Zero(this->capacity+1, this->capacity+1);
    this->Q = MatrixXd::Zero(this->capacity+1, this->capacity+1);

    this->initialized = true;
    return;
}

void SOGP::save(std::string filename) {
    std::ofstream out;
    try {
        out.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    out << this->kernel->getName() << std::endl;
    kernel->save(out);

    out << this->state_dim << std::endl;
    out << this->output_dim << std::endl;
    out << this->noise << std::endl;
    out << this->epsilon << std::endl;
    out << this->capacity << std::endl;
    out << this->current_size << std::endl;
    out << this->initialized << std::endl;

    //initialise the matrices : not that the capacity is +1 since
    //we allow it to grow one more before reducing.
    saveMatrixToStream(out, this->alpha);
    saveMatrixToStream(out, this->C);
    saveMatrixToStream(out, this->Q);

    for (unsigned int i=0; i<basis_vectors.size(); i++) {
        saveVectorToStream(out, basis_vectors[i]);
    }

    out.close();
}

void SOGP::load(std::string filename) {
    this->load(filename, this->kernel_factory);
}

void SOGP::load(std::string filename, const KernelFactory &kernel_factory) {
    std::ifstream in;
    try {
        in.open(filename.c_str());
    } catch (std::exception &e) {
        std::string error_msg = "Cannot open ";
        error_msg += filename + " for writing";
        throw OTLException(error_msg);
    }

    std::string kernel_name;
    in >> kernel_name;

    delete kernel;
    this->kernel = kernel_factory.get(kernel_name);
    this->kernel->load(in);

    in >> this->state_dim;
    in >> this->output_dim;
    in >> this->noise;
    in >> this->epsilon;
    in >> this->capacity;
    in >> this->current_size;
    in >> this->initialized;

    readMatrixFromStream(in, this->alpha);
    readMatrixFromStream(in, this->C);
    readMatrixFromStream(in, this->Q);

    for (unsigned int i=0; i<this->current_size; i++) {
        VectorXd temp;
        readVectorFromStream(in, temp);
        this->basis_vectors.push_back(temp);
    }

    in.close();
}


void SOGP::init(unsigned int state_dim, unsigned int output_dim,
                Kernel &kernel,
                double noise,
                double epsilon,
                unsigned int capacity) {

    this->kernel = kernel.createCopy();
    this->state_dim = state_dim;
    this->output_dim = output_dim;
    this->noise = noise;
    this->epsilon = epsilon;
    this->capacity = capacity;
    this->current_size = 0;

    //initialise the matrices : not that the capacity is +1 since
    //we allow it to grow one more before reducing.
    this->alpha = MatrixXd::Zero(this->capacity+1, this->output_dim);
    this->C = MatrixXd::Zero(this->capacity+1, this->capacity+1);
    this->Q = MatrixXd::Zero(this->capacity+1, this->capacity+1);

    this->initialized = true;
}

unsigned int SOGP::getCurrentSize(void) {
    return this->current_size;
}

void SOGP::setKernelFactory(KernelFactory &kernel_factory) {
    this->kernel_factory = kernel_factory;
}

}
