#ifndef OTL_RLS_H_329079579082190890387902619047
#define OTLRLS_H_329079579082190890387902619047

#include "otl_exception.h"
#include "otl_learning_algs.h"
#include <eigen3/Eigen/Dense>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class RLS : public LearningAlgorithm {
public:
    virtual void train(const VectorXd &input);
    virtual void predict(VectorXd &prediction, VectorXd &prediction_variance);
    virtual void reset();

    virtual void setParameters(double delta = 0.1, double lambda = 0.99, double noise = 0.0);
private:
    double delta;
    double lambda;
    double noise;
};


void RLS::train(const VectorXd &input) {
    return;
}

void RLS::predict(VectorXd &prediction, VectorXd &prediction_variance) {
    return;
}

void RLS::reset() {
    return;
}

void RLS::setParameters(double delta, double lambda, double noise) {

}

}
#endif
