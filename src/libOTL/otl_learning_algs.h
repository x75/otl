#ifndef LEARNING_ALGS_H_329079579082190890387902619047
#define LEARNING_ALGS_H_329079579082190890387902619047

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

class LearningAlgorithm {
public:
    virtual void train(const VectorXd &input) = 0;
    virtual void predict(VectorXd &prediction, VectorXd &prediction_variance) = 0;
    virtual void reset() = 0;
};

}


#endif
