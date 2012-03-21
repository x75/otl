#ifndef LEARNING_ALGS_H_329079579082190890387902619047
#define LEARNING_ALGS_H_329079579082190890387902619047

#include "otl_exception.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>


namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;


class LearningAlgorithm {
public:
    virtual void train(const VectorXd &state, const VectorXd &output) = 0;
    virtual void predict(const VectorXd &state, VectorXd &prediction, VectorXd &prediction_variance) = 0;
    virtual void reset() = 0;

    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename) = 0;

};

}


#endif
