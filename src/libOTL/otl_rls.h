/**
  OTL Recursive Least Squares Class
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the basic recursive least squares algoritm

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_RLS_H_329079579082190890387902619047
#define OTL_RLS_H_329079579082190890387902619047

#include "otl_exception.h"
#include "otl_learning_algs.h"
#include "otl_helpers.h"
#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;


class RLS : public LearningAlgorithm {
public:
    RLS();
    RLS(RLS &rhs);

    virtual void train(const VectorXd &state, const VectorXd &output);
    virtual void predict(const VectorXd &state, VectorXd &prediction, VectorXd &prediction_variance);
    virtual void reset();
    virtual void save(std::string filename);
    virtual void load(std::string filename);

    virtual void init(unsigned int state_dim, unsigned int output_dim, double delta = 0.1, double lambda = 0.99, double noise = 0.0);

private:
    bool initialized; //initialised

    double delta;
    double lambda;
    double noise;

    unsigned int state_dim;
    unsigned int output_dim;

    std::vector <MatrixXd> P_rls;
    std::vector <VectorXd> w_rls;
};



}
#endif
