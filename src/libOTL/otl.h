/**
  Online Temporal Learning (OTL)
  A simple library for online time-series learning.

  Copyright 2012 Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com

  This code is free for use in non-commercial settings: at academic institutions, teaching institutes,
  research purposes and non-profit research.

  If you do use this code in your research, please cite:

    @inproceedings{Soh2012,
        Author = {Harold Soh and Yiannis Demiris},
        Booktitle = {International Joint Conference on Neural Networks (IJCNN)},
        Title = {Iterative Temporal Learning and Prediction with the Sparse Online Echo State Gaussian Process},
        Year = {2012}}

  Also free for use by hobbyists and students who are not making money from it.

  If you use this code for commercial use, you must contact the author for commercial
  licenses, i.e., if you make some money out of this software, I think it's fair that I get some of it
  so that I can support my family and the causes I believe in.

  As usual, no warranties are implied nor guaranteed. Use at your own risk. That said, bug reports are
  welcome.

  **/

#ifndef OTL_H_83920174821689372901758964786849732189371289371289461
#define OTL_H_83920174821689372901758964786849732189371289371289461

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include <string>
#include "otl_exception.h"
#include "otl_aug_state.h"

#include "otl_learning_algs.h"
#include "otl_rls.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

//class OTL {
//public:
//    void update(const VectorXd &input);
//    void predict(VectorXd &prediction, VectorXd &pred_variance);
//    void train(const VectorXd &target);

//    void save(const std::string &filename);
//    void load(const std::string &filename);

//};


}

#endif






