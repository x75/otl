/**
  OTL Augmented state virtual class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Use this class if you want to derive your own augmented states.

  Please see LICENSE.txt for licensing.

  **/


#ifndef OTL_AUGMENTED_STATE_H_283901790654792078109839078943267862190875436284789210
#define OTL_AUGMENTED_STATE_H_283901790654792078109839078943267862190875436284789210

#include "otl_exception.h"
#include "eigen3/Eigen/Dense"
#include <string>
#include <fstream>

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
  \brief AugmentedState is a virtual class that specifies the necessary
  functions that any derived state maker should implement i.e., update,
  getState, setState and reset methods.
  **/
class AugmentedState {
public:
    virtual void update(const VectorXd &input) = 0;
    virtual void getState(VectorXd &State) = 0;
    virtual void setState(const VectorXd &State) = 0;
    virtual void reset(void) = 0;
    virtual unsigned int getStateSize() = 0;

    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename) = 0;
};

}

#endif
