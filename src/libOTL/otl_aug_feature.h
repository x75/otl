#ifndef OTL_AUGMENTED_FEATURE_H_283901790654792078109839078943267862190875436284789210
#define OTL_AUGMENTED_FEATURE_H_283901790654792078109839078943267862190875436284789210

#include "otl_exception.h"
#include <string>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace OTL {

/**
  \brief AugmentedFeatures is a virtual class that specifies the necessary
  functions that any derived feature maker should implement i.e., update,
  getFeatures, setFeatures and reset methods.
  **/
class AugmentedFeatures {
public:
    virtual void update(const VectorXd &input) = 0;
    virtual void getFeatures(VectorXd &features) = 0;
    virtual void setFeatures(const VectorXd &features) = 0;
    virtual void reset(void) = 0;

    virtual void save(std::string filename) = 0;
    virtual void load(std::string filename) = 0;
};

}

#endif
