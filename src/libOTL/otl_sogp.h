/**
  OTL Sparse Online Gaussian Process Class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the Sparse online gaussian process (SOGP) proposed by Csato
  and Opper.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_SOGP_H_750279082490178956479784073190
#define OTL_SOGP_H_750279082490178956479784073190

#include "otl_exception.h"
#include "otl_learning_algs.h"
#include "otl_helpers.h"
#include "otl_kernel.h"
#include "otl_kernel_factory.h"

#include <eigen3/Eigen/Dense>
#include <string>
#include <fstream>
#include <vector>


namespace OTL {


using Eigen::MatrixXd;
using Eigen::VectorXd;


class SOGP : public LearningAlgorithm {
public:
    /**
      \brief choice of regression or classification mode
      **/
    enum { REGRESSION, CLASSIFICATION };

    /**
      \brief deletion of basis vector criteria
      **/
    enum { NORM, MINIMAX};

    SOGP();
    SOGP(SOGP &rhs);
    ~SOGP();

    /**
      \brief Training Function. Trains the SOGP using a state and output
      \param state the "input" state of type VectorXd.
      \param output the desired output of type VectorXd.
      **/
    virtual void train(const VectorXd &state, const VectorXd &output);

    /**
      \brief Prediction using state
      \param state the "input" state of type VectorXd.
      \param prediction the predicted output of type VectorXd.
      \param prediction_variance the variance (uncertainty) of the predicted output
      **/
    virtual void predict(const VectorXd &state, VectorXd &prediction, VectorXd &prediction_variance);

    /**
      \brief resets the SOGP
      */
    virtual void reset();

    /**
      \brief saves the sogp to a file with filename
      */
    virtual void save(std::string filename);

    /**
      \brief loads the sogp from filename using the default kernel factory
      */
    virtual void load(std::string filename);


    /**
      \brief loads the sogp from filename
      \param kernel_factory an OTL::KernelFactory object with a list of kernels.
                This is used for reading and writing to disk. You MUST provide
                a valid kernel factory to allow for correct reading/writing.
      */
    virtual void load(std::string filename, const KernelFactory &kernel_factory);

    /**
      \brief Initialises the SOGP
      \param state_dim how big is the state that we want to regress from
      \param output_dim how big is the output state
            (the number of classes if the problem type is CLASSIFICATION)
      \param kernel an OTL::Kernel object with the kernel you want to use
      \param noise the noise parameter (application dependent)
      \param epsilon threshold parameter (typically small 1e-4)
      \param capacity the capacity of the SOGP (application dependent)
      \param problem_type SOGP::REGRESSION or SOGP::CLASSIFICATION
      \param deletion_criteria SOGP::NORM or SOGP::MINIMAX (this also applies for
                multi-dimensional outputs)
      **/
    virtual void init(unsigned int state_dim, unsigned int output_dim,
                      Kernel &kernel,
                      double noise,
                      double epsilon,
                      unsigned int capacity,
                      int problem_type = SOGP::REGRESSION,
                      int deletion_criteria = SOGP::NORM);


    /**
      \brief returns the current size of the SOGP (number of basis vectors)
      */
    virtual unsigned int getCurrentSize(void);


    /**
      \brief sets the kernel factory
      */
    virtual void setKernelFactory(KernelFactory &kernel_factory);



private:
    bool initialized; //initialised?

    unsigned int state_dim;
    unsigned int output_dim;
    unsigned int current_size;

    int problem_type;
    int deletion_criteria;

    Kernel *kernel;
    KernelFactory kernel_factory;

    double epsilon;
    double noise;
    unsigned int capacity;

    MatrixXd alpha;
    MatrixXd C;
    MatrixXd Q;

    std::vector<VectorXd> basis_vectors;

    void reduceBasisVectorSet(unsigned int index);

};


}
#endif
