/**
  Online Temporal Learning (OTL)
  A library for online time-series learning. OTL was originally
  created to implement the OESGP and STORKGP algorithms. It can be used/extended
  to create other novel online temporal learning algorithms. Its only dependecy
  is the Eigen C++ library.

  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  This code is DUAL-LICENSED. Please see LICENSE.txt for details.

  **/

#ifndef OTL_H_83920174821689372901758964786849732189371289371289461
#define OTL_H_83920174821689372901758964786849732189371289371289461

#include <eigen3/Eigen/Dense>

#include "otl_exception.h"
#include "otl_helpers.h"
#include "otl_aug_state.h"
#include "otl_window.h"
#include "otl_reservoir.h"

#include "otl_learning_algs.h"
#include "otl_rls.h"
#include "otl_sogp.h"

#include "otl_kernel_factory.h"
#include "otl_kernel.h"
#include "otl_kernel_gaussian.h"
#include "otl_kernel_recursive_gaussian.h"
#include "otl_kernel_recursive_equality_gaussian.h"

namespace OTL {

using Eigen::MatrixXd;
using Eigen::VectorXd;

}
#endif






