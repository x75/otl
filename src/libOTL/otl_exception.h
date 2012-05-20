/**
  OTL Exception class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements a very simple exception class. This should be improved in
  future iterations.

  Please see LICENSE.txt for licensing.

  **/

#ifndef OTL_EXCEPTION_43890928098790178928390128908493
#define OTL_EXCEPTION_43890928098790178928390128908493

#include <string>
#include <iostream>

namespace OTL {

class OTLException {
public:
    OTLException(std::string error_msg) {
        this->error_msg = error_msg;
    }
    void showError() {
        std::cerr << error_msg << std::endl;
    }
private:
    std::string error_msg;
};
}

#endif
