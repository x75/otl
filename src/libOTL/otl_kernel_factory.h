/**
  OTL Kernel Factory Class.
  Copyright 2012 All Rights Reserved, Harold Soh
    haroldsoh@imperial.ac.uk
    haroldsoh@gmail.com
    http://www.haroldsoh.com

  Implements the kernel factory that provides a means of writing and reading
  kernels from kernel learning methods (like SOGP) in such a way that they
  don't have to "know" about the different kernel types.

  Please see LICENSE.txt for licensing.

  **/


#ifndef KERNEL_FACTORY_H_32189758397284931290843219832198
#define KERNEL_FACTORY_H_32189758397284931290843219832198

#include <map>
#include <string>
#include "otl_kernel.h"
#include "otl_kernel_gaussian.h"
#include "otl_kernel_recursive_gaussian.h"
#include "otl_kernel_recursive_equality_gaussian.h"

namespace OTL {

class KernelFactory
{
public:

    KernelFactory() {}

    ~KernelFactory() {
        kernels_type::iterator itr;
        for (itr = kernels.begin(); itr != kernels.end(); ++itr) {
            if (itr->second != NULL) {
                Kernel *k = itr->second;
                delete k; //CANNOT seem to delete this: to investigate
            }
        }
    }


    /**
    \brief gets the kernel using a string name
    \id a valid string containing the kernel's name
    */
    Kernel* get(const std::string& id) const {
        kernels_type::const_iterator itr = kernels.find(id);
        if (itr == kernels.end()) {
            throw OTLException("Did not find proper kernel");
        }
        return itr->second->createCopy();
    }

    /**
    \brief sets a kernel with a string name so that it can be retrived in the
    future
    \param id a string containing the kernel's name.
    \param kernel a pointer to the kernel you want to save.
    */
    void set(std::string id, Kernel* kernel) {
        kernels[id] = kernel->createCopy();
    }

    /**
      \brief overloaded assignment operator
      */
    KernelFactory& operator=(const KernelFactory &rhs) {
        kernels.clear();
        kernels_type::const_iterator itr;
        for (itr = rhs.kernels.begin(); itr != rhs.kernels.end(); ++itr) {
            this->kernels[(*itr).first] = (*itr).second->createCopy();
        }
        return *this;
    }

    /**
      \brief Copy constructor
      */
    KernelFactory(KernelFactory &rhs) {
        kernels_type::iterator itr;
        for (itr = rhs.kernels.begin(); itr != rhs.kernels.end(); ++itr) {
            this->kernels[(*itr).first] = (*itr).second->createCopy();
        }
    }

    /**
      \brief Returns a pointer to a copy of this kernel factory
      */
    KernelFactory *createCopy(void) {
        return new KernelFactory(*this);
    }


private:
    typedef std::map < std::string, Kernel* > kernels_type;
    kernels_type kernels;
};


}
#endif
