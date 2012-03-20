#ifndef KERNEL_FACTORY_H_32189758397284931290843219832198
#define KERNEL_FACTORY_H_32189758397284931290843219832198

#include <map>
#include <string>
#include "otl_kernel.h"
#include "otl_kernel_gaussian.h"
#include "otl_kernel_recursive_gaussian.h"
namespace OTL {
// Kernel Factory
class KernelFactory
{
public:
  Kernel* get(std::string id) {
      return kernels[id];
  }
  void set(std::string id, Kernel* kernel) {
      kernels[id] = kernel->createCopy();
  }

  KernelFactory() {}

  KernelFactory(KernelFactory &rhs) {
      kernels = rhs.kernels;
  }

  KernelFactory *createCopy(void) {
      return new KernelFactory(*this);
  }


private:
  typedef std::map < std::string, Kernel* > kernels_type;
  kernels_type kernels;
};


/**
  \brief initialises the kernel factory
  **/
void initKernelFactory(KernelFactory *kfact) {
    GaussianKernel gk;
    RecursiveGaussianKernel rgk;

    kfact->set(gk.getName(), &gk);
    kfact->set(rgk.getName(), &rgk);
}


}
#endif
