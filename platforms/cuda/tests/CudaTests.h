/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_CUDATESTS_H_
#define OPENMM_CUDATESTS_H_

#include "openmm/Platform.h"

extern "C" void registerTholeDipoleCudaKernelFactories();
extern "C" void registerTholeDipoleReferenceKernelFactories();

using namespace OpenMM;

extern Platform* cudaPlatform;
extern Platform* referencePlatform;

inline void setupKernels(int argc, char* argv[]) {
    registerTholeDipoleReferenceKernelFactories();
    registerTholeDipoleCudaKernelFactories();
    cudaPlatform = &Platform::getPlatformByName("CUDA");
    referencePlatform = &Platform::getPlatformByName("Reference");
    if (argc > 1)
        cudaPlatform->setPropertyDefaultValue("CudaPrecision", std::string(argv[1]));
}

#endif // OPENMM_CUDATESTS_H_
