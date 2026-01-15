/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_OPENCLTESTS_H_
#define OPENMM_OPENCLTESTS_H_

#include "openmm/Platform.h"

extern "C" void registerTholeDipoleOpenCLKernelFactories();
extern "C" void registerTholeDipoleReferenceKernelFactories();

using namespace OpenMM;

extern Platform* openclPlatform;
extern Platform* referencePlatform;

inline void setupKernels(int argc, char* argv[]) {
    registerTholeDipoleReferenceKernelFactories();
    registerTholeDipoleOpenCLKernelFactories();
    openclPlatform = &Platform::getPlatformByName("OpenCL");
    referencePlatform = &Platform::getPlatformByName("Reference");
    if (argc > 1)
        openclPlatform->setPropertyDefaultValue("OpenCLPrecision", std::string(argv[1]));
}

#endif // OPENMM_OPENCLTESTS_H_
