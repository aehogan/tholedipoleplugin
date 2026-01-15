#ifndef OPENCL_THOLEDIPOLE_KERNELS_H_
#define OPENCL_THOLEDIPOLE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "CommonTholeDipoleKernels.h"
#include "openmm/opencl/OpenCLContext.h"

namespace OpenMM {
    class OpenCLFFT3D;
}

namespace TholeDipolePlugin {

class OpenCLCalcTholeDipoleForceKernel : public CommonCalcTholeDipoleForceKernel {
public:
    OpenCLCalcTholeDipoleForceKernel(const std::string& name, const OpenMM::Platform& platform,
                                     OpenMM::OpenCLContext& cl, const OpenMM::System& system) :
            CommonCalcTholeDipoleForceKernel(name, platform, cl, system), fft(NULL) {
    }
    ~OpenCLCalcTholeDipoleForceKernel();
    void initialize(const OpenMM::System& system, const TholeDipoleForce& force);
    void computeFFT(bool forward);
    bool useFixedPointChargeSpreading() const {
        return true;
    }
private:
    OpenMM::OpenCLFFT3D* fft;
};

} // namespace TholeDipolePlugin

#endif // OPENCL_THOLEDIPOLE_KERNELS_H_
