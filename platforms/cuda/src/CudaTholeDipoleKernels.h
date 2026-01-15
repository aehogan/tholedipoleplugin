#ifndef CUDA_THOLEDIPOLE_KERNELS_H_
#define CUDA_THOLEDIPOLE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "CommonTholeDipoleKernels.h"
#include "openmm/cuda/CudaContext.h"
#include <cufft.h>

namespace TholeDipolePlugin {

class CudaCalcTholeDipoleForceKernel : public CommonCalcTholeDipoleForceKernel {
public:
    CudaCalcTholeDipoleForceKernel(const std::string& name, const OpenMM::Platform& platform,
                                   OpenMM::CudaContext& cu, const OpenMM::System& system) :
            CommonCalcTholeDipoleForceKernel(name, platform, cu, system), hasInitializedFFT(false) {
    }
    ~CudaCalcTholeDipoleForceKernel();
    void initialize(const OpenMM::System& system, const TholeDipoleForce& force);
    void computeFFT(bool forward);
    bool useFixedPointChargeSpreading() const {
        return cc.getUseDoublePrecision();
    }
private:
    bool hasInitializedFFT;
    cufftHandle fft;
};

} // namespace TholeDipolePlugin

#endif // CUDA_THOLEDIPOLE_KERNELS_H_
