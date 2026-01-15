/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "CudaTholeDipoleKernels.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/OpenMMException.h"

using namespace TholeDipolePlugin;
using namespace OpenMM;
using namespace std;

CudaCalcTholeDipoleForceKernel::~CudaCalcTholeDipoleForceKernel() {
    ContextSelector selector(cc);
    if (hasInitializedFFT)
        cufftDestroy(fft);
}

void CudaCalcTholeDipoleForceKernel::initialize(const System& system, const TholeDipoleForce& force) {
    CommonCalcTholeDipoleForceKernel::initialize(system, force);
    if (usePME) {
        ContextSelector selector(cc);
        cufftResult result = cufftPlan3d(&fft, gridSizeX, gridSizeY, gridSizeZ,
                                         cc.getUseDoublePrecision() ? CUFFT_Z2Z : CUFFT_C2C);
        if (result != CUFFT_SUCCESS)
            throw OpenMMException("Error initializing FFT: " + cc.intToString(result));
        hasInitializedFFT = true;
    }
}

void CudaCalcTholeDipoleForceKernel::computeFFT(bool forward) {
    CudaArray& grid1 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid1);
    CudaArray& grid2 = dynamic_cast<CudaContext&>(cc).unwrap(pmeGrid2);
    if (forward) {
        if (cc.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) grid1.getDevicePointer(), (double2*) grid2.getDevicePointer(), CUFFT_FORWARD);
        else
            cufftExecC2C(fft, (float2*) grid1.getDevicePointer(), (float2*) grid2.getDevicePointer(), CUFFT_FORWARD);
    }
    else {
        if (cc.getUseDoublePrecision())
            cufftExecZ2Z(fft, (double2*) grid2.getDevicePointer(), (double2*) grid1.getDevicePointer(), CUFFT_INVERSE);
        else
            cufftExecC2C(fft, (float2*) grid2.getDevicePointer(), (float2*) grid1.getDevicePointer(), CUFFT_INVERSE);
    }
}
