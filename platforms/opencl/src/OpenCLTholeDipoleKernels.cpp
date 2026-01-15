/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTholeDipoleKernels.h"
#include "openmm/opencl/OpenCLArray.h"
#include "openmm/opencl/OpenCLFFT3D.h"

using namespace TholeDipolePlugin;
using namespace OpenMM;

OpenCLCalcTholeDipoleForceKernel::~OpenCLCalcTholeDipoleForceKernel() {
    if (fft != NULL)
        delete fft;
}

void OpenCLCalcTholeDipoleForceKernel::initialize(const System& system, const TholeDipoleForce& force) {
    CommonCalcTholeDipoleForceKernel::initialize(system, force);
    if (usePME) {
        OpenCLContext& cl = dynamic_cast<OpenCLContext&>(cc);
        fft = new OpenCLFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, false);
    }
}

void OpenCLCalcTholeDipoleForceKernel::computeFFT(bool forward) {
    OpenCLArray& grid1 = dynamic_cast<OpenCLContext&>(cc).unwrap(pmeGrid1);
    OpenCLArray& grid2 = dynamic_cast<OpenCLContext&>(cc).unwrap(pmeGrid2);
    if (forward)
        fft->execFFT(grid1, grid2, true);
    else
        fft->execFFT(grid2, grid1, false);
}
