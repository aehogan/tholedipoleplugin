#ifndef COMMON_THOLEDIPOLE_KERNELS_H_
#define COMMON_THOLEDIPOLE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "TholeDipoleKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/common/ComputeArray.h"

namespace TholeDipolePlugin {

class CommonCalcTholeDipoleForceKernel : public CalcTholeDipoleForceKernel {
public:
    CommonCalcTholeDipoleForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc, const OpenMM::System& system) :
            CalcTholeDipoleForceKernel(name, platform), cc(cc), system(system), hasInitializedKernels(false), multipolesAreValid(false) {
    }
    ~CommonCalcTholeDipoleForceKernel();
    void initialize(const OpenMM::System& system, const TholeDipoleForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const TholeDipoleForce& force);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getTotalDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getElectrostaticPotential(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& inputGrid,
                                   std::vector<double>& outputElectrostaticPotential);
    void getSystemMultipoleMoments(OpenMM::ContextImpl& context, std::vector<double>& outputMultipoleMoments);

protected:
    class ForceInfo;
    void initializeScaleFactors();
    void initializeBSplineModuli();
    void computeInducedFieldDirect();
    bool iterateDipolesByDIIS(int iteration);
    void ensureMultipolesValid(OpenMM::ContextImpl& context);
    void setPeriodicBoxArgs(OpenMM::ContextImpl& context, OpenMM::ComputeKernel kernel, int index);
    void computeReciprocalBoxVectors(OpenMM::ContextImpl& context);
    double computePmeReciprocalField(OpenMM::ContextImpl& context, bool isFixedField);
    double computePmeSelfEnergy();
    double computePmeInducedRecipEnergy();
    virtual void computeFFT(bool forward) = 0;
    virtual bool useFixedPointChargeSpreading() const = 0;
    template <class T, class T4> void computeSystemMultipoleMomentsImpl(OpenMM::ContextImpl& context, std::vector<double>& outputMultipoleMoments);

    int numParticles, maxInducedIterations;
    int gridSizeX, gridSizeY, gridSizeZ;
    int diisBlockSize;  // Block size for DIIS kernel execution
    double pmeAlpha, inducedEpsilon;
    bool usePME, hasInitializedKernels, hasInitializedScaleFactors, multipolesAreValid;
    TholeDipoleForce::PolarizationType polarizationType;
    TholeDipoleForce::TholeDampingType dampingType;
    double tholeDampingParameter;
    double cutoffDistance;

    OpenMM::ComputeContext& cc;
    const OpenMM::System& system;

    std::vector<OpenMM::mm_int4> covalentFlagValues;
    std::vector<float> originalScaleFactors;  // Scale factors in original atom order

    OpenMM::ComputeArray multipoleParticles;
    OpenMM::ComputeArray localDipoles;
    OpenMM::ComputeArray labDipoles;
    OpenMM::ComputeArray inverseAtomIndex;  // Maps original index to GPU index
    OpenMM::ComputeArray polarizability;
    OpenMM::ComputeArray dampingAndThole;
    OpenMM::ComputeArray field;
    OpenMM::ComputeArray inducedField;
    OpenMM::ComputeArray inducedDipole;
    OpenMM::ComputeArray torque;
    OpenMM::ComputeArray covalentFlags;
    OpenMM::ComputeArray pairScaleFactors;  // mScale for each pair (i,j), indexed as i*paddedNumAtoms+j
    OpenMM::ComputeArray lastPositions;

    // Mutual polarization DIIS arrays
    OpenMM::ComputeArray inducedDipoleErrors;
    OpenMM::ComputeArray prevDipoles;
    OpenMM::ComputeArray prevErrors;
    OpenMM::ComputeArray diisMatrix;
    OpenMM::ComputeArray diisCoefficients;

    // PME arrays
    OpenMM::ComputeArray sphericalDipoles;
    OpenMM::ComputeArray fracDipoles;
    OpenMM::ComputeArray pmeGrid1;
    OpenMM::ComputeArray pmeGrid2;
    OpenMM::ComputeArray pmeGridLong;
    OpenMM::ComputeArray pmeBsplineModuliX;
    OpenMM::ComputeArray pmeBsplineModuliY;
    OpenMM::ComputeArray pmeBsplineModuliZ;
    OpenMM::ComputeArray pmePhi;
    OpenMM::ComputeArray pmePhid;
    OpenMM::ComputeArray pmeCphi;
    OpenMM::ComputeArray pmeEnergyBuffer;

    OpenMM::ComputeKernel computeMomentsKernel;
    OpenMM::ComputeKernel computeFixedFieldKernel;
    OpenMM::ComputeKernel computeInducedFieldKernel;
    OpenMM::ComputeKernel recordInducedDipolesKernel;
    OpenMM::ComputeKernel updateInducedFieldKernel;
    OpenMM::ComputeKernel electrostaticsKernel;
    OpenMM::ComputeKernel mapTorqueKernel;
    OpenMM::ComputeKernel computePotentialKernel;

    // DIIS kernels
    OpenMM::ComputeKernel recordDIISDipolesKernel;
    OpenMM::ComputeKernel buildMatrixKernel;
    OpenMM::ComputeKernel solveMatrixKernel;

    // PME kernels
    OpenMM::ComputeKernel pmeSpreadFixedMultipolesKernel;
    OpenMM::ComputeKernel pmeSpreadInducedDipolesKernel;
    OpenMM::ComputeKernel pmeFinishSpreadChargeKernel;
    OpenMM::ComputeKernel pmeConvolutionKernel;
    OpenMM::ComputeKernel pmeFixedPotentialKernel;
    OpenMM::ComputeKernel pmeInducedPotentialKernel;
    OpenMM::ComputeKernel pmeFixedForceKernel;
    OpenMM::ComputeKernel pmeInducedForceKernel;
    OpenMM::ComputeKernel pmeRecordInducedFieldDipolesKernel;
    OpenMM::ComputeKernel pmeTransformMultipolesKernel;
    OpenMM::ComputeKernel pmeTransformPotentialKernel;
    OpenMM::ComputeKernel pmeRecipForceKernel;

    // Cached box vectors for PME
    OpenMM::mm_double4 recipBoxVecXD, recipBoxVecYD, recipBoxVecZD;
    OpenMM::mm_double4 periodicBoxVecXD, periodicBoxVecYD, periodicBoxVecZD, periodicBoxSizeD;
    OpenMM::mm_float4 recipBoxVecXF, recipBoxVecYF, recipBoxVecZF;
    OpenMM::mm_float4 periodicBoxVecXF, periodicBoxVecYF, periodicBoxVecZF, periodicBoxSizeF;

    static constexpr int PmeOrder = 5;
    static constexpr int MaxPrevDIISDipoles = 20;
};

} // namespace TholeDipolePlugin

#endif /*COMMON_THOLEDIPOLE_KERNELS_H_*/
