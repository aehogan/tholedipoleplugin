#ifndef REFERENCE_THOLEDIPOLE_KERNELS_H_
#define REFERENCE_THOLEDIPOLE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                            OpenMMTholeDipole                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2024 Stanford University and the Authors.           *
 * Authors:                                                                   *
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
#include "openmm/Platform.h"
#include "openmm/Vec3.h"
#include <vector>

namespace TholeDipolePlugin {

class ReferenceTholeDipoleForce;

/**
 * This kernel is invoked by TholeDipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcTholeDipoleForceKernel : public CalcTholeDipoleForceKernel {
public:
    ReferenceCalcTholeDipoleForceKernel(const std::string& name, const OpenMM::Platform& platform, const OpenMM::System& system);
    ~ReferenceCalcTholeDipoleForceKernel();
    
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the TholeDipoleForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const TholeDipoleForce& force);
    
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    
    /**
     * Calculate the induced dipoles.
     *
     * @param context        the context in which to execute this kernel
     * @param outputDipoles  induced dipoles are copied into this
     */
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& outputDipoles);
    
    /**
     * Get the permanent dipoles rotated into the lab frame.
     *
     * @param context        the context in which to execute this kernel
     * @param outputDipoles  permanent dipoles in the lab frame are copied into this
     */
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& outputDipoles);
    
    /**
     * Get the total dipoles (permanent + induced).
     *
     * @param context        the context in which to execute this kernel
     * @param outputDipoles  total dipoles are copied into this
     */
    void getTotalDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& outputDipoles);
    
    /**
     * Calculate the electrostatic potential at a set of points.
     *
     * @param context                      the context in which to execute this kernel
     * @param inputGrid                    input grid points
     * @param outputElectrostaticPotential output potential
     */
    void getElectrostaticPotential(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& inputGrid,
                                    std::vector<double>& outputElectrostaticPotential);
    
    /**
     * Calculate the system multipole moments.
     *
     * @param context                      the context in which to execute this kernel
     * @param outputMultipoleMoments       output multipole moments
     */
    void getSystemMultipoleMoments(OpenMM::ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the TholeDipoleForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const TholeDipoleForce& force);
    
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

private:
    ReferenceTholeDipoleForce* setupReferenceTholeDipoleForce(OpenMM::ContextImpl& context);
    
    const OpenMM::System& system;
    int numParticles;
    int mutualInducedMaxIterations;
    double mutualInducedTargetEpsilon;
    bool usePme;
    double alphaEwald;
    double cutoffDistance;
    
    TholeDipoleForce::NonbondedMethod nonbondedMethod;
    TholeDipoleForce::PolarizationType polarizationType;
    
    std::vector<double> charges;
    std::vector<double> dipoles;
    std::vector<double> tholes;
    std::vector<double> polarity;
    std::vector<int> axisTypes;
    std::vector<int> multipoleAtomZs;
    std::vector<int> multipoleAtomXs;
    std::vector<int> multipoleAtomYs;
    std::vector<std::vector<std::vector<int> > > covalentInfo;
    std::vector<double> extrapolationCoefficients;
    std::vector<int> pmeGridDimension;
};

} // namespace TholeDipolePlugin

#endif /*REFERENCE_THOLEDIPOLE_KERNELS_H_*/
