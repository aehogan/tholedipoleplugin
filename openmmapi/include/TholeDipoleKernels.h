#ifndef THOLEDIPOLE_KERNELS_H_
#define THOLEDIPOLE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
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

#include "TholeDipoleForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace TholeDipolePlugin {

/**
 * This kernel is invoked by TholeDipoleForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcTholeDipoleForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcTholeDipoleForce";
    }
    CalcTholeDipoleForceKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the TholeDipoleForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const TholeDipoleForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the TholeDipoleForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const TholeDipoleForce& force) = 0;
    /**
     * Get the PME parameters being used for the current context.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
    /**
     * Get the induced dipole moments of all particles.
     *
     * @param context         the context for which to get the induced dipoles
     * @param dipoles         the induced dipole moment of particle i is stored into the i'th element
     */
    virtual void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    /**
     * Get the fixed dipole moments of all particles in the global reference frame.
     *
     * @param context         the context for which to get the fixed dipoles
     * @param dipoles         the fixed dipole moment of particle i is stored into the i'th element
     */
    virtual void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    /**
     * Get the total dipole moments (fixed plus induced) of all particles.
     *
     * @param context         the context for which to get the total dipoles
     * @param dipoles         the total dipole moment of particle i is stored into the i'th element
     */
    virtual void getTotalDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles) = 0;
    /**
     * Get the electrostatic potential.
     *
     * @param context                           the context
     * @param inputGrid                         input grid points over which the potential is to be evaluated
     * @param outputElectrostaticPotential     output potential
     */
    virtual void getElectrostaticPotential(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& inputGrid,
                                           std::vector<double>& outputElectrostaticPotential) = 0;
    /**
     * Get the system multipole moments.
     *
     * @param context                           the context
     * @param outputMultipoleMoments            output multipole moments
     */
    virtual void getSystemMultipoleMoments(OpenMM::ContextImpl& context, std::vector<double>& outputMultipoleMoments) = 0;
};

} // namespace TholeDipolePlugin

#endif /*THOLEDIPOLE_KERNELS_H_*/
