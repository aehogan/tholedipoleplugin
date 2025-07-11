#ifndef OPENMM_THOLEDIPOLEFORCEIMPL_H_
#define OPENMM_THOLEDIPOLEFORCEIMPL_H_

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
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

namespace TholeDipolePlugin {

/**
 * This is the internal implementation of TholeDipoleForce.
 */

class OPENMM_EXPORT_THOLEDIPOLE TholeDipoleForceImpl : public OpenMM::ForceImpl {
public:
    TholeDipoleForceImpl(const TholeDipoleForce& owner);
    ~TholeDipoleForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const TholeDipoleForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getInducedDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getLabFramePermanentDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getTotalDipoles(OpenMM::ContextImpl& context, std::vector<OpenMM::Vec3>& dipoles);
    void getElectrostaticPotential(OpenMM::ContextImpl& context, const std::vector<OpenMM::Vec3>& inputGrid,
                                   std::vector<double>& outputElectrostaticPotential);
    void getSystemMultipoleMoments(OpenMM::ContextImpl& context, std::vector<double>& outputMultipoleMoments);
    
    // Static helper methods
    static const int* getCovalentDegrees();
    static void getCovalentRange(const TholeDipoleForce& force, int atomIndex,
                                const std::vector<TholeDipoleForce::CovalentType>& lists,
                                int* minCovalentIndex, int* maxCovalentIndex);
    static void getCovalentDegree(const TholeDipoleForce& force, std::vector<int>& covalentDegree);
    
private:
    const TholeDipoleForce& owner;
    OpenMM::Kernel kernel;
    
    // Static members for covalent degree handling
    static bool initializedCovalentDegrees;
    static int CovalentDegrees[TholeDipoleForce::CovalentEnd];
};

} // namespace TholeDipolePlugin

#endif /*OPENMM_THOLEDIPOLEFORCEIMPL_H_*/
