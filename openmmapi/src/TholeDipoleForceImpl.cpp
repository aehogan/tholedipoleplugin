/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs, Peter Eastman                                    *
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

#include "openmm/internal/ContextImpl.h"
#include "internal/TholeDipoleForceImpl.h"
#include "openmm/internal/Messages.h"
#include "TholeDipoleKernels.h"
#include <stdio.h>

using namespace OpenMM;
using namespace TholeDipolePlugin;

using std::vector;

bool TholeDipoleForceImpl::initializedCovalentDegrees = false;
int TholeDipoleForceImpl::CovalentDegrees[] = { 1,2,3,4 };

TholeDipoleForceImpl::TholeDipoleForceImpl(const TholeDipoleForce& owner) : owner(owner) {
}

TholeDipoleForceImpl::~TholeDipoleForceImpl() {
}

void TholeDipoleForceImpl::initialize(ContextImpl& context) {
    const OpenMM::System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (owner.getNumParticles() != numParticles)
        throw OpenMMException("TholeDipoleForce must have exactly as many particles as the System it belongs to.");

    // Check cutoff < 0.5*boxSize
    if (owner.getNonbondedMethod() == TholeDipoleForce::PME) {
        Vec3 boxVectors[3];
        system.getDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);
        double cutoff = owner.getCutoffDistance();
        if (cutoff > 0.5*boxVectors[0][0] || cutoff > 0.5*boxVectors[1][1] || cutoff > 0.5*boxVectors[2][2])
            throw OpenMMException("TholeDipoleForce: "+Messages::cutoffTooLarge);
    }

    // Validate particle parameters
    for (int ii = 0; ii < numParticles; ii++) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, polarizability, tholeDamping;
        std::vector<double> molecularDipole;

        owner.getParticleParameters(ii, charge, molecularDipole, polarizability, tholeDamping, 
                                   axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY);

        // Only 'Z-then-X', 'Bisector', Z-Bisect, ThreeFold, ZOnly currently handled
        if (axisType != TholeDipoleForce::ZThenX     && axisType != TholeDipoleForce::Bisector &&
            axisType != TholeDipoleForce::ZBisect    && axisType != TholeDipoleForce::ThreeFold &&
            axisType != TholeDipoleForce::ZOnly      && axisType != TholeDipoleForce::NoAxisType) {
             std::stringstream buffer;
             buffer << "TholeDipoleForce: axis type=" << axisType;
             buffer << " not currently handled - only axisTypes[ ";
             buffer << TholeDipoleForce::ZThenX   << ", " << TholeDipoleForce::Bisector  << ", ";
             buffer << TholeDipoleForce::ZBisect  << ", " << TholeDipoleForce::ThreeFold << ", ";
             buffer << TholeDipoleForce::ZOnly    << ", " << TholeDipoleForce::NoAxisType;
             buffer << "] (ZThenX, Bisector, Z-Bisect, ThreeFold, ZOnly, NoAxisType) currently handled.";
             throw OpenMMException(buffer.str());
        }
        
        // Validate reference atoms
        if (axisType != TholeDipoleForce::NoAxisType && (multipoleAtomZ < 0 || multipoleAtomZ >= numParticles)) {
            std::stringstream buffer;
            buffer << "TholeDipoleForce: invalid z axis particle: " << multipoleAtomZ;
            throw OpenMMException(buffer.str());
        }
        if (axisType != TholeDipoleForce::NoAxisType && axisType != TholeDipoleForce::ZOnly &&
                (multipoleAtomX < 0 || multipoleAtomX >= numParticles)) {
            std::stringstream buffer;
            buffer << "TholeDipoleForce: invalid x axis particle: " << multipoleAtomX;
            throw OpenMMException(buffer.str());
        }
        if ((axisType == TholeDipoleForce::ZBisect || axisType == TholeDipoleForce::ThreeFold) &&
                (multipoleAtomY < 0 || multipoleAtomY >= numParticles)) {
            std::stringstream buffer;
            buffer << "TholeDipoleForce: invalid y axis particle: " << multipoleAtomY;
            throw OpenMMException(buffer.str());
        }
    }
    
    kernel = context.getPlatform().createKernel(CalcTholeDipoleForceKernel::Name(), context);
    kernel.getAs<CalcTholeDipoleForceKernel>().initialize(context.getSystem(), owner);
}

double TholeDipoleForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcTholeDipoleForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> TholeDipoleForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcTholeDipoleForceKernel::Name());
    return names;
}

const int* TholeDipoleForceImpl::getCovalentDegrees() {
    if (!initializedCovalentDegrees) {
        initializedCovalentDegrees = true;
        CovalentDegrees[TholeDipoleForce::Covalent12] = 1;
        CovalentDegrees[TholeDipoleForce::Covalent13] = 2;
        CovalentDegrees[TholeDipoleForce::Covalent14] = 3;
        CovalentDegrees[TholeDipoleForce::Covalent15] = 4;
    }
    return CovalentDegrees;
}

void TholeDipoleForceImpl::getCovalentRange(const TholeDipoleForce& force, int atomIndex, 
                                            const std::vector<TholeDipoleForce::CovalentType>& lists,
                                            int* minCovalentIndex, int* maxCovalentIndex) {
    *minCovalentIndex =  999999999;
    *maxCovalentIndex = -999999999;
    for (unsigned int kk = 0; kk < lists.size(); kk++) {
        TholeDipoleForce::CovalentType jj = lists[kk];
        std::vector<int> covalentList;
        force.getCovalentMap(atomIndex, jj, covalentList);
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
            if (*minCovalentIndex > covalentList[ii]) {
               *minCovalentIndex = covalentList[ii];
            }
            if (*maxCovalentIndex < covalentList[ii]) {
               *maxCovalentIndex = covalentList[ii];
            }
        }
    }
    return;
}

void TholeDipoleForceImpl::getCovalentDegree(const TholeDipoleForce& force, std::vector<int>& covalentDegree) {
    covalentDegree.resize(TholeDipoleForce::CovalentEnd);
    const int* CovalentDegrees = TholeDipoleForceImpl::getCovalentDegrees();
    for (unsigned int kk = 0; kk < TholeDipoleForce::CovalentEnd; kk++) {
        covalentDegree[kk] = CovalentDegrees[kk];
    }
    return;
}

void TholeDipoleForceImpl::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcTholeDipoleForceKernel>().getLabFramePermanentDipoles(context, dipoles);
}

void TholeDipoleForceImpl::getInducedDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcTholeDipoleForceKernel>().getInducedDipoles(context, dipoles);
}

void TholeDipoleForceImpl::getTotalDipoles(ContextImpl& context, vector<Vec3>& dipoles) {
    kernel.getAs<CalcTholeDipoleForceKernel>().getTotalDipoles(context, dipoles);
}

void TholeDipoleForceImpl::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                     std::vector< double >& outputElectrostaticPotential) {
    kernel.getAs<CalcTholeDipoleForceKernel>().getElectrostaticPotential(context, inputGrid, outputElectrostaticPotential);
}

void TholeDipoleForceImpl::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {
    kernel.getAs<CalcTholeDipoleForceKernel>().getSystemMultipoleMoments(context, outputMultipoleMoments);
}

void TholeDipoleForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcTholeDipoleForceKernel>().copyParametersToContext(context, owner);
    context.systemChanged();
}

void TholeDipoleForceImpl::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    kernel.getAs<CalcTholeDipoleForceKernel>().getPMEParameters(alpha, nx, ny, nz);
}

