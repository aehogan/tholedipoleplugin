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
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "ReferenceTholeDipoleKernels.h"
#include "ReferenceTholeDipoleForce.h"
#include "internal/TholeDipoleForceImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "TholeDipoleForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/NonbondedForceImpl.h"

#include <cmath>
#include <memory>
#ifdef _MSC_VER
#include <windows.h>
#endif

using namespace OpenMM;
using namespace TholeDipolePlugin;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

/* -------------------------------------------------------------------------- *
 *                             TholeDipole                                    *
 * -------------------------------------------------------------------------- */

ReferenceCalcTholeDipoleForceKernel::ReferenceCalcTholeDipoleForceKernel(const std::string& name, const Platform& platform, const System& system) :
         CalcTholeDipoleForceKernel(name, platform), system(system), numParticles(0), mutualInducedMaxIterations(60), mutualInducedTargetEpsilon(1.0e-05),
                                                         usePme(false), alphaEwald(0.0), cutoffDistance(1.0) {  

}

ReferenceCalcTholeDipoleForceKernel::~ReferenceCalcTholeDipoleForceKernel() {
}

void ReferenceCalcTholeDipoleForceKernel::initialize(const System& system, const TholeDipolePlugin::TholeDipoleForce& force) {

    numParticles = force.getNumParticles();

    charges.resize(numParticles);
    dipoles.resize(3*numParticles);
    polarity.resize(numParticles);
    axisTypes.resize(numParticles);
    multipoleAtomZs.resize(numParticles);
    multipoleAtomXs.resize(numParticles);
    multipoleAtomYs.resize(numParticles);
    covalentInfo.resize(numParticles);

    int dipoleIndex = 0;
    for (int ii = 0; ii < numParticles; ii++) {

        // Get particle parameters
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, polarityD;
        std::vector<double> dipolesD;
        force.getParticleParameters(ii, charge, dipolesD, polarityD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY);

        axisTypes[ii] = axisType;
        multipoleAtomZs[ii] = multipoleAtomZ;
        multipoleAtomXs[ii] = multipoleAtomX;
        multipoleAtomYs[ii] = multipoleAtomY;

        charges[ii] = charge;
        polarity[ii] = polarityD;

        if (dipolesD.size() != 3)
            throw OpenMMException("Dipole vector must have exactly 3 components for particle " + std::to_string(ii));
        dipoles[dipoleIndex++] = dipolesD[0];
        dipoles[dipoleIndex++] = dipolesD[1];
        dipoles[dipoleIndex++] = dipolesD[2];

        // covalent info
        std::vector< std::vector<int> > covalentLists;
        force.getCovalentMaps(ii, covalentLists);
        covalentInfo[ii] = covalentLists;
    }

    // Get global Thole damping parameters
    tholeDampingType = force.getTholeDampingType();
    tholeDampingParameter = force.getTholeDampingParameter();

    polarizationType = force.getPolarizationType();
    if (polarizationType == TholeDipolePlugin::TholeDipoleForce::Mutual) {
        mutualInducedMaxIterations = force.getMutualInducedMaxIterations();
        mutualInducedTargetEpsilon = force.getMutualInducedTargetEpsilon();
    }
    else if (polarizationType == TholeDipolePlugin::TholeDipoleForce::Extrapolated) {
        extrapolationCoefficients = force.getExtrapolationCoefficients();
    }

    // PME
    nonbondedMethod = force.getNonbondedMethod();
    if (nonbondedMethod == TholeDipoleForce::PME) {
        usePme = true;
        pmeGridDimension.resize(3);
        force.getPMEParameters(alphaEwald, pmeGridDimension[0], pmeGridDimension[1], pmeGridDimension[2]);
        cutoffDistance = force.getCutoffDistance();
        if (pmeGridDimension[0] == 0 || alphaEwald == 0.0) {
            NonbondedForce nb;
            nb.setEwaldErrorTolerance(force.getEwaldErrorTolerance());
            nb.setCutoffDistance(force.getCutoffDistance());
            int gridSizeX, gridSizeY, gridSizeZ;
            NonbondedForceImpl::calcPMEParameters(system, nb, alphaEwald, gridSizeX, gridSizeY, gridSizeZ, false);
            pmeGridDimension[0] = gridSizeX;
            pmeGridDimension[1] = gridSizeY;
            pmeGridDimension[2] = gridSizeZ;
        }    
    }
    else {
        usePme = false;
    }
    return;
}

TholeDipolePlugin::ReferenceTholeDipoleForce* ReferenceCalcTholeDipoleForceKernel::setupReferenceTholeDipoleForce(ContextImpl& context)
{
    TholeDipolePlugin::ReferenceTholeDipoleForce* referenceTholeDipoleForce = NULL;
    
    if (usePme) {
        ReferencePMETholeDipoleForce* referencePMETholeDipoleForce = new ReferencePMETholeDipoleForce();
        referencePMETholeDipoleForce->setAlphaEwald(alphaEwald);
        referencePMETholeDipoleForce->setCutoffDistance(cutoffDistance);
        referencePMETholeDipoleForce->setPmeGridDimensions(pmeGridDimension);
        Vec3* boxVectors = extractBoxVectors(context);
        double minAllowedSize = 1.999999*cutoffDistance;
        if (boxVectors[0][0] < minAllowedSize || boxVectors[1][1] < minAllowedSize || boxVectors[2][2] < minAllowedSize) {
            throw OpenMMException("The periodic box size has decreased to less than twice the nonbonded cutoff.");
        }
        referencePMETholeDipoleForce->setPeriodicBoxSize(boxVectors);
        referenceTholeDipoleForce = static_cast<ReferenceTholeDipoleForce*>(referencePMETholeDipoleForce);
    }
    else {
        referenceTholeDipoleForce = new ReferenceTholeDipoleForce(ReferenceTholeDipoleForce::NoCutoff);
    }

    // set polarization type
    if (polarizationType == TholeDipoleForce::Mutual) {
        referenceTholeDipoleForce->setPolarizationType(ReferenceTholeDipoleForce::Mutual);
        referenceTholeDipoleForce->setMutualInducedDipoleTargetEpsilon(mutualInducedTargetEpsilon);
        referenceTholeDipoleForce->setMaximumMutualInducedDipoleIterations(mutualInducedMaxIterations);
    }
    else if (polarizationType == TholeDipoleForce::Direct) {
        referenceTholeDipoleForce->setPolarizationType(ReferenceTholeDipoleForce::Direct);
    }
    else if (polarizationType == TholeDipoleForce::Extrapolated) {
        referenceTholeDipoleForce->setPolarizationType(ReferenceTholeDipoleForce::Extrapolated);
        referenceTholeDipoleForce->setExtrapolationCoefficients(extrapolationCoefficients);
    }
    else {
        throw OpenMMException("Polarization type not recognized.");
    }

    // Set global Thole damping parameters
    referenceTholeDipoleForce->setTholeDampingType(tholeDampingType);
    referenceTholeDipoleForce->setTholeDampingParameter(tholeDampingParameter);

    return referenceTholeDipoleForce;
}

double ReferenceCalcTholeDipoleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    unique_ptr<ReferenceTholeDipoleForce> referenceTholeDipoleForce(setupReferenceTholeDipoleForce(context));

    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& forceData = extractForces(context);
    double energy = referenceTholeDipoleForce->calculateForceAndEnergy(posData, charges, dipoles, polarity,
                                                                       axisTypes,
                                                                       multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                                       covalentInfo, forceData);

    return energy;
}

void ReferenceCalcTholeDipoleForceKernel::getInducedDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    unique_ptr<ReferenceTholeDipoleForce> referenceTholeDipoleForce(setupReferenceTholeDipoleForce(context));
    vector<Vec3>& posData = extractPositions(context);

    referenceTholeDipoleForce->calculateInducedDipoles(posData, charges, dipoles, polarity,
            axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, covalentInfo, outputDipoles);
}

void ReferenceCalcTholeDipoleForceKernel::getLabFramePermanentDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    unique_ptr<ReferenceTholeDipoleForce> referenceTholeDipoleForce(setupReferenceTholeDipoleForce(context));
    vector<Vec3>& posData = extractPositions(context);

    referenceTholeDipoleForce->calculateLabFramePermanentDipoles(posData, charges, dipoles, polarity,
            axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, covalentInfo, outputDipoles);
}

void ReferenceCalcTholeDipoleForceKernel::getTotalDipoles(ContextImpl& context, vector<Vec3>& outputDipoles) {
    unique_ptr<ReferenceTholeDipoleForce> referenceTholeDipoleForce(setupReferenceTholeDipoleForce(context));
    vector<Vec3>& posData = extractPositions(context);

    referenceTholeDipoleForce->calculateTotalDipoles(posData, charges, dipoles, polarity,
            axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs, covalentInfo, outputDipoles);
}

void ReferenceCalcTholeDipoleForceKernel::getElectrostaticPotential(ContextImpl& context, const std::vector< Vec3 >& inputGrid,
                                                                    std::vector< double >& outputElectrostaticPotential) {

    unique_ptr<ReferenceTholeDipoleForce> referenceTholeDipoleForce(setupReferenceTholeDipoleForce(context));
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3> grid = inputGrid;
    referenceTholeDipoleForce->calculateElectrostaticPotential(posData, charges, dipoles, polarity,
                                                               axisTypes,
                                                               multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                               covalentInfo, grid, outputElectrostaticPotential);
}

void ReferenceCalcTholeDipoleForceKernel::getSystemMultipoleMoments(ContextImpl& context, std::vector< double >& outputMultipoleMoments) {

    const System& sys = context.getSystem();
    vector<double> masses;
    masses.reserve(sys.getNumParticles());
    for (int i = 0; i < sys.getNumParticles(); ++i) {
        masses.push_back(sys.getParticleMass(i));
    }

    unique_ptr<ReferenceTholeDipoleForce> referenceTholeDipoleForce(setupReferenceTholeDipoleForce(context));
    vector<Vec3>& posData = extractPositions(context);
    referenceTholeDipoleForce->calculateTholeDipoleSystemMultipoleMoments(masses, posData, charges, dipoles, polarity,
                                                               axisTypes,
                                                               multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
                                                               covalentInfo, outputMultipoleMoments);
}

void ReferenceCalcTholeDipoleForceKernel::copyParametersToContext(ContextImpl& context, const TholeDipolePlugin::TholeDipoleForce& force) {
    if (numParticles != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    // Update global Thole damping parameters
    tholeDampingType = force.getTholeDampingType();
    tholeDampingParameter = force.getTholeDampingParameter();

    // Record the particle values.
    int dipoleIndex = 0;
    for (int i = 0; i < numParticles; ++i) {
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;
        double charge, polarityD;
        std::vector<double> dipolesD;
        force.getParticleParameters(i, charge, dipolesD, polarityD, axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY);
        axisTypes[i] = axisType;
        multipoleAtomZs[i] = multipoleAtomZ;
        multipoleAtomXs[i] = multipoleAtomX;
        multipoleAtomYs[i] = multipoleAtomY;
        charges[i] = charge;
        polarity[i] = polarityD;
        if (dipolesD.size() != 3)
            throw OpenMMException("Dipole vector must have exactly 3 components for particle " + std::to_string(i));
        dipoles[dipoleIndex++] = dipolesD[0];
        dipoles[dipoleIndex++] = dipolesD[1];
        dipoles[dipoleIndex++] = dipolesD[2];

        std::vector<std::vector<int>> covalentLists;
        force.getCovalentMaps(i, covalentLists);
        covalentInfo[i] = covalentLists;
    }
}

void ReferenceCalcTholeDipoleForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    if (!usePme)
        throw OpenMMException("getPMEParametersInContext: This Context is not using PME");
    alpha = alphaEwald;
    nx = pmeGridDimension[0];
    ny = pmeGridDimension[1];
    nz = pmeGridDimension[2];
}
