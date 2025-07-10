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

#include "ReferenceTholeDipoleForce.h"
#include "openmm/OpenMMException.h"

using namespace TholeDipolePlugin;
using namespace OpenMM;

ReferenceTholeDipoleForce::ReferenceTholeDipoleForce() : _nonbondedMethod(NoCutoff) {
    initialize();
}

ReferenceTholeDipoleForce::ReferenceTholeDipoleForce(NonbondedMethod nonbondedMethod) : _nonbondedMethod(nonbondedMethod) {
    initialize();
}

void ReferenceTholeDipoleForce::initialize() {
    _electric = 1.0;
    _dielectric = 1.0;
    _mutualInducedDipoleTargetEpsilon = 1.0e-03;
    _maximumMutualInducedDipoleIterations = 60;
    _mutualInducedDipoleEpsilon = 1.0e+50;
    _mutualInducedDipoleConverged = 0;
    _mutualInducedDipoleIterations = 0;
    _debye = 0.4803;

    // Initialize scale factors
    _scaleMaps.resize(LAST_SCALE_TYPE_INDEX);
    _maxScaleIndex.resize(LAST_SCALE_TYPE_INDEX);
    for (int i = 0; i < LAST_SCALE_TYPE_INDEX; i++) {
        _maxScaleIndex[i] = 5;
    }

    // Set default scale factors
    _mScale[0] = 0.0; // 1-1 excluded
    _mScale[1] = 0.0; // 1-2 excluded
    _mScale[2] = 0.0; // 1-3 excluded
    _mScale[3] = 0.5; // 1-4 scaled
    _mScale[4] = 1.0; // 1-5+ full

    _iScale[0] = 0.0; // 1-1 excluded
    _iScale[1] = 1.0; // 1-2+ full
    _iScale[2] = 1.0;
    _iScale[3] = 1.0;
    _iScale[4] = 1.0;

    _polarizationType = Direct;
}

ReferenceTholeDipoleForce::NonbondedMethod ReferenceTholeDipoleForce::getNonbondedMethod() const {
    return _nonbondedMethod;
}

void ReferenceTholeDipoleForce::setNonbondedMethod(NonbondedMethod nonbondedMethod) {
    _nonbondedMethod = nonbondedMethod;
}

ReferenceTholeDipoleForce::PolarizationType ReferenceTholeDipoleForce::getPolarizationType() const {
    return _polarizationType;
}

void ReferenceTholeDipoleForce::setPolarizationType(PolarizationType polarizationType) {
    _polarizationType = polarizationType;
}

int ReferenceTholeDipoleForce::getMutualInducedDipoleConverged() const {
    return _mutualInducedDipoleConverged;
}

int ReferenceTholeDipoleForce::getMutualInducedDipoleIterations() const {
    return _mutualInducedDipoleIterations;
}

double ReferenceTholeDipoleForce::getMutualInducedDipoleEpsilon() const {
    return _mutualInducedDipoleEpsilon;
}

void ReferenceTholeDipoleForce::setExtrapolationCoefficients(const std::vector<double> &coefficients) {
    _extrapolationCoefficients = coefficients;
}

void ReferenceTholeDipoleForce::setMutualInducedDipoleTargetEpsilon(double targetEpsilon) {
    _mutualInducedDipoleTargetEpsilon = targetEpsilon;
}

double ReferenceTholeDipoleForce::getMutualInducedDipoleTargetEpsilon() const {
    return _mutualInducedDipoleTargetEpsilon;
}

void ReferenceTholeDipoleForce::setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations) {
    _maximumMutualInducedDipoleIterations = maximumMutualInducedDipoleIterations;
}

int ReferenceTholeDipoleForce::getMaximumMutualInducedDipoleIterations() const {
    return _maximumMutualInducedDipoleIterations;
}

double ReferenceTholeDipoleForce::calculateForceAndEnergy(const std::vector<Vec3>& particlePositions,
                                                          const std::vector<double>& charges,
                                                          const std::vector<double>& dipoles,
                                                          const std::vector<double>& polarizabilities,
                                                          const std::vector<double>& tholeDampingFactors,
                                                          const std::vector<int>& axisTypes,
                                                          const std::vector<int>& multipoleAtomZs,
                                                          const std::vector<int>& multipoleAtomXs,
                                                          const std::vector<int>& multipoleAtomYs,
                                                          const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                          std::vector<Vec3>& forces) {
    // TODO: Implement the actual force calculation
    throw OpenMMException("TholeDipole force calculation not yet implemented");
    return 0.0;
}

// Stub implementations for other methods
void ReferenceTholeDipoleForce::calculateInducedDipoles(const std::vector<Vec3>& particlePositions,
                                                        const std::vector<double>& charges,
                                                        const std::vector<double>& dipoles,
                                                        const std::vector<double>& polarizabilities,
                                                        const std::vector<double>& tholeDampingFactors,
                                                        const std::vector<int>& axisTypes,
                                                        const std::vector<int>& multipoleAtomZs,
                                                        const std::vector<int>& multipoleAtomXs,
                                                        const std::vector<int>& multipoleAtomYs,
                                                        const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                        std::vector<Vec3>& outputInducedDipoles) {
    throw OpenMMException("calculateInducedDipoles not yet implemented");
}

void ReferenceTholeDipoleForce::calculateLabFramePermanentDipoles(const std::vector<Vec3>& particlePositions,
                                                                  const std::vector<double>& charges,
                                                                  const std::vector<double>& dipoles,
                                                                  const std::vector<double>& polarizabilities,
                                                                  const std::vector<double>& tholeDampingFactors,
                                                                  const std::vector<int>& axisTypes,
                                                                  const std::vector<int>& multipoleAtomZs,
                                                                  const std::vector<int>& multipoleAtomXs,
                                                                  const std::vector<int>& multipoleAtomYs,
                                                                  const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                                  std::vector<Vec3>& outputRotatedPermanentDipoles) {
    throw OpenMMException("calculateLabFramePermanentDipoles not yet implemented");
}

void ReferenceTholeDipoleForce::calculateTotalDipoles(const std::vector<Vec3>& particlePositions,
                                                      const std::vector<double>& charges,
                                                      const std::vector<double>& dipoles,
                                                      const std::vector<double>& polarizabilities,
                                                      const std::vector<double>& tholeDampingFactors,
                                                      const std::vector<int>& axisTypes,
                                                      const std::vector<int>& multipoleAtomZs,
                                                      const std::vector<int>& multipoleAtomXs,
                                                      const std::vector<int>& multipoleAtomYs,
                                                      const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                      std::vector<Vec3>& outputTotalDipoles) {
    throw OpenMMException("calculateTotalDipoles not yet implemented");
}

void ReferenceTholeDipoleForce::calculateTholeDipoleSystemMultipoleMoments(const std::vector<double>& masses,
                                                                           const std::vector<Vec3>& particlePositions,
                                                                           const std::vector<double>& charges,
                                                                           const std::vector<double>& dipoles,
                                                                           const std::vector<double>& polarizabilities,
                                                                           const std::vector<double>& tholeDampingFactors,
                                                                           const std::vector<int>& axisTypes,
                                                                           const std::vector<int>& multipoleAtomZs,
                                                                           const std::vector<int>& multipoleAtomXs,
                                                                           const std::vector<int>& multipoleAtomYs,
                                                                           const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                                           std::vector<double>& outputMultipoleMoments) {
    throw OpenMMException("calculateTholeDipoleSystemMultipoleMoments not yet implemented");
}

void ReferenceTholeDipoleForce::calculateElectrostaticPotential(const std::vector<Vec3>& particlePositions,
                                                                const std::vector<double>& charges,
                                                                const std::vector<double>& dipoles,
                                                                const std::vector<double>& polarizabilities,
                                                                const std::vector<double>& tholeDampingFactors,
                                                                const std::vector<int>& axisTypes,
                                                                const std::vector<int>& multipoleAtomZs,
                                                                const std::vector<int>& multipoleAtomXs,
                                                                const std::vector<int>& multipoleAtomYs,
                                                                const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                                const std::vector<Vec3>& inputGrid,
                                                                std::vector<double>& outputPotential) {
    throw OpenMMException("calculateElectrostaticPotential not yet implemented");
}

// PME class implementation
ReferencePmeTholeDipoleForce::ReferencePmeTholeDipoleForce() : ReferenceTholeDipoleForce(ReferenceTholeDipoleForce::PME) {
    _cutoffDistance = 1.0;
    _alphaEwald = 0.0;
    _pmeGridDimensions.resize(3, 0);
}

ReferencePmeTholeDipoleForce::~ReferencePmeTholeDipoleForce() {
}

void ReferencePmeTholeDipoleForce::setCutoffDistance(double cutoffDistance) {
    _cutoffDistance = cutoffDistance;
}

double ReferencePmeTholeDipoleForce::getCutoffDistance() const {
    return _cutoffDistance;
}

void ReferencePmeTholeDipoleForce::setAlphaEwald(double alphaEwald) {
    _alphaEwald = alphaEwald;
}

double ReferencePmeTholeDipoleForce::getAlphaEwald() const {
    return _alphaEwald;
}

void ReferencePmeTholeDipoleForce::setPmeGridDimensions(const std::vector<int>& pmeGridDimensions) {
    _pmeGridDimensions = pmeGridDimensions;
}

void ReferencePmeTholeDipoleForce::getPmeGridDimensions(std::vector<int>& pmeGridDimensions) const {
    pmeGridDimensions = _pmeGridDimensions;
}

void ReferencePmeTholeDipoleForce::setPeriodicBoxSize(Vec3* boxVectors) {
    for (int i = 0; i < 3; i++) {
        _periodicBoxVectors[i] = boxVectors[i];
    }
}

void ReferencePmeTholeDipoleForce::getPeriodicDelta(Vec3& deltaR) const {
    // Apply periodic boundary conditions
    // This is a simplified implementation - full PBC would be more complex
    for (int i = 0; i < 3; i++) {
        double boxSize = _periodicBoxVectors[i][i];
        if (boxSize > 0) {
            deltaR[i] -= boxSize * floor(deltaR[i] / boxSize + 0.5);
        }
    }
}