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

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "TholeDipoleForce.h"
#include "internal/TholeDipoleForceImpl.h"
#include <stdio.h>

using namespace OpenMM;
using namespace TholeDipolePlugin;
using std::string;
using std::vector;

TholeDipoleForce::TholeDipoleForce() : nonbondedMethod(NoCutoff), polarizationType(Mutual), pmeBSplineOrder(5), 
                                       cutoffDistance(1.0), ewaldErrorTol(1e-4), mutualInducedMaxIterations(60),
                                       mutualInducedTargetEpsilon(1e-5), alpha(0.0), nx(0), ny(0), nz(0) {
    // Default extrapolation coefficients for induced dipoles
    extrapolationCoefficients.push_back(-0.154);
    extrapolationCoefficients.push_back(0.017);
    extrapolationCoefficients.push_back(0.658);
    extrapolationCoefficients.push_back(0.474);
}

TholeDipoleForce::NonbondedMethod TholeDipoleForce::getNonbondedMethod() const {
    return nonbondedMethod;
}

void TholeDipoleForce::setNonbondedMethod(TholeDipoleForce::NonbondedMethod method) {
    if (method < 0 || method > 1)
        throw OpenMMException("TholeDipoleForce: Illegal value for nonbonded method");
    nonbondedMethod = method;
}

TholeDipoleForce::PolarizationType TholeDipoleForce::getPolarizationType() const {
    return polarizationType;
}

void TholeDipoleForce::setPolarizationType(TholeDipoleForce::PolarizationType type) {
    polarizationType = type;
}

void TholeDipoleForce::setExtrapolationCoefficients(const std::vector<double> &coefficients) {
    extrapolationCoefficients = coefficients;
}

const std::vector<double> & TholeDipoleForce::getExtrapolationCoefficients() const {
    return extrapolationCoefficients;
}

double TholeDipoleForce::getCutoffDistance() const {
    return cutoffDistance;
}

void TholeDipoleForce::setCutoffDistance(double distance) {
    cutoffDistance = distance;
}

void TholeDipoleForce::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    alpha = this->alpha;
    nx = this->nx;
    ny = this->ny;
    nz = this->nz;
}

void TholeDipoleForce::setPMEParameters(double alpha, int nx, int ny, int nz) {
    this->alpha = alpha;
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
}

int TholeDipoleForce::getPmeBSplineOrder() const { 
    return pmeBSplineOrder; 
} 

void TholeDipoleForce::getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const TholeDipoleForceImpl&>(getImplInContext(context)).getPMEParameters(alpha, nx, ny, nz);
}

int TholeDipoleForce::getMutualInducedMaxIterations() const {
    return mutualInducedMaxIterations;
}

void TholeDipoleForce::setMutualInducedMaxIterations(int inputMutualInducedMaxIterations) {
    mutualInducedMaxIterations = inputMutualInducedMaxIterations;
}

double TholeDipoleForce::getMutualInducedTargetEpsilon() const {
    return mutualInducedTargetEpsilon;
}

void TholeDipoleForce::setMutualInducedTargetEpsilon(double inputMutualInducedTargetEpsilon) {
    mutualInducedTargetEpsilon = inputMutualInducedTargetEpsilon;
}

double TholeDipoleForce::getEwaldErrorTolerance() const {
    return ewaldErrorTol;
}

void TholeDipoleForce::setEwaldErrorTolerance(double tol) {
    ewaldErrorTol = tol;
}

int TholeDipoleForce::addParticle(double charge, const std::vector<double>& molecularDipole, double polarizability,
                                  double tholeDamping, int axisType, int multipoleAtomZ, 
                                  int multipoleAtomX, int multipoleAtomY) {
    particles.push_back(ParticleInfo(charge, molecularDipole, polarizability, tholeDamping, 
                                    axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY));
    return particles.size()-1;
}

void TholeDipoleForce::getParticleParameters(int index, double& charge, std::vector<double>& molecularDipole,
                                             double& polarizability, double& tholeDamping, int& axisType,
                                             int& multipoleAtomZ, int& multipoleAtomX, int& multipoleAtomY) const {
    charge              = particles[index].charge;
    polarizability      = particles[index].polarizability;
    tholeDamping        = particles[index].tholeDamping;

    molecularDipole.resize(3);
    molecularDipole[0]  = particles[index].molecularDipole[0];
    molecularDipole[1]  = particles[index].molecularDipole[1];
    molecularDipole[2]  = particles[index].molecularDipole[2];

    axisType            = particles[index].axisType;
    multipoleAtomZ      = particles[index].multipoleAtomZ;
    multipoleAtomX      = particles[index].multipoleAtomX;
    multipoleAtomY      = particles[index].multipoleAtomY;
}

void TholeDipoleForce::setParticleParameters(int index, double charge, const std::vector<double>& molecularDipole,
                                             double polarizability, double tholeDamping, int axisType,
                                             int multipoleAtomZ, int multipoleAtomX, int multipoleAtomY) {
    particles[index].charge              = charge;
    particles[index].polarizability      = polarizability;
    particles[index].tholeDamping        = tholeDamping;

    particles[index].molecularDipole[0]  = molecularDipole[0];
    particles[index].molecularDipole[1]  = molecularDipole[1];
    particles[index].molecularDipole[2]  = molecularDipole[2];

    particles[index].axisType            = axisType;
    particles[index].multipoleAtomZ      = multipoleAtomZ;
    particles[index].multipoleAtomX      = multipoleAtomX;
    particles[index].multipoleAtomY      = multipoleAtomY;
}

void TholeDipoleForce::setCovalentMap(int index, CovalentType typeId, const std::vector<int>& covalentAtoms) {
    std::vector<int>& covalentList = particles[index].covalentInfo[typeId];
    covalentList.resize(covalentAtoms.size());
    for (unsigned int ii = 0; ii < covalentAtoms.size(); ii++) {
       covalentList[ii] = covalentAtoms[ii];
    }
}

void TholeDipoleForce::getCovalentMap(int index, CovalentType typeId, std::vector<int>& covalentAtoms) const {
    // load covalent atom index entries for atomId==index and covalentId==typeId into covalentAtoms
    std::vector<int> covalentList = particles[index].covalentInfo[typeId];
    covalentAtoms.resize(covalentList.size());
    for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
       covalentAtoms[ii] = covalentList[ii];
    }
}

void TholeDipoleForce::getCovalentMaps(int index, std::vector< std::vector<int> >& covalentLists) const {
    covalentLists.resize(CovalentEnd);
    for (unsigned int jj = 0; jj < CovalentEnd; jj++) {
        std::vector<int> covalentList = particles[index].covalentInfo[jj];
        std::vector<int> covalentAtoms;
        covalentAtoms.resize(covalentList.size());
        for (unsigned int ii = 0; ii < covalentList.size(); ii++) {
           covalentAtoms[ii] = covalentList[ii];
        }
        covalentLists[jj] = covalentAtoms;
    }
}

void TholeDipoleForce::getInducedDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<TholeDipoleForceImpl&>(getImplInContext(context)).getInducedDipoles(getContextImpl(context), dipoles);
}

void TholeDipoleForce::getLabFramePermanentDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<TholeDipoleForceImpl&>(getImplInContext(context)).getLabFramePermanentDipoles(getContextImpl(context), dipoles);
}

void TholeDipoleForce::getTotalDipoles(Context& context, vector<Vec3>& dipoles) {
    dynamic_cast<TholeDipoleForceImpl&>(getImplInContext(context)).getTotalDipoles(getContextImpl(context), dipoles);
}

void TholeDipoleForce::getElectrostaticPotential(const std::vector< Vec3 >& inputGrid, Context& context, 
                                                 std::vector< double >& outputElectrostaticPotential) {
    dynamic_cast<TholeDipoleForceImpl&>(getImplInContext(context)).getElectrostaticPotential(getContextImpl(context), 
                                                                                             inputGrid, outputElectrostaticPotential);
}

void TholeDipoleForce::getSystemMultipoleMoments(Context& context, std::vector< double >& outputMultipoleMoments) {
    dynamic_cast<TholeDipoleForceImpl&>(getImplInContext(context)).getSystemMultipoleMoments(getContextImpl(context), 
                                                                                             outputMultipoleMoments);
}

ForceImpl* TholeDipoleForce::createImpl() const {
    return new TholeDipoleForceImpl(*this);
}

void TholeDipoleForce::updateParametersInContext(Context& context) {
    dynamic_cast<TholeDipoleForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}
