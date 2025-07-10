/* -------------------------------------------------------------------------- *
 *                             OpenMMTholeDipole                                 *
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
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace TholeDipolePlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerTholeDipoleSerializationProxies();

void testSerialization() {
    // Create a Force.

    TholeDipoleForce force;
    force.setForceGroup(3);
    force.setNonbondedMethod(TholeDipoleForce::PME);
    force.setPolarizationType(TholeDipoleForce::Mutual);
    force.setCutoffDistance(0.9);
    force.setEwaldErrorTolerance(1.0e-05);
    force.setMutualInducedMaxIterations(200);
    force.setMutualInducedTargetEpsilon(1.0e-05);
    force.setPMEParameters(0.5, 32, 32, 32);
    
    vector<double> extrapolationCoefficients;
    extrapolationCoefficients.push_back(0.0);
    extrapolationCoefficients.push_back(-0.1);
    extrapolationCoefficients.push_back(1.1);
    force.setExtrapolationCoefficients(extrapolationCoefficients);

    // Add some particles
    for (int i = 0; i < 4; i++) {
        vector<double> molecularDipole;
        molecularDipole.push_back(0.1 * i);
        molecularDipole.push_back(0.2 * i);
        molecularDipole.push_back(0.3 * i);
        
        int particleIndex = force.addParticle(i + 1.0, molecularDipole, 0.5 + 0.1 * i, 0.8 + 0.05 * i,
                                             TholeDipoleForce::ZThenX, i + 1, i + 2, i + 3);
        
        // Add some covalent maps
        for (int j = 0; j < TholeDipoleForce::CovalentEnd; j++) {
            vector<int> covalentMap;
            covalentMap.push_back(i * j);
            covalentMap.push_back(i * j + 1);
            force.setCovalentMap(particleIndex, static_cast<TholeDipoleForce::CovalentType>(j), covalentMap);
        }
    }

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<TholeDipoleForce>(&force, "Force", buffer);
    TholeDipoleForce* copy = XmlSerializer::deserialize<TholeDipoleForce>(buffer);

    // Compare the two forces to see if they are identical.

    TholeDipoleForce& force2 = *copy;
    ASSERT_EQUAL(force.getForceGroup(), force2.getForceGroup());
    ASSERT_EQUAL(force.getNonbondedMethod(), force2.getNonbondedMethod());
    ASSERT_EQUAL(force.getPolarizationType(), force2.getPolarizationType());
    ASSERT_EQUAL(force.getCutoffDistance(), force2.getCutoffDistance());
    ASSERT_EQUAL(force.getEwaldErrorTolerance(), force2.getEwaldErrorTolerance());
    ASSERT_EQUAL(force.getMutualInducedMaxIterations(), force2.getMutualInducedMaxIterations());
    ASSERT_EQUAL(force.getMutualInducedTargetEpsilon(), force2.getMutualInducedTargetEpsilon());
    ASSERT_EQUAL(force.getPmeBSplineOrder(), force2.getPmeBSplineOrder());
    
    // Check PME parameters
    double alpha1, alpha2;
    int nx1, ny1, nz1, nx2, ny2, nz2;
    force.getPMEParameters(alpha1, nx1, ny1, nz1);
    force2.getPMEParameters(alpha2, nx2, ny2, nz2);
    ASSERT_EQUAL(alpha1, alpha2);
    ASSERT_EQUAL(nx1, nx2);
    ASSERT_EQUAL(ny1, ny2);
    ASSERT_EQUAL(nz1, nz2);
    
    // Check extrapolation coefficients
    const vector<double>& extrapCoeff1 = force.getExtrapolationCoefficients();
    const vector<double>& extrapCoeff2 = force2.getExtrapolationCoefficients();
    ASSERT_EQUAL(extrapCoeff1.size(), extrapCoeff2.size());
    for (int i = 0; i < (int)extrapCoeff1.size(); i++) {
        ASSERT_EQUAL(extrapCoeff1[i], extrapCoeff2[i]);
    }
    
    // Check particles
    ASSERT_EQUAL(force.getNumParticles(), force2.getNumParticles());
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge1, charge2, polarizability1, polarizability2, tholeDamping1, tholeDamping2;
        vector<double> molecularDipole1, molecularDipole2;
        int axisType1, axisType2, multipoleAtomZ1, multipoleAtomZ2;
        int multipoleAtomX1, multipoleAtomX2, multipoleAtomY1, multipoleAtomY2;
        
        force.getParticleParameters(i, charge1, molecularDipole1, polarizability1, tholeDamping1,
                                   axisType1, multipoleAtomZ1, multipoleAtomX1, multipoleAtomY1);
        force2.getParticleParameters(i, charge2, molecularDipole2, polarizability2, tholeDamping2,
                                    axisType2, multipoleAtomZ2, multipoleAtomX2, multipoleAtomY2);
        
        ASSERT_EQUAL(charge1, charge2);
        ASSERT_EQUAL(polarizability1, polarizability2);
        ASSERT_EQUAL(tholeDamping1, tholeDamping2);
        ASSERT_EQUAL(axisType1, axisType2);
        ASSERT_EQUAL(multipoleAtomZ1, multipoleAtomZ2);
        ASSERT_EQUAL(multipoleAtomX1, multipoleAtomX2);
        ASSERT_EQUAL(multipoleAtomY1, multipoleAtomY2);
        
        ASSERT_EQUAL(molecularDipole1.size(), molecularDipole2.size());
        for (int j = 0; j < (int)molecularDipole1.size(); j++) {
            ASSERT_EQUAL(molecularDipole1[j], molecularDipole2[j]);
        }
        
        // Check covalent maps
        for (int j = 0; j < TholeDipoleForce::CovalentEnd; j++) {
            vector<int> covalentMap1, covalentMap2;
            force.getCovalentMap(i, static_cast<TholeDipoleForce::CovalentType>(j), covalentMap1);
            force2.getCovalentMap(i, static_cast<TholeDipoleForce::CovalentType>(j), covalentMap2);
            ASSERT_EQUAL(covalentMap1.size(), covalentMap2.size());
            for (int k = 0; k < (int)covalentMap1.size(); k++) {
                ASSERT_EQUAL(covalentMap1[k], covalentMap2[k]);
            }
        }
    }
}

int main() {
    try {
        registerTholeDipoleSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
