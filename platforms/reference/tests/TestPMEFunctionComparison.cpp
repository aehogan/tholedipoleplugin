/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs                                                   *
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

/**
 * This test compares individual PME functions between TholeDipole and AMOEBA
 * to identify where implementations diverge.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "ReferencePMETholeDipoleForce.h"
#include <complex>
#include <iomanip>

using namespace TholeDipolePlugin;

/**
 * Test B-spline moduli calculation
 *
 * This function tests whether the B-spline moduli are calculated identically
 * between TholeDipole and AMOEBA implementations.
 */
void testBSplineModuli() {
    cout << "\n=== Testing B-spline Moduli Calculation ===" << endl;

    // Create simple O-H pair system with PME
    System tholeDipoleSystem;
    double boxDimension = 5.0;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Direct);
    tholeDipoleForce->setCutoffDistance(0.70);
    tholeDipoleForce->setPMEParameters(5.4459052e+00, 20, 20, 20);

    // Add O-H pair
    tholeDipoleSystem.addParticle(1.5995000e+01);  // O
    tholeDipoleSystem.addParticle(1.0080000e+00);  // H

    std::vector<double> oxygenDipole = {0.0, 0.0, 0.001};
    std::vector<double> hydrogenDipole = {0.0, 0.0, -0.001};

    tholeDipoleForce->addParticle(-5.1966000e-1, oxygenDipole, 8.3700000e-4, 4, 1, -1, -1);
    tholeDipoleForce->addParticle(2.5983000e-1, hydrogenDipole, 4.9600000e-4, 4, 0, -1, -1);

    std::vector<int> covalentMap;
    covalentMap.push_back(1);
    tholeDipoleForce->setCovalentMap(0, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap.clear();
    covalentMap.push_back(0);
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent12, covalentMap);

    tholeDipoleSystem.addForce(tholeDipoleForce);

    // Create equivalent AMOEBA system
    System amoebaSystem;
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);
    amoebaSystem.addParticle(1.5995000e+01);
    amoebaSystem.addParticle(1.0080000e+00);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
    amoebaSystem.addForce(amoebaForce);

    // Create contexts
    LangevinIntegrator integ1(0.0, 0.1, 0.01);
    LangevinIntegrator integ2(0.0, 0.1, 0.01);

    Context tholeContext(tholeDipoleSystem, integ1, *platform);
    Context amoebaContext(amoebaSystem, integ2);

    // Set positions
    std::vector<Vec3> positions(2);
    positions[0] = Vec3(-8.7387270e-01, 5.3220410e-01, 7.4214000e-03);
    positions[1] = Vec3(-9.6050090e-01, 5.1173410e-01, -2.2202700e-02);

    tholeContext.setPositions(positions);
    amoebaContext.setPositions(positions);

    // Force evaluation to initialize PME
    cout << "\n--- TholeDipole Calculation ---" << endl;
    State tholeState = tholeContext.getState(State::Energy);

    cout << "\n--- AMOEBA Calculation ---" << endl;
    State amoebaState = amoebaContext.getState(State::Energy);

    cout << "\n--- Energy Comparison ---" << endl;
    cout << "TholeDipole Energy: " << tholeState.getPotentialEnergy() << " kJ/mol" << endl;
    cout << "AMOEBA Energy:      " << amoebaState.getPotentialEnergy() << " kJ/mol" << endl;
    cout << "Difference:         " << fabs(tholeState.getPotentialEnergy() - amoebaState.getPotentialEnergy()) << " kJ/mol" << endl;

    cout << "\nB-spline moduli values are printed in debug output above." << endl;
    cout << "Compare the 'First 5 B-spline moduli[0]' values from both runs." << endl;
}

/**
 * Test reciprocal energy decomposition
 */
void testReciprocalEnergyBreakdown() {
    cout << "\n=== Testing Reciprocal Energy Breakdown ===" << endl;

    cout << "Key observations to investigate:" << endl;
    cout << "1. PME reciprocal energy calculation" << endl;
    cout << "2. Self energy calculation" << endl;
    cout << "3. Direct space energy calculation" << endl;
    cout << "4. Energy component magnitudes and signs" << endl;
    cout << endl;

    cout << "From debug output above, check:" << endl;
    cout << "- Are B-spline moduli identical? (should be ~1.0, 0.96, 0.85, 0.69, 0.51)" << endl;
    cout << "- Does grid magnitude after spreading match?" << endl;
    cout << "- Does FFT output magnitude match?" << endl;
    cout << "- Does reciprocal energy match?" << endl;
    cout << "- Do phi values (potential) match?" << endl;
    cout << endl;
}

/**
 * Main test runner
 */
void runPMEFunctionTests() {
    cout << "\n======================================" << endl;
    cout << "PME Function-Level Comparison Tests" << endl;
    cout << "======================================" << endl;

    testBSplineModuli();
    testReciprocalEnergyBreakdown();

    cout << "\n======================================" << endl;
    cout << "PME Function Tests Complete" << endl;
    cout << "======================================\n" << endl;
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        runPMEFunctionTests();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
