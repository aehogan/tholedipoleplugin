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
 * This tests TholeDipoleForce with a single water molecule to examine
 * intramolecular interaction scaling.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testSingleWater() {
    std::string testName = "testSingleWater";

    // Create TholeDipole system with single water molecule
    System tholeDipoleSystem;
    tholeDipoleSystem.addParticle(1.5995000e+01);  // Oxygen
    tholeDipoleSystem.addParticle(1.0080000e+00);  // Hydrogen 1
    tholeDipoleSystem.addParticle(1.0080000e+00);  // Hydrogen 2

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Direct);

    // Water parameters
    std::vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    std::vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    // Add oxygen (Bisector axis type, uses both H as references)
    tholeDipoleForce->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4,
                                  3.9000000e-1, 1, 1, 2, -1);
    // Add hydrogens (ZOnly axis type, uses O as reference)
    tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4,
                                  3.9000000e-1, 0, 0, 2, -1);
    tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4,
                                  3.9000000e-1, 0, 0, 1, -1);

    // Set up covalent maps for 1-2 bonded interactions
    std::vector<int> covalentMap;

    // Oxygen is bonded to both hydrogens (1-2 bonds)
    covalentMap.clear();
    covalentMap.push_back(1);
    covalentMap.push_back(2);
    tholeDipoleForce->setCovalentMap(0, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

    // Hydrogen 1 is bonded to oxygen (1-2 bond)
    covalentMap.clear();
    covalentMap.push_back(0);
    tholeDipoleForce->setCovalentMap(1, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

    // Hydrogen 2 is bonded to oxygen (1-2 bond)
    covalentMap.clear();
    covalentMap.push_back(0);
    tholeDipoleForce->setCovalentMap(2, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

    // Hydrogen 1 is 1-3 to Hydrogen 2 (through oxygen)
    covalentMap.clear();
    covalentMap.push_back(2);
    tholeDipoleForce->setCovalentMap(1, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);

    // Hydrogen 2 is 1-3 to Hydrogen 1 (through oxygen)
    covalentMap.clear();
    covalentMap.push_back(1);
    tholeDipoleForce->setCovalentMap(2, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);

    tholeDipoleSystem.addForce(tholeDipoleForce);

    // Debug: Print TholeDipole covalent maps
    cout << "\n=== TholeDipole Covalent Maps ===" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "Particle " << i << ":" << endl;
        for (int typeId = 0; typeId < TholeDipoleForce::CovalentEnd; typeId++) {
            std::vector<int> covalentAtoms;
            tholeDipoleForce->getCovalentMap(i, static_cast<TholeDipoleForce::CovalentType>(typeId), covalentAtoms);
            if (!covalentAtoms.empty()) {
                cout << "  Type " << typeId << " (";
                if (typeId == 0) cout << "1-2";
                else if (typeId == 1) cout << "1-3";
                else if (typeId == 2) cout << "1-4";
                else if (typeId == 3) cout << "1-5";
                cout << "): ";
                for (size_t j = 0; j < covalentAtoms.size(); j++) {
                    if (j > 0) cout << ", ";
                    cout << covalentAtoms[j];
                }
                cout << endl;
            }
        }
    }

    // Water geometry (typical water structure)
    std::vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);           // Oxygen at origin
    positions[1] = Vec3(0.09572, 0.0, 0.0);       // H1 along x
    positions[2] = Vec3(-0.023999, 0.092662, 0.0); // H2 at 104.52° angle

    cout << "\n=== Single Water Molecule Test ===" << endl;
    cout << "Testing intramolecular interaction scaling" << endl;
    cout << "Positions:" << endl;
    cout << "  O:  " << positions[0] << endl;
    cout << "  H1: " << positions[1] << endl;
    cout << "  H2: " << positions[2] << endl;

    // Calculate O-H1 distance
    Vec3 diff = positions[1] - positions[0];
    double oh1_dist = sqrt(diff.dot(diff));
    cout << "  O-H1 distance: " << oh1_dist << " nm" << endl;

    // Calculate O-H2 distance
    diff = positions[2] - positions[0];
    double oh2_dist = sqrt(diff.dot(diff));
    cout << "  O-H2 distance: " << oh2_dist << " nm" << endl;

    // Calculate H1-H2 distance
    diff = positions[2] - positions[1];
    double h1h2_dist = sqrt(diff.dot(diff));
    cout << "  H1-H2 distance: " << h1h2_dist << " nm" << endl;

    // Test TholeDipole system
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeDipoleSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    cout << "\nTholeDipole Results:" << endl;
    cout << "  Energy: " << energy << " kJ/mol" << endl;
    cout << "  Forces:" << endl;
    cout << "    O:  " << forces[0] << " kJ/mol/nm" << endl;
    cout << "    H1: " << forces[1] << " kJ/mol/nm" << endl;
    cout << "    H2: " << forces[2] << " kJ/mol/nm" << endl;

    // Basic sanity check
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 3; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    // Compare with AMOEBA
    cout << "\n=== AMOEBA Comparison ===" << endl;

    // Create equivalent AMOEBA system
    System amoebaSystem;
    amoebaSystem.addParticle(1.5995000e+01);
    amoebaSystem.addParticle(1.0080000e+00);
    amoebaSystem.addParticle(1.0080000e+00);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);

    // Debug: Print AMOEBA covalent maps
    cout << "\n=== AMOEBA Covalent Maps ===" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "Particle " << i << ":" << endl;
        for (int typeId = 0; typeId < AmoebaMultipoleForce::PolarizationCovalent11; typeId++) {
            std::vector<int> covalentAtoms;
            amoebaForce->getCovalentMap(i, static_cast<AmoebaMultipoleForce::CovalentType>(typeId), covalentAtoms);
            if (!covalentAtoms.empty()) {
                cout << "  Type " << typeId << " (";
                if (typeId == AmoebaMultipoleForce::Covalent12) cout << "1-2";
                else if (typeId == AmoebaMultipoleForce::Covalent13) cout << "1-3";
                else if (typeId == AmoebaMultipoleForce::Covalent14) cout << "1-4";
                else if (typeId == AmoebaMultipoleForce::Covalent15) cout << "1-5";
                cout << "): ";
                for (size_t j = 0; j < covalentAtoms.size(); j++) {
                    if (j > 0) cout << ", ";
                    cout << covalentAtoms[j];
                }
                cout << endl;
            }
        }
    }

    amoebaSystem.addForce(amoebaForce);

    // Use compareForces for full AMOEBA comparison - this will throw if comparison fails
    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 0.01, 0.01);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testSingleWater();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
