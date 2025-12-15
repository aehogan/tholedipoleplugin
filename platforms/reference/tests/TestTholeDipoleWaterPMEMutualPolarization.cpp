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
 * This tests TholeDipoleForce waterpmemutualpolarization.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testTholeDipoleWaterPMEMutualPolarization() {

    std::string testName      = "testTholeDipoleWaterMutualPolarization";

    int numberOfParticles     = 12;
    int inputPmeGridDimension = 20;
    double cutoff             = 0.70;

    // Get initial forces/energy and positions using existing setup
    std::vector<Vec3> forces;
    double energy;
    setupAndGetForcesEnergyTholeDipoleWater(TholeDipoleForce::PME, TholeDipoleForce::Mutual,
                                            cutoff, inputPmeGridDimension, forces, energy);

    cout << "TholeDipole standalone energy: " << energy << " kJ/mol" << endl;

    // Test TholeDipole system standalone first - basic sanity check
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < numberOfParticles; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    // Compare with AMOEBA - need to recreate systems for comparison
    cout << "Starting full AMOEBA comparison..." << endl;

    // Create TholeDipole system
    System tholeDipoleSystem;
    double boxDimension = 1.8643;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Mutual);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-6);
    tholeDipoleForce->setMutualInducedMaxIterations(500);
    tholeDipoleForce->setPMEParameters(5.4459052e+00, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    // Add particles
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleSystem.addParticle(1.5995000e+01);
        tholeDipoleSystem.addParticle(1.0080000e+00);
        tholeDipoleSystem.addParticle(1.0080000e+00);
    }

    std::vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    std::vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleForce->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4, 1, jj+1, jj+2, -1);
        tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, jj, jj+2, -1);
        tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, jj, jj+1, -1);
    }

    // Covalent maps (same as in setupAndGetForcesEnergyTholeDipoleWater)
    std::vector<int> covalentMap;
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        covalentMap.clear();
        covalentMap.push_back(jj+1);
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj);
        tholeDipoleForce->setCovalentMap(jj+1, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);
        tholeDipoleForce->setCovalentMap(jj+2, static_cast<TholeDipoleForce::CovalentType>(0), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj+1, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+1);
        tholeDipoleForce->setCovalentMap(jj+2, static_cast<TholeDipoleForce::CovalentType>(1), covalentMap);
    }

    tholeDipoleSystem.addForce(tholeDipoleForce);

    std::vector<Vec3> positions(numberOfParticles);
    positions[0]  = Vec3(-8.7387270e-01,  5.3220410e-01,  7.4214000e-03);
    positions[1]  = Vec3(-9.6050090e-01,  5.1173410e-01, -2.2202700e-02);
    positions[2]  = Vec3(-8.5985900e-01,  4.9658230e-01,  1.0283390e-01);
    positions[3]  = Vec3( 9.1767100e-02, -7.8956650e-01,  4.3804200e-01);
    positions[4]  = Vec3( 1.2333420e-01, -7.0267430e-01,  4.2611550e-01);
    positions[5]  = Vec3( 1.7267090e-01, -8.2320810e-01,  4.8124750e-01);
    positions[6]  = Vec3( 8.6290110e-01,  6.2153500e-02,  4.1280850e-01);
    positions[7]  = Vec3( 8.6385200e-01,  1.2684730e-01,  3.3887060e-01);
    positions[8]  = Vec3( 9.5063550e-01,  5.3173300e-02,  4.4799160e-01);
    positions[9]  = Vec3( 5.0844930e-01,  2.8684740e-01, -6.9293750e-01);
    positions[10] = Vec3( 6.0459330e-01,  3.0620510e-01, -7.0100130e-01);
    positions[11] = Vec3( 5.0590640e-01,  1.8880920e-01, -6.8813470e-01);

    // Create equivalent AMOEBA system
    System amoebaSystem;
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < numberOfParticles; i++)
        amoebaSystem.addParticle(tholeDipoleSystem.getParticleMass(i));

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Mutual);
    amoebaSystem.addForce(amoebaForce);

    // Use compareForces for full AMOEBA comparison including dipoles
    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 0.02, 0.5);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testTholeDipoleWaterPMEMutualPolarization();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}