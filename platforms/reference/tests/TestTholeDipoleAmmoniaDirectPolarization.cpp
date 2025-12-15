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
 * This tests TholeDipoleForce ammonia direct polarization.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

static void testTholeDipoleAmmoniaDirectPolarization() {
    std::string testName      = "testTholeDipoleAmmoniaDirectPolarization";

    int numberOfParticles     = 8;
    int inputPmeGridDimension = 0;
    double cutoff             = 9000000.0;

    // Create TholeDipole system
    System tholeDipoleSystem;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(tholeDipoleSystem, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct,
                            cutoff, inputPmeGridDimension);

    // Get positions from the setup
    LangevinIntegrator tempIntegrator(0.0, 0.1, 0.01);
    Context tempContext(tholeDipoleSystem, tempIntegrator, *platform);
    std::vector<Vec3> forces;
    double energy;
    getForcesEnergyTholeDipoleAmmonia(tempContext, forces, energy);
    std::vector<Vec3> positions = tempContext.getState(State::Positions).getPositions();

    cout << "TholeDipole standalone energy: " << energy << " kJ/mol" << endl;

    // Test TholeDipole system standalone first - basic sanity check
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < numberOfParticles; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    // Compare with AMOEBA
    cout << "Starting full AMOEBA comparison..." << endl;

    // Create equivalent AMOEBA system
    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++)
        amoebaSystem.addParticle(tholeDipoleSystem.getParticleMass(i));

    // Set periodic box
    Vec3 a, b, c;
    tholeDipoleSystem.getDefaultPeriodicBoxVectors(a, b, c);
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
    amoebaSystem.addForce(amoebaForce);

    // DEBUG: Print particle parameters
    cout << "\n=== DEBUG: Particle Parameters ===" << endl;
    for (int i = 0; i < numberOfParticles; i++) {
        double charge;
        vector<double> dipole;
        double polarizability;
        int axisType, atomZ, atomX, atomY;
        tholeDipoleForce->getParticleParameters(i, charge, dipole, polarizability, axisType, atomZ, atomX, atomY);
        cout << "Particle " << i << ": q=" << charge << " pol=" << polarizability
             << " axisType=" << axisType << " Z=" << atomZ << " X=" << atomX << " Y=" << atomY << endl;
        cout << "  molDipole=(" << dipole[0] << ", " << dipole[1] << ", " << dipole[2] << ")" << endl;
    }

    // DEBUG: Print covalent maps
    cout << "\n=== DEBUG: Covalent Maps ===" << endl;
    for (int i = 0; i < numberOfParticles; i++) {
        cout << "Particle " << i << ":" << endl;
        for (int t = 0; t < 4; t++) {
            vector<int> cov;
            tholeDipoleForce->getCovalentMap(i, static_cast<TholeDipoleForce::CovalentType>(t), cov);
            if (!cov.empty()) {
                cout << "  Covalent" << (12+t) << ": ";
                for (int j : cov) cout << j << " ";
                cout << endl;
            }
        }
    }

    // DEBUG: Compute and print pair interactions
    cout << "\n=== DEBUG: Pair Distances and Scale Factors ===" << endl;
    for (int i = 0; i < numberOfParticles; i++) {
        for (int j = i+1; j < numberOfParticles; j++) {
            Vec3 delta = positions[j] - positions[i];
            double r = sqrt(delta.dot(delta));

            // Check if in covalent map
            bool is12 = false, is13 = false, is14 = false;
            vector<int> cov12, cov13, cov14;
            tholeDipoleForce->getCovalentMap(i, TholeDipoleForce::Covalent12, cov12);
            tholeDipoleForce->getCovalentMap(i, TholeDipoleForce::Covalent13, cov13);
            tholeDipoleForce->getCovalentMap(i, TholeDipoleForce::Covalent14, cov14);
            for (int k : cov12) if (k == j) is12 = true;
            for (int k : cov13) if (k == j) is13 = true;
            for (int k : cov14) if (k == j) is14 = true;

            double mScale = 1.0;
            if (is12) mScale = 0.0;
            else if (is13) mScale = 0.0;
            else if (is14) mScale = 0.5;

            cout << "Pair (" << i << "," << j << "): r=" << r << " nm, mScale=" << mScale;
            if (is12) cout << " [1-2]";
            if (is13) cout << " [1-3]";
            if (is14) cout << " [1-4]";
            cout << endl;
        }
    }

    // Use compareForces for full AMOEBA comparison including dipoles
    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 0.01, 0.05);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testTholeDipoleAmmoniaDirectPolarization();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}