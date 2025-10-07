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
 * This tests TholeDipoleForce ammonia mutual polarization.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

static void testTholeDipoleAmmoniaMutualPolarization() {
    std::string testName      = "testTholeDipoleAmmoniaMutualPolarization";

    int numberOfParticles     = 8;
    int inputPmeGridDimension = 0;
    double cutoff             = 9000000.0;

    // Create TholeDipole system
    System tholeDipoleSystem;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(tholeDipoleSystem, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Mutual,
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
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Mutual);
    amoebaSystem.addForce(amoebaForce);

    // Use compareForces for full AMOEBA comparison including dipoles
    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 0.01, 0.01);

    // Test parameter update functionality
    double tolerance = 1.0e-4;
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeDipoleSystem, integrator, *platform);
    context.setPositions(positions);

    // Try changing the particle parameters and make sure it's still correct.
    for (int i = 0; i < numberOfParticles; i++) {
        double charge, polarizability;
        int axisType, atomX, atomY, atomZ;
        std::vector<double> dipole;
        tholeDipoleForce->getParticleParameters(i, charge, dipole, polarizability, axisType, atomZ, atomX, atomY);
        dipole[0] *= 0.7;
        tholeDipoleForce->setParticleParameters(i, 1.1*charge, dipole, 1.5*polarizability, axisType, atomZ, atomX, atomY);
    }
    LangevinIntegrator integrator2(0.0, 0.1, 0.01);
    Context context2(tholeDipoleSystem, integrator2, context.getPlatform());
    context2.setPositions(context.getState(State::Positions).getPositions());
    State state1 = context.getState(State::Forces | State::Energy);
    State state2 = context2.getState(State::Forces | State::Energy);
    bool exceptionThrown = false;
    try {
        // This should throw an exception.
        compareForcesEnergy(testName, state2.getPotentialEnergy(), state1.getPotentialEnergy(), state2.getForces(), state1.getForces(), tolerance);
        for (int i = 0; i < numberOfParticles; i++)
            ASSERT_EQUAL_VEC(state1.getForces()[i], state2.getForces()[i], tolerance);
    }
    catch (std::exception ex) {
        exceptionThrown = true;
    }
    ASSERT(exceptionThrown);
    tholeDipoleForce->updateParametersInContext(context);
    state1 = context.getState(State::Forces | State::Energy);
    compareForcesEnergy(testName, state2.getPotentialEnergy(), state1.getPotentialEnergy(), state2.getForces(), state1.getForces(), tolerance);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testTholeDipoleAmmoniaMutualPolarization();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}