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
 * This tests TholeDipoleForce neutralizingplasmacorrection.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testNeutralizingPlasmaCorrection() {
    // Verify that the energy of a system with nonzero charge doesn't depend on alpha.

    System system;
    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::PME);
    system.addForce(force);
    vector<double> d(3, 0.0);
    for (int i = 0; i < 2; i++) {
        system.addParticle(1.0);
        force->addParticle(1.0, d, 0.001, 0.39, TholeDipoleForce::NoAxisType, 0, 0, 0);
    }
    vector<Vec3> positions(2);
    positions[0] = Vec3();
    positions[1] = Vec3(0.3, 0.4, 0.0);

    // Compute the energy.

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);
    double energy1 = context.getState(State::Energy).getPotentialEnergy();

    // Change the cutoff distance, which will change alpha, and see if the energy is the same.

    force->setCutoffDistance(0.7);
    context.reinitialize(true);
    double energy2 = context.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy1, energy2, 1e-4);

    // Try changing a particle charge with updateParametersInContext() and make sure the
    // energy changes by the correct amount.

    force->setParticleParameters(0, 2.0, d, 0.001, 0.39, TholeDipoleForce::NoAxisType, 0, 0, 0);
    force->updateParametersInContext(context);
    double energy3 = context.getState(State::Energy).getPotentialEnergy();
    force->setCutoffDistance(1.0);
    context.reinitialize(true);
    double energy4 = context.getState(State::Energy).getPotentialEnergy();
    ASSERT_EQUAL_TOL(energy3, energy4, 1e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testNeutralizingPlasmaCorrection();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}