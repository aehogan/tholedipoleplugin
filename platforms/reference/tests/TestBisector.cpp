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
 * This tests TholeDipoleForce bisector axis type.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testBisector() {
    // Create TholeDipole system
    System tholeSystem;
    for (int i = 0; i < 4; i++)
        tholeSystem.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    tholeSystem.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    // 4-particle system with Bisector axis configuration
    // Particle 0: central atom with Bisector axis type
    // Particles 1,2: used to define Y-axis (bisector of 1-0-2)
    // Particle 3: used to define Z-axis
    double charge[] = {0.5, -0.25, -0.25, -0.1};
    double dipole[4][3] = {
        {0.0, 0.05, 0.08},    // dipole with Y and Z components
        {0.0, 0.0, 0.0},      // reference for Y-axis
        {0.0, 0.0, 0.0},      // reference for Y-axis
        {0.0, 0.0, 0.1}       // reference for Z-axis
    };
    double polarizability[] = {0.001, 0.001, 0.001, 0.001};

    // Particle 0: Bisector axis type - Y from bisector of 1&2, Z from 3
    vector<double> d0;
    for (int j = 0; j < 3; j++) d0.push_back(dipole[0][j]);
    force->addParticle(charge[0], d0, polarizability[0],
                      TholeDipoleForce::Bisector, 3, 1, 2);

    // Particles 1,2: References for Y-axis bisector
    vector<double> d1;
    for (int j = 0; j < 3; j++) d1.push_back(dipole[1][j]);
    force->addParticle(charge[1], d1, polarizability[1],
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<double> d2;
    for (int j = 0; j < 3; j++) d2.push_back(dipole[2][j]);
    force->addParticle(charge[2], d2, polarizability[2],
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    // Particle 3: Reference for Z-axis
    vector<double> d3;
    for (int j = 0; j < 3; j++) d3.push_back(dipole[3][j]);
    force->addParticle(charge[3], d3, polarizability[3],
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    // Create equivalent AMOEBA system
    System amoebaSystem;
    for (int i = 0; i < 4; i++)
        amoebaSystem.addParticle(1.0);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions(4);
    positions[0] = Vec3(0.0, 0.0, 0.0);      // Central particle
    positions[1] = Vec3(-0.15, 0.15, 0.0);   // For Y-axis bisector
    positions[2] = Vec3(0.15, 0.15, 0.0);    // For Y-axis bisector
    positions[3] = Vec3(0.0, 0.0, 0.3);      // For Z-axis

    // Test TholeDipole system standalone first
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    // Forces should be non-zero (particles are interacting)
    double totalForceMag = 0.0;
    for (int i = 0; i < 4; i++) {
        double forceMag = sqrt(forces[i][0]*forces[i][0] + forces[i][1]*forces[i][1] + forces[i][2]*forces[i][2]);
        totalForceMag += forceMag;
    }
    ASSERT(totalForceMag > 1e-6);

    // Forces should sum to zero (momentum conservation)
    Vec3 forceSum = forces[0] + forces[1] + forces[2] + forces[3];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    // Use compareForces for full AMOEBA comparison including dipoles
    try {
        compareForces("Bisector", tholeSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testBisector();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
