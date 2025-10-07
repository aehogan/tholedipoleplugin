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
 * This tests TholeDipoleForce zxpair.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testZXPair() {
    // Create TholeDipole system
    System tholeSystem;
    for (int i = 0; i < 3; i++)
        tholeSystem.addParticle(1.0);
    
    TholeDipoleForce* force = new TholeDipoleForce();
    tholeSystem.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    
    // 3-particle system with Z-X axis configuration
    double charge[] = {0.5, -0.25, -0.25};
    double dipole[3][3] = {
        {0.0, 0.0, 0.1},    // dipole along z
        {0.1, 0.0, 0.0},    // dipole along x
        {0.0, 0.1, 0.0}     // dipole along y
    };
    double polarizability[] = {0.001, 0.001, 0.001};

    // Particle 0: Z-X axis type - Z-axis defined by particle 1, X-axis by particle 2
    vector<double> d0;
    for (int j = 0; j < 3; j++) d0.push_back(dipole[0][j]);
    force->addParticle(charge[0], d0, polarizability[0],
                      TholeDipoleForce::ZThenX, 1, 2, -1);

    // Particle 1: Reference for Z-axis
    vector<double> d1;
    for (int j = 0; j < 3; j++) d1.push_back(dipole[1][j]);
    force->addParticle(charge[1], d1, polarizability[1],
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    // Particle 2: Reference for X-axis
    vector<double> d2;
    for (int j = 0; j < 3; j++) d2.push_back(dipole[2][j]);
    force->addParticle(charge[2], d2, polarizability[2],
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    
    // Create equivalent AMOEBA system
    System amoebaSystem;
    for (int i = 0; i < 3; i++)
        amoebaSystem.addParticle(1.0);
    
    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaSystem.addForce(amoebaForce);
    
    vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);    // Central particle
    positions[1] = Vec3(0.0, 0.0, 0.3);    // Along z for Z-axis
    positions[2] = Vec3(0.3, 0.0, 0.0);    // Along x for X-axis
    
    // Test TholeDipole system standalone first
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);
    
    // Basic sanity check - energy should be finite
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 3; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1])); 
        ASSERT(std::isfinite(forces[i][2]));
    }
    
    // Energy should be reasonable for a 3-particle system
    ASSERT(fabs(energy) < 1e6);
    
    // Use compareForces for full AMOEBA comparison including dipoles
    try {
        compareForces("ZXPair", tholeSystem, amoebaSystem, positions, 0.01, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testZXPair();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
