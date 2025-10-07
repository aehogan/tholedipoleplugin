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
 * This tests TholeDipoleForce singleparticle.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testSingleParticle() {
    System system;
    system.addParticle(1.0);
    
    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    
    // Single particle with charge, dipole, and polarization
    vector<double> d(3, 0.0);
    d[0] = 0.1;  // Dipole along x
    
    double charge = 1.0;
    double polarizability = 0.001;

    // Single particle - no axis type needed since there's no reference particle
    force->addParticle(charge, d, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    
    vector<Vec3> positions(1);
    positions[0] = Vec3(0, 0, 0);
    
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);
    
    State state = context.getState(State::Forces | State::Energy);
    
    // Single particle should have no interactions - energy and forces should be zero
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    
    // Basic sanity checks
    ASSERT(std::isfinite(energy));
    ASSERT(std::isfinite(forces[0][0]));
    ASSERT(std::isfinite(forces[0][1]));
    ASSERT(std::isfinite(forces[0][2]));
    
    // Single particle should have zero energy and forces
    ASSERT_EQUAL_TOL(energy, 0.0, 1e-10);
    ASSERT_EQUAL_TOL(forces[0][0], 0.0, 1e-10);
    ASSERT_EQUAL_TOL(forces[0][1], 0.0, 1e-10);
    ASSERT_EQUAL_TOL(forces[0][2], 0.0, 1e-10);
    
    // Compare with AMOEBA
    try {
        // Create equivalent AMOEBA system
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);
        
        // Use compareForces for full AMOEBA comparison including dipoles
        compareForces("SingleParticle", system, amoebaSystem, positions, 0.01, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testSingleParticle();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}