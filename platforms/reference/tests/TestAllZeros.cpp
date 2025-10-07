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
 * This tests TholeDipoleForce allzeros.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

void testAllZeros() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    
    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    
    // System with all parameters set to zero
    vector<double> zeroDipole(3, 0.0);  // Zero dipole moments
    double zeroCharge = 0.0;  // Zero charges
    double zeroPolarizability = 0.0;  // Zero polarizabilities

    // Both particles have everything set to zero
    force->addParticle(zeroCharge, zeroDipole, zeroPolarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(zeroCharge, zeroDipole, zeroPolarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    
    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0.3, 0, 0);
    
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);
    
    State state = context.getState(State::Forces | State::Energy);
    
    // Should have no interactions at all - energy should be zero or negligible
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();
    
    // Basic sanity checks
    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 2; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }
    
    // Energy should be zero or very close to zero
    ASSERT_EQUAL_TOL(energy, 0.0, 1e-10);
    
    // Forces should be zero or very close to zero
    for (int i = 0; i < 2; i++) {
        ASSERT_EQUAL_TOL(forces[i][0], 0.0, 1e-10);
        ASSERT_EQUAL_TOL(forces[i][1], 0.0, 1e-10);
        ASSERT_EQUAL_TOL(forces[i][2], 0.0, 1e-10);
    }
    
    // Compare with AMOEBA
    try {
        // Create equivalent AMOEBA system
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);
        
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);
        
        // Use compareForces for full AMOEBA comparison including dipoles
        compareForces("AllZeros", system, amoebaSystem, positions, 0.01, 0.01);
    } catch (const std::exception& e) {
        cout << "Full AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testAllZeros();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}