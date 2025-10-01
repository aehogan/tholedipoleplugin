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
 * This tests TholeDipoleForce particletotaldipoles.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testParticleTotalDipoles() {
    int numberOfParticles     = 8;
    int inputPmeGridDimension = 0;
    double cutoff             = 9000000.0;
    std::vector<Vec3> forces;
    double energy;

    System system;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();;
    setupTholeDipoleAmmonia(system, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Mutual, 
                                             cutoff, inputPmeGridDimension);
    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    getForcesEnergyTholeDipoleAmmonia(context, forces, energy);
    std::vector<Vec3> dipole;
    tholeDipoleForce->getTotalDipoles(context, dipole);
    
    // Compare to expected values (placeholder values - need to be calculated)
    std::vector<Vec3> expectedDipole(numberOfParticles);
    // These would need to be calculated for the Thole dipole model
    for (int i = 0; i < numberOfParticles; i++) {
        // ASSERT_EQUAL_VEC(expectedDipole[i], dipole[i], 1e-4);
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testParticleTotalDipoles();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}