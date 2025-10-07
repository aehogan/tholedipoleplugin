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
 * This tests TholeDipoleForce triclinic.
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testTriclinic() {
    // Create a triclinic box containing eight water molecules.

    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(1.8643, 0, 0), Vec3(-0.16248445120445926, 1.8572057756524414, 0), Vec3(0.16248445120445906, -0.14832299817478897, 1.8512735025730875));
    for (int i = 0; i < 24; i++)
        system.addParticle(1.0);
    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::PME);
    force->setPolarizationType(TholeDipoleForce::Mutual);
    force->setCutoffDistance(0.7);
    force->setMutualInducedTargetEpsilon(1e-6);
    force->setPMEParameters(5.4459051633620055, 24, 24, 24);
    double o_charge = -0.42616, h_charge = 0.21308;
    vector<double> o_dipole(3), h_dipole(3);
    o_dipole[0] = 0;
    o_dipole[1] = 0;
    o_dipole[2] = 0.0033078867454609203;
    h_dipole[0] = -0.0053536858428776405;
    h_dipole[1] = 0;
    h_dipole[2] = -0.014378273997907321;

    for (int i = 0; i < 8; i++) {
        int atom1 = 3*i, atom2 = 3*i+1, atom3 = 3*i+2;
        force->addParticle(o_charge, o_dipole, 0.001*0.92, 1, atom2, atom3, -1);
        force->addParticle(h_charge, h_dipole, 0.001*0.539, 0, atom1, atom3, -1);
        force->addParticle(h_charge, h_dipole, 0.001*0.539, 0, atom1, atom2, -1);
        vector<int> coval1_12(2);
        coval1_12[0] = atom2;
        coval1_12[1] = atom3;
        force->setCovalentMap(atom1, TholeDipoleForce::Covalent12, coval1_12);
        vector<int> coval2_12(1);
        coval2_12[0] = atom1;
        force->setCovalentMap(atom2, TholeDipoleForce::Covalent12, coval2_12);
        force->setCovalentMap(atom3, TholeDipoleForce::Covalent12, coval2_12);
        vector<int> coval2_13(1);
        coval2_13[0] = atom3;
        force->setCovalentMap(atom2, TholeDipoleForce::Covalent13, coval2_13);
        vector<int> coval3_13(1);
        coval3_13[0] = atom2;
        force->setCovalentMap(atom3, TholeDipoleForce::Covalent13, coval3_13);
    }
    vector<Vec3> positions(24);
    positions[0] = Vec3(0.867966, 0.708769, -0.0696862);
    positions[1] = Vec3(0.780946, 0.675579, -0.0382259);
    positions[2] = Vec3(0.872223, 0.681424, -0.161756);
    positions[3] = Vec3(-0.0117313, 0.824445, 0.683762);
    positions[4] = Vec3(0.0216892, 0.789544, 0.605003);
    positions[5] = Vec3(0.0444268, 0.782601, 0.75302);
    positions[6] = Vec3(0.837906, -0.0092611, 0.681463);
    positions[7] = Vec3(0.934042, 0.0098069, 0.673406);
    positions[8] = Vec3(0.793962, 0.0573676, 0.626984);
    positions[9] = Vec3(0.658995, 0.184432, -0.692317);
    positions[10] = Vec3(0.588543, 0.240231, -0.671793);
    positions[11] = Vec3(0.618153, 0.106275, -0.727368);
    positions[12] = Vec3(0.71466, 0.575358, 0.233152);
    positions[13] = Vec3(0.636812, 0.612604, 0.286268);
    positions[14] = Vec3(0.702502, 0.629465, 0.15182);
    positions[15] = Vec3(-0.242658, -0.850419, -0.250483);
    positions[16] = Vec3(-0.169206, -0.836825, -0.305829);
    positions[17] = Vec3(-0.279321, -0.760247, -0.24031);
    positions[18] = Vec3(-0.803838, -0.360559, 0.230369);
    positions[19] = Vec3(-0.811375, -0.424813, 0.301849);
    positions[20] = Vec3(-0.761939, -0.2863, 0.270962);
    positions[21] = Vec3(-0.148063, 0.824409, -0.827221);
    positions[22] = Vec3(-0.20902, 0.868798, -0.7677);
    positions[23] = Vec3(-0.0700878, 0.882333, -0.832221);

    // Compute the forces and energy.

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);

    std::cout << "TholeDipole Energy: " << state.getPotentialEnergy() << " kJ/mol" << std::endl;

    // Basic sanity checks
    ASSERT(std::isfinite(state.getPotentialEnergy()));
    const vector<Vec3>& forces = state.getForces();
    for (int i = 0; i < 24; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    // Compare with AMOEBA
    System amoebaSystem;
    amoebaSystem.setDefaultPeriodicBoxVectors(Vec3(1.8643, 0, 0), Vec3(-0.16248445120445926, 1.8572057756524414, 0), Vec3(0.16248445120445906, -0.14832299817478897, 1.8512735025730875));
    for (int i = 0; i < 24; i++)
        amoebaSystem.addParticle(1.0);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Mutual);
    amoebaForce->setCutoffDistance(0.7);
    amoebaForce->setMutualInducedTargetEpsilon(1e-6);
    amoebaForce->setPMEParameters(5.4459051633620055, 24, 24, 24);
    amoebaSystem.addForce(amoebaForce);

    compareForces("testTriclinic", system, amoebaSystem, positions, 0.1, 0.1);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testTriclinic();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}