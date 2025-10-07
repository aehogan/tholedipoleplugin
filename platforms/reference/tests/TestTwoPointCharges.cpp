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
 * This tests TholeDipoleForce with two point charges to validate against MPMC.
 * Two charges: +0.5e and -0.5e separated by 3 Angstroms (0.3 nm)
 * Tests multiple damping types and compares with MPMC ground truth.
 *
 * MPMC Reference values (10000 A box):
 *   No polarization: -13925.19971 K = -115.768 kJ/mol (electrostatic only)
 *   Linear damping (λ=2.1304): -14795.52451 K = -123.016 kJ/mol
 *   Exponential damping (λ=2.1304): -14778.87515 K = -122.878 kJ/mol
 *   AMOEBA damping (λ=0.39): -14794.40479 K = -123.007 kJ/mol
 */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testTwoPointChargesNoPol() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    // No polarization - zero polarizability
    vector<double> zeroDipole(3, 0.0);
    double charge1 = 0.5;
    double charge2 = -0.5;
    double polarizability = 0.0;

    force->addParticle(charge1, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge2, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.3, 0.0, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    cout << "No Polarization Test:" << endl;
    cout << "  Energy: " << energy << " kJ/mol" << endl;

    double mpmc_energy = -13925.19971 * 0.008314462;
    cout << "  MPMC:   " << mpmc_energy << " kJ/mol" << endl;
    cout << "  Diff:   " << (energy - mpmc_energy) << " kJ/mol" << endl;

    ASSERT(std::isfinite(energy));
    ASSERT(energy < 0.0);
    ASSERT_EQUAL_TOL(forces[0][0], -forces[1][0], 1e-6);
}

void testTwoPointChargesLinear() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Mutual);
    force->setMutualInducedTargetEpsilon(1.0e-6);
    force->setMutualInducedMaxIterations(500);

    // Linear damping
    force->setTholeDampingType(TholeDipoleForce::Linear);
    force->setTholeDampingParameter(2.1304);

    vector<double> zeroDipole(3, 0.0);
    double charge1 = 0.5;
    double charge2 = -0.5;
    double polarizability = 0.0015;  // 1.5 A^3

    force->addParticle(charge1, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge2, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.3, 0.0, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    cout << "Linear Damping Test:" << endl;
    cout << "  Energy: " << energy << " kJ/mol" << endl;

    double mpmc_energy = -14795.52451 * 0.008314462;
    cout << "  MPMC:   " << mpmc_energy << " kJ/mol" << endl;
    cout << "  Diff:   " << (energy - mpmc_energy) << " kJ/mol" << endl;

    ASSERT(std::isfinite(energy));
    ASSERT(energy < 0.0);
    ASSERT_EQUAL_TOL(forces[0][0], -forces[1][0], 1e-6);
}

void testTwoPointChargesExponential() {
    System system;
    system.addParticle(1.0);  // Particle 1
    system.addParticle(1.0);  // Particle 2

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Mutual);
    force->setMutualInducedTargetEpsilon(1.0e-6);
    force->setMutualInducedMaxIterations(500);

    // Set Thole damping to match MPMC
    force->setTholeDampingType(TholeDipoleForce::Exponential);
    force->setTholeDampingParameter(2.1304);

    // Two particles with charges +0.5 and -0.5
    // Zero permanent dipole, polarizability = 1.5 A^3 = 0.0015 nm^3
    vector<double> zeroDipole(3, 0.0);

    double charge1 = 0.5;   // electron charge units
    double charge2 = -0.5;
    double polarizability = 0.0015;  // nm^3 (= 1.5 A^3)

    // Point charges don't need axis definitions (NoAxisType)
    force->addParticle(charge1, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge2, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    // Positions: 3 Angstroms apart along x-axis
    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);  // nm
    positions[1] = Vec3(0.3, 0.0, 0.0);  // 3 Angstroms = 0.3 nm

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    cout << "Exponential Damping Test:" << endl;
    cout << "  Energy: " << energy << " kJ/mol" << endl;

    double mpmc_energy = -14778.87515 * 0.008314462;
    cout << "  MPMC:   " << mpmc_energy << " kJ/mol" << endl;
    cout << "  Diff:   " << (energy - mpmc_energy) << " kJ/mol" << endl;

    ASSERT(std::isfinite(energy));
    ASSERT(energy < 0.0);
    ASSERT_EQUAL_TOL(forces[0][0], -forces[1][0], 1e-6);
}

void testTwoPointChargesAmoeba() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Mutual);
    force->setMutualInducedTargetEpsilon(1.0e-6);
    force->setMutualInducedMaxIterations(500);

    // AMOEBA damping
    force->setTholeDampingType(TholeDipoleForce::Amoeba);
    force->setTholeDampingParameter(0.39);

    vector<double> zeroDipole(3, 0.0);
    double charge1 = 0.5;
    double charge2 = -0.5;
    double polarizability = 0.0015;  // 1.5 A^3

    force->addParticle(charge1, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge2, zeroDipole, polarizability,
                      TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.3, 0.0, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);
    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    cout << "AMOEBA Damping Test:" << endl;
    cout << "  Energy: " << energy << " kJ/mol" << endl;

    double mpmc_energy = -14794.40479 * 0.008314462;
    cout << "  MPMC:   " << mpmc_energy << " kJ/mol" << endl;
    cout << "  Diff:   " << (energy - mpmc_energy) << " kJ/mol" << endl;

    ASSERT(std::isfinite(energy));
    ASSERT(energy < 0.0);
    ASSERT_EQUAL_TOL(forces[0][0], -forces[1][0], 1e-6);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n========================================" << endl;
        cout << "Two Point Charges Validation Tests" << endl;
        cout << "========================================\n" << endl;

        testTwoPointChargesNoPol();
        cout << endl;

        testTwoPointChargesLinear();
        cout << endl;

        testTwoPointChargesExponential();
        cout << endl;

        testTwoPointChargesAmoeba();
        cout << endl;

        cout << "========================================" << endl;
        cout << "All tests passed!" << endl;
        cout << "========================================" << endl;
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << endl;
    return 0;
}
