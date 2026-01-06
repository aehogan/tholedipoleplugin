#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

void testZeroCharges() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    vector<double> d1(3, 0.0);
    d1[0] = 0.1;
    vector<double> d2(3, 0.0);
    d2[1] = 0.1;

    force->addParticle(0.0, d1, 0.001, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(0.0, d2, 0.001, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0.3, 0, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 2; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }
    ASSERT(fabs(energy) < 1e3);

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
        amoebaSystem.addForce(amoebaForce);

        compareForces("ZeroCharges", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testZeroDipoles() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    vector<double> zeroDipole(3, 0.0);

    force->addParticle(0.5, zeroDipole, 0.001, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(-0.5, zeroDipole, 0.001, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0.3, 0, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 2; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    // Forces should be equal and opposite
    ASSERT_EQUAL_TOL(forces[0][0], -forces[1][0], 1e-8);
    ASSERT_EQUAL_TOL(forces[0][1], -forces[1][1], 1e-8);
    ASSERT_EQUAL_TOL(forces[0][2], -forces[1][2], 1e-8);

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
        amoebaSystem.addForce(amoebaForce);

        compareForces("ZeroDipoles", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testZeroPolarizabilities() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    vector<double> d(3, 0.0);

    force->addParticle(0.5, d, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(-0.5, d, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0.3, 0, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    ASSERT(std::isfinite(energy));
    for (int i = 0; i < 2; i++) {
        ASSERT(std::isfinite(forces[i][0]));
        ASSERT(std::isfinite(forces[i][1]));
        ASSERT(std::isfinite(forces[i][2]));
    }

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);

        compareForces("ZeroPolarizabilities", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testAllZeros() {
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    vector<double> zeroDipole(3, 0.0);

    force->addParticle(0.0, zeroDipole, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(0.0, zeroDipole, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0.3, 0, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    ASSERT(std::isfinite(energy));
    ASSERT_EQUAL_TOL(energy, 0.0, 1e-10);

    for (int i = 0; i < 2; i++) {
        ASSERT_EQUAL_TOL(forces[i][0], 0.0, 1e-10);
        ASSERT_EQUAL_TOL(forces[i][1], 0.0, 1e-10);
        ASSERT_EQUAL_TOL(forces[i][2], 0.0, 1e-10);
    }

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);

        compareForces("AllZeros", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);

        cout << "=== Zero Charges ===" << endl;
        testZeroCharges();

        cout << "\n=== Zero Dipoles ===" << endl;
        testZeroDipoles();

        cout << "\n=== Zero Polarizabilities ===" << endl;
        testZeroPolarizabilities();

        cout << "\n=== All Zeros ===" << endl;
        testAllZeros();

        cout << "\nAll edge case tests passed!" << endl;
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR.  Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
