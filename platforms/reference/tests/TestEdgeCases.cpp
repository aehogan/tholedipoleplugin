#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

void testSingleParticle() {
    System system;
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    vector<double> d(3, 0.0);
    d[0] = 0.1;

    force->addParticle(1.0, d, 0.001, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(1);
    positions[0] = Vec3(0, 0, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), 0.0, 1e-10);
    ASSERT_EQUAL_TOL(state.getForces()[0][0], 0.0, 1e-10);
    ASSERT_EQUAL_TOL(state.getForces()[0][1], 0.0, 1e-10);
    ASSERT_EQUAL_TOL(state.getForces()[0][2], 0.0, 1e-10);

    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaSystem.addForce(amoebaForce);
        compareForces("SingleParticle", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testSingleWater() {
    System tholeDipoleSystem;
    tholeDipoleSystem.addParticle(1.5995000e+01);
    tholeDipoleSystem.addParticle(1.0080000e+00);
    tholeDipoleSystem.addParticle(1.0080000e+00);

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Direct);

    std::vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    std::vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    tholeDipoleForce->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4, 1, 1, 2, -1);
    tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, 0, 2, -1);
    tholeDipoleForce->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, 0, 1, -1);

    std::vector<int> covalentMap;
    covalentMap = {1, 2};
    tholeDipoleForce->setCovalentMap(0, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent12, covalentMap);
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {2};
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent13, covalentMap);
    covalentMap = {1};
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent13, covalentMap);

    tholeDipoleSystem.addForce(tholeDipoleForce);

    std::vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.09572, 0.0, 0.0);
    positions[2] = Vec3(-0.023999, 0.092662, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeDipoleSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Forces | State::Energy);

    ASSERT_EQUAL_TOL(state.getPotentialEnergy(), 0.0, 1e-10);

    const vector<Vec3>& forces = state.getForces();
    Vec3 forceSum = forces[0] + forces[1] + forces[2];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    for (int i = 0; i < 3; i++) {
        ASSERT_EQUAL_TOL(forces[i][0], 0.0, 1e-10);
        ASSERT_EQUAL_TOL(forces[i][1], 0.0, 1e-10);
        ASSERT_EQUAL_TOL(forces[i][2], 0.0, 1e-10);
    }

    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.5995000e+01);
        amoebaSystem.addParticle(1.0080000e+00);
        amoebaSystem.addParticle(1.0080000e+00);
        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
        amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
        amoebaSystem.addForce(amoebaForce);
        compareForces("SingleWater", tholeDipoleSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

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

    ASSERT_EQUAL_TOL(forces[0][0], -forces[1][0], 1e-8);
    ASSERT_EQUAL_TOL(forces[0][1], -forces[1][1], 1e-8);
    ASSERT_EQUAL_TOL(forces[0][2], -forces[1][2], 1e-8);

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

        cout << "=== Single Particle ===" << endl;
        testSingleParticle();

        cout << "\n=== Single Water ===" << endl;
        testSingleWater();

        cout << "\n=== Zero Charges ===" << endl;
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
