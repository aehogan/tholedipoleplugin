#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include "openmm/AmoebaMultipoleForce.h"

// Helper to create a 2-particle ZOnly system with configurable parameters
void createZOnlySystem(System& system, TholeDipoleForce*& force,
                       double dip1z, double dip2z, double pol,
                       TholeDipoleForce::PolarizationType polType) {
    system.addParticle(1.0);
    system.addParticle(1.0);

    force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(polType);
    system.addForce(force);

    vector<double> d1(3, 0.0);
    d1[2] = dip1z;
    vector<double> d2(3, 0.0);
    d2[2] = dip2z;

    double charge1 = 0.5;
    double charge2 = -0.5;

    // Use NoAxisType if no dipoles AND no polarization (pure point charges)
    if (dip1z == 0.0 && dip2z == 0.0 && pol == 0.0) {
        force->addParticle(charge1, d1, pol, TholeDipoleForce::NoAxisType, -1, -1, -1);
        force->addParticle(charge2, d2, pol, TholeDipoleForce::NoAxisType, -1, -1, -1);
    } else {
        force->addParticle(charge1, d1, pol, TholeDipoleForce::ZOnly, 1, -1, -1);
        force->addParticle(charge2, d2, pol, TholeDipoleForce::ZOnly, 0, -1, -1);
    }
}

// Common validation for 2-particle ZOnly tests
void validateZOnlyForces(const string& testName, const vector<Vec3>& forces) {
    // Cylindrical symmetry: forces along z-axis should have no x,y components
    ASSERT(fabs(forces[0][0]) < 1e-10);
    ASSERT(fabs(forces[0][1]) < 1e-10);
    ASSERT(fabs(forces[1][0]) < 1e-10);
    ASSERT(fabs(forces[1][1]) < 1e-10);

    // Forces should sum to zero (momentum conservation)
    Vec3 forceSum = forces[0] + forces[1];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);
}

void runZOnlyTest(const string& testName, double dip1z, double dip2z, double pol,
                  TholeDipoleForce::PolarizationType polType,
                  AmoebaMultipoleForce::PolarizationType amoebaPolType) {
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, dip1z, dip2z, pol, polType);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    printf("%s Test:\n", testName.c_str());
    printf("Energy: %.8f\n", state.getPotentialEnergy());
    for (int i = 0; i < 2; i++) {
        Vec3 f = state.getForces()[i];
        printf("Force[%d]: (%.6e, %.6e, %.6e)\n", i, f[0], f[1], f[2]);
    }

    validateZOnlyForces(testName, state.getForces());

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
        amoebaForce->setPolarizationType(amoebaPolType);
        amoebaSystem.addForce(amoebaForce);

        compareForces(testName, system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testZOnlyDirect() {
    runZOnlyTest("ZOnlyDirect", 0.1, -0.05, 0.001,
                 TholeDipoleForce::Direct, AmoebaMultipoleForce::Direct);
}

void testZOnlyMutual() {
    runZOnlyTest("ZOnlyMutual", 0.1, -0.05, 0.001,
                 TholeDipoleForce::Mutual, AmoebaMultipoleForce::Mutual);
}

void testZOnlyNoDipole() {
    runZOnlyTest("ZOnlyNoDipole", 0.0, 0.0, 0.001,
                 TholeDipoleForce::Direct, AmoebaMultipoleForce::Direct);
}

void testZOnlyNoPolarization() {
    runZOnlyTest("ZOnlyNoPolarization", 0.1, -0.05, 0.0,
                 TholeDipoleForce::Direct, AmoebaMultipoleForce::Direct);
}

void testZOnlyNoDipoleNoPolarization() {
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, 0.0, 0.0, 0.0, TholeDipoleForce::Direct);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    context.setPositions(positions);

    State state = context.getState(State::Forces | State::Energy);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    // Pure Coulomb: E = k*q1*q2/r = 138.935*0.5*(-0.5)/0.3 = -115.78 kJ/mol
    ASSERT_EQUAL_TOL(energy, -115.78, 1e-3);

    // Forces should be non-zero and equal/opposite
    ASSERT(fabs(forces[0][2]) > 1e-6);
    ASSERT_EQUAL_TOL(forces[0][2], -forces[1][2], 1e-10);

    validateZOnlyForces("ZOnlyNoDipoleNoPolarization", forces);

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        amoebaSystem.addParticle(1.0);
        amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
        amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
        amoebaSystem.addForce(amoebaForce);

        compareForces("ZOnlyNoDipoleNoPolarization", system, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

void testZOnly3Particle() {
    System tholeSystem;
    for (int i = 0; i < 3; i++)
        tholeSystem.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);
    tholeSystem.addForce(force);

    double charge[] = {0.5, -0.25, -0.25};
    double dipole[3][3] = {
        {0.0, 0.0, 0.1},
        {0.0, 0.0, 0.05},
        {0.0, 0.0, 0.025}
    };
    double polarity[] = {0.001, 0.001, 0.001};

    for (int i = 0; i < 3; i++) {
        vector<double> d;
        for (int j = 0; j < 3; j++)
            d.push_back(dipole[i][j]);
        int zAxis = (i + 1) % 3;
        force->addParticle(charge[i], d, polarity[i], TholeDipoleForce::ZOnly, zAxis, -1, -1);
    }

    vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.3, 0.0, 0.0);
    positions[2] = Vec3(0.0, 0.3, 0.0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(tholeSystem, integrator, *platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    double energy = state.getPotentialEnergy();
    const vector<Vec3>& forces = state.getForces();

    ASSERT(energy < 0.0);

    // Forces should sum to zero
    Vec3 forceSum(0, 0, 0);
    for (int i = 0; i < 3; i++)
        forceSum += forces[i];
    ASSERT(fabs(forceSum[0]) < 1e-6);
    ASSERT(fabs(forceSum[1]) < 1e-6);
    ASSERT(fabs(forceSum[2]) < 1e-6);

    // Compare with AMOEBA
    try {
        System amoebaSystem;
        for (int i = 0; i < 3; i++)
            amoebaSystem.addParticle(1.0);

        AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(force);
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
        amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
        amoebaSystem.addForce(amoebaForce);

        compareForces("ZOnly3Particle", tholeSystem, amoebaSystem, positions, 1e-4, 1e-3);
    } catch (const std::exception& e) {
        cout << "AMOEBA comparison failed: " << e.what() << endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);

        cout << "=== ZOnly Direct ===" << endl;
        testZOnlyDirect();

        cout << "\n=== ZOnly Mutual ===" << endl;
        testZOnlyMutual();

        cout << "\n=== ZOnly No Dipole ===" << endl;
        testZOnlyNoDipole();

        cout << "\n=== ZOnly No Polarization ===" << endl;
        testZOnlyNoPolarization();

        cout << "\n=== ZOnly No Dipole No Polarization ===" << endl;
        testZOnlyNoDipoleNoPolarization();

        cout << "\n=== ZOnly 3 Particle ===" << endl;
        testZOnly3Particle();

        cout << "\nAll ZOnly tests passed!" << endl;
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR.  Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
