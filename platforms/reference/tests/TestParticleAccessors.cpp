#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testInducedDipolesDirect() {
    int numberOfParticles = 8;
    double cutoff = 9000000.0;
    std::vector<Vec3> forces;
    double energy;

    System system;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, cutoff, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    getForcesEnergyTholeDipoleAmmonia(context, forces, energy);

    std::vector<Vec3> tholeDipole;
    tholeDipoleForce->getInducedDipoles(context, tholeDipole);

    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* equivalentForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaSystem.addForce(equivalentForce);

    LangevinIntegrator amoebaIntegrator(0.0, 0.1, 0.01);
    Context amoebaContext(amoebaSystem, amoebaIntegrator);
    amoebaContext.setPositions(context.getState(State::Positions).getPositions());
    amoebaContext.getState(State::Forces | State::Energy);

    std::vector<Vec3> amoebaDipole;
    equivalentForce->getInducedDipoles(amoebaContext, amoebaDipole);

    double maxDiff = 0.0;
    for (int i = 0; i < numberOfParticles; i++) {
        Vec3 diff = tholeDipole[i] - amoebaDipole[i];
        maxDiff = std::max(maxDiff, sqrt(diff.dot(diff)));
    }
    ASSERT_EQUAL_TOL(0.0, maxDiff, 1e-4);
}

void testInducedDipolesExtrapolated() {
    int numberOfParticles = 8;
    double cutoff = 9000000.0;
    std::vector<Vec3> forces;
    double energy;

    System system;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Extrapolated, cutoff, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    getForcesEnergyTholeDipoleAmmonia(context, forces, energy);

    std::vector<Vec3> tholeDipole;
    tholeDipoleForce->getInducedDipoles(context, tholeDipole);

    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* equivalentForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaSystem.addForce(equivalentForce);

    LangevinIntegrator amoebaIntegrator(0.0, 0.1, 0.01);
    Context amoebaContext(amoebaSystem, amoebaIntegrator);
    amoebaContext.setPositions(context.getState(State::Positions).getPositions());
    amoebaContext.getState(State::Forces | State::Energy);

    std::vector<Vec3> amoebaDipole;
    equivalentForce->getInducedDipoles(amoebaContext, amoebaDipole);

    double maxDiff = 0.0;
    for (int i = 0; i < numberOfParticles; i++) {
        Vec3 diff = tholeDipole[i] - amoebaDipole[i];
        maxDiff = std::max(maxDiff, sqrt(diff.dot(diff)));
    }
    ASSERT_EQUAL_TOL(0.0, maxDiff, 1e-4);
}

void testInducedDipolesMutual() {
    int numberOfParticles = 8;
    double cutoff = 9000000.0;
    std::vector<Vec3> forces;
    double energy;

    System system;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Mutual, cutoff, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    getForcesEnergyTholeDipoleAmmonia(context, forces, energy);

    std::vector<Vec3> tholeDipole;
    tholeDipoleForce->getInducedDipoles(context, tholeDipole);

    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* equivalentForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaSystem.addForce(equivalentForce);

    LangevinIntegrator amoebaIntegrator(0.0, 0.1, 0.01);
    Context amoebaContext(amoebaSystem, amoebaIntegrator);
    amoebaContext.setPositions(context.getState(State::Positions).getPositions());
    amoebaContext.getState(State::Forces | State::Energy);

    std::vector<Vec3> amoebaDipole;
    equivalentForce->getInducedDipoles(amoebaContext, amoebaDipole);

    double maxDiff = 0.0;
    for (int i = 0; i < numberOfParticles; i++) {
        Vec3 diff = tholeDipole[i] - amoebaDipole[i];
        maxDiff = std::max(maxDiff, sqrt(diff.dot(diff)));
    }
    ASSERT_EQUAL_TOL(0.0, maxDiff, 1e-4);
}

void testLabFramePermanentDipoles() {
    int numberOfParticles = 8;
    double cutoff = 9000000.0;
    std::vector<Vec3> forces;
    double energy;

    System system;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Mutual, cutoff, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    getForcesEnergyTholeDipoleAmmonia(context, forces, energy);

    std::vector<Vec3> tholeDipole;
    tholeDipoleForce->getLabFramePermanentDipoles(context, tholeDipole);

    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* equivalentForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaSystem.addForce(equivalentForce);

    LangevinIntegrator amoebaIntegrator(0.0, 0.1, 0.01);
    Context amoebaContext(amoebaSystem, amoebaIntegrator);
    amoebaContext.setPositions(context.getState(State::Positions).getPositions());
    amoebaContext.getState(State::Forces | State::Energy);

    std::vector<Vec3> amoebaDipole;
    equivalentForce->getLabFramePermanentDipoles(amoebaContext, amoebaDipole);

    double maxDiff = 0.0;
    for (int i = 0; i < numberOfParticles; i++) {
        Vec3 diff = tholeDipole[i] - amoebaDipole[i];
        maxDiff = std::max(maxDiff, sqrt(diff.dot(diff)));
    }
    ASSERT_EQUAL_TOL(0.0, maxDiff, 1e-5);
}

void testTotalDipoles() {
    int numberOfParticles = 8;
    double cutoff = 9000000.0;
    std::vector<Vec3> forces;
    double energy;

    System system;
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    setupTholeDipoleAmmonia(system, tholeDipoleForce, TholeDipoleForce::NoCutoff, TholeDipoleForce::Direct, cutoff, 0);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);
    getForcesEnergyTholeDipoleAmmonia(context, forces, energy);

    std::vector<Vec3> tholeDipole;
    tholeDipoleForce->getTotalDipoles(context, tholeDipole);

    System amoebaSystem;
    for (int i = 0; i < numberOfParticles; i++) amoebaSystem.addParticle(1.0);
    AmoebaMultipoleForce* equivalentForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaSystem.addForce(equivalentForce);

    LangevinIntegrator amoebaIntegrator(0.0, 0.1, 0.01);
    Context amoebaContext(amoebaSystem, amoebaIntegrator);
    amoebaContext.setPositions(context.getState(State::Positions).getPositions());
    amoebaContext.getState(State::Forces | State::Energy);

    std::vector<Vec3> amoebaDipole;
    equivalentForce->getTotalDipoles(amoebaContext, amoebaDipole);

    double maxDiff = 0.0;
    for (int i = 0; i < numberOfParticles; i++) {
        Vec3 diff = tholeDipole[i] - amoebaDipole[i];
        maxDiff = std::max(maxDiff, sqrt(diff.dot(diff)));
    }
    ASSERT_EQUAL_TOL(0.0, maxDiff, 1e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);

        cout << "=== Induced Dipoles (Direct) ===" << endl;
        testInducedDipolesDirect();

        cout << "\n=== Induced Dipoles (Extrapolated) ===" << endl;
        testInducedDipolesExtrapolated();

        cout << "\n=== Induced Dipoles (Mutual) ===" << endl;
        testInducedDipolesMutual();

        cout << "\n=== Lab Frame Permanent Dipoles ===" << endl;
        testLabFramePermanentDipoles();

        cout << "\n=== Total Dipoles ===" << endl;
        testTotalDipoles();

        cout << "\nAll particle accessor tests passed!" << endl;
    }
    catch (const std::exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR.  Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
