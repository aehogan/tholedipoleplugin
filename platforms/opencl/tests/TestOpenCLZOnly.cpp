/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

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

    if (dip1z == 0.0 && dip2z == 0.0 && pol == 0.0) {
        force->addParticle(charge1, d1, pol, TholeDipoleForce::NoAxisType, -1, -1, -1);
        force->addParticle(charge2, d2, pol, TholeDipoleForce::NoAxisType, -1, -1, -1);
    } else {
        force->addParticle(charge1, d1, pol, TholeDipoleForce::ZOnly, 1, -1, -1);
        force->addParticle(charge2, d2, pol, TholeDipoleForce::ZOnly, 0, -1, -1);
    }
}

void testZOnlyDirect() {
    cout << "  ZOnly Direct..." << endl;
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, 0.1, -0.05, 0.001, TholeDipoleForce::Direct);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZOnlyMutual() {
    cout << "  ZOnly Mutual..." << endl;
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, 0.1, -0.05, 0.001, TholeDipoleForce::Mutual);
    force->setMutualInducedMaxIterations(200);
    force->setMutualInducedTargetEpsilon(1.0e-9);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZOnlyNoDipole() {
    cout << "  ZOnly NoDipole..." << endl;
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, 0.0, 0.0, 0.001, TholeDipoleForce::Direct);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZOnlyNoPolarization() {
    cout << "  ZOnly NoPolarization..." << endl;
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, 0.1, -0.05, 0.0, TholeDipoleForce::Direct);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZOnlyNoDipoleNoPolarization() {
    cout << "  ZOnly NoDipoleNoPolarization..." << endl;
    System system;
    TholeDipoleForce* force;
    createZOnlySystem(system, force, 0.0, 0.0, 0.0, TholeDipoleForce::Direct);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0, 0, 0);
    positions[1] = Vec3(0, 0, 0.3);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZOnly3Particle() {
    cout << "  ZOnly 3Particle..." << endl;
    System system;
    for (int i = 0; i < 3; i++)
        system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);
    system.addForce(force);

    double charge[] = {0.5, -0.25, -0.25};
    double dipole[3][3] = {{0.0, 0.0, 0.1}, {0.0, 0.0, 0.05}, {0.0, 0.0, 0.025}};
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

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== ZOnly Tests ===" << endl;
        testZOnlyDirect();
        testZOnlyMutual();
        testZOnlyNoDipole();
        testZOnlyNoPolarization();
        testZOnlyNoDipoleNoPolarization();
        testZOnly3Particle();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
