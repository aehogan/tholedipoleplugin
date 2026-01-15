/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

void runTwoDipoleTest(const string& name,
                       double charge1, const Vec3& dipole1,
                       double charge2, const Vec3& dipole2,
                       const Vec3& pos1, const Vec3& pos2,
                       double boxSize,
                       double energyTol, double forceTol) {
    cout << "  TwoDipoles " << name << "..." << endl;

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    Vec3 a(boxSize, 0.0, 0.0);
    Vec3 b(0.0, boxSize, 0.0);
    Vec3 c(0.0, 0.0, boxSize);
    system.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::PME);
    force->setPolarizationType(TholeDipoleForce::Direct);
    force->setCutoffDistance(0.7);
    force->setPMEParameters(5.0, 64, 64, 64);

    vector<double> dip1 = {dipole1[0], dipole1[1], dipole1[2]};
    vector<double> dip2 = {dipole2[0], dipole2[1], dipole2[2]};

    force->addParticle(charge1, dip1, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(charge2, dip2, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);

    system.addForce(force);

    vector<Vec3> positions = {pos1, pos2};

    assertForcesAndEnergiesMatch(system, positions, energyTol, forceTol);
}

void testTwoDipolesNoCutoff() {
    cout << "  TwoDipoles NoCutoff..." << endl;

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    vector<double> dip = {0.01, 0, 0};
    force->addParticle(0.0, dip, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(0.0, dip, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    system.addForce(force);

    vector<Vec3> positions = {Vec3(0.2, 0.5, 0.5), Vec3(0.5, 0.5, 0.5)};

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testTwoDipolesChargesOnly() {
    runTwoDipoleTest("ChargesOnly",
                     0.5, Vec3(0, 0, 0),
                     -0.3, Vec3(0, 0, 0),
                     Vec3(0.2, 0.2, 0.2),
                     Vec3(0.5, 0.2, 0.2),
                     1.5, 5e-5, 5e-4);
}

void testTwoDipolesDipolesAlongAxis() {
    runTwoDipoleTest("DipolesAlongAxis",
                     0.0, Vec3(0.01, 0, 0),
                     0.0, Vec3(0.01, 0, 0),
                     Vec3(0.2, 0.5, 0.5),
                     Vec3(0.5, 0.5, 0.5),
                     1.5, 5e-5, 5e-4);
}

void testTwoDipolesChargesAndDipoles() {
    runTwoDipoleTest("ChargesAndDipoles",
                     0.5, Vec3(0.01, 0, 0),
                     -0.5, Vec3(0.01, 0, 0),
                     Vec3(0.2, 0.5, 0.5),
                     Vec3(0.5, 0.5, 0.5),
                     1.5, 5e-5, 5e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== Two Dipoles Tests ===" << endl;
        testTwoDipolesNoCutoff();
        testTwoDipolesChargesOnly();
        testTwoDipolesDipolesAlongAxis();
        testTwoDipolesChargesAndDipoles();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
