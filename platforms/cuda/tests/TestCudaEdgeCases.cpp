/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "CudaTests.h"
#include "CudaTestCommon.h"

void testSingleParticle() {
    cout << "  SingleParticle..." << endl;
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

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testSingleWater() {
    cout << "  SingleWater..." << endl;
    System system;
    system.addParticle(1.5995000e+01);
    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);

    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    force->setPolarizationType(TholeDipoleForce::Direct);

    vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    force->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4, 1, 1, 2, -1);
    force->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, 0, 2, -1);
    force->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, 0, 1, -1);

    vector<int> covalentMap;
    covalentMap = {1, 2};
    force->setCovalentMap(0, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {0};
    force->setCovalentMap(1, TholeDipoleForce::Covalent12, covalentMap);
    force->setCovalentMap(2, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {2};
    force->setCovalentMap(1, TholeDipoleForce::Covalent13, covalentMap);
    covalentMap = {1};
    force->setCovalentMap(2, TholeDipoleForce::Covalent13, covalentMap);

    system.addForce(force);

    vector<Vec3> positions(3);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(0.09572, 0.0, 0.0);
    positions[2] = Vec3(-0.023999, 0.092662, 0.0);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZeroCharges() {
    cout << "  ZeroCharges..." << endl;
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

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZeroDipoles() {
    cout << "  ZeroDipoles..." << endl;
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

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testZeroPolarizabilities() {
    cout << "  ZeroPolarizabilities..." << endl;
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

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void testAllZeros() {
    cout << "  AllZeros..." << endl;
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

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== Edge Case Tests ===" << endl;
        testSingleParticle();
        testSingleWater();
        testZeroCharges();
        testZeroDipoles();
        testZeroPolarizabilities();
        testAllZeros();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
