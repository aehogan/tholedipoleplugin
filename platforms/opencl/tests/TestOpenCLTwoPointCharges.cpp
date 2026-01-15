/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- */

#include "OpenCLTests.h"
#include "OpenCLTestCommon.h"

static const double CHARGE1 = 0.5;
static const double CHARGE2 = -0.5;
static const double SEPARATION = 0.3;
static const double POLARIZABILITY = 0.0015;
static const double BOX_SIZE = 5.0;
static const double CUTOFF = 2.0;

void testTwoPointChargesNoPol() {
    cout << "  TwoPointCharges NoPol..." << endl;
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);
    force->setNonbondedMethod(TholeDipoleForce::NoCutoff);

    vector<double> zeroDipole(3, 0.0);
    force->addParticle(CHARGE1, zeroDipole, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(CHARGE2, zeroDipole, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    positions[0] = Vec3(0.0, 0.0, 0.0);
    positions[1] = Vec3(SEPARATION, 0.0, 0.0);

    assertForcesAndEnergiesMatch(system, positions, 5e-5, 5e-4);
}

void runTwoPointChargesTest(TholeDipoleForce::TholeDampingType dampingType,
                             double dampingParam,
                             TholeDipoleForce::NonbondedMethod nbMethod,
                             TholeDipoleForce::PolarizationType polType,
                             const string& name) {
    cout << "  TwoPointCharges " << name << "..." << endl;

    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);

    if (nbMethod == TholeDipoleForce::PME) {
        system.setDefaultPeriodicBoxVectors(Vec3(BOX_SIZE, 0, 0),
                                            Vec3(0, BOX_SIZE, 0),
                                            Vec3(0, 0, BOX_SIZE));
    }

    TholeDipoleForce* force = new TholeDipoleForce();
    system.addForce(force);

    force->setNonbondedMethod(nbMethod);
    force->setPolarizationType(polType);
    force->setTholeDampingType(dampingType);
    force->setTholeDampingParameter(dampingParam);
    force->setDampPermanentInducedField(false);

    if (nbMethod == TholeDipoleForce::PME) {
        force->setCutoffDistance(CUTOFF);
    }

    if (polType == TholeDipoleForce::Mutual) {
        force->setMutualInducedTargetEpsilon(1.0e-9);
        force->setMutualInducedMaxIterations(500);
    }

    vector<double> zeroDipole(3, 0.0);
    force->addParticle(CHARGE1, zeroDipole, POLARIZABILITY, TholeDipoleForce::NoAxisType, -1, -1, -1);
    force->addParticle(CHARGE2, zeroDipole, POLARIZABILITY, TholeDipoleForce::NoAxisType, -1, -1, -1);

    vector<Vec3> positions(2);
    if (nbMethod == TholeDipoleForce::NoCutoff) {
        positions[0] = Vec3(0.0, 0.0, 0.0);
        positions[1] = Vec3(SEPARATION, 0.0, 0.0);
    } else {
        positions[0] = Vec3(BOX_SIZE/2 - SEPARATION/2, BOX_SIZE/2, BOX_SIZE/2);
        positions[1] = Vec3(BOX_SIZE/2 + SEPARATION/2, BOX_SIZE/2, BOX_SIZE/2);
    }

    double energyTol = 5e-5;
    double forceTol = 5e-4;

    assertForcesAndEnergiesMatch(system, positions, energyTol, forceTol);
}

void testTwoPointChargesCombinations() {
    struct DampingConfig {
        TholeDipoleForce::TholeDampingType type;
        double parameter;
        const char* name;
    };

    DampingConfig dampings[] = {
        {TholeDipoleForce::NoDamping,    0.0,    "NoDamping"},
        {TholeDipoleForce::Exponential,  21.304, "Exponential"},
        {TholeDipoleForce::Amoeba,       0.39,   "Amoeba"},
        {TholeDipoleForce::Linear,       2.1304, "Linear"},
    };

    TholeDipoleForce::NonbondedMethod methods[] = {
        TholeDipoleForce::NoCutoff,
        TholeDipoleForce::PME
    };

    TholeDipoleForce::PolarizationType pols[] = {
        TholeDipoleForce::Direct,
        TholeDipoleForce::Mutual
    };

    for (auto& damp : dampings) {
        for (auto method : methods) {
            for (auto pol : pols) {
                string methodName = (method == TholeDipoleForce::NoCutoff) ? "NoCutoff" : "PME";
                string polName = (pol == TholeDipoleForce::Direct) ? "Direct" : "Mutual";
                string name = string(damp.name) + "_" + methodName + "_" + polName;
                runTwoPointChargesTest(damp.type, damp.parameter, method, pol, name);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n=== Two Point Charges Tests ===" << endl;
        testTwoPointChargesNoPol();
        testTwoPointChargesCombinations();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
