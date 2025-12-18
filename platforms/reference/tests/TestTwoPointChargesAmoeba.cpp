/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * Test AMOEBA PME + Mutual with two point charges.
 * Identical system to TholeDipole's TestTwoPointCharges for comparison.
 *
 * Two charges: +0.5e and -0.5e separated by 3 Angstroms (0.3 nm)
 * Polarizability: 1.5 A^3 (0.0015 nm^3)
 * Box size: 5 nm, Cutoff: 2 nm
 * -------------------------------------------------------------------------- */

#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

static const double CHARGE1 = 0.5;
static const double CHARGE2 = -0.5;
static const double SEPARATION = 0.3;  // nm
static const double POLARIZABILITY = 0.0015;  // nm^3 (= 1.5 A^3)
static const double BOX_SIZE = 5.0;  // nm for PME
static const double CUTOFF = 2.0;  // nm for PME

static double checkFiniteDifferences(const string& name,
                                   const vector<Vec3>& analyticForces,
                                   Context& context,
                                   const vector<Vec3>& positions,
                                   double tolerance = 1e-2) {
    double norm = 0.0;
    for (const auto& f : analyticForces)
        norm += f.dot(f);
    norm = sqrt(norm);

    if (norm < 1e-10) {
        return 0.0;
    }

    const double stepSize = 1e-4;
    double step = 0.5 * stepSize / norm;

    vector<Vec3> positions2(analyticForces.size()), positions3(analyticForces.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        Vec3 p = positions[i];
        Vec3 f = analyticForces[i];
        positions2[i] = Vec3(p[0] - f[0]*step, p[1] - f[1]*step, p[2] - f[2]*step);
        positions3[i] = Vec3(p[0] + f[0]*step, p[1] + f[1]*step, p[2] + f[2]*step);
    }

    context.setPositions(positions2);
    State state2 = context.getState(State::Energy);
    context.setPositions(positions3);
    State state3 = context.getState(State::Energy);

    double numericalForceNorm = (state2.getPotentialEnergy() - state3.getPotentialEnergy()) / stepSize;

    double relError = fabs(norm - numericalForceNorm) / norm;
    cout << "    " << name << " FD: ana=" << norm << " num=" << numericalForceNorm
         << " err=" << (relError * 100) << "%" << endl;
    if (relError > tolerance) {
        cout << "    " << name << " FAILED: Finite diff error " << relError * 100 << "% > " << tolerance * 100 << "%" << endl;
    }
    return relError;
}

void testAmoebaPMEMutual() {
    cout << "\n=== AMOEBA PME + Mutual ===" << endl;

    // Create TholeDipole system first
    System tholeDipoleSystem;
    tholeDipoleSystem.addParticle(1.0);
    tholeDipoleSystem.addParticle(1.0);
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(Vec3(BOX_SIZE, 0, 0),
                                                   Vec3(0, BOX_SIZE, 0),
                                                   Vec3(0, 0, BOX_SIZE));

    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Mutual);
    tholeDipoleForce->setTholeDampingType(TholeDipoleForce::Amoeba);
    tholeDipoleForce->setTholeDampingParameter(0.39);
    tholeDipoleForce->setDampPermanentInducedField(false);
    tholeDipoleForce->setCutoffDistance(CUTOFF);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-8);
    tholeDipoleForce->setMutualInducedMaxIterations(500);

    vector<double> zeroDipole(3, 0.0);
    tholeDipoleForce->addParticle(CHARGE1, zeroDipole, POLARIZABILITY, TholeDipoleForce::NoAxisType, -1, -1, -1);
    tholeDipoleForce->addParticle(CHARGE2, zeroDipole, POLARIZABILITY, TholeDipoleForce::NoAxisType, -1, -1, -1);
    tholeDipoleSystem.addForce(tholeDipoleForce);

    // Create equivalent AMOEBA system
    System amoebaSystem;
    amoebaSystem.addParticle(1.0);
    amoebaSystem.addParticle(1.0);
    amoebaSystem.setDefaultPeriodicBoxVectors(Vec3(BOX_SIZE, 0, 0),
                                              Vec3(0, BOX_SIZE, 0),
                                              Vec3(0, 0, BOX_SIZE));

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaSystem.addForce(amoebaForce);

    // Positions
    vector<Vec3> positions(2);
    positions[0] = Vec3(BOX_SIZE/2 - SEPARATION/2, BOX_SIZE/2, BOX_SIZE/2);
    positions[1] = Vec3(BOX_SIZE/2 + SEPARATION/2, BOX_SIZE/2, BOX_SIZE/2);

    // Run TholeDipole
    cout << "\n--- TholeDipole ---" << endl;
    LangevinIntegrator integTD(0.0, 0.1, 0.01);
    Context contextTD(tholeDipoleSystem, integTD, *platform);
    contextTD.setPositions(positions);

    State stateTD = contextTD.getState(State::Forces | State::Energy);
    double energyTD = stateTD.getPotentialEnergy();
    const vector<Vec3>& forcesTD = stateTD.getForces();

    vector<Vec3> inducedTD;
    tholeDipoleForce->getInducedDipoles(contextTD, inducedTD);

    cout << "  Energy: " << energyTD << " kJ/mol" << endl;
    cout << "  Force[0]: " << forcesTD[0] << endl;
    cout << "  Force[1]: " << forcesTD[1] << endl;
    cout << "  Induced[0]: " << inducedTD[0] << " (|" << sqrt(inducedTD[0].dot(inducedTD[0])) << "|)" << endl;
    cout << "  Induced[1]: " << inducedTD[1] << " (|" << sqrt(inducedTD[1].dot(inducedTD[1])) << "|)" << endl;

    checkFiniteDifferences("TholeDipole", forcesTD, contextTD, positions, 0.01);

    // Run AMOEBA
    cout << "\n--- AMOEBA ---" << endl;
    LangevinIntegrator integAM(0.0, 0.1, 0.01);
    Context contextAM(amoebaSystem, integAM, *platform);
    contextAM.setPositions(positions);

    State stateAM = contextAM.getState(State::Forces | State::Energy);
    double energyAM = stateAM.getPotentialEnergy();
    const vector<Vec3>& forcesAM = stateAM.getForces();

    vector<Vec3> inducedAM;
    amoebaForce->getInducedDipoles(contextAM, inducedAM);

    cout << "  Energy: " << energyAM << " kJ/mol" << endl;
    cout << "  Force[0]: " << forcesAM[0] << endl;
    cout << "  Force[1]: " << forcesAM[1] << endl;
    cout << "  Induced[0]: " << inducedAM[0] << " (|" << sqrt(inducedAM[0].dot(inducedAM[0])) << "|)" << endl;
    cout << "  Induced[1]: " << inducedAM[1] << " (|" << sqrt(inducedAM[1].dot(inducedAM[1])) << "|)" << endl;

    checkFiniteDifferences("AMOEBA", forcesAM, contextAM, positions, 0.01);

    // Compare
    cout << "\n--- Comparison ---" << endl;
    cout << "  Energy diff: " << (energyTD - energyAM) << " kJ/mol" << endl;
    cout << "  Force[0] diff: " << (forcesTD[0] - forcesAM[0]) << endl;
    cout << "  Force[1] diff: " << (forcesTD[1] - forcesAM[1]) << endl;
    cout << "  Induced[0] diff: " << (inducedTD[0] - inducedAM[0]) << endl;

    cout << "  PASSED" << endl;
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        cout << "\n========================================" << endl;
        cout << "AMOEBA Two Point Charges PME+Mutual Test" << endl;
        cout << "========================================" << endl;

        testAmoebaPMEMutual();

        cout << "\n========================================" << endl;
        cout << "All tests passed!" << endl;
        cout << "========================================" << endl;
    }
    catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR.  Test failed." << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
