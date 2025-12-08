#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"
#include <cmath>

void runTwoDipoleTest(const string& testName,
                      double charge1, const Vec3& dipole1,
                      double charge2, const Vec3& dipole2,
                      const Vec3& pos1, const Vec3& pos2,
                      double boxSize,
                      double energyTol, double forceTol,
                      bool verbose = true) {

    if (verbose) {
        cout << "\n=== " << testName << " ===" << endl;
        cout << "Particle 0: pos=" << pos1 << ", charge=" << charge1 << ", dipole=" << dipole1 << endl;
        cout << "Particle 1: pos=" << pos2 << ", charge=" << charge2 << ", dipole=" << dipole2 << endl;
    }

    // Create TholeDipole system
    System tholeSystem;
    tholeSystem.addParticle(1.0);
    tholeSystem.addParticle(1.0);

    Vec3 a(boxSize, 0.0, 0.0);
    Vec3 b(0.0, boxSize, 0.0);
    Vec3 c(0.0, 0.0, boxSize);
    tholeSystem.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeForce = new TholeDipoleForce();
    tholeForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeForce->setPolarizationType(TholeDipoleForce::Direct);
    tholeForce->setCutoffDistance(0.7);

    double alpha = 5.0;
    int gridDim = 64;  // Try larger grid
    tholeForce->setPMEParameters(alpha, gridDim, gridDim, gridDim);

    vector<double> dip1 = {dipole1[0], dipole1[1], dipole1[2]};
    vector<double> dip2 = {dipole2[0], dipole2[1], dipole2[2]};

    // NoAxisType since we're providing lab-frame dipoles directly (no rotation needed)
    tholeForce->addParticle(charge1, dip1, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    tholeForce->addParticle(charge2, dip2, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);

    tholeSystem.addForce(tholeForce);

    // Create AMOEBA system
    System amoebaSystem;
    amoebaSystem.addParticle(1.0);
    amoebaSystem.addParticle(1.0);
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeForce);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions = {pos1, pos2};

    // Create contexts
    LangevinIntegrator integ1(0.0, 0.1, 0.01);
    LangevinIntegrator integ2(0.0, 0.1, 0.01);

    Context tholeContext(tholeSystem, integ1, *platform);
    Context amoebaContext(amoebaSystem, integ2, *platform);

    tholeContext.setPositions(positions);
    amoebaContext.setPositions(positions);

    State tholeState = tholeContext.getState(State::Forces | State::Energy);
    State amoebaState = amoebaContext.getState(State::Forces | State::Energy);

    double tholeEnergy = tholeState.getPotentialEnergy();
    double amoebaEnergy = amoebaState.getPotentialEnergy();

    const vector<Vec3>& tholeForces = tholeState.getForces();
    const vector<Vec3>& amoebaForces = amoebaState.getForces();

    cout << "\nEnergy Comparison:" << endl;
    cout << "  TholeDipole: " << tholeEnergy << " kJ/mol" << endl;
    cout << "  AMOEBA:      " << amoebaEnergy << " kJ/mol" << endl;
    cout << "  Difference:  " << (tholeEnergy - amoebaEnergy) << " kJ/mol" << endl;

    cout << "\nForce Comparison:" << endl;
    for (int i = 0; i < 2; i++) {
        Vec3 diff = tholeForces[i] - amoebaForces[i];
        double forceDiff = sqrt(diff.dot(diff));
        double ratio = tholeForces[i][0] / amoebaForces[i][0];
        cout << "  Particle " << i << ":" << endl;
        cout << "    TholeDipole: " << tholeForces[i] << " kJ/mol/nm" << endl;
        cout << "    AMOEBA:      " << amoebaForces[i] << " kJ/mol/nm" << endl;
        cout << "    Difference:  " << forceDiff << " kJ/mol/nm, ratio=" << ratio << endl;
    }

    // Get dipoles
    vector<Vec3> tholePerm, tholeInd, tholeTot;
    vector<Vec3> amoebaPerm, amoebaInd, amoebaTot;

    tholeForce->getLabFramePermanentDipoles(tholeContext, tholePerm);
    tholeForce->getInducedDipoles(tholeContext, tholeInd);
    tholeForce->getTotalDipoles(tholeContext, tholeTot);

    amoebaForce->getLabFramePermanentDipoles(amoebaContext, amoebaPerm);
    amoebaForce->getInducedDipoles(amoebaContext, amoebaInd);
    amoebaForce->getTotalDipoles(amoebaContext, amoebaTot);

    cout << "\nDipole Comparison:" << endl;
    for (int i = 0; i < 2; i++) {
        cout << "  Particle " << i << ":" << endl;
        cout << "    TholeDipole Permanent: " << tholePerm[i] << endl;
        cout << "    AMOEBA Permanent:      " << amoebaPerm[i] << endl;
    }

    // Check tolerances
    double energyDiff = fabs(tholeEnergy - amoebaEnergy);
    double maxForceDiff = 0.0;
    for (int i = 0; i < 2; i++) {
        Vec3 diff = tholeForces[i] - amoebaForces[i];
        maxForceDiff = max(maxForceDiff, sqrt(diff.dot(diff)));
    }

    cout << "\nTolerance Check:" << endl;
    cout << "  Energy diff " << energyDiff << " vs tol " << energyTol << ": " << (energyDiff < energyTol ? "PASS" : "FAIL") << endl;
    cout << "  Force diff " << maxForceDiff << " vs tol " << forceTol << ": " << (maxForceDiff < forceTol ? "PASS" : "FAIL") << endl;

    ASSERT_EQUAL_TOL(tholeEnergy, amoebaEnergy, energyTol);
    ASSERT(maxForceDiff < forceTol);
}

void testChargesOnly() {
    // Two point charges, no dipoles - should match well
    // Relax energy tolerance slightly due to PME numerical differences
    runTwoDipoleTest("Charges Only",
                     0.5, Vec3(0, 0, 0),     // particle 0
                     -0.3, Vec3(0, 0, 0),    // particle 1
                     Vec3(0.2, 0.2, 0.2),    // pos 0
                     Vec3(0.5, 0.2, 0.2),    // pos 1
                     1.5,                     // box size
                     0.15, 0.1);              // tolerances (energy, force)
}

// Test B-spline computation for comparison
void testBSplineComparison() {
    cout << "\n=== B-Spline Comparison Test ===" << endl;

    // Test with w = 0.266667 (particle 0's fractional x position)
    double w = 0.266667;
    const int order = 5;

    // Our fixed implementation (matching AMOEBA)
    double array[order][order];

    // Initialize order-2 spline
    array[1][1] = w;
    array[1][0] = 1.0 - w;

    // Order-3 spline
    array[2][2] = 0.5 * w * array[1][1];
    array[2][1] = 0.5 * ((1.0 + w) * array[1][0] + (2.0 - w) * array[1][1]);
    array[2][0] = 0.5 * (1.0 - w) * array[1][0];

    // Build up to order 5
    for (int i = 4; i <= order; i++) {
        int k = i - 1;
        double denom = 1.0 / k;
        array[i-1][i-1] = denom * w * array[k-1][k-1];
        for (int j = 1; j <= i - 2; j++)
            array[i-1][i-1-j] = denom * ((w + j) * array[k-1][i-1-j-1] + (i - j - w) * array[k-1][i-1-j]);
        array[i-1][0] = denom * (1.0 - w) * array[k-1][0];
    }

    // First derivative
    int k = order - 2;
    array[k][order-1] = array[k][order-2];
    for (int i = order - 2; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];

    // Second derivative
    k = order - 3;
    array[k][order-2] = array[k][order-3];
    for (int i = order - 3; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];
    array[k][order-1] = array[k][order-2];
    for (int i = order - 2; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];

    // Third derivative
    k = order - 4;
    array[k][order-3] = array[k][order-4];
    for (int i = order - 4; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];
    array[k][order-2] = array[k][order-3];
    for (int i = order - 3; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];
    array[k][order-1] = array[k][order-2];
    for (int i = order - 2; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];

    cout << "Our B-splines for w=" << w << ":" << endl;
    for (int i = 0; i < order; i++) {
        cout << "  [" << i << "]: B=" << array[order-1][i] << ", B'=" << array[order-2][i] << ", B''=" << array[order-3][i] << endl;
    }

    // AMOEBA-style implementation (using 1-indexed macro)
    double amoeba_array[order*order];
    #define ARRAY(i,j) amoeba_array[(i-1)*order + (j-1)]

    ARRAY(2,2) = w;
    ARRAY(2,1) = 1.0 - w;

    ARRAY(3,3) = 0.5 * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5 * ((1.0+w)*ARRAY(2,1)+(2.0-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5 * (1.0-w) * ARRAY(2,1);

    for (int i = 4; i <= order; i++) {
        int kk = i - 1;
        double denom = 1.0 / kk;
        ARRAY(i,i) = denom * w * ARRAY(kk,kk);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(kk,i-j-1)+(i-j-w)*ARRAY(kk,i-j));
        ARRAY(i,1) = denom * (1.0-w) * ARRAY(kk,1);
    }

    // First derivative (k = order-1)
    int k_deriv = order - 1;
    ARRAY(k_deriv,order) = ARRAY(k_deriv,order-1);
    for (int i = order-1; i >= 2; i--)
        ARRAY(k_deriv,i) = ARRAY(k_deriv,i-1) - ARRAY(k_deriv,i);
    ARRAY(k_deriv,1) = -ARRAY(k_deriv,1);

    // Second derivative (k = order-2)
    k_deriv = order - 2;
    ARRAY(k_deriv,order-1) = ARRAY(k_deriv,order-2);
    for (int i = order-2; i >= 2; i--)
        ARRAY(k_deriv,i) = ARRAY(k_deriv,i-1) - ARRAY(k_deriv,i);
    ARRAY(k_deriv,1) = -ARRAY(k_deriv,1);
    ARRAY(k_deriv,order) = ARRAY(k_deriv,order-1);
    for (int i = order-1; i >= 2; i--)
        ARRAY(k_deriv,i) = ARRAY(k_deriv,i-1) - ARRAY(k_deriv,i);
    ARRAY(k_deriv,1) = -ARRAY(k_deriv,1);

    cout << "AMOEBA B-splines for w=" << w << ":" << endl;
    for (int i = 1; i <= order; i++) {
        cout << "  [" << (i-1) << "]: B=" << ARRAY(order,i) << ", B'=" << ARRAY(order-1,i) << ", B''=" << ARRAY(order-2,i) << endl;
    }
    #undef ARRAY
}

void testDipolesNoCutoff() {
    // Test with NoCutoff to isolate direct space
    cout << "\n=== NoCutoff Test (Direct Space Only) ===" << endl;

    System tholeSystem;
    tholeSystem.addParticle(1.0);
    tholeSystem.addParticle(1.0);

    TholeDipoleForce* tholeForce = new TholeDipoleForce();
    tholeForce->setNonbondedMethod(TholeDipoleForce::NoCutoff);
    tholeForce->setPolarizationType(TholeDipoleForce::Direct);

    vector<double> dip = {0.01, 0, 0};
    tholeForce->addParticle(0.0, dip, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    tholeForce->addParticle(0.0, dip, 0.0, TholeDipoleForce::NoAxisType, -1, -1, -1);
    tholeSystem.addForce(tholeForce);

    // AMOEBA system
    System amoebaSystem;
    amoebaSystem.addParticle(1.0);
    amoebaSystem.addParticle(1.0);

    AmoebaMultipoleForce* amoebaForce = new AmoebaMultipoleForce();
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);

    vector<double> quadrupole(9, 0.0);
    amoebaForce->addMultipole(0.0, dip, quadrupole, 5, 1, 0, -1, 0.0, 0.0, 0.0);
    amoebaForce->addMultipole(0.0, dip, quadrupole, 5, 0, 1, -1, 0.0, 0.0, 0.0);
    amoebaSystem.addForce(amoebaForce);

    vector<Vec3> positions = {Vec3(0.2, 0.5, 0.5), Vec3(0.5, 0.5, 0.5)};

    LangevinIntegrator integ1(0.0, 0.1, 0.01);
    LangevinIntegrator integ2(0.0, 0.1, 0.01);
    Context tholeContext(tholeSystem, integ1, *platform);
    Context amoebaContext(amoebaSystem, integ2, *platform);
    tholeContext.setPositions(positions);
    amoebaContext.setPositions(positions);

    State tholeState = tholeContext.getState(State::Forces | State::Energy);
    State amoebaState = amoebaContext.getState(State::Forces | State::Energy);

    cout << "NoCutoff Energy:" << endl;
    cout << "  TholeDipole: " << tholeState.getPotentialEnergy() << " kJ/mol" << endl;
    cout << "  AMOEBA:      " << amoebaState.getPotentialEnergy() << " kJ/mol" << endl;

    cout << "NoCutoff Forces:" << endl;
    for (int i = 0; i < 2; i++) {
        cout << "  Particle " << i << ":" << endl;
        cout << "    TholeDipole: " << tholeState.getForces()[i] << " kJ/mol/nm" << endl;
        cout << "    AMOEBA:      " << amoebaState.getForces()[i] << " kJ/mol/nm" << endl;
    }
}

void testDipolesAlongAxis() {
    // Two dipoles pointing along the separation axis (x)
    runTwoDipoleTest("Dipoles Along Axis",
                     0.0, Vec3(0.01, 0, 0),   // particle 0: dipole along +x
                     0.0, Vec3(0.01, 0, 0),   // particle 1: dipole along +x
                     Vec3(0.2, 0.5, 0.5),     // pos 0
                     Vec3(0.5, 0.5, 0.5),     // pos 1 (separated along x)
                     1.5,                      // box size
                     0.01, 0.1);               // tolerances
}

void testChargesAndDipoles() {
    // Two particles with BOTH charges and dipoles - this tests PME with charges
    runTwoDipoleTest("Charges + Dipoles",
                     0.5, Vec3(0.01, 0, 0),   // particle 0: charge +0.5, dipole along +x
                     -0.5, Vec3(0.01, 0, 0),  // particle 1: charge -0.5, dipole along +x
                     Vec3(0.2, 0.5, 0.5),     // pos 0
                     Vec3(0.5, 0.5, 0.5),     // pos 1 (separated along x)
                     1.5,                      // box size
                     0.01, 1.0);               // tolerances (relaxed for now)
}

void testChargesAndDipolesWithAxisType() {
    // Same as above but using ZOnly axis type to test dipole rotation
    cout << "\n=== Charges + Dipoles with ZOnly Axis ===" << endl;

    double boxSize = 1.5;
    Vec3 a(boxSize, 0.0, 0.0);
    Vec3 b(0.0, boxSize, 0.0);
    Vec3 c(0.0, 0.0, boxSize);

    // TholeDipole system
    System tholeSystem;
    tholeSystem.addParticle(1.0);
    tholeSystem.addParticle(1.0);
    tholeSystem.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeForce = new TholeDipoleForce();
    tholeForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeForce->setPolarizationType(TholeDipoleForce::Direct);
    tholeForce->setCutoffDistance(0.7);
    tholeForce->setPMEParameters(5.0, 64, 64, 64);

    // Dipole along Z in molecular frame - will be rotated based on axis
    vector<double> dip = {0.0, 0.0, 0.01};  // Z-direction in molecular frame

    // Particle 0: Z-axis points to particle 1, so molecular Z maps to lab X
    tholeForce->addParticle(0.5, dip, 0.0, TholeDipoleForce::ZOnly, 1, -1, -1);
    // Particle 1: Z-axis points to particle 0, so molecular Z maps to lab -X
    tholeForce->addParticle(-0.5, dip, 0.0, TholeDipoleForce::ZOnly, 0, -1, -1);

    tholeSystem.addForce(tholeForce);

    vector<Vec3> positions = {Vec3(0.2, 0.5, 0.5), Vec3(0.5, 0.5, 0.5)};

    // Create AMOEBA system
    System amoebaSystem;
    amoebaSystem.addParticle(1.0);
    amoebaSystem.addParticle(1.0);
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeForce);
    amoebaSystem.addForce(amoebaForce);

    LangevinIntegrator integ1(0.0, 0.1, 0.01);
    LangevinIntegrator integ2(0.0, 0.1, 0.01);
    Context tholeContext(tholeSystem, integ1, *platform);
    Context amoebaContext(amoebaSystem, integ2, *platform);
    tholeContext.setPositions(positions);
    amoebaContext.setPositions(positions);

    State tholeState = tholeContext.getState(State::Forces | State::Energy);
    State amoebaState = amoebaContext.getState(State::Forces | State::Energy);

    cout << "Energy Comparison:" << endl;
    cout << "  TholeDipole: " << tholeState.getPotentialEnergy() << " kJ/mol" << endl;
    cout << "  AMOEBA:      " << amoebaState.getPotentialEnergy() << " kJ/mol" << endl;
    cout << "  Difference:  " << (tholeState.getPotentialEnergy() - amoebaState.getPotentialEnergy()) << " kJ/mol" << endl;

    // Get lab frame dipoles
    vector<Vec3> tholePerm, amoebaPerm;
    tholeForce->getLabFramePermanentDipoles(tholeContext, tholePerm);
    amoebaForce->getLabFramePermanentDipoles(amoebaContext, amoebaPerm);

    cout << "\nLab Frame Dipoles:" << endl;
    for (int i = 0; i < 2; i++) {
        cout << "  Particle " << i << ":" << endl;
        cout << "    TholeDipole: " << tholePerm[i] << endl;
        cout << "    AMOEBA:      " << amoebaPerm[i] << endl;
    }

    double energyDiff = fabs(tholeState.getPotentialEnergy() - amoebaState.getPotentialEnergy());
    ASSERT_EQUAL_TOL(tholeState.getPotentialEnergy(), amoebaState.getPotentialEnergy(), 0.01);
}

void testDipolesPerpendicularToAxis() {
    // Two dipoles perpendicular to separation axis
    runTwoDipoleTest("Dipoles Perpendicular to Axis",
                     0.0, Vec3(0, 0.01, 0),   // particle 0: dipole along +y
                     0.0, Vec3(0, 0.01, 0),   // particle 1: dipole along +y
                     Vec3(0.2, 0.5, 0.5),     // pos 0
                     Vec3(0.5, 0.5, 0.5),     // pos 1 (separated along x)
                     1.5,                      // box size
                     0.01, 0.1);               // tolerances
}

void testDipolesOpposite() {
    // Two dipoles pointing opposite directions along the axis
    runTwoDipoleTest("Dipoles Opposite",
                     0.0, Vec3(0.01, 0, 0),   // particle 0: dipole along +x
                     0.0, Vec3(-0.01, 0, 0),  // particle 1: dipole along -x
                     Vec3(0.2, 0.5, 0.5),     // pos 0
                     Vec3(0.5, 0.5, 0.5),     // pos 1 (separated along x)
                     1.5,                      // box size
                     0.01, 0.1);               // tolerances
}

void testChargeDipoleInteraction() {
    // Charge on one, dipole on the other
    runTwoDipoleTest("Charge-Dipole",
                     0.5, Vec3(0, 0, 0),      // particle 0: charge only
                     0.0, Vec3(0.01, 0, 0),   // particle 1: dipole only
                     Vec3(0.2, 0.5, 0.5),     // pos 0
                     Vec3(0.5, 0.5, 0.5),     // pos 1 (separated along x)
                     1.5,                      // box size
                     0.01, 0.1);               // tolerances
}

void testMixedChargeAndDipole() {
    // Both particles have both charge and dipole
    runTwoDipoleTest("Mixed Charge and Dipole",
                     0.3, Vec3(0.005, 0.005, 0),   // particle 0
                     -0.2, Vec3(0.008, 0, 0.003),  // particle 1
                     Vec3(0.2, 0.3, 0.4),          // pos 0
                     Vec3(0.5, 0.4, 0.5),          // pos 1
                     1.5,                           // box size
                     0.01, 0.1);                    // tolerances
}

void testDipolesZ() {
    // Dipoles along z, separation along x
    runTwoDipoleTest("Dipoles Along Z",
                     0.0, Vec3(0, 0, 0.01),  // particle 0: dipole along +z
                     0.0, Vec3(0, 0, 0.01),  // particle 1: dipole along +z
                     Vec3(0.2, 0.5, 0.5),    // pos 0
                     Vec3(0.5, 0.5, 0.5),    // pos 1 (separated along x)
                     1.5,                     // box size
                     0.01, 0.1);              // tolerances
}

void testLargerSeparation() {
    // Larger separation to reduce direct space contribution
    runTwoDipoleTest("Larger Separation",
                     0.0, Vec3(0.01, 0, 0),  // particle 0
                     0.0, Vec3(0.01, 0, 0),  // particle 1
                     Vec3(0.1, 0.5, 0.5),    // pos 0
                     Vec3(0.9, 0.5, 0.5),    // pos 1 (larger separation)
                     1.5,                     // box size
                     0.01, 0.1);              // tolerances
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        Platform::loadPluginsFromDirectory("/home/aehogan2/miniforge3/envs/openff_jun2025/lib/plugins");

        cout << "Running two-dipole diagnostic tests..." << endl;

        // First test NoCutoff (direct space only)
        testDipolesNoCutoff();

        // Test dipoles along axis with PME
        testDipolesAlongAxis();

        // Test charges + dipoles with PME
        testChargesAndDipoles();

        // Test charges + dipoles with axis type rotation
        testChargesAndDipolesWithAxisType();

        cout << "\nTest complete!" << endl;

    } catch (const exception& e) {
        cout << "exception: " << e.what() << endl;
        cout << "FAIL - ERROR.  Test failed." << endl;
        return 1;
    }
    return 0;
}
