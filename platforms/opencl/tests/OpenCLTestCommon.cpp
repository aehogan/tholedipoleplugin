#include "OpenCLTestCommon.h"
#include "OpenCLTests.h"

Platform* openclPlatform;
Platform* referencePlatform;

// Get tolerance multiplier based on OpenCL precision
static double getPrecisionMultiplier() {
    string precision = openclPlatform->getPropertyDefaultValue("OpenCLPrecision");
    if (precision == "single")
        return 5.0;   // Single precision: 5x looser tolerances
    else if (precision == "mixed")
        return 5.0;   // Mixed precision: 5x looser tolerances
    else
        return 1.0;   // Double precision: no multiplier
}

TholeDipoleForce* cloneTholeDipoleForce(const TholeDipoleForce* origForce) {
    TholeDipoleForce* force = new TholeDipoleForce();
    force->setNonbondedMethod(origForce->getNonbondedMethod());
    force->setPolarizationType(origForce->getPolarizationType());
    force->setTholeDampingType(origForce->getTholeDampingType());
    force->setTholeDampingParameter(origForce->getTholeDampingParameter());
    force->setCutoffDistance(origForce->getCutoffDistance());
    force->setMutualInducedMaxIterations(origForce->getMutualInducedMaxIterations());
    force->setMutualInducedTargetEpsilon(origForce->getMutualInducedTargetEpsilon());
    force->setEwaldErrorTolerance(origForce->getEwaldErrorTolerance());
    force->setDampPermanentInducedField(origForce->getDampPermanentInducedField());
    double alpha;
    int nx, ny, nz;
    origForce->getPMEParameters(alpha, nx, ny, nz);
    force->setPMEParameters(alpha, nx, ny, nz);
    force->setExtrapolationCoefficients(origForce->getExtrapolationCoefficients());

    for (int i = 0; i < origForce->getNumParticles(); i++) {
        double charge, polarity;
        int axisType, atomZ, atomX, atomY;
        vector<double> dipole;
        origForce->getParticleParameters(i, charge, dipole, polarity, axisType, atomZ, atomX, atomY);
        force->addParticle(charge, dipole, polarity, axisType, atomZ, atomX, atomY);

        for (int t = 0; t < 4; t++) {
            vector<int> covalent;
            origForce->getCovalentMap(i, TholeDipoleForce::CovalentType(t), covalent);
            force->setCovalentMap(i, TholeDipoleForce::CovalentType(t), covalent);
        }
    }
    return force;
}

void assertForcesAndEnergiesMatch(System& system, vector<Vec3>& positions,
                                   double energyTol, double forceTol) {
    // Adjust tolerances based on OpenCL precision
    double multiplier = getPrecisionMultiplier();
    energyTol *= multiplier;
    forceTol *= multiplier;

    System openclSystem;
    System refSystem;
    for (int i = 0; i < system.getNumParticles(); i++) {
        openclSystem.addParticle(system.getParticleMass(i));
        refSystem.addParticle(system.getParticleMass(i));
    }

    Vec3 a, b, c;
    system.getDefaultPeriodicBoxVectors(a, b, c);
    if (a[0] > 0) {
        openclSystem.setDefaultPeriodicBoxVectors(a, b, c);
        refSystem.setDefaultPeriodicBoxVectors(a, b, c);
    }

    const TholeDipoleForce* origForce = NULL;
    for (int i = 0; i < system.getNumForces(); i++) {
        origForce = dynamic_cast<const TholeDipoleForce*>(&system.getForce(i));
        if (origForce != NULL) break;
    }

    TholeDipoleForce* openclForce = cloneTholeDipoleForce(origForce);
    TholeDipoleForce* refForce = cloneTholeDipoleForce(origForce);

    openclSystem.addForce(openclForce);

    VerletIntegrator integOpenCL(1.0);
    VerletIntegrator integRef(1.0);

    // Create OpenCL context first to get calculated PME parameters
    Context openclContext(openclSystem, integOpenCL, *openclPlatform);
    openclContext.setPositions(positions);

    // For PME, sync the grid parameters so both platforms use identical settings
    if (openclForce->getNonbondedMethod() == TholeDipoleForce::PME) {
        double alpha;
        int nx, ny, nz;
        openclForce->getPMEParametersInContext(openclContext, alpha, nx, ny, nz);
        refForce->setPMEParameters(alpha, nx, ny, nz);
    }

    refSystem.addForce(refForce);
    Context refContext(refSystem, integRef, *referencePlatform);
    refContext.setPositions(positions);

    State openclState = openclContext.getState(State::Energy | State::Forces);
    State refState = refContext.getState(State::Energy | State::Forces);

    double refEnergy = refState.getPotentialEnergy();
    double openclEnergy = openclState.getPotentialEnergy();

    ASSERT_EQUAL_TOL(refEnergy, openclEnergy, energyTol);

    int numParticles = openclSystem.getNumParticles();
    for (int i = 0; i < numParticles; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_EQUAL_TOL(refState.getForces()[i][j], openclState.getForces()[i][j], forceTol);
        }
    }
}

void setupTholeDipoleAmmonia(System& system, TholeDipoleForce* tholeDipoleForce,
                            TholeDipoleForce::NonbondedMethod nonbondedMethod,
                            TholeDipoleForce::PolarizationType polarizationType,
                            double cutoff, int inputPmeGridDimension) {

    double boxDimension = 0.6;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    system.setDefaultPeriodicBoxVectors(a, b, c);

    tholeDipoleForce->setNonbondedMethod(nonbondedMethod);
    tholeDipoleForce->setPolarizationType(polarizationType);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-9);
    tholeDipoleForce->setMutualInducedMaxIterations(500);
    tholeDipoleForce->setPMEParameters(1.4024714e+01, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    vector<double> nitrogenMolecularDipole(3);
    nitrogenMolecularDipole[0] = 0.0;
    nitrogenMolecularDipole[1] = 0.0;
    nitrogenMolecularDipole[2] = 3.4e-3;

    system.addParticle(1.4007000e+01);
    tholeDipoleForce->addParticle(-5.7960000e-1, nitrogenMolecularDipole, 1.0730000e-3, TholeDipoleForce::ThreeFold, 1, 2, 3);

    vector<double> hydrogenMolecularDipole(3);
    hydrogenMolecularDipole[0] = 0.0;
    hydrogenMolecularDipole[1] = 0.0;
    hydrogenMolecularDipole[2] = -4.7e-3;

    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 0, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 0, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 0, -1, -1);

    system.addParticle(1.4007000e+01);
    tholeDipoleForce->addParticle(-5.796e-1, nitrogenMolecularDipole, 1.073e-3, TholeDipoleForce::ThreeFold, 5, 6, 7);

    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 4, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 4, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 4, -1, -1);

    vector<int> covalentMap;
    covalentMap = {1, 2, 3};
    tholeDipoleForce->setCovalentMap(0, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {2, 3};
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {1, 3};
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap = {0};
    tholeDipoleForce->setCovalentMap(3, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {1, 2};
    tholeDipoleForce->setCovalentMap(3, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap = {5, 6, 7};
    tholeDipoleForce->setCovalentMap(4, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap = {4};
    tholeDipoleForce->setCovalentMap(5, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {6, 7};
    tholeDipoleForce->setCovalentMap(5, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap = {4};
    tholeDipoleForce->setCovalentMap(6, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {5, 7};
    tholeDipoleForce->setCovalentMap(6, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap = {4};
    tholeDipoleForce->setCovalentMap(7, TholeDipoleForce::Covalent12, covalentMap);
    covalentMap = {5, 6};
    tholeDipoleForce->setCovalentMap(7, TholeDipoleForce::Covalent13, covalentMap);

    system.addForce(tholeDipoleForce);
}

vector<Vec3> getAmmoniaPositions() {
    vector<Vec3> positions(8);
    positions[0] = Vec3(1.5927280e-01, 1.7000000e-06, 1.6491000e-03);
    positions[1] = Vec3(2.0805540e-01, -8.1258800e-02, 3.7282500e-02);
    positions[2] = Vec3(2.0843610e-01, 8.0953200e-02, 3.7462200e-02);
    positions[3] = Vec3(1.7280780e-01, 2.0730000e-04, -9.8741700e-02);
    positions[4] = Vec3(-1.6743680e-01, 1.5900000e-05, -6.6149000e-03);
    positions[5] = Vec3(-2.0428260e-01, 8.1071500e-02, 4.1343900e-02);
    positions[6] = Vec3(-6.7308300e-02, 1.2800000e-05, 1.0623300e-02);
    positions[7] = Vec3(-2.0426290e-01, -8.1231400e-02, 4.1033500e-02);
    return positions;
}

void setupWaterPME(System& system, TholeDipoleForce* force,
                   TholeDipoleForce::PolarizationType polType,
                   double cutoff, int pmeGrid) {
    int numberOfParticles = 12;
    double boxDimension = 1.8643;
    system.setDefaultPeriodicBoxVectors(Vec3(boxDimension, 0, 0),
                                        Vec3(0, boxDimension, 0),
                                        Vec3(0, 0, boxDimension));

    force->setNonbondedMethod(TholeDipoleForce::PME);
    force->setPolarizationType(polType);
    force->setCutoffDistance(cutoff);
    force->setMutualInducedTargetEpsilon(1.0e-9);
    force->setMutualInducedMaxIterations(500);
    force->setPMEParameters(5.4459052e+00, pmeGrid, pmeGrid, pmeGrid);
    force->setEwaldErrorTolerance(1.0e-4);

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        system.addParticle(1.5995000e+01);
        system.addParticle(1.0080000e+00);
        system.addParticle(1.0080000e+00);
    }

    vector<double> oxygenMolecularDipole = {0.0, 0.0, 7.5561214e-3};
    vector<double> hydrogenMolecularDipole = {-2.0420949e-3, 0.0, -3.0787530e-3};

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        force->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4, 1, jj+1, jj+2, -1);
        force->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, jj, jj+2, -1);
        force->addParticle(2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, 0, jj, jj+1, -1);
    }

    vector<int> covalentMap;
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        covalentMap.clear();
        covalentMap.push_back(jj+1);
        covalentMap.push_back(jj+2);
        force->setCovalentMap(jj, TholeDipoleForce::Covalent12, covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj);
        force->setCovalentMap(jj+1, TholeDipoleForce::Covalent12, covalentMap);
        force->setCovalentMap(jj+2, TholeDipoleForce::Covalent12, covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+2);
        force->setCovalentMap(jj+1, TholeDipoleForce::Covalent13, covalentMap);

        covalentMap.clear();
        covalentMap.push_back(jj+1);
        force->setCovalentMap(jj+2, TholeDipoleForce::Covalent13, covalentMap);
    }

    system.addForce(force);
}

vector<Vec3> getWaterPMEPositions() {
    vector<Vec3> positions(12);
    positions[0]  = Vec3(-8.7387270e-01,  5.3220410e-01,  7.4214000e-03);
    positions[1]  = Vec3(-9.6050090e-01,  5.1173410e-01, -2.2202700e-02);
    positions[2]  = Vec3(-8.5985900e-01,  4.9658230e-01,  1.0283390e-01);
    positions[3]  = Vec3( 9.1767100e-02, -7.8956650e-01,  4.3804200e-01);
    positions[4]  = Vec3( 1.2333420e-01, -7.0267430e-01,  4.2611550e-01);
    positions[5]  = Vec3( 1.7267090e-01, -8.2320810e-01,  4.8124750e-01);
    positions[6]  = Vec3( 8.6290110e-01,  6.2153500e-02,  4.1280850e-01);
    positions[7]  = Vec3( 8.6385200e-01,  1.2684730e-01,  3.3887060e-01);
    positions[8]  = Vec3( 9.5063550e-01,  5.3173300e-02,  4.4799160e-01);
    positions[9]  = Vec3( 5.0844930e-01,  2.8684740e-01, -6.9293750e-01);
    positions[10] = Vec3( 6.0459330e-01,  3.0620510e-01, -7.0100130e-01);
    positions[11] = Vec3( 5.0590640e-01,  1.8880920e-01, -6.8813470e-01);
    return positions;
}
