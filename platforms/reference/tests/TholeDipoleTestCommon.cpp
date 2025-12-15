#include "TholeDipoleTestCommon.h"

void setupTholeDipoleAmmonia(System& system, TholeDipoleForce* tholeDipoleForce, TholeDipoleForce::NonbondedMethod nonbondedMethod,
                            TholeDipoleForce::PolarizationType polarizationType,
                            double cutoff, int inputPmeGridDimension) {

    // box
    double boxDimension                               = 0.6;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    system.setDefaultPeriodicBoxVectors(a, b, c);

    int numberOfParticles                             = 8;

    tholeDipoleForce->setNonbondedMethod(nonbondedMethod);
    tholeDipoleForce->setPolarizationType(polarizationType);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-6);
    tholeDipoleForce->setMutualInducedMaxIterations(500);
    tholeDipoleForce->setPMEParameters(1.4024714e+01, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    std::vector<double> nitrogenMolecularDipole(3);
    nitrogenMolecularDipole[0]     =   0.0;
    nitrogenMolecularDipole[1]     =   0.0;
    nitrogenMolecularDipole[2]     =   3.4e-3;

    // first N
    system.addParticle(1.4007000e+01);
    tholeDipoleForce->addParticle(-5.7960000e-1, nitrogenMolecularDipole, 1.0730000e-3, TholeDipoleForce::ThreeFold, 1, 2, 3);

    // 3 H attached to first N
    std::vector<double> hydrogenMolecularDipole(3);
    hydrogenMolecularDipole[0]     =   0.0;
    hydrogenMolecularDipole[1]     =   0.0;
    hydrogenMolecularDipole[2]     =  -4.7e-3;

    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 0, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 0, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 0, -1, -1);

    // second N
    system.addParticle(1.4007000e+01);
    tholeDipoleForce->addParticle(-5.796e-1, nitrogenMolecularDipole, 1.073e-3, TholeDipoleForce::ThreeFold, 5, 6, 7);

    // 3 H attached to second N
    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    system.addParticle(1.0080000e+00);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 4, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 4, -1, -1);
    tholeDipoleForce->addParticle(1.932e-1, hydrogenMolecularDipole, 4.96e-4, TholeDipoleForce::ZOnly, 4, -1, -1);

    // covalent maps
    std::vector< int > covalentMap;
    covalentMap.resize(0);
    covalentMap.push_back(1);
    covalentMap.push_back(2);
    covalentMap.push_back(3);
    tholeDipoleForce->setCovalentMap(0, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(0);
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(2);
    covalentMap.push_back(3);
    tholeDipoleForce->setCovalentMap(1, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(0);
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(1);
    covalentMap.push_back(3);
    tholeDipoleForce->setCovalentMap(2, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(0);
    tholeDipoleForce->setCovalentMap(3, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(1);
    covalentMap.push_back(2);
    tholeDipoleForce->setCovalentMap(3, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(5);
    covalentMap.push_back(6);
    covalentMap.push_back(7);
    tholeDipoleForce->setCovalentMap(4, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(4);
    tholeDipoleForce->setCovalentMap(5, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(6);
    covalentMap.push_back(7);
    tholeDipoleForce->setCovalentMap(5, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(4);
    tholeDipoleForce->setCovalentMap(6, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(5);
    covalentMap.push_back(7);
    tholeDipoleForce->setCovalentMap(6, TholeDipoleForce::Covalent13, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(4);
    tholeDipoleForce->setCovalentMap(7, TholeDipoleForce::Covalent12, covalentMap);

    covalentMap.resize(0);
    covalentMap.push_back(5);
    covalentMap.push_back(6);
    tholeDipoleForce->setCovalentMap(7, TholeDipoleForce::Covalent13, covalentMap);

    system.addForce(tholeDipoleForce);
}

void getForcesEnergyTholeDipoleAmmonia(Context& context, std::vector<Vec3>& forces, double& energy) {
    std::vector<Vec3> positions(context.getSystem().getNumParticles());

    positions[0]              = Vec3(  1.5927280e-01,  1.7000000e-06,   1.6491000e-03);
    positions[1]              = Vec3(  2.0805540e-01, -8.1258800e-02,   3.7282500e-02);
    positions[2]              = Vec3(  2.0843610e-01,  8.0953200e-02,   3.7462200e-02);
    positions[3]              = Vec3(  1.7280780e-01,  2.0730000e-04,  -9.8741700e-02);
    positions[4]              = Vec3( -1.6743680e-01,  1.5900000e-05,  -6.6149000e-03);
    positions[5]              = Vec3( -2.0428260e-01,  8.1071500e-02,   4.1343900e-02);
    positions[6]              = Vec3( -6.7308300e-02,  1.2800000e-05,   1.0623300e-02);
    positions[7]              = Vec3( -2.0426290e-01, -8.1231400e-02,   4.1033500e-02);

    context.setPositions(positions);
    State state                      = context.getState(State::Forces | State::Energy);
    forces                           = state.getForces();
    energy                           = state.getPotentialEnergy();
}

void compareForcesEnergy(std::string& testName, double expectedEnergy, double energy,
                        const std::vector<Vec3>& expectedForces,
                        const std::vector<Vec3>& forces, double tolerance) {
    for (unsigned int ii = 0; ii < forces.size(); ii++) {
        ASSERT_EQUAL_VEC_MOD(expectedForces[ii], forces[ii], tolerance, testName);
    }
    ASSERT_EQUAL_TOL_MOD(expectedEnergy, energy, tolerance, testName);
}

void setupAndGetForcesEnergyTholeDipoleWater(TholeDipoleForce::NonbondedMethod nonbondedMethod,
                                           TholeDipoleForce::PolarizationType polarizationType,
                                           double cutoff, int inputPmeGridDimension, std::vector<Vec3>& forces,
                                           double& energy) {

    // beginning of TholeDipole setup
    System system;

    // box dimensions
    double boxDimension                               = 1.8643;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    system.setDefaultPeriodicBoxVectors(a, b, c);

    TholeDipoleForce* tholeDipoleForce        = new TholeDipoleForce();;
    int numberOfParticles                             = 12;
    tholeDipoleForce->setNonbondedMethod(nonbondedMethod);
    tholeDipoleForce->setPolarizationType(polarizationType);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-6);
    tholeDipoleForce->setMutualInducedMaxIterations(500);
    tholeDipoleForce->setPMEParameters(5.4459052e+00, inputPmeGridDimension, inputPmeGridDimension, inputPmeGridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        system.addParticle(1.5995000e+01);
        system.addParticle(1.0080000e+00);
        system.addParticle(1.0080000e+00);
    }

    std::vector<double> oxygenMolecularDipole(3);
    oxygenMolecularDipole[0]     =   0.0;
    oxygenMolecularDipole[1]     =   0.0;
    oxygenMolecularDipole[2]     =   7.5561214e-3;

    std::vector<double> hydrogenMolecularDipole(3);
    hydrogenMolecularDipole[0]     =  -2.0420949e-3;
    hydrogenMolecularDipole[1]     =   0.0;
    hydrogenMolecularDipole[2]     =  -3.0787530e-3;

    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        tholeDipoleForce->addParticle(-5.1966000e-1, oxygenMolecularDipole, 8.3700000e-4, TholeDipoleForce::Bisector, jj+1, jj+2, -1);
        tholeDipoleForce->addParticle( 2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, TholeDipoleForce::ZThenX, jj, jj+2, -1);
        tholeDipoleForce->addParticle( 2.5983000e-1, hydrogenMolecularDipole, 4.9600000e-4, TholeDipoleForce::ZThenX, jj, jj+1, -1);
    }

    // CovalentMaps
    std::vector< int > covalentMap;
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        covalentMap.resize(0);
        covalentMap.push_back(jj+1);
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj, TholeDipoleForce::Covalent12, covalentMap);
    
        covalentMap.resize(0);
        covalentMap.push_back(jj);
        tholeDipoleForce->setCovalentMap(jj+1, TholeDipoleForce::Covalent12, covalentMap);
        tholeDipoleForce->setCovalentMap(jj+2, TholeDipoleForce::Covalent12, covalentMap);
    
        covalentMap.resize(0);
        covalentMap.push_back(jj+2);
        tholeDipoleForce->setCovalentMap(jj+1, TholeDipoleForce::Covalent13, covalentMap);
    
        covalentMap.resize(0);
        covalentMap.push_back(jj+1);
        tholeDipoleForce->setCovalentMap(jj+2, TholeDipoleForce::Covalent13, covalentMap);
    
    } 
 
    // 1-2 bonds needed
    CustomBondForce* bondForce  = new CustomBondForce("k*(d^2 - 25.5*d^3 + 379.3125*d^4); d=r-r0");
    bondForce->addPerBondParameter("r0");
    bondForce->addPerBondParameter("k");

    // addBond: particle1, particle2, length, quadraticK
    for (unsigned int jj = 0; jj < numberOfParticles; jj += 3) {
        bondForce->addBond(jj, jj+1, {0.0, 0.0});
        bondForce->addBond(jj, jj+2, {0.0, 0.0});
    }

    system.addForce(bondForce);

    std::vector<Vec3> positions(numberOfParticles);

    positions[0]              = Vec3( -8.7387270e-01,   5.3220410e-01,    7.4214000e-03);
    positions[1]              = Vec3( -9.6050090e-01,   5.1173410e-01,   -2.2202700e-02);
    positions[2]              = Vec3( -8.5985900e-01,   4.9658230e-01,    1.0283390e-01);
    positions[3]              = Vec3(  9.1767100e-02,  -7.8956650e-01,    4.3804200e-01);
    positions[4]              = Vec3(  1.2333420e-01,  -7.0267430e-01,    4.2611550e-01);
    positions[5]              = Vec3(  1.7267090e-01,  -8.2320810e-01,    4.8124750e-01);
    positions[6]              = Vec3(  8.6290110e-01,   6.2153500e-02,    4.1280850e-01);
    positions[7]              = Vec3(  8.6385200e-01,   1.2684730e-01,    3.3887060e-01);
    positions[8]              = Vec3(  9.5063550e-01,   5.3173300e-02,    4.4799160e-01);
    positions[9]              = Vec3(  5.0844930e-01,   2.8684740e-01,   -6.9293750e-01);
    positions[10]             = Vec3(  6.0459330e-01,   3.0620510e-01,   -7.0100130e-01);
    positions[11]             = Vec3(  5.0590640e-01,   1.8880920e-01,   -6.8813470e-01);

    system.addForce(tholeDipoleForce);

    LangevinIntegrator integrator(0.0, 0.1, 0.01);
    Context context(system, integrator, *platform);

    context.setPositions(positions);
    State state                      = context.getState(State::Forces | State::Energy);
    forces                           = state.getForces();
    energy                           = state.getPotentialEnergy();
}

AmoebaMultipoleForce* createEquivalentAmoebaForce(TholeDipoleForce* tholeDipoleForce) {
    // Load custom AMOEBA plugin with debug output
    Platform::loadPluginsFromDirectory("/home/aehogan2/PycharmProjects/tholedipoleplugin/amoeba_reference/build/plugins");
    
    cout << "Creating AMOEBA force..." << endl;
    AmoebaMultipoleForce* amoebaForce = new AmoebaMultipoleForce();
    cout << "AMOEBA force created successfully" << endl;
    
    // Copy method settings
    if (tholeDipoleForce->getNonbondedMethod() == TholeDipoleForce::NoCutoff) {
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::NoCutoff);
    } else if (tholeDipoleForce->getNonbondedMethod() == TholeDipoleForce::PME) {
        amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
        amoebaForce->setCutoffDistance(tholeDipoleForce->getCutoffDistance());
        
        double alpha;
        int nx, ny, nz;
        tholeDipoleForce->getPMEParameters(alpha, nx, ny, nz);
        amoebaForce->setPMEParameters(alpha, nx, ny, nz);
    }
    
    // Copy polarization settings
    switch(tholeDipoleForce->getPolarizationType()) {
        case TholeDipoleForce::Direct:
            amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
            break;
        case TholeDipoleForce::Mutual:
            amoebaForce->setPolarizationType(AmoebaMultipoleForce::Mutual);
            amoebaForce->setMutualInducedMaxIterations(
                tholeDipoleForce->getMutualInducedMaxIterations());
            amoebaForce->setMutualInducedTargetEpsilon(
                tholeDipoleForce->getMutualInducedTargetEpsilon());
            break;
        case TholeDipoleForce::Extrapolated:
            amoebaForce->setPolarizationType(AmoebaMultipoleForce::Extrapolated);
            vector<double> coeffs = tholeDipoleForce->getExtrapolationCoefficients();
            amoebaForce->setExtrapolationCoefficients(coeffs);
            break;
    }
    
    // Convert particles
    int numParticles = tholeDipoleForce->getNumParticles();
    double thole = tholeDipoleForce->getTholeDampingParameter();
    for (int i = 0; i < numParticles; i++) {
        double charge;
        vector<double> dipole;
        double polarizability;
        int axisType, multipoleAtomZ, multipoleAtomX, multipoleAtomY;

        tholeDipoleForce->getParticleParameters(i, charge, dipole, polarizability,
                                                axisType,
                                                multipoleAtomZ, multipoleAtomX, multipoleAtomY);
        
        // Create zero quadrupole (9 components: XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ)
        vector<double> quadrupole(9, 0.0);
        
        // Handle axis particle mapping - AMOEBA doesn't allow -1 for any axis
        int amoebaAxisType = axisType;
        int amoebaAtomZ = multipoleAtomZ;
        int amoebaAtomX = multipoleAtomX;
        int amoebaAtomY = multipoleAtomY;
        
        // For ZOnly axis type, if X or Y are -1, we need to handle this
        if (axisType == 4) { // ZOnly
            if (multipoleAtomX == -1) {
                // For ZOnly, find a suitable X axis particle (not self, not Z axis)
                for (int k = 0; k < numParticles; k++) {
                    if (k != i && k != multipoleAtomZ) {
                        amoebaAtomX = k;
                        break;
                    }
                }
            }
            if (multipoleAtomY == -1) {
                amoebaAtomY = -1; // AMOEBA allows -1 for Y axis in ZOnly type
            }
        }
        
        // Add to AMOEBA force with dampingFactor = polarizability^(1/6)
        double dampingFactor = (polarizability > 0) ? pow(polarizability, 1.0/6.0) : 0.0;
        cout << "Adding particle " << i << " to AMOEBA: axisType=" << amoebaAxisType
             << " atomZ=" << amoebaAtomZ << " atomX=" << amoebaAtomX << " atomY=" << amoebaAtomY << endl;
        amoebaForce->addMultipole(charge, dipole, quadrupole,
                                  amoebaAxisType, amoebaAtomZ, amoebaAtomX, amoebaAtomY,
                                  thole, dampingFactor, polarizability);
        cout << "Particle " << i << " added successfully" << endl;
    }
    
    // Copy covalent maps
    // TholeDipole uses Covalent12/13/14/15 for electrostatic scaling
    // AMOEBA also uses Covalent12/13/14/15 (same indices 0-3) for electrostatic scaling
    // and PolarizationCovalent11 for polarization scaling
    for (int i = 0; i < numParticles; i++) {
        vector<int> allCovalentAtoms;

        // Copy each covalent type directly (12, 13, 14, 15) from TholeDipole to AMOEBA
        for (int j = 0; j < TholeDipoleForce::CovalentEnd; j++) {
            vector<int> covalentAtoms;
            tholeDipoleForce->getCovalentMap(i,
                static_cast<TholeDipoleForce::CovalentType>(j), covalentAtoms);

            // Set the same covalent type in AMOEBA (indices match: 0=12, 1=13, 2=14, 3=15)
            if (!covalentAtoms.empty()) {
                amoebaForce->setCovalentMap(i,
                    static_cast<AmoebaMultipoleForce::CovalentType>(j), covalentAtoms);
            }

            // Also collect all atoms for PolarizationCovalent11
            allCovalentAtoms.insert(allCovalentAtoms.end(),
                                   covalentAtoms.begin(), covalentAtoms.end());
        }

        // Set all collected covalent atoms as PolarizationCovalent11 in AMOEBA
        if (!allCovalentAtoms.empty()) {
            amoebaForce->setCovalentMap(i,
                AmoebaMultipoleForce::PolarizationCovalent11, allCovalentAtoms);
        }
    }
    
    cout << "AMOEBA force setup completed with " << numParticles << " particles" << endl;
    return amoebaForce;
}

void compareForces(const string& testName,
                   System& tholeDipoleSystem,
                   System& amoebaSystem,
                   const vector<Vec3>& positions,
                   double energyTolerance,
                   double forceTolerance) {
    
    // Create contexts for both systems
    LangevinIntegrator integ1(0.0, 0.1, 0.01);
    LangevinIntegrator integ2(0.0, 0.1, 0.01);
    
    cout << "Creating TholeDipole context..." << endl;
    Context tholeContext(tholeDipoleSystem, integ1, *platform);
    cout << "TholeDipole context created successfully" << endl;

    // Use the same platform for AMOEBA for fair comparison
    cout << "Creating AMOEBA context..." << endl;
    Context amoebaContext(amoebaSystem, integ2, *platform);
    cout << "AMOEBA context created successfully" << endl;
    
    cout << "TholeDipole platform: " << tholeContext.getPlatform().getName() << endl;
    cout << "AMOEBA platform: " << amoebaContext.getPlatform().getName() << endl;
    
    // Set positions
    tholeContext.setPositions(positions);
    amoebaContext.setPositions(positions);
    
    // Get states
    State tholeState = tholeContext.getState(State::Forces | State::Energy);
    State amoebaState = amoebaContext.getState(State::Forces | State::Energy);
    
    // Compare energies
    double tholeEnergy = tholeState.getPotentialEnergy();
    double amoebaEnergy = amoebaState.getPotentialEnergy();
    double energyDiff = fabs(tholeEnergy - amoebaEnergy);
    
    cout << testName << " Energy Comparison:" << endl;
    cout << "  TholeDipole Energy: " << tholeEnergy << " kJ/mol" << endl;
    cout << "  AMOEBA Energy:      " << amoebaEnergy << " kJ/mol" << endl;
    cout << "  Difference:         " << energyDiff << " kJ/mol" << endl;
    
    // Get forces for later comparison  
    const vector<Vec3>& tholeForces = tholeState.getForces();
    const vector<Vec3>& amoebaForces = amoebaState.getForces();
    
    double maxForceDiff = 0.0;
    for (size_t i = 0; i < positions.size(); i++) {
        Vec3 diff = tholeForces[i] - amoebaForces[i];
        double forceDiff = sqrt(diff.dot(diff));
        maxForceDiff = max(maxForceDiff, forceDiff);
    }
    
    // Compare dipoles
    try {
        // Get TholeDipole forces to extract dipoles
        TholeDipoleForce* tholeForce = nullptr;
        AmoebaMultipoleForce* amoebaForce = nullptr;
        
        // Find the forces in the systems
        for (int i = 0; i < tholeDipoleSystem.getNumForces(); i++) {
            if (dynamic_cast<TholeDipoleForce*>(&tholeDipoleSystem.getForce(i)) != nullptr) {
                tholeForce = dynamic_cast<TholeDipoleForce*>(&tholeDipoleSystem.getForce(i));
                break;
            }
        }
        
        for (int i = 0; i < amoebaSystem.getNumForces(); i++) {
            if (dynamic_cast<AmoebaMultipoleForce*>(&amoebaSystem.getForce(i)) != nullptr) {
                amoebaForce = dynamic_cast<AmoebaMultipoleForce*>(&amoebaSystem.getForce(i));
                break;
            }
        }
        
        if (tholeForce && amoebaForce) {
            // Extract dipoles from both systems
            vector<Vec3> tholePermanentDipoles, tholeInducedDipoles, tholeTotalDipoles;
            vector<Vec3> amoebaInducedDipoles, amoebaPermanentDipoles, amoebaTotalDipoles;
            
            tholeForce->getLabFramePermanentDipoles(tholeContext, tholePermanentDipoles);
            tholeForce->getInducedDipoles(tholeContext, tholeInducedDipoles);
            tholeForce->getTotalDipoles(tholeContext, tholeTotalDipoles);
            
            amoebaForce->getInducedDipoles(amoebaContext, amoebaInducedDipoles);
            amoebaForce->getLabFramePermanentDipoles(amoebaContext, amoebaPermanentDipoles);
            amoebaForce->getTotalDipoles(amoebaContext, amoebaTotalDipoles);
            
            // Get system multipole moments from both
            vector<double> tholeSystemMoments, amoebaSystemMoments;
            tholeForce->getSystemMultipoleMoments(tholeContext, tholeSystemMoments);
            amoebaForce->getSystemMultipoleMoments(amoebaContext, amoebaSystemMoments);
            
            cout << testName << " Per-Particle Comparison:" << endl;
            
            // Compare permanent dipoles first
            double maxPermanentDipoleDiff = 0.0;
            for (size_t i = 0; i < positions.size(); i++) {
                Vec3 diff = tholePermanentDipoles[i] - amoebaPermanentDipoles[i];
                double dipoleDiff = sqrt(diff.dot(diff));
                maxPermanentDipoleDiff = max(maxPermanentDipoleDiff, dipoleDiff);
            }
            
            // Compare induced dipoles and include force comparison
            double maxInducedDipoleDiff = 0.0;
            for (size_t i = 0; i < positions.size(); i++) {
                Vec3 diff = tholeInducedDipoles[i] - amoebaInducedDipoles[i];
                double dipoleDiff = sqrt(diff.dot(diff));
                maxInducedDipoleDiff = max(maxInducedDipoleDiff, dipoleDiff);

                // Only print detailed output for first 2 particles
                if (i < 2) {
                    // Calculate force difference for this particle
                    Vec3 forceDiff = tholeForces[i] - amoebaForces[i];
                    double forceDiffMag = sqrt(forceDiff.dot(forceDiff));

                    cout << "  Particle " << i << ":" << endl;
                    cout << "    TholeDipole Force:     (" << tholeForces[i][0] << ", " << tholeForces[i][1] << ", " << tholeForces[i][2] << ") kJ/mol/nm" << endl;
                    cout << "    AMOEBA Force:          (" << amoebaForces[i][0] << ", " << amoebaForces[i][1] << ", " << amoebaForces[i][2] << ") kJ/mol/nm" << endl;
                    cout << "    Force Difference:      " << forceDiffMag << " kJ/mol/nm" << endl;
                    cout << "    TholeDipole Permanent: (" << tholePermanentDipoles[i][0] << ", " << tholePermanentDipoles[i][1] << ", " << tholePermanentDipoles[i][2] << ")" << endl;
                    cout << "    AMOEBA Permanent:      (" << amoebaPermanentDipoles[i][0] << ", " << amoebaPermanentDipoles[i][1] << ", " << amoebaPermanentDipoles[i][2] << ")" << endl;
                    cout << "    TholeDipole Induced:   (" << tholeInducedDipoles[i][0] << ", " << tholeInducedDipoles[i][1] << ", " << tholeInducedDipoles[i][2] << ")" << endl;
                    cout << "    AMOEBA Induced:        (" << amoebaInducedDipoles[i][0] << ", " << amoebaInducedDipoles[i][1] << ", " << amoebaInducedDipoles[i][2] << ")" << endl;
                    cout << "    TholeDipole Total:     (" << tholeTotalDipoles[i][0] << ", " << tholeTotalDipoles[i][1] << ", " << tholeTotalDipoles[i][2] << ")" << endl;
                    cout << "    AMOEBA Total:          (" << amoebaTotalDipoles[i][0] << ", " << amoebaTotalDipoles[i][1] << ", " << amoebaTotalDipoles[i][2] << ")" << endl;
                }
            }
            
            cout << "  Max Force Difference: " << maxForceDiff << " kJ/mol/nm" << endl;
            cout << "  Max Permanent Dipole Difference: " << maxPermanentDipoleDiff << endl;
            cout << "  Max Induced Dipole Difference: " << maxInducedDipoleDiff << endl;
            
            // Compare system multipole moments
            if (tholeSystemMoments.size() >= 4 && amoebaSystemMoments.size() >= 4) {
                cout << "  System Multipole Moments:" << endl;
                cout << "    TholeDipole - Charge: " << tholeSystemMoments[0] << ", Dipole: (" 
                     << tholeSystemMoments[1] << ", " << tholeSystemMoments[2] << ", " << tholeSystemMoments[3] << ")" << endl;
                cout << "    AMOEBA      - Charge: " << amoebaSystemMoments[0] << ", Dipole: (" 
                     << amoebaSystemMoments[1] << ", " << amoebaSystemMoments[2] << ", " << amoebaSystemMoments[3] << ")" << endl;
                
                double chargeDiff = fabs(tholeSystemMoments[0] - amoebaSystemMoments[0]);
                Vec3 tholeSysDipole(tholeSystemMoments[1], tholeSystemMoments[2], tholeSystemMoments[3]);
                Vec3 amoebaSysDipole(amoebaSystemMoments[1], amoebaSystemMoments[2], amoebaSystemMoments[3]);
                Vec3 sysDipoleDiff = tholeSysDipole - amoebaSysDipole;
                double sysDipoleDiffMag = sqrt(sysDipoleDiff.dot(sysDipoleDiff));
                
                cout << "    System Charge Difference: " << chargeDiff << endl;
                cout << "    System Dipole Difference: " << sysDipoleDiffMag << endl;
            }
        } else {
            cout << "Could not extract forces for dipole comparison" << endl;
        }
    } catch (const std::exception& e) {
        cout << "Dipole comparison failed: " << e.what() << endl;
    }
    
    // Check energy and force tolerances after dipole comparison
    cout << "Checking energy tolerance: " << energyDiff << " vs " << energyTolerance << endl;
    cout << "Checking force tolerance: " << maxForceDiff << " vs " << forceTolerance << endl;
    
    ASSERT_EQUAL_TOL(tholeEnergy, amoebaEnergy, energyTolerance);
    ASSERT(maxForceDiff < forceTolerance);
}
