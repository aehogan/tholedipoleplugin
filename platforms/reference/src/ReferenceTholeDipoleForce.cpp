#include "ReferenceTholeDipoleForce.h"
#include "openmm/OpenMMException.h"

using namespace TholeDipolePlugin;
using namespace OpenMM;
using std::vector;

ReferenceTholeDipoleForce::ReferenceTholeDipoleForce() : _nonbondedMethod(NoCutoff) {
    initialize();
}

ReferenceTholeDipoleForce::ReferenceTholeDipoleForce(NonbondedMethod nonbondedMethod) : _nonbondedMethod(nonbondedMethod) {
    initialize();
}

ReferenceTholeDipoleForce::~ReferenceTholeDipoleForce() {
}

void ReferenceTholeDipoleForce::initialize() {
    _electric = 1.0;
    _dielectric = 1.0;
    _mutualInducedDipoleTargetEpsilon = 1.0e-03;
    _maximumMutualInducedDipoleIterations = 60;
    _mutualInducedDipoleEpsilon = 1.0e+50;
    _mutualInducedDipoleConverged = 0;
    _mutualInducedDipoleIterations = 0;
    _debye = 0.4803;

    // Initialize scale factors
    _scaleMaps.resize(LAST_SCALE_TYPE_INDEX);
    _maxScaleIndex.resize(LAST_SCALE_TYPE_INDEX);
    for (int i = 0; i < LAST_SCALE_TYPE_INDEX; i++) {
        _maxScaleIndex[i] = 5;
    }

    // Set default scale factors
    _mScale[0] = 0.0; // 1-1 excluded
    _mScale[1] = 0.0; // 1-2 excluded
    _mScale[2] = 0.0; // 1-3 excluded
    _mScale[3] = 0.5; // 1-4 scaled
    _mScale[4] = 1.0; // 1-5+ full

    _iScale[0] = 0.0; // 1-1 excluded
    _iScale[1] = 1.0; // 1-2+ full
    _iScale[2] = 1.0;
    _iScale[3] = 1.0;
    _iScale[4] = 1.0;

    _polarizationType = Direct;
}

ReferenceTholeDipoleForce::NonbondedMethod ReferenceTholeDipoleForce::getNonbondedMethod() const {
    return _nonbondedMethod;
}

void ReferenceTholeDipoleForce::setNonbondedMethod(NonbondedMethod nonbondedMethod) {
    _nonbondedMethod = nonbondedMethod;
}

ReferenceTholeDipoleForce::PolarizationType ReferenceTholeDipoleForce::getPolarizationType() const {
    return _polarizationType;
}

void ReferenceTholeDipoleForce::setPolarizationType(PolarizationType polarizationType) {
    _polarizationType = polarizationType;
}

int ReferenceTholeDipoleForce::getMutualInducedDipoleConverged() const {
    return _mutualInducedDipoleConverged;
}

int ReferenceTholeDipoleForce::getMutualInducedDipoleIterations() const {
    return _mutualInducedDipoleIterations;
}

double ReferenceTholeDipoleForce::getMutualInducedDipoleEpsilon() const {
    return _mutualInducedDipoleEpsilon;
}

void ReferenceTholeDipoleForce::setExtrapolationCoefficients(const std::vector<double> &coefficients) {
    _extrapolationCoefficients = coefficients;
}

void ReferenceTholeDipoleForce::setMutualInducedDipoleTargetEpsilon(double targetEpsilon) {
    _mutualInducedDipoleTargetEpsilon = targetEpsilon;
}

double ReferenceTholeDipoleForce::getMutualInducedDipoleTargetEpsilon() const {
    return _mutualInducedDipoleTargetEpsilon;
}

void ReferenceTholeDipoleForce::setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations) {
    _maximumMutualInducedDipoleIterations = maximumMutualInducedDipoleIterations;
}

int ReferenceTholeDipoleForce::getMaximumMutualInducedDipoleIterations() const {
    return _maximumMutualInducedDipoleIterations;
}

double ReferenceTholeDipoleForce::calculateForceAndEnergy(const vector<Vec3>& particlePositions,
                                                          const vector<double>& charges,
                                                          const vector<double>& dipoles,
                                                          const vector<double>& polarizabilities,
                                                          const vector<double>& tholeDampingFactors,
                                                          const vector<int>& axisTypes,
                                                          const vector<int>& multipoleAtomZs,
                                                          const vector<int>& multipoleAtomXs,
                                                          const vector<int>& multipoleAtomYs,
                                                          const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                          vector<Vec3>& forces) {
    // Setup, including calculating induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities, tholeDampingFactors,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);

    // Calculate electrostatic interactions including torques
    vector<Vec3> torques;
    initializeVec3Vector(torques);
    double energy = calculateElectrostatic(particleData, torques, forces);

    // Debug accumulated torques before mapping to forces
    for (unsigned int i = 0; i < std::min((size_t)6, particleData.size()); i++) {
        if (torques[i][0] != 0.0 || torques[i][1] != 0.0 || torques[i][2] != 0.0) {
            printf("DEBUG Input Torque: particle %d = (%.6f, %.6f, %.6f)\n", 
                   i, torques[i][0], torques[i][1], torques[i][2]);
        }
    }

    // Map torques to forces
    mapTorqueToForce(particleData, multipoleAtomXs, multipoleAtomYs, multipoleAtomZs, 
                     axisTypes, torques, forces);

    return energy;
}

double ReferenceTholeDipoleForce::calculateElectrostaticPairIxn(
    const TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData& particleK,
    double mScale,
    double iScale,
    vector<Vec3>& forces,
    vector<Vec3>& torques) const {
    
    unsigned int iIndex = particleI.particleIndex;
    unsigned int kIndex = particleK.particleIndex;
    
    Vec3 deltaR = particleK.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);
    double r = sqrt(r2);
    
    // Build rotation matrix to transform dipoles to QI frame
    double qiRotationMatrix[3][3];
    formQIRotationMatrix(particleI.position, particleK.position, deltaR, r, qiRotationMatrix);
    
    // Debug rotation matrix for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG Rotation Matrix: particles %d-%d, r=%.6f\n", iIndex, kIndex, r);
        printf("  deltaR: [%.6f, %.6f, %.6f]\n", deltaR[0], deltaR[1], deltaR[2]);
        printf("  qiRotationMatrix:\n");
        for (int i = 0; i < 3; i++) {
            printf("    [%.6f, %.6f, %.6f]\n", qiRotationMatrix[i][0], qiRotationMatrix[i][1], qiRotationMatrix[i][2]);
        }
    }
    
    // Force rotation matrix transforms QI forces back to lab frame
    double forceRotationMatrix[3][3];
    forceRotationMatrix[0][0] = qiRotationMatrix[1][1];
    forceRotationMatrix[0][1] = qiRotationMatrix[2][1];
    forceRotationMatrix[0][2] = qiRotationMatrix[0][1];
    forceRotationMatrix[1][0] = qiRotationMatrix[1][2];
    forceRotationMatrix[1][1] = qiRotationMatrix[2][2];
    forceRotationMatrix[1][2] = qiRotationMatrix[0][2];
    forceRotationMatrix[2][0] = qiRotationMatrix[1][0];
    forceRotationMatrix[2][1] = qiRotationMatrix[2][0];
    forceRotationMatrix[2][2] = qiRotationMatrix[0][0];
    
    // Rotate induced dipoles to QI frame
    double qiUindI[3], qiUindJ[3];
    
    // Debug lab frame induced dipoles for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG Lab Frame Induced Dipoles: particles %d-%d\n", iIndex, kIndex);
        printf("  _inducedDipole[%d]: [%.6f, %.6f, %.6f]\n", iIndex, 
               _inducedDipole[iIndex][0], _inducedDipole[iIndex][1], _inducedDipole[iIndex][2]);
        printf("  _inducedDipole[%d]: [%.6f, %.6f, %.6f]\n", kIndex,
               _inducedDipole[kIndex][0], _inducedDipole[kIndex][1], _inducedDipole[kIndex][2]);
    }
    
    for (int i = 0; i < 3; i++) {
        qiUindI[i] = 0.0;
        qiUindJ[i] = 0.0;
        for (int j = 0; j < 3; j++) {
            qiUindI[i] += qiRotationMatrix[i][j] * _inducedDipole[iIndex][j];
            qiUindJ[i] += qiRotationMatrix[i][j] * _inducedDipole[kIndex][j];
        }
    }
    
    // Debug QI frame induced dipoles for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG QI Frame Induced Dipoles: particles %d-%d\n", iIndex, kIndex);
        printf("  qiUindI: [%.6f, %.6f, %.6f]\n", qiUindI[0], qiUindI[1], qiUindI[2]);
        printf("  qiUindJ: [%.6f, %.6f, %.6f]\n", qiUindJ[0], qiUindJ[1], qiUindJ[2]);
    }
    
    // QI frame multipoles for atoms I and J
    double qiQI[4], qiQJ[4];
    qiQI[0] = particleI.charge;
    qiQJ[0] = particleK.charge;
    
    // Debug lab frame dipoles for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG Lab Frame Dipoles: particles %d-%d\n", iIndex, kIndex);
        printf("  particleI.dipole: [%.6f, %.6f, %.6f]\n", particleI.dipole[0], particleI.dipole[1], particleI.dipole[2]);
        printf("  particleK.dipole: [%.6f, %.6f, %.6f]\n", particleK.dipole[0], particleK.dipole[1], particleK.dipole[2]);
    }
    
    // Rotate permanent dipoles to QI frame
    for (int i = 0; i < 3; i++) {
        qiQI[i+1] = 0.0;
        qiQJ[i+1] = 0.0;
        for (int j = 0; j < 3; j++) {
            qiQI[i+1] += qiRotationMatrix[i][j] * particleI.dipole[j];
            qiQJ[i+1] += qiRotationMatrix[i][j] * particleK.dipole[j];
        }
    }
    
    // Debug QI frame multipoles for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG QI Frame Multipoles: particles %d-%d\n", iIndex, kIndex);
        printf("  qiQI: [%.6f, %.6f, %.6f, %.6f] (charge, dipole_x, dipole_y, dipole_z)\n", 
               qiQI[0], qiQI[1], qiQI[2], qiQI[3]);
        printf("  qiQJ: [%.6f, %.6f, %.6f, %.6f] (charge, dipole_x, dipole_y, dipole_z)\n", 
               qiQJ[0], qiQJ[1], qiQJ[2], qiQJ[3]);
    }
    
    // Torque intermediates for permanent dipoles
    double qiQIX[4] = {0.0, qiQI[3], 0.0, -qiQI[1]};
    double qiQIY[4] = {0.0, -qiQI[2], qiQI[1], 0.0};
    double qiQIZ[4] = {0.0, 0.0, -qiQI[3], qiQI[2]};
    double qiQJX[4] = {0.0, qiQJ[3], 0.0, -qiQJ[1]};
    double qiQJY[4] = {0.0, -qiQJ[2], qiQJ[1], 0.0};
    double qiQJZ[4] = {0.0, 0.0, -qiQJ[3], qiQJ[2]};

    // Debug multipole derivatives for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG Multipole Derivatives: particles %d-%d\n", iIndex, kIndex);
        printf("  qiQI input: [%.6f, %.6f, %.6f, %.6f]\n", qiQI[0], qiQI[1], qiQI[2], qiQI[3]);
        printf("  qiQIX calculated: [%.6f, %.6f, %.6f, %.6f] = [0, qiQI[3], 0, -qiQI[1]]\n", 
               qiQIX[0], qiQIX[1], qiQIX[2], qiQIX[3]);
        printf("  Note: qiQIX[1] = qiQI[3] = %.6f (Z-dipole becomes X-derivative of Y-dipole)\n", qiQI[3]);
    }
    
    // Get Thole-damped interaction tensors
    vector<double> rInvVec(4);
    double rInv = 1.0 / r;
    double prefac = _electric / _dielectric;
    rInvVec[1] = prefac * rInv;
    for (int i = 2; i < 4; i++) {
        rInvVec[i] = rInvVec[i-1] * rInv;
    }
    
    // Thole damping parameters
    double dmp = particleI.tholeDamping * particleK.tholeDamping;
    double a = particleI.tholeDamping < particleK.tholeDamping ? 
               particleI.tholeDamping : particleK.tholeDamping;
    double u = r / dmp;
    double au3 = fabs(dmp) > 1.0e-5 ? a * u * u * u : 0.0;
    double expau3 = fabs(dmp) > 1.0e-5 ? exp(-au3) : 0.0;
    
    // Thole damping factors for energies
    double thole_c = 1.0 - expau3;
    double thole_d0 = 1.0 - expau3 * (1.0 + 1.5 * au3);
    double thole_d1 = 1.0 - expau3;
    
    // Thole damping factors for derivatives
    double dthole_c = 1.0 - expau3 * (1.0 + 1.5 * au3);
    double dthole_d0 = 1.0 - expau3 * (1.0 + au3 + 1.5 * au3 * au3);
    double dthole_d1 = 1.0 - expau3 * (1.0 + au3);
    
    // Field derivatives at I due to J and vice versa
    double Vij[4], Vji[4], VijR[4], VjiR[4];
    double Vijp[3], Vijd[3], Vjip[3], Vjid[3];
    
    // Initialize arrays
    for (int i = 0; i < 4; i++) {
        Vij[i] = 0.0; Vji[i] = 0.0; VijR[i] = 0.0; VjiR[i] = 0.0;
    }
    for (int i = 0; i < 3; i++) {
        Vijp[i] = 0.0; Vijd[i] = 0.0; Vjip[i] = 0.0; Vjid[i] = 0.0;
    }
    
    // C-C interaction (m=0)
    double ePermCoef = rInvVec[1] * mScale;
    double dPermCoef = -0.5 * mScale * rInvVec[2];
    Vij[0] = ePermCoef * qiQJ[0];
    Vji[0] = ePermCoef * qiQI[0];
    VijR[0] = dPermCoef * qiQJ[0];
    VjiR[0] = dPermCoef * qiQI[0];
    
    // C-D and C-Uind interactions (m=0)
    ePermCoef = rInvVec[2] * mScale;
    double eUIndCoef = rInvVec[2] * iScale * thole_c;
    dPermCoef = -rInvVec[3] * mScale;
    double dUIndCoef = -2.0 * rInvVec[3] * iScale * dthole_c;
    
    Vij[0] += -(ePermCoef * qiQJ[1] + eUIndCoef * qiUindJ[0]);
    Vji[1] = -(ePermCoef * qiQI[0]);
    VijR[0] += -(dPermCoef * qiQJ[1] + dUIndCoef * qiUindJ[0]);
    VjiR[1] = -(dPermCoef * qiQI[0]);
    Vjid[0] = -(eUIndCoef * qiQI[0]);
    
    // D-C and Uind-C interactions (m=0)
    Vij[1] = ePermCoef * qiQJ[0];
    
    // Debug Vij[1] calculation for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG Vij[1] Calc: particles %d-%d\n", iIndex, kIndex);
        printf("  Vij[1] = ePermCoef * qiQJ[0] = %.6f * %.6f = %.6f\n", 
               ePermCoef, qiQJ[0], Vij[1]);
        printf("  ePermCoef = rInvVec[2] * mScale = %.6f * %.6f = %.6f\n", 
               rInvVec[2], mScale, ePermCoef);
        printf("  qiQJ[0] = %.6f (charge of particle J)\n", qiQJ[0]);
    }
    
    Vji[0] += ePermCoef * qiQI[1] + eUIndCoef * qiUindI[0];
    VijR[1] = dPermCoef * qiQJ[0];
    VjiR[0] += dPermCoef * qiQI[1] + dUIndCoef * qiUindI[0];
    Vijd[0] = eUIndCoef * qiQJ[0];
    
    // D-D and D-Uind interactions (m=0)
    ePermCoef = -2.0 * rInvVec[3] * mScale;
    eUIndCoef = -2.0 * rInvVec[3] * iScale * thole_d0;
    dPermCoef = 3.0 * rInvVec[4] * mScale;
    dUIndCoef = 6.0 * rInvVec[4] * iScale * dthole_d0;
    
    Vij[1] += ePermCoef * qiQJ[1] + eUIndCoef * qiUindJ[0];
    Vji[1] += ePermCoef * qiQI[1] + eUIndCoef * qiUindI[0];
    VijR[1] += dPermCoef * qiQJ[1] + dUIndCoef * qiUindJ[0];
    VjiR[1] += dPermCoef * qiQI[1] + dUIndCoef * qiUindI[0];
    Vijd[0] += eUIndCoef * qiQJ[1];
    Vjid[0] += eUIndCoef * qiQI[1];
    
    // D-D and D-Uind interactions (m=1)
    ePermCoef = rInvVec[3] * mScale;
    eUIndCoef = rInvVec[3] * iScale * thole_d1;
    dPermCoef = -1.5 * rInvVec[4] * mScale;
    dUIndCoef = -3.0 * rInvVec[4] * iScale * dthole_d1;
    
    Vij[2] = ePermCoef * qiQJ[2] + eUIndCoef * qiUindJ[1];
    Vji[2] = ePermCoef * qiQI[2] + eUIndCoef * qiUindI[1];
    VijR[2] = dPermCoef * qiQJ[2] + dUIndCoef * qiUindJ[1];
    VjiR[2] = dPermCoef * qiQI[2] + dUIndCoef * qiUindI[1];
    Vijd[1] = eUIndCoef * qiQJ[2];
    Vjid[1] = eUIndCoef * qiQI[2];
    
    Vij[3] = ePermCoef * qiQJ[3] + eUIndCoef * qiUindJ[2];
    Vji[3] = ePermCoef * qiQI[3] + eUIndCoef * qiUindI[2];
    VijR[3] = dPermCoef * qiQJ[3] + dUIndCoef * qiUindJ[2];
    VjiR[3] = dPermCoef * qiQI[3] + dUIndCoef * qiUindI[2];
    Vijd[2] = eUIndCoef * qiQJ[3];
    Vjid[2] = eUIndCoef * qiQI[3];
    
    // Calculate energy, forces and torques
    double energy = 0.5 * (qiQI[0] * Vij[0] + qiQJ[0] * Vji[0]);
    double fIZ = qiQI[0] * VijR[0];
    double fJZ = qiQJ[0] * VjiR[0];
    double EIX = 0.0, EIY = 0.0, EIZ = 0.0;
    double EJX = 0.0, EJY = 0.0, EJZ = 0.0;
    
    for (int i = 1; i < 4; i++) {
        energy += 0.5 * (qiQI[i] * Vij[i] + qiQJ[i] * Vji[i]);
        fIZ += qiQI[i] * VijR[i];
        fJZ += qiQJ[i] * VjiR[i];
        EIX += qiQIX[i] * Vij[i];
        EIY += qiQIY[i] * Vij[i];
        EIZ += qiQIZ[i] * Vij[i];
        EJX += qiQJX[i] * Vji[i];
        EJY += qiQJY[i] * Vji[i];
        EJZ += qiQJZ[i] * Vji[i];
    }
    
    // Induced dipole torques
    double iEIX = qiUindI[2] * Vijd[0] - qiUindI[0] * Vijd[2];
    double iEJX = qiUindJ[2] * Vjid[0] - qiUindJ[0] * Vjid[2];
    double iEIY = qiUindI[0] * Vijd[1] - qiUindI[1] * Vijd[0];
    double iEJY = qiUindJ[0] * Vjid[1] - qiUindJ[1] * Vjid[0];
    
    // Add Uind-Uind interactions for mutual polarization
    if (_polarizationType == Mutual) {
        // Uind-Uind (m=0)
        double eCoef = -4.0 * rInvVec[3] * iScale * thole_d0;
        double dCoef = 6.0 * rInvVec[4] * iScale * dthole_d0;
        iEIX += eCoef * qiUindI[2] * qiUindJ[0];
        iEJX += eCoef * qiUindJ[2] * qiUindI[0];
        iEIY -= eCoef * qiUindI[1] * qiUindJ[0];
        iEJY -= eCoef * qiUindJ[1] * qiUindI[0];
        fIZ += dCoef * qiUindI[0] * qiUindJ[0];
        fJZ += dCoef * qiUindJ[0] * qiUindI[0];
        
        // Uind-Uind (m=1)
        eCoef = 2.0 * rInvVec[3] * iScale * thole_d1;
        dCoef = -3.0 * rInvVec[4] * iScale * dthole_d1;
        iEIX -= eCoef * qiUindI[0] * qiUindJ[2];
        iEJX -= eCoef * qiUindJ[0] * qiUindI[2];
        iEIY += eCoef * qiUindI[0] * qiUindJ[1];
        iEJY += eCoef * qiUindJ[0] * qiUindI[1];
        fIZ += dCoef * (qiUindI[1] * qiUindJ[1] + qiUindI[2] * qiUindJ[2]);
        fJZ += dCoef * (qiUindJ[1] * qiUindI[1] + qiUindJ[2] * qiUindI[2]);
    }
    
    // QI frame forces and torques
    double qiForce[3] = {rInv * (EIY + EJY + iEIY + iEJY), 
                         -rInv * (EIX + EJX + iEIX + iEJX), 
                         -(fJZ + fIZ)};
    double qiTorqueI[3] = {-EIX, -EIY, -EIZ};
    double qiTorqueJ[3] = {-EJX, -EJY, -EJZ};
    
    // Debug torque calculation for small systems
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG Torque Calc: particles %d-%d\n", iIndex, kIndex);
        printf("  Lab positions: I=(%.6f,%.6f,%.6f), J=(%.6f,%.6f,%.6f)\n", 
               particleI.position[0], particleI.position[1], particleI.position[2],
               particleK.position[0], particleK.position[1], particleK.position[2]);
        printf("  Lab dipoles: I=(%.6f,%.6f,%.6f), J=(%.6f,%.6f,%.6f)\n",
               particleI.dipole[0], particleI.dipole[1], particleI.dipole[2],
               particleK.dipole[0], particleK.dipole[1], particleK.dipole[2]);
        printf("  deltaR=(%.6f,%.6f,%.6f), r=%.6f\n", deltaR[0], deltaR[1], deltaR[2], r);
        printf("  EIX=%.6f, EIY=%.6f, EIZ=%.6f\n", EIX, EIY, EIZ);
        printf("  EJX=%.6f, EJY=%.6f, EJZ=%.6f\n", EJX, EJY, EJZ);
        printf("  qiQI: [%.6f, %.6f, %.6f, %.6f]\n", qiQI[0], qiQI[1], qiQI[2], qiQI[3]);
        printf("  qiQJ: [%.6f, %.6f, %.6f, %.6f]\n", qiQJ[0], qiQJ[1], qiQJ[2], qiQJ[3]);
        printf("  qiQIX: [%.6f, %.6f, %.6f, %.6f]\n", qiQIX[0], qiQIX[1], qiQIX[2], qiQIX[3]);
        printf("  qiQIY: [%.6f, %.6f, %.6f, %.6f]\n", qiQIY[0], qiQIY[1], qiQIY[2], qiQIY[3]);
        printf("  qiQIZ: [%.6f, %.6f, %.6f, %.6f]\n", qiQIZ[0], qiQIZ[1], qiQIZ[2], qiQIZ[3]);
        printf("  Vij: [%.6f, %.6f, %.6f, %.6f]\n", Vij[0], Vij[1], Vij[2], Vij[3]);
        printf("  Vji: [%.6f, %.6f, %.6f, %.6f]\n", Vji[0], Vji[1], Vji[2], Vji[3]);
        printf("  QI Rotation Matrix:\n");
        for (int row = 0; row < 3; row++) {
            printf("    [%.6f, %.6f, %.6f]\n", forceRotationMatrix[row][0], forceRotationMatrix[row][1], forceRotationMatrix[row][2]);
        }
    }
    
    // Debug QI frame torques before rotation
    if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
        printf("DEBUG QI Frame Torques: particles %d-%d\n", iIndex, kIndex);
        printf("  qiTorqueI (QI): (%.6f, %.6f, %.6f)\n", qiTorqueI[0], qiTorqueI[1], qiTorqueI[2]);
        printf("  qiTorqueJ (QI): (%.6f, %.6f, %.6f)\n", qiTorqueJ[0], qiTorqueJ[1], qiTorqueJ[2]);
    }

    // Rotate forces and torques back to lab frame
    for (int i = 0; i < 3; i++) {
        double forceVal = 0.0;
        double torqueIVal = 0.0;
        double torqueJVal = 0.0;
        for (int j = 0; j < 3; j++) {
            forceVal += forceRotationMatrix[i][j] * qiForce[j];
            torqueIVal += forceRotationMatrix[i][j] * qiTorqueI[j];
            torqueJVal += forceRotationMatrix[i][j] * qiTorqueJ[j];
        }
        
        // Debug output for all small systems (first few particles)
        if ((iIndex <= 2 || kIndex <= 2) && (iIndex <= 5 && kIndex <= 5)) {
            printf("DEBUG Electrostatic: particles %d-%d, component %d\n", iIndex, kIndex, i);
            printf("  qiTorqueI: %.6f, qiTorqueJ: %.6f\n", qiTorqueI[i], qiTorqueJ[i]);
            printf("  torqueIVal: %.6f, torqueJVal: %.6f\n", torqueIVal, torqueJVal);
        }
        
        torques[iIndex][i] += torqueIVal;
        torques[kIndex][i] += torqueJVal;
        forces[iIndex][i] -= forceVal;
        forces[kIndex][i] += forceVal;
    }
    
    return energy;
}

double ReferenceTholeDipoleForce::calculateElectrostatic(
    const vector<TholeDipoleParticleData>& particleData,
    vector<Vec3>& torques,
    vector<Vec3>& forces) {
    
    double energy = 0.0;
    
    // Calculate pairwise interactions
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            double mScale = 1.0;
            double iScale = 1.0;
            
            // Get scaling factors if within cutoff
            if (j <= _maxScaleIndex[i]) {
                mScale = getScaleFactor(i, j, M_SCALE);
                iScale = getScaleFactor(i, j, I_SCALE);
            }
            
            energy += calculateElectrostaticPairIxn(particleData[i], particleData[j],
                                                    mScale, iScale, forces, torques);
        }
    }
    
    return energy;
}

void ReferenceTholeDipoleForce::loadParticleData(const vector<Vec3>& particlePositions,
                                                 const vector<double>& charges,
                                                 const vector<double>& dipoles,
                                                 const vector<double>& polarizabilities,
                                                 const vector<double>& tholeDampingFactors,
                                                 vector<TholeDipoleParticleData>& particleData) const {
    particleData.resize(_numParticles);
    for (unsigned int i = 0; i < _numParticles; i++) {
        particleData[i].particleIndex = i;
        particleData[i].position = particlePositions[i];
        particleData[i].charge = charges[i];
        
        // Load dipole components
        particleData[i].dipole = Vec3(dipoles[3*i], dipoles[3*i+1], dipoles[3*i+2]);
        
        // Initialize induced dipole to zero
        particleData[i].inducedDipole = Vec3(0.0, 0.0, 0.0);
        
        particleData[i].polarizability = polarizabilities[i];
        particleData[i].tholeDamping = tholeDampingFactors[i];
    }
}

void ReferenceTholeDipoleForce::zeroFixedDipoleFields() {
    _fixedDipoleField.resize(_numParticles);
    for (unsigned int i = 0; i < _numParticles; i++) {
        _fixedDipoleField[i] = Vec3(0.0, 0.0, 0.0);
    }
}

void ReferenceTholeDipoleForce::initializeVec3Vector(vector<Vec3>& vec3Vector) const {
    vec3Vector.resize(_numParticles);
    for (unsigned int i = 0; i < _numParticles; i++) {
        vec3Vector[i] = Vec3(0.0, 0.0, 0.0);
    }
}

void ReferenceTholeDipoleForce::checkChiralCenterAtParticle(TholeDipoleParticleData& particleI, 
                                                            int axisType,
                                                            const TholeDipoleParticleData& particleZ, 
                                                            const TholeDipoleParticleData& particleX,
                                                            const TholeDipoleParticleData& particleY) const {
    if (axisType != TholeDipoleForce::ZThenX || particleY.particleIndex == -1) {
        return;
    }

    Vec3 deltaAD = particleI.position - particleY.position;
    Vec3 deltaBD = particleZ.position - particleY.position;
    Vec3 deltaCD = particleX.position - particleY.position;

    Vec3 deltaC = deltaBD.cross(deltaCD);
    double volume = deltaC.dot(deltaAD);

    if (volume < 0.0) {
        // Flip the y-component of the dipole
        particleI.dipole[1] *= -1.0;
    }
}

void ReferenceTholeDipoleForce::checkChiral(vector<TholeDipoleParticleData>& particleData,
                                            const vector<int>& multipoleAtomXs,
                                            const vector<int>& multipoleAtomYs,
                                            const vector<int>& multipoleAtomZs,
                                            const vector<int>& axisTypes) const {
    for (unsigned int i = 0; i < _numParticles; i++) {
        if (multipoleAtomYs[i] > -1) {
            checkChiralCenterAtParticle(particleData[i], axisTypes[i],
                                        particleData[multipoleAtomZs[i]],
                                        particleData[multipoleAtomXs[i]],
                                        particleData[multipoleAtomYs[i]]);
        }
    }
}

double ReferenceTholeDipoleForce::normalizeVec3(Vec3& vector) const {
    double norm = sqrt(vector.dot(vector));
    if (norm > 0.0) {
        vector *= (1.0/norm);
    }
    return norm;
}

void ReferenceTholeDipoleForce::applyRotationMatrixToParticle(
    TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData* particleZ,
    const TholeDipoleParticleData* particleX,
    const TholeDipoleParticleData* particleY,
    int axisType) const {
    
    // Get the z-axis vector
    Vec3 vectorZ = particleZ->position - particleI.position;
    normalizeVec3(vectorZ);
    
    Vec3 vectorX, vectorY;
    
    // Determine x-axis based on axis type
    if (axisType == TholeDipoleForce::ZOnly) {
        // z-only: choose an arbitrary x-axis perpendicular to z
        if (fabs(vectorZ[0]) < 0.866) {
            vectorX = Vec3(1.0, 0.0, 0.0);
        } else {
            vectorX = Vec3(0.0, 1.0, 0.0);
        }
        
        // Debug output for axis setup
        if (particleI.particleIndex <= 5) {
            printf("DEBUG Axis Setup: particle %d, axisType %d (ZOnly)\n", particleI.particleIndex, axisType);
            printf("  Original dipole: (%.6f, %.6f, %.6f)\n", particleI.dipole[0], particleI.dipole[1], particleI.dipole[2]);
            printf("  Z-axis vector: (%.6f, %.6f, %.6f)\n", vectorZ[0], vectorZ[1], vectorZ[2]);
            printf("  Final X-axis: (%.6f, %.6f, %.6f)\n", vectorX[0], vectorX[1], vectorX[2]);
        }
        
    } else {
        vectorX = particleX->position - particleI.position;
        
        if (axisType == TholeDipoleForce::Bisector) {
            // bisector: z-axis is the average of the two vectors
            normalizeVec3(vectorX);
            vectorZ += vectorX;
            normalizeVec3(vectorZ);
        } else if (axisType == TholeDipoleForce::ZBisect) {
            // z-bisect: x-axis is the average of the two vectors
            normalizeVec3(vectorX);
            vectorY = particleY->position - particleI.position;
            normalizeVec3(vectorY);
            vectorX += vectorY;
            normalizeVec3(vectorX);
        } else if (axisType == TholeDipoleForce::ThreeFold) {
            // 3-fold: z-axis is the average of three vectors
            normalizeVec3(vectorX);
            vectorY = particleY->position - particleI.position;
            normalizeVec3(vectorY);
            vectorZ += vectorX + vectorY;
            normalizeVec3(vectorZ);
        }
    }
    
    // Orthogonalize x-axis to z-axis
    double dot = vectorZ.dot(vectorX);
    vectorX -= vectorZ * dot;
    normalizeVec3(vectorX);
    
    // y-axis is the cross product of z and x
    vectorY = vectorZ.cross(vectorX);
    
    // Debug output for all axis types
    if (particleI.particleIndex <= 5 && axisType != TholeDipoleForce::ZOnly) {
        printf("DEBUG Axis Setup: particle %d, axisType %d\n", particleI.particleIndex, axisType);
        printf("  Original dipole: (%.6f, %.6f, %.6f)\n", particleI.dipole[0], particleI.dipole[1], particleI.dipole[2]);
        printf("  Final X-axis: (%.6f, %.6f, %.6f)\n", vectorX[0], vectorX[1], vectorX[2]);
        printf("  Final Y-axis: (%.6f, %.6f, %.6f)\n", vectorY[0], vectorY[1], vectorY[2]);
        printf("  Final Z-axis: (%.6f, %.6f, %.6f)\n", vectorZ[0], vectorZ[1], vectorZ[2]);
    }
    
    // Build rotation matrix (each row is a basis vector)
    Vec3 rotationMatrix[3];
    rotationMatrix[0] = vectorX;
    rotationMatrix[1] = vectorY;
    rotationMatrix[2] = vectorZ;
    
    // Rotate the dipole from molecular frame to lab frame
    Vec3 labDipole;
    for (int i = 0; i < 3; i++) {
        labDipole[i] = 0.0;
        for (int j = 0; j < 3; j++) {
            labDipole[i] += particleI.dipole[j] * rotationMatrix[j][i];
        }
    }
    
    // More debug output for small systems
    if (particleI.particleIndex <= 5) {
        printf("  Final transformed dipole: (%.6f, %.6f, %.6f)\n", labDipole[0], labDipole[1], labDipole[2]);
    }
    
    particleI.dipole = labDipole;
}

void ReferenceTholeDipoleForce::applyRotationMatrix(
    vector<TholeDipoleParticleData>& particleData,
    const vector<int>& axisTypes,
    const vector<int>& multipoleAtomZs,
    const vector<int>& multipoleAtomXs,
    const vector<int>& multipoleAtomYs) const {
    
    for (unsigned int i = 0; i < _numParticles; i++) {
        if (multipoleAtomZs[i] >= 0 && multipoleAtomZs[i] != i) {
            TholeDipoleParticleData* particleZ = &particleData[multipoleAtomZs[i]];
            TholeDipoleParticleData* particleX = nullptr;
            TholeDipoleParticleData* particleY = nullptr;
            
            if (multipoleAtomXs[i] >= 0) {
                particleX = &particleData[multipoleAtomXs[i]];
            }
            if (multipoleAtomYs[i] >= 0) {
                particleY = &particleData[multipoleAtomYs[i]];
            }
            
            applyRotationMatrixToParticle(particleData[i], particleZ, particleX, particleY, axisTypes[i]);
        }
    }
}

void ReferenceTholeDipoleForce::formQIRotationMatrix(
    const Vec3& iPosition,
    const Vec3& jPosition,
    const Vec3& deltaR,
    double r,
    double (&rotationMatrix)[3][3]) const {
    
    Vec3 vectorZ = deltaR / r;
    Vec3 vectorX(vectorZ);
    
    // Choose an arbitrary vector not parallel to vectorZ
    if ((iPosition[1] != jPosition[1]) || (iPosition[2] != jPosition[2])) {
        vectorX[0] += 1.0;
    } else {
        vectorX[1] += 1.0;
    }
    
    // Orthogonalize and normalize
    double dot = vectorZ.dot(vectorX);
    vectorX -= vectorZ * dot;
    normalizeVec3(vectorX);
    
    Vec3 vectorY = vectorZ.cross(vectorX);
    
    // Build rotation matrix
    rotationMatrix[0][0] = vectorX[0];
    rotationMatrix[0][1] = vectorX[1];
    rotationMatrix[0][2] = vectorX[2];
    rotationMatrix[1][0] = vectorY[0];
    rotationMatrix[1][1] = vectorY[1];
    rotationMatrix[1][2] = vectorY[2];
    rotationMatrix[2][0] = vectorZ[0];
    rotationMatrix[2][1] = vectorZ[1];
    rotationMatrix[2][2] = vectorZ[2];
}

void ReferenceTholeDipoleForce::setupScaleMaps(const vector<vector<vector<int>>>& multipoleCovalentInfo) {
    _scaleMaps[M_SCALE].resize(_numParticles);
    _scaleMaps[I_SCALE].resize(_numParticles);
    _maxScaleIndex.resize(_numParticles);
    
    for (unsigned int i = 0; i < _numParticles; i++) {
        _maxScaleIndex[i] = 0;
        
        // Process covalent info for M_SCALE
        for (unsigned int j = 0; j < multipoleCovalentInfo[i][M_SCALE].size(); j++) {
            int atom = multipoleCovalentInfo[i][M_SCALE][j];
            _scaleMaps[M_SCALE][i][atom] = _mScale[j];
            if (atom > _maxScaleIndex[i]) {
                _maxScaleIndex[i] = atom;
            }
        }
        
        // Process covalent info for I_SCALE  
        for (unsigned int j = 0; j < multipoleCovalentInfo[i][I_SCALE].size(); j++) {
            int atom = multipoleCovalentInfo[i][I_SCALE][j];
            _scaleMaps[I_SCALE][i][atom] = _iScale[j];
            if (atom > _maxScaleIndex[i]) {
                _maxScaleIndex[i] = atom;
            }
        }
    }
}

double ReferenceTholeDipoleForce::getScaleFactor(unsigned int particleI, unsigned int particleJ, 
                                                 ScaleType scaleType) const {
    MapIntRealOpenMMCI iter = _scaleMaps[scaleType][particleI].find(particleJ);
    if (iter != _scaleMaps[scaleType][particleI].end()) {
        return iter->second;
    }
    return 1.0;
}

void ReferenceTholeDipoleForce::getAndScaleInverseRs(double dampI, double dampJ,
                                                     double tholeI, double tholeJ,
                                                     double r, vector<double>& rrI) const {
    double rI = 1.0 / r;
    double r2I = rI * rI;
    
    rrI[0] = rI * r2I;  // 1/r^3
    double constantFactor = 3.0;
    for (unsigned int i = 1; i < rrI.size(); i++) {
        rrI[i] = constantFactor * rrI[i-1] * r2I;
        constantFactor += 2.0;
    }
    
    // Apply Thole damping
    double damp = dampI * dampJ;
    if (damp != 0.0) {
        double pgamma = tholeI < tholeJ ? tholeI : tholeJ;
        double ratio = r / damp;
        ratio = ratio * ratio * ratio;
        damp = -pgamma * ratio;
        
        if (damp > -50.0) {
            double dampExp = exp(damp);
            rrI[0] *= 1.0 - dampExp;
            rrI[1] *= 1.0 - (1.0 - damp) * dampExp;
            if (rrI.size() > 2) {
                rrI[2] *= 1.0 - (1.0 - damp + (0.6 * damp * damp)) * dampExp;
            }
        }
    }
}

void ReferenceTholeDipoleForce::calculateFixedDipoleFieldPairIxn(
    const TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData& particleJ,
    double mScale, double iScale) {
    
    if (particleI.particleIndex == particleJ.particleIndex) {
        return;
    }
    
    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r = sqrt(deltaR.dot(deltaR));
    
    vector<double> rrI(2);  // Need 1/r^3 and 1/r^5
    getAndScaleInverseRs(particleI.tholeDamping, particleJ.tholeDamping, 
                         particleI.tholeDamping, particleJ.tholeDamping, r, rrI);
    
    double rr3 = rrI[0];
    double rr5 = rrI[1];
    
    // Field at particle I due to charge and dipole at particle J
    double dipoleDeltaJ = particleJ.dipole.dot(deltaR);
    double factorJ = rr3 * particleJ.charge - rr5 * dipoleDeltaJ;
    Vec3 fieldJ = deltaR * factorJ + particleJ.dipole * rr3;
    
    unsigned int indexI = particleI.particleIndex;
    _fixedDipoleField[indexI] -= fieldJ * mScale;
    
    // Field at particle J due to charge and dipole at particle I  
    double dipoleDeltaI = particleI.dipole.dot(deltaR);
    double factorI = rr3 * particleI.charge + rr5 * dipoleDeltaI;
    Vec3 fieldI = deltaR * factorI - particleI.dipole * rr3;
    
    unsigned int indexJ = particleJ.particleIndex;
    _fixedDipoleField[indexJ] += fieldI * mScale;
}

void ReferenceTholeDipoleForce::calculateFixedDipoleField(
    const vector<TholeDipoleParticleData>& particleData) {
    
    // Calculate fixed dipole fields from permanent charges and dipoles
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            double mScale = 1.0;
            double iScale = 1.0;
            
            // Get scaling factors if within cutoff
            if (j <= _maxScaleIndex[i]) {
                mScale = getScaleFactor(i, j, M_SCALE);
                iScale = getScaleFactor(i, j, I_SCALE);
            }
            
            calculateFixedDipoleFieldPairIxn(particleData[i], particleData[j], mScale, iScale);
        }
    }
}

void ReferenceTholeDipoleForce::calculateInducedDipoles(
    const vector<TholeDipoleParticleData>& particleData) {
    
    // Zero and calculate fixed dipole fields
    zeroFixedDipoleFields();
    calculateFixedDipoleField(particleData);
    
    // Scale fields by polarizability and initialize induced dipoles
    _inducedDipole.resize(_numParticles);
    for (unsigned int i = 0; i < _numParticles; i++) {
        _inducedDipole[i] = _fixedDipoleField[i] * particleData[i].polarizability;
    }
    
    // For Direct polarization, we're done
    if (_polarizationType == Direct) {
        _mutualInducedDipoleConverged = 1;
        _mutualInducedDipoleIterations = 0;
        return;
    }
    
    // For Mutual polarization, iterate until convergence
    if (_polarizationType == Mutual) {
        convergeInducedDipolesByDIIS(particleData);
    }
    
    // For Extrapolated polarization, use perturbation theory
    if (_polarizationType == Extrapolated) {
        convergeInducedDipolesByExtrapolation(particleData);
    }
}

void ReferenceTholeDipoleForce::calculateInducedDipolePairIxn(
    unsigned int particleI,
    unsigned int particleJ,
    double rr3,
    double rr5,
    const Vec3& deltaR,
    const vector<Vec3>& inducedDipole,
    vector<Vec3>& field) const {
    
    double dDotDelta = rr5 * (inducedDipole[particleJ].dot(deltaR));
    field[particleI] += inducedDipole[particleJ] * rr3 + deltaR * dDotDelta;
    
    dDotDelta = rr5 * (inducedDipole[particleI].dot(deltaR));
    field[particleJ] += inducedDipole[particleI] * rr3 + deltaR * dDotDelta;
}

void ReferenceTholeDipoleForce::calculateInducedDipoleFields(
    const vector<TholeDipoleParticleData>& particleData,
    const vector<Vec3>& inducedDipoles,
    vector<Vec3>& inducedDipoleField) {
    
    // Initialize field to zero
    initializeVec3Vector(inducedDipoleField);
    
    // Calculate induced dipole fields
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            Vec3 deltaR = particleData[j].position - particleData[i].position;
            getPeriodicDelta(deltaR);
            double r = sqrt(deltaR.dot(deltaR));
            
            vector<double> rrI(2);
            getAndScaleInverseRs(particleData[i].tholeDamping, particleData[j].tholeDamping,
                                 particleData[i].tholeDamping, particleData[j].tholeDamping, r, rrI);
            
            double rr3 = -rrI[0];  // Note the negative sign
            double rr5 = rrI[1];
            
            // Get scaling factor
            double iScale = 1.0;
            if (j <= _maxScaleIndex[i]) {
                iScale = getScaleFactor(i, j, I_SCALE);
            }
            
            // Calculate mutual field with scaling
            if (iScale != 0.0) {
                Vec3 scaledDeltaR = deltaR;
                calculateInducedDipolePairIxn(i, j, rr3 * iScale, rr5 * iScale, 
                                              scaledDeltaR, inducedDipoles, inducedDipoleField);
            }
        }
    }
}

void ReferenceTholeDipoleForce::convergeInducedDipolesByDIIS(
    const vector<TholeDipoleParticleData>& particleData) {
    
    // Simple iterative convergence for mutual polarization
    // This is a simplified version - full DIIS would be more complex
    
    vector<Vec3> inducedDipoleField;
    double epsilon = 1.0;
    
    for (_mutualInducedDipoleIterations = 0; 
         _mutualInducedDipoleIterations < _maximumMutualInducedDipoleIterations && 
         epsilon > _mutualInducedDipoleTargetEpsilon;
         _mutualInducedDipoleIterations++) {
        
        // Calculate induced dipole field
        calculateInducedDipoleFields(particleData, _inducedDipole, inducedDipoleField);
        
        // Update induced dipoles and check convergence
        epsilon = 0.0;
        for (unsigned int i = 0; i < _numParticles; i++) {
            Vec3 newDipole = (_fixedDipoleField[i] + inducedDipoleField[i]) * 
                             particleData[i].polarizability;
            Vec3 delta = newDipole - _inducedDipole[i];
            epsilon += delta.dot(delta);
            _inducedDipole[i] = newDipole;
        }
        epsilon = sqrt(epsilon / _numParticles);
    }
    
    _mutualInducedDipoleEpsilon = epsilon;
    _mutualInducedDipoleConverged = epsilon < _mutualInducedDipoleTargetEpsilon ? 1 : 0;
}

void ReferenceTholeDipoleForce::convergeInducedDipolesByExtrapolation(
    const vector<TholeDipoleParticleData>& particleData) {
    
    // Check if extrapolation coefficients are set
    if (_extrapolationCoefficients.size() == 0) {
        // Default to OPT4 coefficients if not set
        _extrapolationCoefficients.resize(4);
        _extrapolationCoefficients[0] = -0.154;
        _extrapolationCoefficients[1] = 0.017;
        _extrapolationCoefficients[2] = 0.658;
        _extrapolationCoefficients[3] = 0.474;
    }
    
    int maxPTOrder = _extrapolationCoefficients.size();
    
    // Storage for perturbation theory orders
    vector<vector<Vec3>> extrapolatedDipoles(maxPTOrder);
    vector<Vec3> inducedDipoleField(_numParticles);
    
    // PT0: Direct dipoles (initial induced dipoles from fixed field only)
    extrapolatedDipoles[0] = _inducedDipole;
    
    // Generate higher order PT terms by recursively applying the dipole interaction operator
    for (int order = 1; order < maxPTOrder; order++) {
        // Calculate field from current induced dipoles
        calculateInducedDipoleFields(particleData, _inducedDipole, inducedDipoleField);
        
        // Update induced dipoles: µ_n = α * (E_fixed + E_induced)
        extrapolatedDipoles[order].resize(_numParticles);
        for (unsigned int i = 0; i < _numParticles; i++) {
            _inducedDipole[i] = (_fixedDipoleField[i] + inducedDipoleField[i]) * 
                                particleData[i].polarizability;
            extrapolatedDipoles[order][i] = _inducedDipole[i];
        }
    }
    
    // Form the final induced dipoles as a linear combination of PT orders
    for (unsigned int i = 0; i < _numParticles; i++) {
        _inducedDipole[i] = Vec3(0.0, 0.0, 0.0);
        for (int order = 0; order < maxPTOrder; order++) {
            _inducedDipole[i] += extrapolatedDipoles[order][i] * _extrapolationCoefficients[order];
        }
    }
    
    // Calculate final error for reporting
    calculateInducedDipoleFields(particleData, _inducedDipole, inducedDipoleField);
    double epsilon = 0.0;
    for (unsigned int i = 0; i < _numParticles; i++) {
        Vec3 finalDipole = (_fixedDipoleField[i] + inducedDipoleField[i]) * 
                           particleData[i].polarizability;
        Vec3 delta = finalDipole - _inducedDipole[i];
        epsilon += delta.dot(delta);
    }
    _mutualInducedDipoleEpsilon = sqrt(epsilon / _numParticles);
    _mutualInducedDipoleConverged = 1;
    _mutualInducedDipoleIterations = maxPTOrder;
}

const vector<double>& ReferenceTholeDipoleForce::getExtrapolationCoefficients() const {
    return _extrapolationCoefficients;
}

void ReferenceTholeDipoleForce::mapTorqueToForceForParticle(
    const TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData& particleU,
    const TholeDipoleParticleData& particleV,
    const TholeDipoleParticleData* particleW,
    int axisType,
    const Vec3& torque,
    vector<Vec3>& forces) const {

    // Get coordinates of this atom and the axis atoms
    if (axisType == TholeDipoleForce::NoAxisType) {
        return;
    }
    
    // Debug output for small systems
    if (particleI.particleIndex <= 5) {
        printf("DEBUG Torque Mapping: particle %d, axisType %d\n", particleI.particleIndex, axisType);
        printf("  Torque: (%.6f, %.6f, %.6f)\n", torque[0], torque[1], torque[2]);
        printf("  particleU: %d, particleV: %d", particleU.particleIndex, particleV.particleIndex);
        if (particleW) printf(", particleW: %d", particleW->particleIndex);
        printf("\n");
    }

    Vec3 vectorU = particleU.position - particleI.position;
    double normU = normalizeVec3(vectorU);

    Vec3 vectorV = particleV.position - particleI.position;
    double normV = normalizeVec3(vectorV);

    Vec3 vectorW;
    double normW;
    if (particleW && (axisType == TholeDipoleForce::ZBisect || axisType == TholeDipoleForce::ThreeFold)) {
        vectorW = particleW->position - particleI.position;
    } else {
        vectorW = vectorU.cross(vectorV);
    }
    normW = normalizeVec3(vectorW);

    Vec3 vectorUV = vectorV.cross(vectorU);
    Vec3 vectorUW = vectorW.cross(vectorU);
    Vec3 vectorVW = vectorW.cross(vectorV);

    normalizeVec3(vectorUV);
    normalizeVec3(vectorUW);
    normalizeVec3(vectorVW);

    // Calculate angles
    double cosUV = vectorU.dot(vectorV);
    double sinUV = sqrt(1.0 - cosUV*cosUV);

    double cosUW = vectorU.dot(vectorW);
    double sinUW = sqrt(1.0 - cosUW*cosUW);

    double cosVW = vectorV.dot(vectorW);
    double sinVW = sqrt(1.0 - cosVW*cosVW);

    // Project torque onto local axes
    Vec3 dphi;
    dphi[0] = vectorU.dot(torque);
    dphi[1] = vectorV.dot(torque);
    dphi[2] = vectorW.dot(torque);
    dphi *= -1.0;

    // Branch based on axis type
    if (axisType == TholeDipoleForce::ZThenX || axisType == TholeDipoleForce::Bisector) {
        double factor1 = dphi[1]/(normU*sinUV);
        double factor2 = dphi[2]/normU;
        double factor3 = -dphi[0]/(normV*sinUV);
        double factor4;

        if (axisType == TholeDipoleForce::Bisector) {
            factor2 *= 0.5;
            factor4 = 0.5*dphi[2]/normV;
        } else {
            factor4 = 0.0;
        }

        for (int i = 0; i < 3; i++) {
            double forceU = vectorUV[i]*factor1 + factor2*vectorUW[i];
            forces[particleU.particleIndex][i] -= forceU;

            double forceV = vectorUV[i]*factor3 + factor4*vectorVW[i];
            forces[particleV.particleIndex][i] -= forceV;

            forces[particleI.particleIndex][i] += (forceU + forceV);
        }

    } else if (axisType == TholeDipoleForce::ZBisect) {
        Vec3 vectorR = vectorV + vectorW;
        Vec3 vectorS = vectorU.cross(vectorR);

        double normR = normalizeVec3(vectorR);
        double normS = normalizeVec3(vectorS);

        Vec3 vectorUR = vectorR.cross(vectorU);
        Vec3 vectorUS = vectorS.cross(vectorU);
        Vec3 vectorVS = vectorS.cross(vectorV);
        Vec3 vectorWS = vectorS.cross(vectorW);

        normalizeVec3(vectorUR);
        normalizeVec3(vectorUS);
        normalizeVec3(vectorVS);
        normalizeVec3(vectorWS);

        double cosUR = vectorU.dot(vectorR);
        double sinUR = sqrt(1.0 - cosUR*cosUR);

        double cosUS = vectorU.dot(vectorS);
        double sinUS = sqrt(1.0 - cosUS*cosUS);

        double cosVS = vectorV.dot(vectorS);
        double sinVS = sqrt(1.0 - cosVS*cosVS);

        double cosWS = vectorW.dot(vectorS);
        double sinWS = sqrt(1.0 - cosWS*cosWS);

        Vec3 t1 = vectorV - vectorS*cosVS;
        Vec3 t2 = vectorW - vectorS*cosWS;

        normalizeVec3(t1);
        normalizeVec3(t2);

        double ut1cos = vectorU.dot(t1);
        double ut1sin = sqrt(1.0 - ut1cos*ut1cos);

        double ut2cos = vectorU.dot(t2);
        double ut2sin = sqrt(1.0 - ut2cos*ut2cos);

        double dphiR = vectorR.dot(torque)*(-1.0);
        double dphiS = vectorS.dot(torque)*(-1.0);

        double factor1 = dphiR/(normU*sinUR);
        double factor2 = dphiS/normU;
        double factor3 = dphi[0]/(normV*(ut1sin+ut2sin));
        double factor4 = dphi[0]/(normW*(ut1sin+ut2sin));

        Vec3 forceU = vectorUR*factor1 + vectorUS*factor2;
        forces[particleU.particleIndex] -= forceU;

        Vec3 forceV = (vectorS*sinVS - t1*cosVS)*factor3;
        forces[particleV.particleIndex] -= forceV;

        Vec3 forceW = (vectorS*sinWS - t2*cosWS)*factor4;
        forces[particleW->particleIndex] -= forceW;

        forces[particleI.particleIndex] += (forceU + forceV + forceW);

    } else if (axisType == TholeDipoleForce::ThreeFold) {
        // 3-fold symmetry
        for (int i = 0; i < 3; i++) {
            double du = vectorUW[i]*dphi[2]/(normU*sinUW) +
                       vectorUV[i]*dphi[1]/(normU*sinUV) -
                       vectorUW[i]*dphi[0]/(normU*sinUW) -
                       vectorUV[i]*dphi[0]/(normU*sinUV);

            double dv = vectorVW[i]*dphi[2]/(normV*sinVW) -
                       vectorUV[i]*dphi[0]/(normV*sinUV) -
                       vectorVW[i]*dphi[1]/(normV*sinVW) +
                       vectorUV[i]*dphi[1]/(normV*sinUV);

            double dw = -vectorUW[i]*dphi[0]/(normW*sinUW) -
                       vectorVW[i]*dphi[1]/(normW*sinVW) +
                       vectorUW[i]*dphi[2]/(normW*sinUW) +
                       vectorVW[i]*dphi[2]/(normW*sinVW);

            du /= 3.0;
            dv /= 3.0;
            dw /= 3.0;

            forces[particleU.particleIndex][i] -= du;
            forces[particleV.particleIndex][i] -= dv;
            if (particleW)
                forces[particleW->particleIndex][i] -= dw;
            forces[particleI.particleIndex][i] += (du + dv + dw);
        }

    } else if (axisType == TholeDipoleForce::ZOnly) {
        // Z-only axis
        if (particleI.particleIndex <= 5) {
            printf("DEBUG ZOnly Details: particle %d\n", particleI.particleIndex);
            printf("  vectorU: (%.6f, %.6f, %.6f), normU: %.6f\n", vectorU[0], vectorU[1], vectorU[2], normU);
            printf("  vectorV: (%.6f, %.6f, %.6f)\n", vectorV[0], vectorV[1], vectorV[2]);
            printf("  vectorW: (%.6f, %.6f, %.6f)\n", vectorW[0], vectorW[1], vectorW[2]);
            printf("  vectorUV: (%.6f, %.6f, %.6f), sinUV: %.6f\n", vectorUV[0], vectorUV[1], vectorUV[2], sinUV);
            printf("  vectorUW: (%.6f, %.6f, %.6f)\n", vectorUW[0], vectorUW[1], vectorUW[2]);
            printf("  dphi: (%.6f, %.6f, %.6f)\n", dphi[0], dphi[1], dphi[2]);
        }
        
        for (int i = 0; i < 3; i++) {
            double du = vectorUV[i]*dphi[1]/(normU*sinUV) + vectorUW[i]*dphi[2]/normU;
            if (particleI.particleIndex <= 5) {
                printf("  Force component[%d]: du = %.6f\n", i, du);
            }
            // Don't apply forces to dummy particles (particleIndex = -1)
            if (particleU.particleIndex >= 0) {
                forces[particleU.particleIndex][i] -= du;
            }
            forces[particleI.particleIndex][i] += du;
        }
    }
}

void ReferenceTholeDipoleForce::mapTorqueToForce(
    const vector<TholeDipoleParticleData>& particleData,
    const vector<int>& multipoleAtomXs,
    const vector<int>& multipoleAtomYs,
    const vector<int>& multipoleAtomZs,
    const vector<int>& axisTypes,
    vector<Vec3>& torques,
    vector<Vec3>& forces) const {

    // Map torques to forces
    for (unsigned int ii = 0; ii < particleData.size(); ii++) {
        if (axisTypes[ii] != TholeDipoleForce::NoAxisType) {
            // Debug output for small systems
            if (ii <= 5) {
                printf("DEBUG MapTorque Call: particle %d\n", ii);
                printf("  axisType: %d, multipoleAtomZ: %d, multipoleAtomX: %d, multipoleAtomY: %d\n", 
                       axisTypes[ii], multipoleAtomZs[ii], multipoleAtomXs[ii], multipoleAtomYs[ii]);
            }
            
            // Handle ZOnly case where multipoleAtomX = -1
            TholeDipoleParticleData dummyParticleX;
            if (axisTypes[ii] == TholeDipoleForce::ZOnly && multipoleAtomXs[ii] == -1) {
                // Create a dummy particle for X-axis that gives the same result as the axis setup
                dummyParticleX = particleData[ii];
                Vec3 zAxis = particleData[multipoleAtomZs[ii]].position - particleData[ii].position;
                normalizeVec3(zAxis);
                // Choose perpendicular direction same as in applyRotationMatrixToParticle
                if (fabs(zAxis[0]) < 0.866) {
                    dummyParticleX.position = particleData[ii].position + Vec3(1.0, 0.0, 0.0);
                } else {
                    dummyParticleX.position = particleData[ii].position + Vec3(0.0, 1.0, 0.0);
                }
                dummyParticleX.particleIndex = -1; // Mark as dummy
            }
            
            mapTorqueToForceForParticle(
                particleData[ii],
                particleData[multipoleAtomZs[ii]],
                (axisTypes[ii] == TholeDipoleForce::ZOnly && multipoleAtomXs[ii] == -1) ? dummyParticleX : particleData[multipoleAtomXs[ii]],
                multipoleAtomYs[ii] > -1 ? &particleData[multipoleAtomYs[ii]] : NULL,
                axisTypes[ii],
                torques[ii],
                forces
            );
        }
    }
}

void ReferenceTholeDipoleForce::setup(
    const vector<Vec3>& particlePositions,
    const vector<double>& charges,
    const vector<double>& dipoles,
    const vector<double>& polarizabilities,
    const vector<double>& tholeDampingFactors,
    const vector<int>& axisTypes,
    const vector<int>& multipoleAtomZs,
    const vector<int>& multipoleAtomXs,
    const vector<int>& multipoleAtomYs,
    const vector<vector<vector<int>>>& multipoleCovalentInfo,
    vector<TholeDipoleParticleData>& particleData) {
    
    _numParticles = particlePositions.size();
    
    // Load particle data
    loadParticleData(particlePositions, charges, dipoles, polarizabilities, 
                     tholeDampingFactors, particleData);
    
    // Set axis types and atom indices
    for (unsigned int i = 0; i < _numParticles; i++) {
        particleData[i].axisType = axisTypes[i];
        particleData[i].multipoleAtomZ = multipoleAtomZs[i];
        particleData[i].multipoleAtomX = multipoleAtomXs[i];
        particleData[i].multipoleAtomY = multipoleAtomYs[i];
    }
    
    // Check chirality
    checkChiral(particleData, multipoleAtomXs, multipoleAtomYs, multipoleAtomZs, axisTypes);
    
    // Apply rotation matrices to transform dipoles from molecular to lab frame
    applyRotationMatrix(particleData, axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs);
    
    // Setup scaling factor maps
    setupScaleMaps(multipoleCovalentInfo);
    
    // Calculate induced dipoles
    calculateInducedDipoles(particleData);
}

void ReferenceTholeDipoleForce::calculateInducedDipoles(const vector<Vec3>& particlePositions,
                                                        const vector<double>& charges,
                                                        const vector<double>& dipoles,
                                                        const vector<double>& polarizabilities,
                                                        const vector<double>& tholeDampingFactors,
                                                        const vector<int>& axisTypes,
                                                        const vector<int>& multipoleAtomZs,
                                                        const vector<int>& multipoleAtomXs,
                                                        const vector<int>& multipoleAtomYs,
                                                        const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                        vector<Vec3>& outputInducedDipoles) {
    // Setup, including calculating induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities, tholeDampingFactors,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);
    
    outputInducedDipoles = _inducedDipole;
}

void ReferenceTholeDipoleForce::calculateLabFramePermanentDipoles(const vector<Vec3>& particlePositions,
                                                                  const vector<double>& charges,
                                                                  const vector<double>& dipoles,
                                                                  const vector<double>& polarizabilities,
                                                                  const vector<double>& tholeDampingFactors,
                                                                  const vector<int>& axisTypes,
                                                                  const vector<int>& multipoleAtomZs,
                                                                  const vector<int>& multipoleAtomXs,
                                                                  const vector<int>& multipoleAtomYs,
                                                                  const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                                  vector<Vec3>& outputRotatedPermanentDipoles) {
    // Setup, including rotating permanent dipoles to lab frame
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities, tholeDampingFactors,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);
    
    outputRotatedPermanentDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++) {
        outputRotatedPermanentDipoles[i] = particleData[i].dipole;
    }
}

void ReferenceTholeDipoleForce::calculateTotalDipoles(const vector<Vec3>& particlePositions,
                                                      const vector<double>& charges,
                                                      const vector<double>& dipoles,
                                                      const vector<double>& polarizabilities,
                                                      const vector<double>& tholeDampingFactors,
                                                      const vector<int>& axisTypes,
                                                      const vector<int>& multipoleAtomZs,
                                                      const vector<int>& multipoleAtomXs,
                                                      const vector<int>& multipoleAtomYs,
                                                      const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                      vector<Vec3>& outputTotalDipoles) {
    // Setup, including calculating permanent and induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities, tholeDampingFactors,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);
    
    outputTotalDipoles.resize(_numParticles);
    for (int i = 0; i < _numParticles; i++) {
        for (int j = 0; j < 3; j++) {
            outputTotalDipoles[i][j] = particleData[i].dipole[j] + _inducedDipole[i][j];
        }
    }
}

void ReferenceTholeDipoleForce::calculateTholeDipoleSystemMultipoleMoments(const vector<double>& masses,
                                                                           const vector<Vec3>& particlePositions,
                                                                           const vector<double>& charges,
                                                                           const vector<double>& dipoles,
                                                                           const vector<double>& polarizabilities,
                                                                           const vector<double>& tholeDampingFactors,
                                                                           const vector<int>& axisTypes,
                                                                           const vector<int>& multipoleAtomZs,
                                                                           const vector<int>& multipoleAtomXs,
                                                                           const vector<int>& multipoleAtomYs,
                                                                           const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                                           vector<double>& outputMultipoleMoments) {
    // Setup, including calculating induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities, tholeDampingFactors,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);

    // Calculate center of mass
    double totalMass = 0.0;
    Vec3 centerOfMass(0.0, 0.0, 0.0);
    for (unsigned int i = 0; i < _numParticles; i++) {
        double mass = masses[i];
        totalMass += mass;
        centerOfMass += particleData[i].position * mass;
    }
    
    vector<Vec3> localPositions(_numParticles);
    if (totalMass > 0.0) {
        centerOfMass *= 1.0 / totalMass;
    }
    for (unsigned int i = 0; i < _numParticles; i++) {
        localPositions[i] = particleData[i].position - centerOfMass;
    }

    // Calculate system multipole moments
    double netchg = 0.0;
    Vec3 dpl(0.0, 0.0, 0.0);
    
    // Quadrupole components
    double xxqdp = 0.0, xyqdp = 0.0, xzqdp = 0.0;
    double yyqdp = 0.0, yzqdp = 0.0, zzqdp = 0.0;

    for (unsigned int i = 0; i < _numParticles; i++) {
        double charge = particleData[i].charge;
        Vec3 position = localPositions[i];
        netchg += charge;

        Vec3 netDipole = particleData[i].dipole + _inducedDipole[i];
        dpl += position * charge + netDipole;

        // Quadrupole contributions from charge and dipole
        xxqdp += position[0] * position[0] * charge + 2.0 * position[0] * netDipole[0];
        xyqdp += position[0] * position[1] * charge + position[0] * netDipole[1] + position[1] * netDipole[0];
        xzqdp += position[0] * position[2] * charge + position[0] * netDipole[2] + position[2] * netDipole[0];
        yyqdp += position[1] * position[1] * charge + 2.0 * position[1] * netDipole[1];
        yzqdp += position[1] * position[2] * charge + position[1] * netDipole[2] + position[2] * netDipole[1];
        zzqdp += position[2] * position[2] * charge + 2.0 * position[2] * netDipole[2];
    }

    // Convert quadrupole to traceless form
    outputMultipoleMoments.resize(13);
    double qave = (xxqdp + yyqdp + zzqdp) / 3.0;
    outputMultipoleMoments[4] = 0.5 * (xxqdp - qave);
    outputMultipoleMoments[5] = 0.5 * xyqdp;
    outputMultipoleMoments[6] = 0.5 * xzqdp;
    outputMultipoleMoments[8] = 0.5 * (yyqdp - qave);
    outputMultipoleMoments[9] = 0.5 * yzqdp;
    outputMultipoleMoments[12] = 0.5 * (zzqdp - qave);
    
    // Note: No atomic quadrupoles to add in TholeDipole model
    outputMultipoleMoments[7] = outputMultipoleMoments[5];
    outputMultipoleMoments[10] = outputMultipoleMoments[6];
    outputMultipoleMoments[11] = outputMultipoleMoments[9];

    // Convert to appropriate units
    double debye = 4.80321;
    outputMultipoleMoments[0] = netchg;
    
    dpl *= 10.0 * debye;
    outputMultipoleMoments[1] = dpl[0];
    outputMultipoleMoments[2] = dpl[1];
    outputMultipoleMoments[3] = dpl[2];

    debye *= 3.0;
    for (unsigned int i = 4; i < 13; i++) {
        outputMultipoleMoments[i] *= 100.0 * debye;
    }
}

void ReferenceTholeDipoleForce::calculateElectrostaticPotential(const vector<Vec3>& particlePositions,
                                                                const vector<double>& charges,
                                                                const vector<double>& dipoles,
                                                                const vector<double>& polarizabilities,
                                                                const vector<double>& tholeDampingFactors,
                                                                const vector<int>& axisTypes,
                                                                const vector<int>& multipoleAtomZs,
                                                                const vector<int>& multipoleAtomXs,
                                                                const vector<int>& multipoleAtomYs,
                                                                const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                                const vector<Vec3>& inputGrid,
                                                                vector<double>& outputPotential) {
    // Setup particle data
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities, tholeDampingFactors,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);
    
    outputPotential.resize(inputGrid.size());
    
    // Calculate potential at each grid point
    for (size_t gridIndex = 0; gridIndex < inputGrid.size(); gridIndex++) {
        double potential = 0.0;
        
        for (unsigned int i = 0; i < _numParticles; i++) {
            potential += calculateElectrostaticPotentialForParticleGridPoint(
                particleData[i], inputGrid[gridIndex]);
        }
        
        outputPotential[gridIndex] = potential;
    }
}

double ReferenceTholeDipoleForce::calculateElectrostaticPotentialForParticleGridPoint(
    const TholeDipoleParticleData& particleI, const Vec3& gridPoint) const {
    
    Vec3 deltaR = particleI.position - gridPoint;
    getPeriodicDelta(deltaR);
    
    double r2 = deltaR.dot(deltaR);
    double r = sqrt(r2);
    
    double rr1 = 1.0 / r;
    double rr2 = rr1 * rr1;
    double rr3 = rr1 * rr2;
    
    // Charge contribution
    double potential = particleI.charge * rr1;
    
    // Dipole contributions (permanent and induced)
    double scd = particleI.dipole.dot(deltaR);
    double scu = _inducedDipole[particleI.particleIndex].dot(deltaR);
    potential -= (scd + scu) * rr3;
    
    // Note: No quadrupole contribution in TholeDipole model
    
    return potential;
}

// PME class implementation
ReferencePmeTholeDipoleForce::ReferencePmeTholeDipoleForce() : ReferenceTholeDipoleForce(ReferenceTholeDipoleForce::PME) {
    _cutoffDistance = 1.0;
    _alphaEwald = 0.0;
    _pmeGridDimensions.resize(3, 0);
}

ReferencePmeTholeDipoleForce::~ReferencePmeTholeDipoleForce() {
}

void ReferencePmeTholeDipoleForce::setCutoffDistance(double cutoffDistance) {
    _cutoffDistance = cutoffDistance;
}

double ReferencePmeTholeDipoleForce::getCutoffDistance() const {
    return _cutoffDistance;
}

void ReferencePmeTholeDipoleForce::setAlphaEwald(double alphaEwald) {
    _alphaEwald = alphaEwald;
}

double ReferencePmeTholeDipoleForce::getAlphaEwald() const {
    return _alphaEwald;
}

void ReferencePmeTholeDipoleForce::setPmeGridDimensions(const std::vector<int>& pmeGridDimensions) {
    _pmeGridDimensions = pmeGridDimensions;
}

void ReferencePmeTholeDipoleForce::getPmeGridDimensions(std::vector<int>& pmeGridDimensions) const {
    pmeGridDimensions = _pmeGridDimensions;
}

void ReferencePmeTholeDipoleForce::setPeriodicBoxSize(Vec3* boxVectors) {
    for (int i = 0; i < 3; i++) {
        _periodicBoxVectors[i] = boxVectors[i];
    }
}

void ReferencePmeTholeDipoleForce::getPeriodicDelta(Vec3& deltaR) const {
    // Apply periodic boundary conditions
    // This is a simplified implementation - full PBC would be more complex
    for (int i = 0; i < 3; i++) {
        double boxSize = _periodicBoxVectors[i][i];
        if (boxSize > 0) {
            deltaR[i] -= boxSize * floor(deltaR[i] / boxSize + 0.5);
        }
    }
}
