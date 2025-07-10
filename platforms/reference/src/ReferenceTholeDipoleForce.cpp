#include "ReferenceTholeDipoleForce.h"
#include "openmm/OpenMMException.h"

using namespace TholeDipolePlugin;
using namespace OpenMM;

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

    // Map torques to forces
    mapTorqueToForce(particleData, multipoleAtomXs, multipoleAtomYs, multipoleAtomZs, 
                     axisTypes, torques, forces);

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

void ReferenceTholeDipoleForce::normalizeVec3(Vec3& vector) const {
    double norm = sqrt(vector.dot(vector));
    if (norm > 0.0) {
        vector *= 1.0 / norm;
    }
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
    vector<vector<Vec3>> inducedDipoleField(_numParticles);
    
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

// Also need to add the extrapolation coefficients getter/setter implementations
void ReferenceTholeDipoleForce::setExtrapolationCoefficients(const vector<double>& coefficients) {
    _extrapolationCoefficients = coefficients;
}

const vector<double>& ReferenceTholeDipoleForce::getExtrapolationCoefficients() const {
    return _extrapolationCoefficients;
}

void ReferenceTholeDipoleForce::mapTorqueToForce(
    const vector<TholeDipoleParticleData>& particleData,
    const vector<int>& multipoleAtomXs,
    const vector<int>& multipoleAtomYs,
    const vector<int>& multipoleAtomZs,
    const vector<int>& axisTypes,
    vector<Vec3>& torques,
    vector<Vec3>& forces) const {
    
    // Convert torques on local coordinate frames to forces
    for (unsigned int i = 0; i < _numParticles; i++) {
        if (multipoleAtomZs[i] >= 0 && multipoleAtomZs[i] != i) {
            // This particle has a local coordinate system
            // The torque conversion would go here
            // For now, this is a placeholder
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
