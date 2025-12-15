#include "ReferenceTholeDipoleForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <iostream>

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
    _electric = ONE_4PI_EPS0;
    _dielectric = 1.0;
    _mutualInducedDipoleTargetEpsilon = 1.0e-03;
    _maximumMutualInducedDipoleIterations = 60;
    _mutualInducedDipoleEpsilon = 1.0e+50;
    _mutualInducedDipoleConverged = 0;
    _mutualInducedDipoleIterations = 0;
    _debye = 0.4803;
    _tholeDampingType = TholeDipoleForce::Amoeba;  // Default to Amoeba (ρ₂)
    _tholeDampingParameter = 0.39;  // Default Thole damping parameter


    _scaleMaps.resize(LAST_SCALE_TYPE_INDEX);
    _maxScaleIndex.resize(LAST_SCALE_TYPE_INDEX);
    for (int i = 0; i < LAST_SCALE_TYPE_INDEX; i++) {
        _maxScaleIndex[i] = 5;
    }

    _mScale[0] = 0.0; // Covalent12 (1-2) excluded
    _mScale[1] = 0.0; // Covalent13 (1-3) excluded
    _mScale[2] = 0.5; // Covalent14 (1-4) scaled
    _mScale[3] = 1.0; // Covalent15 (1-5+) full
    _mScale[4] = 1.0; // Guard

    _iScale[0] = 1.0; // Covalent12
    _iScale[1] = 1.0; // Covalent13
    _iScale[2] = 1.0; // Covalent14
    _iScale[3] = 1.0; // Covalent15
    _iScale[4] = 1.0; // Guard

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

TholeDipoleForce::TholeDampingType ReferenceTholeDipoleForce::getTholeDampingType() const {
    return _tholeDampingType;
}

void ReferenceTholeDipoleForce::setTholeDampingType(TholeDipoleForce::TholeDampingType tholeDampingType) {
    _tholeDampingType = tholeDampingType;
}

double ReferenceTholeDipoleForce::getTholeDampingParameter() const {
    return _tholeDampingParameter;
}

void ReferenceTholeDipoleForce::setTholeDampingParameter(double tholeDampingParameter) {
    _tholeDampingParameter = tholeDampingParameter;
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
                                                          const vector<int>& axisTypes,
                                                          const vector<int>& multipoleAtomZs,
                                                          const vector<int>& multipoleAtomXs,
                                                          const vector<int>& multipoleAtomYs,
                                                          const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                          vector<Vec3>& forces) {
    // Setup, including calculating induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);

    // Calculate electrostatic interactions including torques
    vector<Vec3> torques;
    initializeVec3Vector(torques);
    double energy = calculateElectrostatic(particleData, torques, forces);

    // Add polarization energy as -0.5 * μ·E_perm for NoCutoff only
    // For PME, the polarization energy is handled differently in the PME calculation
    // This is the correct form that accounts for both the interaction energy
    // and the cost to create the induced dipole
    double polarizationEnergy = 0.0;
    if (_nonbondedMethod != PME) {
        const double scale_factor = _electric / _dielectric;
        double muDotE = 0.0;
        for (unsigned int i = 0; i < _numParticles; i++) {
            muDotE += _inducedDipole[i].dot(_fixedDipoleField[i]);
        }
        polarizationEnergy = -0.5 * scale_factor * muDotE;
        energy += polarizationEnergy;
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

    const int iIndex = particleI.particleIndex;
    const int kIndex = particleK.particleIndex;

    Vec3 deltaR = particleK.position - particleI.position;
    getPeriodicDelta(deltaR);
    const double r2 = deltaR.dot(deltaR);
    if (r2 == 0.0) return 0.0;
    const double r = sqrt(r2);
    const double rInv = 1.0 / r;
    const double rInv2 = rInv * rInv;
    const double rInv3 = rInv2 * rInv;
    const double rInv4 = rInv3 * rInv;

    const Vec3 rhat = deltaR * rInv;  // points from I to K

    const double qi = particleI.charge;
    const double qk = particleK.charge;
    const Vec3& mi = particleI.dipole;
    const Vec3& mk = particleK.dipole;
    const Vec3& ui = _inducedDipole[iIndex];
    const Vec3& uk = _inducedDipole[kIndex];

    // Calculate damping factors based on damping type
    double damp1, damp2, d_damp1_dr, d_damp2_dr;
    d_damp1_dr = d_damp2_dr = 0.0;

    if (_tholeDampingType == TholeDipoleForce::NoDamping) {
        // No damping
        damp1 = damp2 = 1.0;
        d_damp1_dr = d_damp2_dr = 0.0;
    }
    else {
        // Calculate Thole damping using global damping parameter
        const double a = _tholeDampingParameter;
        double r_pol_scale;
        if (fabs(particleI.polarizability * particleK.polarizability) > 1e-12) {
            r_pol_scale = pow(particleI.polarizability * particleK.polarizability, 1.0/6.0);
        }
        else {
            r_pol_scale = 1.0;
        }
        const double u = r / r_pol_scale;

        if (_tholeDampingType == TholeDipoleForce::Exponential) {
            // ρ₁: Exponential damping
            const double ar = a * r;
            const double exp_ar = (ar < 50.0) ? exp(-ar) : 0.0;
            damp1 = 1.0 - exp_ar * (1.0 + ar + 0.5 * ar * ar);
            damp2 = damp1 - exp_ar * (ar * ar * ar / 6.0);
            if (ar < 50.0) {
                d_damp1_dr = 0.5 * a * a * a * r * r * exp_ar;
                d_damp2_dr = a * a * a * a * r * r * r * exp_ar / 6.0;
            }
        }
        else if (_tholeDampingType == TholeDipoleForce::Amoeba) {
            // ρ₂: Amoeba damping
            const double au3 = a * u * u * u;
            const double exp_au3 = (au3 < 50.0) ? exp(-au3) : 0.0;
            damp1 = 1.0 - exp_au3;
            damp2 = 1.0 - (1.0 + au3) * exp_au3;
            if (au3 < 50.0) {
                d_damp1_dr = exp_au3 * a * 3.0 * u * u / r_pol_scale;
                d_damp2_dr = exp_au3 * a * 3.0 * u * u * au3 / r_pol_scale;
            }
        }
        else { // TholeDipoleForce::Linear
            // ρ₄: Linear damping
            const double s = a * r_pol_scale;
            if (r >= s) {
                damp1 = damp2 = 1.0;
                d_damp1_dr = d_damp2_dr = 0.0;
            } else {
                const double v = r / s;
                const double v2 = v * v;
                const double v3 = v2 * v;
                damp1 = (4.0 - 3.0 * v) * v3;
                damp2 = v3 * v;
                d_damp1_dr = (12.0 * v2 - 12.0 * v3) / s;
                d_damp2_dr = 4.0 * v3 / s;
            }
        }
    }

    double energy = 0.0;
    Vec3 forceK(0.0, 0.0, 0.0);

    const double mi_dot_rhat = mi.dot(rhat);
    const double mk_dot_rhat = mk.dot(rhat);
    const double ui_dot_rhat = ui.dot(rhat);
    const double uk_dot_rhat = uk.dot(rhat);

    const double mi_dot_mk = mi.dot(mk);
    const double mi_dot_uk = mi.dot(uk);
    const double ui_dot_mk = ui.dot(mk);
    const double ui_dot_uk = ui.dot(uk);

    // Initialize electric fields for torque calculation
    Vec3 fieldAtI(0.0, 0.0, 0.0);
    Vec3 fieldAtK(0.0, 0.0, 0.0);

    // --- (1) Charge-Charge (P-P) ---
    double e_cc = mScale * qi * qk * rInv;
    Vec3 f_cc = mScale * qi * qk * rInv2 * rhat;
    energy += e_cc;
    forceK += f_cc;

    // Field at I due to charge K: E = -qk * r̂ / r²  (points from K to I, which is -rhat)
    // Field at K due to charge I: E = qi * r̂ / r²   (points from I to K, which is rhat)
    fieldAtI -= mScale * qk * rInv2 * rhat;
    fieldAtK += mScale * qi * rInv2 * rhat;

    // --- (2) Charge-Dipole (P-P) ---
    Vec3 f_cd = mScale * (
        qk * (3.0 * mi_dot_rhat * rhat - mi) * rInv3
        - qi * (3.0 * mk_dot_rhat * rhat - mk) * rInv3
    );
    double e_cd = mScale * (qk * mi_dot_rhat - qi * mk_dot_rhat) * rInv2;
    energy += e_cd;
    forceK += f_cd;

    // --- (3) Dipole-Dipole (P-P) ---
    Vec3 f_dd = mScale * rInv4 * (
        3.0 * (mi_dot_rhat * mk + mk_dot_rhat * mi + mi_dot_mk * rhat)
        - 15.0 * mi_dot_rhat * mk_dot_rhat * rhat
    );
    double e_dd = mScale * (mi_dot_mk - 3.0 * mi_dot_rhat * mk_dot_rhat) * rInv3;
    energy += e_dd;
    forceK += f_dd;

    // Field at I due to dipole K: E = [3(mk·(-r̂))(-r̂) - mk] / r³ = [3(mk·r̂)r̂ - mk] / r³
    // Field at K due to dipole I: E = [3(mi·r̂)r̂ - mi] / r³
    fieldAtI += mScale * (3.0 * mk_dot_rhat * rhat - mk) * rInv3;
    fieldAtK += mScale * (3.0 * mi_dot_rhat * rhat - mi) * rInv3;

    // --- Interactions involving Induced Dipoles ---
    // The P-I forces use mScale (same as field calculation), not iScale
    // because the polarization force is μ · ∂E/∂r where E was computed with mScale
    // But for energy, we use iScale to allow the P-P only calculation (iScale=0) to work
    if (fabs(iScale) > 0 || fabs(mScale) > 0) {
        // --- (4) Charge-Induced Dipole (P-I) with Thole damping ---
        // Force on induced dipole in damped charge field: F = (μ·∇)(damp1 * q * r/r³)
        // = -damp1 * q * (3(μ·r̂)r̂ - μ)/r³ + d_damp1_dr * q * (μ·r̂) * r̂/r²
        Vec3 f_ci_tensor = -mScale * (
            qi * damp1 * (3.0 * uk_dot_rhat * rhat - uk) * rInv3
            - qk * damp1 * (3.0 * ui_dot_rhat * rhat - ui) * rInv3
        );
        Vec3 f_ci_damp = mScale * d_damp1_dr * rInv2 * (
            qi * uk_dot_rhat - qk * ui_dot_rhat
        ) * rhat;
        Vec3 f_ci = f_ci_tensor + f_ci_damp;
        double e_ci = -iScale * damp1 * (qi * uk_dot_rhat - qk * ui_dot_rhat) * rInv2;
        energy += e_ci;
        forceK += f_ci;

        // --- (5) Permanent Dipole-Induced Dipole (P-I) with Thole damping ---
        // damp1 for isotropic (μ·μ) term, damp2 for anisotropic (μ·r̂)(μ·r̂) term
        Vec3 f_di_tensor = mScale * rInv4 * (
            // Anisotropic terms from -3(mi·r̂)(uk·r̂)/r³ and -3(mk·r̂)(ui·r̂)/r³
            damp2 * 3.0 * (mi_dot_rhat * uk + uk_dot_rhat * mi)
            - damp2 * 15.0 * mi_dot_rhat * uk_dot_rhat * rhat
            + damp2 * 3.0 * (mk_dot_rhat * ui + ui_dot_rhat * mk)
            - damp2 * 15.0 * mk_dot_rhat * ui_dot_rhat * rhat
            // Isotropic terms from (mi·uk)/r³ and (mk·ui)/r³
            + damp1 * 3.0 * (mi_dot_uk + ui_dot_mk) * rhat
        );
        Vec3 f_di_damp = -mScale * rInv3 * (
            d_damp1_dr * (mi_dot_uk + ui_dot_mk)
            - 3.0 * d_damp2_dr * (mi_dot_rhat * uk_dot_rhat + ui_dot_rhat * mk_dot_rhat)
        ) * rhat;
        Vec3 f_di = f_di_tensor + f_di_damp;
        double e_di = iScale * (
            damp1 * (mi_dot_uk + ui_dot_mk) * rInv3
            - 3.0 * damp2 * (mi_dot_rhat * uk_dot_rhat + ui_dot_rhat * mk_dot_rhat) * rInv3
        );
        energy += e_di;
        forceK += f_di;

        // Add induced dipole contributions to fields with damping (use mScale for consistency)
        fieldAtI += mScale * (3.0 * damp2 * uk_dot_rhat * rhat - damp1 * uk) * rInv3;
        fieldAtK += mScale * (3.0 * damp2 * ui_dot_rhat * rhat - damp1 * ui) * rInv3;

        // --- (6) Induced Dipole-Induced Dipole (I-I) ---
        if (_polarizationType == Mutual && particleI.polarizability > 0 && particleK.polarizability > 0) {
            // Energy: E = damp1 * (ui·uk) * r⁻³ - 3 * damp2 * (ui·r̂)(uk·r̂) * r⁻³
            double e_ii = iScale * (
                damp1 * ui_dot_uk * rInv3
                - 3.0 * damp2 * ui_dot_rhat * uk_dot_rhat * rInv3
            );

            // Force tensor: damp1 for isotropic, damp2 for anisotropic
            Vec3 f_ii_tensor = iScale * rInv4 * (
                // Anisotropic terms
                damp2 * 3.0 * (ui_dot_rhat * uk + uk_dot_rhat * ui)
                - damp2 * 15.0 * ui_dot_rhat * uk_dot_rhat * rhat
                // Isotropic term
                + damp1 * 3.0 * ui_dot_uk * rhat
            );

            // Damping derivative contribution
            Vec3 f_ii_damp = -iScale * rInv3 * (
                d_damp1_dr * ui_dot_uk
                - 3.0 * d_damp2_dr * ui_dot_rhat * uk_dot_rhat
            ) * rhat;

            Vec3 f_ii = f_ii_tensor + f_ii_damp;
            energy += e_ii;
            forceK += f_ii;
        }
    }

    // Calculate torques as τ = μ × E
    // These are the torques on the permanent dipoles due to the electric fields
    Vec3 torqueI = mi.cross(fieldAtI);
    Vec3 torqueK = mk.cross(fieldAtK);

    const double energyTotal = _electric * energy / _dielectric;
    const Vec3 forceTotal = _electric * forceK / _dielectric;
    const Vec3 torqueITotal = _electric * torqueI / _dielectric;
    const Vec3 torqueKTotal = _electric * torqueK / _dielectric;

    forces[iIndex] -= forceTotal;
    forces[kIndex] += forceTotal;
    torques[iIndex] += torqueITotal;
    torques[kIndex] += torqueKTotal;

    return energyTotal;
}

double ReferenceTholeDipoleForce::calculateElectrostatic(
    const vector<TholeDipoleParticleData>& particleData,
    vector<Vec3>& torques,
    vector<Vec3>& forces) {

    double energy = 0.0;
    double energyPP = 0.0;
    double energyPI = 0.0;

    // Also compute μ·E directly from pairwise interactions for debugging
    double muDotE_charge_pairwise = 0.0;
    double muDotE_dipole_pairwise = 0.0;

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

            // Calculate P-P only (iScale=0)
            vector<Vec3> tempForces(forces.size());
            vector<Vec3> tempTorques(torques.size());
            double ePP = calculateElectrostaticPairIxn(particleData[i], particleData[j],
                                                       mScale, 0.0, tempForces, tempTorques);
            energyPP += ePP;

            // Calculate full interaction (for forces) but only add P-P to energy
            // The P-I energy is NOT computed pairwise - it's computed as -0.5*μ·E
            double eFull = calculateElectrostaticPairIxn(particleData[i], particleData[j],
                                                         mScale, iScale, forces, torques);
            energyPI += (eFull - ePP);  // For debug only
            energy += ePP;  // Only add P-P energy, not P-I

            // Manually compute μ·E contributions for this pair
            if (_numParticles == 8 && iScale > 0) {
                Vec3 deltaR = particleData[j].position - particleData[i].position;
                getPeriodicDelta(deltaR);
                double r2 = deltaR.dot(deltaR);
                double r = sqrt(r2);
                double rInv = 1.0/r, rInv2 = rInv*rInv, rInv3 = rInv2*rInv;
                Vec3 rhat = deltaR * rInv;

                // Field at i from charge j: E = qj * (-rhat) / r² (rhat points from i to j, so field points from j to i)
                // No wait, let me use the same convention as _fixedDipoleField
                // In calculateFixedDipoleFieldPairIxn: rVec = rI - rJ, rHat = rVec/r (points from J to I)
                // Here: deltaR = rJ - rI, rhat = deltaR/r (points from I to J)
                // So rHat_field = -rhat
                Vec3 rHat_field = -rhat;  // points from J to I (same as field calculation)

                // Field at I from charge J
                Vec3 E_qj_at_i = particleData[j].charge * rHat_field * rInv2;
                // Field at J from charge I
                Vec3 E_qi_at_j = particleData[i].charge * (-rHat_field) * rInv2;

                // μ·E contributions (note: not multiplied by _electric yet)
                double ui_dot_Eqj = _inducedDipole[i].dot(E_qj_at_i);
                double uj_dot_Eqi = _inducedDipole[j].dot(E_qi_at_j);
                muDotE_charge_pairwise += mScale * (ui_dot_Eqj + uj_dot_Eqi);

                // Field at I from dipole J
                double mj_dot_rHat = particleData[j].dipole.dot(rHat_field);
                Vec3 E_mj_at_i = (3.0 * mj_dot_rHat * rHat_field - particleData[j].dipole) * rInv3;
                // Field at J from dipole I
                double mi_dot_rHat = particleData[i].dipole.dot(-rHat_field);
                Vec3 E_mi_at_j = (3.0 * mi_dot_rHat * (-rHat_field) - particleData[i].dipole) * rInv3;

                double ui_dot_Emj = _inducedDipole[i].dot(E_mj_at_i);
                double uj_dot_Emi = _inducedDipole[j].dot(E_mi_at_j);
                muDotE_dipole_pairwise += mScale * (ui_dot_Emj + uj_dot_Emi);
            }
        }
    }

    return energy;
}

void ReferenceTholeDipoleForce::loadParticleData(const vector<Vec3>& particlePositions,
                                                 const vector<double>& charges,
                                                 const vector<double>& dipoles,
                                                 const vector<double>& polarizabilities,
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
    if (norm > 1e-12) {
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
    
    // Debug output for small systems
    // if (particleI.particleIndex <= 5) {
    //     printf("  Particle %d: Original dipole (%.6f, %.6f, %.6f), axisType=%d\n",
    //            particleI.particleIndex, particleI.dipole[0], particleI.dipole[1], particleI.dipole[2], axisType);
    // }
    
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
    
    // Debug output for small systems
    // if (particleI.particleIndex <= 5) {
    //     printf("    vectorX: (%.6f, %.6f, %.6f)\n", vectorX[0], vectorX[1], vectorX[2]);
    //     printf("    vectorY: (%.6f, %.6f, %.6f)\n", vectorY[0], vectorY[1], vectorY[2]);
    //     printf("    vectorZ: (%.6f, %.6f, %.6f)\n", vectorZ[0], vectorZ[1], vectorZ[2]);
    // }
    
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
    // if (particleI.particleIndex <= 5) {
    //     printf("  Final transformed dipole: (%.6f, %.6f, %.6f)\n", labDipole[0], labDipole[1], labDipole[2]);
    // }

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

void ReferenceTholeDipoleForce::setupScaleMaps(const vector<vector<vector<int>>>& multipoleCovalentInfo) {
    _scaleMaps[M_SCALE].resize(_numParticles);
    _scaleMaps[I_SCALE].resize(_numParticles);
    _maxScaleIndex.resize(_numParticles, 0);

    for (unsigned int i = 0; i < _numParticles; i++) {
        for (int t = TholeDipoleForce::Covalent12; t <= TholeDipoleForce::Covalent15; t++) {
            const vector<int>& mList = multipoleCovalentInfo[i][t];
            const vector<int>& iList = multipoleCovalentInfo[i][t];

            for (int atom : mList) {
                if (atom >= 0) {
                    _scaleMaps[M_SCALE][i][atom] = _mScale[t];  // t = 0,1,2,3
                    if (atom > _maxScaleIndex[i]) _maxScaleIndex[i] = atom;
                }
            }
            for (int atom : iList) {
                if (atom >= 0) {
                    _scaleMaps[I_SCALE][i][atom] = _iScale[t];
                    if (atom > _maxScaleIndex[i]) _maxScaleIndex[i] = atom;
                }
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

void ReferenceTholeDipoleForce::calculateFixedDipoleFieldPairIxn(
    const TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData& particleJ,
    double mScale, double iScale) {

    if (particleI.particleIndex == particleJ.particleIndex) {
        return;
    }

    // Vector from source (J) to target (I)
    Vec3 rVec = particleI.position - particleJ.position;
    getPeriodicDelta(rVec);
    double r = sqrt(rVec.dot(rVec));
    if (r == 0.0) return;

    const double rInv = 1.0 / r;
    const double rInv2 = rInv * rInv;
    const double rInv3 = rInv2 * rInv;
    Vec3 rHat = rVec * rInv;

    // Calculate Thole damping factors for the fixed dipole field
    // This matches AMOEBA's getAndScaleInverseRs function
    double damp1 = 1.0, damp2 = 1.0;
    if (_tholeDampingType != TholeDipoleForce::NoDamping) {
        const double a = _tholeDampingParameter;
        double r_pol_scale;
        if (fabs(particleI.polarizability * particleJ.polarizability) > 1e-12) {
            r_pol_scale = pow(particleI.polarizability * particleJ.polarizability, 1.0/6.0);
        }
        else {
            r_pol_scale = 1.0;
        }

        if (_tholeDampingType == TholeDipoleForce::Amoeba) {
            const double u = r / r_pol_scale;
            const double au3 = a * u * u * u;
            const double exp_au3 = (au3 < 50.0) ? exp(-au3) : 0.0;
            damp1 = 1.0 - exp_au3;
            damp2 = 1.0 - (1.0 + au3) * exp_au3;
        }
        else if (_tholeDampingType == TholeDipoleForce::Exponential) {
            const double ar = a * r;
            const double exp_ar = (ar < 50.0) ? exp(-ar) : 0.0;
            damp1 = 1.0 - exp_ar * (1.0 + ar + 0.5 * ar * ar);
            damp2 = damp1 - exp_ar * (ar * ar * ar / 6.0);
        }
        else if (_tholeDampingType == TholeDipoleForce::Linear) {
            const double s = a * r_pol_scale;
            if (r < s) {
                const double v = r / s;
                const double v3 = v * v * v;
                damp1 = (4.0 - 3.0 * v) * v3;
                damp2 = v3 * v;
            }
        }
    }

    // --- Field at I due to J (permanent charge + permanent dipole) ---
    // In AMOEBA, rr3 (damped 1/r³) multiplies both charge and dipole terms
    // But rr3 appears with deltaR for charge: rr3 * q * deltaR = q * deltaR / r³ = q * rHat / r²
    // So for charges, we actually want undamped 1/r² (since rr3*r = 1/r²)
    // However, looking at AMOEBA more carefully: factor = rr3*q*deltaR is damped
    // The "r" that would cancel is already in deltaR, so the charge field IS damped by damp1
    Vec3 fieldAtI(0.0, 0.0, 0.0);
    // Charge contribution with Thole damping (matching AMOEBA)
    fieldAtI += rHat * (damp1 * particleJ.charge * rInv2);
    // Dipole contribution with Thole damping:
    // E = damp1 * μ/r³ - damp2 * 3(μ·rHat)rHat/r³
    // Rewritten: E = [3*damp2*(μ·rHat)rHat - damp1*μ] / r³
    double muJ_dot_rHat = particleJ.dipole.dot(rHat);
    fieldAtI += (3.0 * damp2 * muJ_dot_rHat * rHat - damp1 * particleJ.dipole) * rInv3;

    _fixedDipoleField[particleI.particleIndex] += fieldAtI * mScale;

    // --- Field at J due to I ---
    Vec3 fieldAtJ(0.0, 0.0, 0.0);
    // Vector from I to J is -rVec, so rHatJI = -rHat
    Vec3 rHatJI = -rHat;

    // Charge contribution with Thole damping
    fieldAtJ += rHatJI * (damp1 * particleI.charge * rInv2);
    // Dipole contribution with Thole damping
    double muI_dot_rHatJI = particleI.dipole.dot(rHatJI);
    fieldAtJ += (3.0 * damp2 * muI_dot_rHatJI * rHatJI - damp1 * particleI.dipole) * rInv3;

    _fixedDipoleField[particleJ.particleIndex] += fieldAtJ * mScale;
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
        convergeInducedDipolesByPCG(particleData);
    }

    // For Extrapolated polarization, use perturbation theory
    if (_polarizationType == Extrapolated) {
        convergeInducedDipolesByExtrapolation(particleData);
    }
}

void ReferenceTholeDipoleForce::calculateInducedDipolePairIxn(
    unsigned int particleI,
    unsigned int particleJ,
    double r,
    double rInv3,
    double rInv5,
    const Vec3& deltaR,
    double iScale,
    double polarizabilityI,
    double polarizabilityJ,
    const vector<Vec3>& inducedDipole,
    vector<Vec3>& field) const {

    if (_polarizationType == Direct) {
        // E = [-μ/r³ + 3(μ·r̂)r̂/r³]
        double dDotDelta = rInv5 * (inducedDipole[particleJ].dot(deltaR));
        field[particleI] += -inducedDipole[particleJ] * rInv3 + deltaR * (3.0 * dDotDelta);

        dDotDelta = rInv5 * (inducedDipole[particleI].dot(deltaR));
        field[particleJ] += -inducedDipole[particleI] * rInv3 + deltaR * (3.0 * dDotDelta);
    }
    else {
        // Calculate damping factors based on damping type
        double damp1, damp2;

        if (_tholeDampingType == TholeDipoleForce::NoDamping) {
            // No damping
            damp1 = damp2 = 1.0;
        }
        else {
            // Use global Thole damping parameter
            const double a = _tholeDampingParameter;
            double r_pol_scale;
            if (fabs(polarizabilityI * polarizabilityJ) > 1e-12) {
                r_pol_scale = pow(polarizabilityI * polarizabilityJ, 1.0/6.0);
            }
            else {
                r_pol_scale = 1.0;
            }
            const double u = r / r_pol_scale;

            if (_tholeDampingType == TholeDipoleForce::Exponential) {
                // ρ₁: Exponential damping
                const double ar = a * r;
                const double exp_ar = (ar < 50.0) ? exp(-ar) : 0.0;
                damp1 = 1.0 - exp_ar * (1.0 + ar + 0.5 * ar * ar);
                damp2 = damp1 - exp_ar * (ar * ar * ar / 6.0);
            }
            else if (_tholeDampingType == TholeDipoleForce::Amoeba) {
                // ρ₂: Amoeba damping
                const double au3 = a * u * u * u;
                const double exp_au3 = (au3 < 50.0) ? exp(-au3) : 0.0;
                damp1 = 1.0 - exp_au3;
                damp2 = 1.0 - (1.0 + au3) * exp_au3;
            }
            else { // TholeDipoleForce::Linear
                // ρ₄: Linear damping
                const double s = a * r_pol_scale;
                if (r >= s) {
                    damp1 = damp2 = 1.0;
                } else {
                    const double v = r / s;
                    damp1 = (4.0 - 3.0 * v) * v * v * v;
                    damp2 = v * v * v * v;
                }
            }
        }

        // Get induced dipoles
        const Vec3& uj = inducedDipole[particleJ];
        const Vec3& ui = inducedDipole[particleI];

        // Field on I due to J with correct damping
        // E = [-μ/r³ + 3(μ·r̂)r̂/r³] with Thole damping
        Vec3 fieldI = -damp1 * uj * rInv3 + 3.0 * damp2 * deltaR * (uj.dot(deltaR)) * rInv5;

        // Field on J due to I with correct damping
        Vec3 fieldJ = -damp1 * ui * rInv3 + 3.0 * damp2 * deltaR * (ui.dot(deltaR)) * rInv5;

        // Apply scaling
        field[particleI] += iScale * fieldI;
        field[particleJ] += iScale * fieldJ;
    }
}

void ReferenceTholeDipoleForce::calculateInducedDipoleFields(
    const vector<TholeDipoleParticleData>& particleData,
    const vector<Vec3>& inducedDipoles,
    vector<Vec3>& inducedDipoleField) {

    initializeVec3Vector(inducedDipoleField);
    
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            Vec3 deltaR = particleData[j].position - particleData[i].position;
            getPeriodicDelta(deltaR);
            double r = sqrt(deltaR.dot(deltaR));

            if (r < 1e-8) continue;

            const double rInv = 1.0 / r;
            const double rInv3 = rInv * rInv * rInv;
            const double rInv5 = rInv3 * rInv * rInv;
            
            double iScale = 1.0;
            if (j <= _maxScaleIndex[i]) {
                iScale = getScaleFactor(i, j, I_SCALE);
            }
            
            if (iScale != 0.0) {
                calculateInducedDipolePairIxn(
                    i, j,
                    r, rInv3, rInv5,
                    deltaR,
                    iScale,
                    particleData[i].polarizability,
                    particleData[j].polarizability,
                    inducedDipoles,
                    inducedDipoleField
                );
            }
        }
    }
}

void ReferenceTholeDipoleForce::convergeInducedDipolesByPCG(
    const vector<TholeDipoleParticleData>& particleData) {

    const int maxIter = _maximumMutualInducedDipoleIterations;
    const double tol = _mutualInducedDipoleTargetEpsilon;
    const int n = _numParticles;

    // Allocate working vectors
    vector<Vec3> r(n);          // residual
    vector<Vec3> z(n);          // preconditioned residual
    vector<Vec3> p(n);          // search direction
    vector<Vec3> Ap(n);         // A * p
    vector<Vec3> inducedDipoleField(n);

    // Initial guess is already in _inducedDipole (from direct term)

    // Compute initial residual: r = b - A * mu
    // where b = alpha * E_fixed, and A*mu = alpha * (E_fixed + E_induced(mu)) - alpha*E_induced(mu) ??? 
    // Simpler: define residual as:
    //   r = alpha * (E_fixed + E_induced(mu)) - mu
    // because at convergence: mu = alpha*(E_fixed + E_induced(mu)) → r = 0

    calculateInducedDipoleFields(particleData, _inducedDipole, inducedDipoleField);
    double r_dot_z = 0.0;
    for (int i = 0; i < n; ++i) {
        Vec3 E_total = _fixedDipoleField[i] + inducedDipoleField[i];
        Vec3 b = particleData[i].polarizability * E_total;
        r[i] = b - _inducedDipole[i];  // residual

        // Diagonal preconditioner: M_ii = 1 (since system is mu = alpha*(...), precondition with 1/alpha is natural)
        // But better: use M^{-1} = 1 / (1 + alpha * T_ii) ≈ 1, or simply M = I.
        // However, a common choice is to precondition with polarizability:
        //   z = r / (1.0)  → no preconditioning
        // OR: since mu_i ≈ alpha_i * E_i, scale by alpha_i
        // We'll use diagonal preconditioner: z_i = r_i / (1.0) → identity (simplest)
        // But for better convergence, use: z_i = r_i * (1.0 / (1.0 - 0))? Not helpful.
        // Instead, note: the diagonal of A is ~1/alpha_i in some formulations.
        // Here, we precondition by polarizability: z = r * (1.0 / alpha_i) if alpha_i > 0
        double alpha_i = particleData[i].polarizability;
        if (alpha_i > 1e-12) {
            z[i] = r[i] / alpha_i;
        } else {
            z[i] = Vec3(0.0, 0.0, 0.0);
        }
        p[i] = z[i];
        r_dot_z += r[i].dot(z[i]);
    }

    double epsilon = sqrt(r_dot_z / (3.0 * n)); // 3 components per particle

    if (epsilon <= tol) {
        _mutualInducedDipoleConverged = 1;
        _mutualInducedDipoleIterations = 0;
        _mutualInducedDipoleEpsilon = epsilon;
        return;
    }

    for (_mutualInducedDipoleIterations = 0;
         _mutualInducedDipoleIterations < maxIter;
         _mutualInducedDipoleIterations++) {

        // Compute Ap = A * p
        // A*p = p - alpha * T * p   → but easier: 
        // Since A mu = mu - alpha*(E_induced_from_mu)
        // So A p = p - alpha * E_induced(p)
        calculateInducedDipoleFields(particleData, p, Ap); // Ap_temp = T * p
        for (int i = 0; i < n; ++i) {
            // A*p = p - alpha_i * (T * p)_i
            Ap[i] = p[i] - particleData[i].polarizability * Ap[i];
        }

        // Compute alpha_cg = (r^T z) / (p^T A p)
        double p_dot_Ap = 0.0;
        for (int i = 0; i < n; ++i) {
            p_dot_Ap += p[i].dot(Ap[i]);
        }

        if (fabs(p_dot_Ap) < 1e-16) break; // avoid division by zero

        double alpha_cg = r_dot_z / p_dot_Ap;

        // Update solution: mu = mu + alpha_cg * p
        for (int i = 0; i < n; ++i) {
            _inducedDipole[i] += alpha_cg * p[i];
        }

        // Update residual: r = r - alpha_cg * A p
        double r_new_dot_z_new = 0.0;
        for (int i = 0; i < n; ++i) {
            r[i] -= alpha_cg * Ap[i];
            double alpha_i = particleData[i].polarizability;
            if (alpha_i > 1e-12) {
                z[i] = r[i] / alpha_i;
            } else {
                z[i] = Vec3(0.0, 0.0, 0.0);
            }
            r_new_dot_z_new += r[i].dot(z[i]);
        }

        epsilon = sqrt(r_new_dot_z_new / (3.0 * n));
        if (epsilon <= tol) {
            _mutualInducedDipoleConverged = 1;
            _mutualInducedDipoleEpsilon = epsilon;
            return;
        }

        // Compute beta = (r_new^T z_new) / (r^T z)
        double beta = r_new_dot_z_new / r_dot_z;
        r_dot_z = r_new_dot_z_new;

        // Update search direction: p = z + beta * p
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
    }

    _mutualInducedDipoleEpsilon = epsilon;
    _mutualInducedDipoleConverged = (epsilon <= tol) ? 1 : 0;
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
        
        for (int i = 0; i < 3; i++) {
            double du = vectorUV[i]*dphi[1]/(normU*sinUV) + vectorUW[i]*dphi[2]/normU;
            forces[particleU.particleIndex][i] -= du;
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
    const vector<int>& axisTypes,
    const vector<int>& multipoleAtomZs,
    const vector<int>& multipoleAtomXs,
    const vector<int>& multipoleAtomYs,
    const vector<vector<vector<int>>>& multipoleCovalentInfo,
    vector<TholeDipoleParticleData>& particleData) {

    _numParticles = particlePositions.size();

    // Load particle data
    loadParticleData(particlePositions, charges, dipoles, polarizabilities, particleData);
    
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
                                                        const vector<int>& axisTypes,
                                                        const vector<int>& multipoleAtomZs,
                                                        const vector<int>& multipoleAtomXs,
                                                        const vector<int>& multipoleAtomYs,
                                                        const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                        vector<Vec3>& outputInducedDipoles) {
    // Setup, including calculating induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities,
          axisTypes, multipoleAtomZs, multipoleAtomXs, multipoleAtomYs,
          multipoleCovalentInfo, particleData);

    outputInducedDipoles = _inducedDipole;

}

void ReferenceTholeDipoleForce::calculateLabFramePermanentDipoles(const vector<Vec3>& particlePositions,
                                                                  const vector<double>& charges,
                                                                  const vector<double>& dipoles,
                                                                  const vector<double>& polarizabilities,
                                                                  const vector<int>& axisTypes,
                                                                  const vector<int>& multipoleAtomZs,
                                                                  const vector<int>& multipoleAtomXs,
                                                                  const vector<int>& multipoleAtomYs,
                                                                  const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                                  vector<Vec3>& outputRotatedPermanentDipoles) {
    // Setup, including rotating permanent dipoles to lab frame
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities,
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
                                                      const vector<int>& axisTypes,
                                                      const vector<int>& multipoleAtomZs,
                                                      const vector<int>& multipoleAtomXs,
                                                      const vector<int>& multipoleAtomYs,
                                                      const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                      vector<Vec3>& outputTotalDipoles) {
    // Setup, including calculating permanent and induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities,
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
                                                                           const vector<int>& axisTypes,
                                                                           const vector<int>& multipoleAtomZs,
                                                                           const vector<int>& multipoleAtomXs,
                                                                           const vector<int>& multipoleAtomYs,
                                                                           const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                                           vector<double>& outputMultipoleMoments) {
    // Setup, including calculating induced dipoles
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities,
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
                                                                const vector<int>& axisTypes,
                                                                const vector<int>& multipoleAtomZs,
                                                                const vector<int>& multipoleAtomXs,
                                                                const vector<int>& multipoleAtomYs,
                                                                const vector<vector<vector<int>>>& multipoleCovalentInfo,
                                                                const vector<Vec3>& inputGrid,
                                                                vector<double>& outputPotential) {
    // Setup particle data
    vector<TholeDipoleParticleData> particleData;
    setup(particlePositions, charges, dipoles, polarizabilities,
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

