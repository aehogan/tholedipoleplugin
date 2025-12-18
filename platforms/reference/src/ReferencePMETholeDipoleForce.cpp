
#include "ReferencePMETholeDipoleForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "pocketfft_hdronly.h"
#include "openmm/internal/MSVC_erfc.h"
#include <cmath>
#include <sstream>

using namespace TholeDipolePlugin;
using namespace OpenMM;
using namespace std;

const int ReferencePMETholeDipoleForce::THOLE_PME_ORDER = 5;
const double ReferencePMETholeDipoleForce::SQRT_PI = 1.77245385091;

ReferencePMETholeDipoleForce::ReferencePMETholeDipoleForce() :
               ReferenceTholeDipoleForce(),
               _cutoffDistance(1.0), _cutoffDistanceSquared(1.0),
               _pmeGridSize(0), _totalGridSize(0), _alphaEwald(0.0),
               _fixedMultipoleRecipEnergy(0.0), _inducedDipoleRecipEnergy(0.0)
{
    _pmeGrid = NULL;
    _pmeGridDimensions = IntVec(-1, -1, -1);

    setNonbondedMethod(PME);
}

ReferencePMETholeDipoleForce::~ReferencePMETholeDipoleForce()
{
    if (_pmeGrid) {
        delete[] _pmeGrid;
    }
}

double ReferencePMETholeDipoleForce::getCutoffDistance() const
{
     return _cutoffDistance;
}

void ReferencePMETholeDipoleForce::setCutoffDistance(double cutoffDistance)
{
     _cutoffDistance        = cutoffDistance;
     _cutoffDistanceSquared = cutoffDistance*cutoffDistance;
}

double ReferencePMETholeDipoleForce::getAlphaEwald() const
{
     return _alphaEwald;
}

void ReferencePMETholeDipoleForce::setAlphaEwald(double alphaEwald)
{
     _alphaEwald = alphaEwald;
}

void ReferencePMETholeDipoleForce::computePmeTholeDampingFactors(double r, double polarizabilityI, double polarizabilityJ,
                                                                  double& thole_c, double& thole_d0, double& thole_d1,
                                                                  double& dthole_c, double& dthole_d0, double& dthole_d1) const
{
    thole_c = thole_d0 = thole_d1 = 1.0;
    dthole_c = dthole_d0 = dthole_d1 = 1.0;

    if (_tholeDampingType == TholeDipoleForce::NoDamping) {
        return;
    }

    const double a = _tholeDampingParameter;
    double r_pol_scale;
    if (fabs(polarizabilityI * polarizabilityJ) > 1e-12) {
        r_pol_scale = pow(polarizabilityI * polarizabilityJ, 1.0/6.0);
    } else {
        r_pol_scale = 1.0;
    }
    const double u = r / r_pol_scale;

    if (_tholeDampingType == TholeDipoleForce::Exponential) {
        const double ar = a * r;
        const double exp_ar = (ar < 50.0) ? exp(-ar) : 0.0;
        thole_c = 1.0 - exp_ar * (1.0 + ar + 0.5 * ar * ar);
        thole_d0 = thole_c;
        thole_d1 = thole_c - exp_ar * (ar * ar * ar / 6.0);
        dthole_c = thole_c;
        dthole_d0 = thole_d0;
        dthole_d1 = thole_d1;
    }
    else if (_tholeDampingType == TholeDipoleForce::Amoeba) {
        const double au3 = a * u * u * u;
        const double exp_au3 = (au3 < 50.0) ? exp(-au3) : 0.0;
        const double a2u6 = au3 * au3;
        thole_c = 1.0 - exp_au3;
        thole_d0 = 1.0 - exp_au3 * (1.0 + 1.5 * au3);
        thole_d1 = 1.0 - exp_au3;
        dthole_c = 1.0 - exp_au3 * (1.0 + 1.5 * au3);
        dthole_d0 = 1.0 - exp_au3 * (1.0 + au3 + 1.5 * a2u6);
        dthole_d1 = 1.0 - exp_au3 * (1.0 + au3);
    }
    else { // TholeDipoleForce::Linear
        const double s = a * r_pol_scale;
        if (r < s) {
            const double v = r / s;
            const double v2 = v * v;
            const double v3 = v2 * v;
            thole_c = (4.0 - 3.0 * v) * v3;
            thole_d0 = thole_c;
            thole_d1 = v3 * v;
            dthole_c = thole_c;
            dthole_d0 = thole_d0;
            dthole_d1 = thole_d1;
        }
    }
}

void ReferencePMETholeDipoleForce::getPmeGridDimensions(vector<int>& pmeGridDimensions) const
{
    pmeGridDimensions.resize(3);
    pmeGridDimensions[0] = _pmeGridDimensions[0];
    pmeGridDimensions[1] = _pmeGridDimensions[1];
    pmeGridDimensions[2] = _pmeGridDimensions[2];
}

void ReferencePMETholeDipoleForce::setPmeGridDimensions(vector<int>& pmeGridDimensions)
{
    if ((pmeGridDimensions[0] == _pmeGridDimensions[0]) &&
        (pmeGridDimensions[1] == _pmeGridDimensions[1]) &&
        (pmeGridDimensions[2] == _pmeGridDimensions[2]))
        return;

    _pmeGridDimensions[0] = pmeGridDimensions[0];
    _pmeGridDimensions[1] = pmeGridDimensions[1];
    _pmeGridDimensions[2] = pmeGridDimensions[2];

    initializeBSplineModuli();
}

void ReferencePMETholeDipoleForce::setPeriodicBoxSize(OpenMM::Vec3* vectors)
{
    if (vectors[0][0] == 0.0 || vectors[1][1] == 0.0 || vectors[2][2] == 0.0) {
        stringstream message;
        message << "Box size of zero is invalid.";
        throw OpenMMException(message.str());
    }

    _periodicBoxVectors[0] = vectors[0];
    _periodicBoxVectors[1] = vectors[1];
    _periodicBoxVectors[2] = vectors[2];

    double determinant = _computeBoxVolume();

    if (determinant <= 0) {
        stringstream message;
        message << "Box vectors form invalid cell with non-positive volume.";
        throw OpenMMException(message.str());
    }

    double scale = 1.0/determinant;

    _recipBoxVectors[0] = Vec3(vectors[1][1]*vectors[2][2] - vectors[1][2]*vectors[2][1],
                                vectors[0][2]*vectors[2][1] - vectors[0][1]*vectors[2][2],
                                vectors[0][1]*vectors[1][2] - vectors[0][2]*vectors[1][1])*scale;
    _recipBoxVectors[1] = Vec3(vectors[1][2]*vectors[2][0] - vectors[1][0]*vectors[2][2],
                                vectors[0][0]*vectors[2][2] - vectors[0][2]*vectors[2][0],
                                vectors[0][2]*vectors[1][0] - vectors[0][0]*vectors[1][2])*scale;
    _recipBoxVectors[2] = Vec3(vectors[1][0]*vectors[2][1] - vectors[1][1]*vectors[2][0],
                                vectors[0][1]*vectors[2][0] - vectors[0][0]*vectors[2][1],
                                vectors[0][0]*vectors[1][1] - vectors[0][1]*vectors[1][0])*scale;
}

void ReferencePMETholeDipoleForce::resizePmeArrays()
{
    _totalGridSize = _pmeGridDimensions[0]*_pmeGridDimensions[1]*_pmeGridDimensions[2];
    if (_pmeGridSize < _totalGridSize) {
        if (_pmeGrid) {
            delete[] _pmeGrid;
        }
        _pmeGrid      = new complex<double>[_totalGridSize];
        _pmeGridSize  = _totalGridSize;
    }

    for (unsigned int ii = 0; ii < 3; ii++) {
       _pmeBsplineModuli[ii].resize(_pmeGridDimensions[ii]);
       _thetai[ii].resize(THOLE_PME_ORDER*_numParticles);
    }

    _iGrid.resize(_numParticles);
    _particleFraction.resize(_numParticles);

    _phi.resize(10*_numParticles);
    _phid.resize(10*_numParticles);  // potential, 3 first derivs, 6 second derivs
}

void ReferencePMETholeDipoleForce::initializePmeGrid()
{
    if (_pmeGrid == NULL)
        return;

    for (int jj = 0; jj < _totalGridSize; jj++)
        _pmeGrid[jj] = complex<double>(0, 0);
}

void ReferencePMETholeDipoleForce::getPeriodicDelta(Vec3& deltaR) const
{
    double lambda2 = deltaR[0]*_recipBoxVectors[2][0] + deltaR[1]*_recipBoxVectors[2][1] + deltaR[2]*_recipBoxVectors[2][2];
    double lambda1 = deltaR[0]*_recipBoxVectors[1][0] + deltaR[1]*_recipBoxVectors[1][1] + deltaR[2]*_recipBoxVectors[1][2];
    double lambda0 = deltaR[0]*_recipBoxVectors[0][0] + deltaR[1]*_recipBoxVectors[0][1] + deltaR[2]*_recipBoxVectors[0][2];

    int n2 = (int) floor(lambda2 + 0.5);
    deltaR -= _periodicBoxVectors[2] * n2;

    int n1 = (int) floor(lambda1 + 0.5);
    deltaR -= _periodicBoxVectors[1] * n1;

    int n0 = (int) floor(lambda0 + 0.5);
    deltaR -= _periodicBoxVectors[0] * n0;
}

void ReferencePMETholeDipoleForce::initializeBSplineModuli()
{
    const int order = THOLE_PME_ORDER;
    int maxSize = max(max(_pmeGridDimensions[0], _pmeGridDimensions[1]), _pmeGridDimensions[2]);
    if (maxSize < order)
        throw OpenMMException("PME grid dimensions must be at least " + std::to_string(order));

    for (int dim = 0; dim < 3; dim++) {
        _pmeBsplineModuli[dim].resize(_pmeGridDimensions[dim]);
    }

    double array[order];
    vector<double> bsarray(maxSize + 1, 0.0);

    double x = 0.0;
    array[0] = 1.0 - x;
    array[1] = x;
    for (int k = 2; k < order; k++) {
        double denom = 1.0/k;
        array[k] = x*array[k-1]*denom;
        for (int i = 1; i < k; i++) {
            array[k-i] = ((x+i)*array[k-i-1] + ((k-i+1)-x)*array[k-i])*denom;
        }
        array[0] = (1.0-x)*array[0]*denom;
    }

    for (int i = 2; i <= order+1; i++) {
        bsarray[i] = array[i-2];
    }

    for (int dim = 0; dim < 3; dim++) {
        int ndata = _pmeGridDimensions[dim];
        double factor = 2.0 * M_PI / ndata;

        for (int i = 0; i < ndata; i++) {
            double sum1 = 0.0, sum2 = 0.0;
            for (int j = 1; j <= ndata; j++) {
                double arg = factor * i * (j-1);
                sum1 += bsarray[j] * cos(arg);
                sum2 += bsarray[j] * sin(arg);
            }
            _pmeBsplineModuli[dim][i] = sum1*sum1 + sum2*sum2;
        }

        double eps = 1.0e-7;
        if (_pmeBsplineModuli[dim][0] < eps) {
            _pmeBsplineModuli[dim][0] = 0.5 * _pmeBsplineModuli[dim][1];
        }
        for (int i = 1; i < ndata-1; i++) {
            if (_pmeBsplineModuli[dim][i] < eps) {
                _pmeBsplineModuli[dim][i] = 0.5 * (_pmeBsplineModuli[dim][i-1] +
                                                   _pmeBsplineModuli[dim][i+1]);
            }
        }
        if (_pmeBsplineModuli[dim][ndata-1] < eps) {
            _pmeBsplineModuli[dim][ndata-1] = 0.5 * _pmeBsplineModuli[dim][ndata-2];
        }

        // Compute and apply the optimal zeta coefficient
        int jcut = 50;
        for (int i = 1; i <= ndata; i++) {
            int k = i - 1;
            if (i > ndata/2)
                k = k - ndata;
            double zeta;
            if (k == 0)
                zeta = 1.0;
            else {
                double sum1 = 1.0;
                double sum2 = 1.0;
                double factor2 = M_PI*k/ndata;
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor2/(factor2+M_PI*j);
                    sum1 = sum1 + pow(arg,   order);
                    sum2 = sum2 + pow(arg, 2*order);
                }
                for (int j = 1; j <= jcut; j++) {
                    double arg  = factor2/(factor2-M_PI*j);
                    sum1 += pow(arg,   order);
                    sum2 += pow(arg, 2*order);
                }
                zeta = sum2/sum1;
            }
            _pmeBsplineModuli[dim][i-1] = _pmeBsplineModuli[dim][i-1]*(zeta*zeta);
        }
    }
}

double ReferencePMETholeDipoleForce::calculateElectrostatic(const vector<TholeDipoleParticleData>& particleData,
                                                            vector<Vec3>& torques, vector<Vec3>& forces)
{
    double energy = 0.0;

    double directEnergy = 0.0;
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            double mScale = 1.0;
            double iScale = 1.0;

            if (j <= _maxScaleIndex[i]) {
                mScale = getScaleFactor(i, j, M_SCALE);
                iScale = getScaleFactor(i, j, I_SCALE);
            }

            directEnergy += calculatePmeDirectElectrostaticPairIxn(particleData[i], particleData[j],
                                                            mScale, iScale, forces, torques);
        }
    }

    double recipEnergy = computeReciprocalSpaceFixedMultipoleForceAndEnergy(particleData, forces, torques);
    double selfEnergy = calculatePmeSelfEnergy(particleData);

    // Compute induced dipole potential on grid for reciprocal force calculation
    initializePmeGrid();
    spreadInducedDipolesOnGrid(_inducedDipole);
    vector<size_t> shape = {(size_t) _pmeGridDimensions[0], (size_t) _pmeGridDimensions[1], (size_t) _pmeGridDimensions[2]};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (_pmeGridDimensions[1]*_pmeGridDimensions[2]*sizeof(std::complex<double>)),
                                (ptrdiff_t) (_pmeGridDimensions[2]*sizeof(std::complex<double>)),
                                (ptrdiff_t) sizeof(std::complex<double>)};
    pocketfft::c2c(shape, stride, stride, axes, true, _pmeGrid, _pmeGrid, 1.0, 0);
    performPmeReciprocalConvolution();
    pocketfft::c2c(shape, stride, stride, axes, false, _pmeGrid, _pmeGrid, 1.0, 0);
    computeInducedPotentialFromGrid();

    double inducedRecipEnergy = computeReciprocalSpaceInducedDipoleForceAndEnergy(particleData, forces, torques);

    energy = directEnergy + recipEnergy + inducedRecipEnergy + selfEnergy;

    fprintf(stderr, "THOLE PME Energy: direct=%.6f recip=%.6f indRecip=%.6f self=%.6f total=%.6f\n",
            directEnergy, recipEnergy, inducedRecipEnergy, selfEnergy, energy);
    fprintf(stderr, "THOLE PME Force[0]: [%.6f, %.6f, %.6f]\n", forces[0][0], forces[0][1], forces[0][2]);
    fprintf(stderr, "THOLE PME Induced[0]: [%.9f, %.9f, %.9f]\n",
            _inducedDipole[0][0], _inducedDipole[0][1], _inducedDipole[0][2]);

    return energy;
}

void ReferencePMETholeDipoleForce::calculateFixedDipoleField(const vector<TholeDipoleParticleData>& particleData)
{
    zeroFixedDipoleFields();
    resizePmeArrays();
    computePmeBSplines(particleData);
    initializePmeGrid();
    spreadFixedMultipolesOntoGrid(particleData);

    vector<size_t> shape = {(size_t) _pmeGridDimensions[0], (size_t) _pmeGridDimensions[1], (size_t) _pmeGridDimensions[2]};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (_pmeGridDimensions[1]*_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) (_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};

    pocketfft::c2c(shape, stride, stride, axes, true, _pmeGrid, _pmeGrid, 1.0, 0);

    _fixedMultipoleRecipEnergy = performPmeReciprocalConvolution();

    pocketfft::c2c(shape, stride, stride, axes, false, _pmeGrid, _pmeGrid, 1.0, 0);

    computeFixedPotentialFromGrid();
    recordFixedMultipoleField();

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (unsigned int jj = 0; jj < _numParticles; jj++) {
        Vec3 selfField = particleData[jj].dipole*term;
        _fixedDipoleField[jj] += selfField;
    }

    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            double mScale = 1.0;
            double iScale = 1.0;

            if (j <= _maxScaleIndex[i]) {
                mScale = getScaleFactor(i, j, M_SCALE);
                iScale = getScaleFactor(i, j, I_SCALE);
            }
            calculateFixedDipoleFieldPairIxn(particleData[i], particleData[j], mScale, iScale);
        }
    }
}

void ReferencePMETholeDipoleForce::calculateInducedDipoleFields(const vector<TholeDipoleParticleData>& particleData,
                                                                const vector<Vec3>& inducedDipoles,
                                                                vector<Vec3>& inducedDipoleField)
{

    for (unsigned int i = 0; i < _numParticles; i++)
        inducedDipoleField[i] = Vec3(0.0, 0.0, 0.0);

    // Use j > i to avoid double counting (each pair processed once, both fields updated)
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            double iScale = 1.0;
            if (j <= _maxScaleIndex[i])
                iScale = getScaleFactor(i, j, I_SCALE);

            calculatePmeDirectInducedDipolePairIxn(particleData[i], particleData[j],
                                                   inducedDipoles, iScale, inducedDipoleField);
        }
    }
    initializePmeGrid();
    spreadInducedDipolesOnGrid(inducedDipoles);

    vector<size_t> shape = {(size_t) _pmeGridDimensions[0], (size_t) _pmeGridDimensions[1], (size_t) _pmeGridDimensions[2]};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (_pmeGridDimensions[1]*_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) (_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};
    pocketfft::c2c(shape, stride, stride, axes, true, _pmeGrid, _pmeGrid, 1.0, 0);
    _inducedDipoleRecipEnergy = performPmeReciprocalConvolution();
    pocketfft::c2c(shape, stride, stride, axes, false, _pmeGrid, _pmeGrid, 1.0, 0);

    computeInducedPotentialFromGrid();
    recordInducedDipoleField(inducedDipoleField);

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (unsigned int jj = 0; jj < _numParticles; jj++) {
        inducedDipoleField[jj] += inducedDipoles[jj]*term;
    }
}

void ReferencePMETholeDipoleForce::calculateFixedDipoleFieldPairIxn(const TholeDipoleParticleData& particleI,
                                                                    const TholeDipoleParticleData& particleJ,
                                                                    double mScale, double iScale)
{
    if (particleI.particleIndex == particleJ.particleIndex)
        return;

    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r = sqrt(r2);

    // Calculate erfc-damped coefficients (bn1, bn2)
    double ralpha = _alphaEwald * r;
    double bn0 = erfc(ralpha) / r;
    double alsq2 = 2.0 * _alphaEwald * _alphaEwald;
    double alsq2n = 1.0 / (SQRT_PI * _alphaEwald);
    double exp2a = exp(-(ralpha * ralpha));
    alsq2n *= alsq2;
    double bn1 = (bn0 + alsq2n * exp2a) / r2;
    alsq2n *= alsq2;
    double bn2 = (3.0 * bn1 + alsq2n * exp2a) / r2;

    double djr = particleJ.dipole.dot(deltaR);
    double dir = particleI.dipole.dot(deltaR);

    // erfc-damped field (fim/fjm in AMOEBA notation)
    Vec3 fim = -particleJ.dipole * bn1 - deltaR * (bn1 * particleJ.charge - bn2 * djr);
    Vec3 fjm = -particleI.dipole * bn1 + deltaR * (bn1 * particleI.charge + bn2 * dir);

    // Thole damping factors - use base class function which handles all damping types
    double scaleFactor3 = 1.0;
    double scaleFactor5 = 1.0;
    double dthole3, dthole5;  // Not used for field calculation
    if (_dampPermanentInducedField && particleI.polarizability > 0 && particleJ.polarizability > 0) {
        computeTholeDampingFactors(r, particleI.polarizability, particleJ.polarizability,
                                   scaleFactor3, scaleFactor5, dthole3, dthole5);
    }

    double dampedMScale3 = scaleFactor3 * mScale;
    double dampedMScale5 = scaleFactor5 * mScale;

    double rInv = 1.0 / r;
    double rInv2 = rInv * rInv;
    double rInv3 = rInv2 * rInv;
    double rInv5 = rInv3 * rInv2;

    double drr3 = (1.0 - dampedMScale3) * rInv3;
    double drr5 = 3.0 * (1.0 - dampedMScale5) * rInv5;

    Vec3 fid = -particleJ.dipole * drr3 - deltaR * (drr3 * particleJ.charge - drr5 * djr);
    Vec3 fjd = -particleI.dipole * drr3 + deltaR * (drr3 * particleI.charge + drr5 * dir);

    Vec3 field_contrib_I = fim - fid;
    Vec3 field_contrib_J = fjm - fjd;

    _fixedDipoleField[particleI.particleIndex] += field_contrib_I;
    _fixedDipoleField[particleJ.particleIndex] += field_contrib_J;
}

void ReferencePMETholeDipoleForce::computeBSplinePoint(double* data, double* ddata,
                                                       double* d2data, double* d3data,
                                                       double w, int order)
{
    if (order != THOLE_PME_ORDER)
        throw OpenMMException("B-spline order must be " + std::to_string(THOLE_PME_ORDER));
    double array[THOLE_PME_ORDER][THOLE_PME_ORDER];

    // Order-2 spline
    array[1][1] = w;
    array[1][0] = 1.0 - w;

    // Order-3 spline
    array[2][2] = 0.5 * w * array[1][1];
    array[2][1] = 0.5 * ((1.0 + w) * array[1][0] + (2.0 - w) * array[1][1]);
    array[2][0] = 0.5 * (1.0 - w) * array[1][0];

    // Higher order recursion
    for (int i = 4; i <= order; i++) {
        int k = i - 1;
        double denom = 1.0 / k;
        array[i-1][i-1] = denom * w * array[k-1][k-1];
        for (int j = 1; j <= i - 2; j++)
            array[i-1][i-1-j] = denom * ((w + j) * array[k-1][i-1-j-1] + (i - j - w) * array[k-1][i-1-j]);
        array[i-1][0] = denom * (1.0 - w) * array[k-1][0];
    }

    // First derivative coefficients
    int k = order - 2;
    array[k][order-1] = array[k][order-2];
    for (int i = order - 2; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];

    // Second derivative coefficients
    k = order - 3;
    array[k][order-2] = array[k][order-3];
    for (int i = order - 3; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];
    array[k][order-1] = array[k][order-2];
    for (int i = order - 2; i >= 1; i--)
        array[k][i] = array[k][i-1] - array[k][i];
    array[k][0] = -array[k][0];

    // Third derivative coefficients
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

    // Copy to output arrays
    for (int i = 0; i < order; i++) {
        data[i] = array[order-1][i];
        ddata[i] = array[order-2][i];
        d2data[i] = array[order-3][i];
        d3data[i] = array[order-4][i];
    }
}

void ReferencePMETholeDipoleForce::updateGridIndexAndFraction(const vector<TholeDipoleParticleData>& particleData)
{
    for (unsigned int i = 0; i < _numParticles; i++) {
        Vec3 position = particleData[i].position;
        for (int d = 0; d < 3; d++) {
            double w = position[0]*_recipBoxVectors[0][d] +
                       position[1]*_recipBoxVectors[1][d] +
                       position[2]*_recipBoxVectors[2][d];
            double fr = _pmeGridDimensions[d] * (w - (int)(w + 0.5) + 0.5);
            int ifr = static_cast<int>(floor(fr));
            _particleFraction[i][d] = fr - ifr;
            int igrid = ifr - THOLE_PME_ORDER + 1;
            if (igrid < 0) igrid += _pmeGridDimensions[d];
            _iGrid[i][d] = igrid;
        }
    }
}

void ReferencePMETholeDipoleForce::computePmeBSplines(const vector<TholeDipoleParticleData>& particleData)
{
    updateGridIndexAndFraction(particleData);

    const int order = THOLE_PME_ORDER;
    double data[order], ddata[order], d2data[order], d3data[order];

    for (unsigned int i = 0; i < _numParticles; i++) {
        for (int j = 0; j < 3; j++) {
            double w = _particleFraction[i][j];

            computeBSplinePoint(data, ddata, d2data, d3data, w, order);

            for (int k = 0; k < order; k++) {
                _thetai[j][i*order + k] = double4(data[k], ddata[k], d2data[k], d3data[k]);
            }
        }
    }
}



void ReferencePMETholeDipoleForce::transformDipolesToFractionalCoordinates(const vector<TholeDipoleParticleData>& particleData)
{
    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    _transformed.resize(particleData.size());
    for (int i = 0; i < (int) particleData.size(); i++) {
        _transformed[i].charge = particleData[i].charge;
        _transformed[i].dipole = Vec3();
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                _transformed[i].dipole[j] += a[j][k]*particleData[i].dipole[k];
    }
}

void ReferencePMETholeDipoleForce::transformPotentialToCartesianCoordinates(const vector<double>& fphi, vector<double>& cphi) const
{
    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    // Transform the potential (10 components: charge + 3 dipole + 6 derivatives)
    for (unsigned int i = 0; i < _numParticles; i++) {
        cphi[10*i] = fphi[10*i];

        cphi[10*i+1] = a[0][0]*fphi[10*i+1] + a[0][1]*fphi[10*i+2] + a[0][2]*fphi[10*i+3];
        cphi[10*i+2] = a[1][0]*fphi[10*i+1] + a[1][1]*fphi[10*i+2] + a[1][2]*fphi[10*i+3];
        cphi[10*i+3] = a[2][0]*fphi[10*i+1] + a[2][1]*fphi[10*i+2] + a[2][2]*fphi[10*i+3];

        cphi[10*i+4] = a[0][0]*a[0][0]*fphi[10*i+4] + a[0][1]*a[0][1]*fphi[10*i+5] + a[0][2]*a[0][2]*fphi[10*i+6];
        cphi[10*i+5] = a[1][0]*a[1][0]*fphi[10*i+4] + a[1][1]*a[1][1]*fphi[10*i+5] + a[1][2]*a[1][2]*fphi[10*i+6];
        cphi[10*i+6] = a[2][0]*a[2][0]*fphi[10*i+4] + a[2][1]*a[2][1]*fphi[10*i+5] + a[2][2]*a[2][2]*fphi[10*i+6];
        cphi[10*i+7] = a[0][0]*a[1][0]*fphi[10*i+4] + a[0][1]*a[1][1]*fphi[10*i+5] + a[0][2]*a[1][2]*fphi[10*i+6];
        cphi[10*i+8] = a[0][0]*a[2][0]*fphi[10*i+4] + a[0][1]*a[2][1]*fphi[10*i+5] + a[0][2]*a[2][2]*fphi[10*i+6];
        cphi[10*i+9] = a[1][0]*a[2][0]*fphi[10*i+4] + a[1][1]*a[2][1]*fphi[10*i+5] + a[1][2]*a[2][2]*fphi[10*i+6];
    }
}

void ReferencePMETholeDipoleForce::spreadFixedMultipolesOntoGrid(const vector<TholeDipoleParticleData>& particleData)
{
    transformDipolesToFractionalCoordinates(particleData);

    const int order = THOLE_PME_ORDER;

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = complex<double>(0, 0);

    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        double atomCharge = _transformed[atomIndex].charge;
        Vec3 atomDipole = _transformed[atomIndex].dipole;

        int x0index = _iGrid[atomIndex][0];
        int y0index = _iGrid[atomIndex][1];
        int z0index = _iGrid[atomIndex][2];

        for (int ix = 0; ix < order; ix++) {
            int xindex = (x0index + ix) % _pmeGridDimensions[0];
            double4 tx = _thetai[0][atomIndex*order + ix];

            for (int iy = 0; iy < order; iy++) {
                int yindex = (y0index + iy) % _pmeGridDimensions[1];
                double4 ty = _thetai[1][atomIndex*order + iy];

                double term0 = atomCharge*tx[0]*ty[0] +
                              atomDipole[0]*tx[1]*ty[0] +
                              atomDipole[1]*tx[0]*ty[1];
                double term1 = atomDipole[2]*tx[0]*ty[0];

                for (int iz = 0; iz < order; iz++) {
                    int zindex = (z0index + iz) % _pmeGridDimensions[2];
                    double4 tz = _thetai[2][atomIndex*order + iz];

                    int index = xindex*_pmeGridDimensions[1]*_pmeGridDimensions[2] +
                               yindex*_pmeGridDimensions[2] + zindex;

                    _pmeGrid[index] += term0*tz[0] + term1*tz[1];
                }
            }
        }
    }
}

double ReferencePMETholeDipoleForce::performPmeReciprocalConvolution()
{
    int nx = _pmeGridDimensions[0];
    int ny = _pmeGridDimensions[1];
    int nz = _pmeGridDimensions[2];

    double factor = M_PI * M_PI / (_alphaEwald * _alphaEwald);
    double volume = fabs(_computeBoxVolume());
    double scaleFactor = 1.0 / (M_PI * volume);

    double esum = 0.0;

    int maxkx = (nx + 1) / 2;
    int maxky = (ny + 1) / 2;
    int maxkz = (nz + 1) / 2;

    for (int kx = 0; kx < nx; kx++) {
        double mx = (kx < maxkx) ? kx : (kx - nx);
        double mhx = mx * _recipBoxVectors[0][0];
        double bx = _pmeBsplineModuli[0][kx];

        for (int ky = 0; ky < ny; ky++) {
            double my = (ky < maxky) ? ky : (ky - ny);
            double mhy = mx*_recipBoxVectors[1][0] + my*_recipBoxVectors[1][1];
            double by = _pmeBsplineModuli[1][ky];

            for (int kz = 0; kz < nz; kz++) {
                int index = kx*ny*nz + ky*nz + kz;

                if (kx == 0 && ky == 0 && kz == 0) {
                    _pmeGrid[index] = complex<double>(0, 0);
                    continue;
                }

                double mz = (kz < maxkz) ? kz : (kz - nz);
                double mhz = mx*_recipBoxVectors[2][0] +
                            my*_recipBoxVectors[2][1] +
                            mz*_recipBoxVectors[2][2];

                double d1 = _pmeGrid[index].real();
                double d2 = _pmeGrid[index].imag();

                double m2 = mhx*mhx + mhy*mhy + mhz*mhz;
                double bz = _pmeBsplineModuli[2][kz];
                double denom = m2 * bx * by * bz;

                double eterm = scaleFactor * exp(-factor*m2) / denom;

                _pmeGrid[index] = complex<double>(d1*eterm, d2*eterm);

                double struct2 = d1*d1 + d2*d2;
                double ets2 = eterm * struct2;
                esum += ets2;
            }
        }
    }

    double result = 0.5 * esum;
    return result;
}

void ReferencePMETholeDipoleForce::computeFixedPotentialFromGrid()
{
    const int order = THOLE_PME_ORDER;

    for (int m = 0; m < _numParticles; m++) {
        int x0index = _iGrid[m][0];
        int y0index = _iGrid[m][1];
        int z0index = _iGrid[m][2];

        double tuv000 = 0.0, tuv100 = 0.0, tuv010 = 0.0, tuv001 = 0.0;
        double tuv200 = 0.0, tuv020 = 0.0, tuv002 = 0.0;
        double tuv110 = 0.0, tuv101 = 0.0, tuv011 = 0.0;

        for (int ix = 0; ix < order; ix++) {
            int xindex = (x0index + ix) % _pmeGridDimensions[0];
            double4 tx = _thetai[0][m*order + ix];

            for (int iy = 0; iy < order; iy++) {
                int yindex = (y0index + iy) % _pmeGridDimensions[1];
                double4 ty = _thetai[1][m*order + iy];

                double tu00 = 0.0, tu01 = 0.0, tu02 = 0.0;

                for (int iz = 0; iz < order; iz++) {
                    int zindex = (z0index + iz) % _pmeGridDimensions[2];
                    int index = xindex*_pmeGridDimensions[1]*_pmeGridDimensions[2] +
                               yindex*_pmeGridDimensions[2] + zindex;

                    double gridvalue = _pmeGrid[index].real();
                    double4 tz = _thetai[2][m*order + iz];

                    tu00 += gridvalue * tz[0];
                    tu01 += gridvalue * tz[1];
                    tu02 += gridvalue * tz[2];
                }

                tuv000 += tx[0] * ty[0] * tu00;
                tuv100 += tx[1] * ty[0] * tu00;
                tuv010 += tx[0] * ty[1] * tu00;
                tuv001 += tx[0] * ty[0] * tu01;
                tuv200 += tx[2] * ty[0] * tu00;
                tuv020 += tx[0] * ty[2] * tu00;
                tuv002 += tx[0] * ty[0] * tu02;
                tuv110 += tx[1] * ty[1] * tu00;
                tuv101 += tx[1] * ty[0] * tu01;
                tuv011 += tx[0] * ty[1] * tu01;
            }
        }

        _phi[10*m + 0] = tuv000;
        _phi[10*m + 1] = tuv100;
        _phi[10*m + 2] = tuv010;
        _phi[10*m + 3] = tuv001;
        _phi[10*m + 4] = tuv200;
        _phi[10*m + 5] = tuv020;
        _phi[10*m + 6] = tuv002;
        _phi[10*m + 7] = tuv110;
        _phi[10*m + 8] = tuv101;
        _phi[10*m + 9] = tuv011;
    }
}

void ReferencePMETholeDipoleForce::computeInducedPotentialFromGrid()
{
    for (int m = 0; m < _numParticles; m++) {
        IntVec gridPoint = _iGrid[m];
        double tuv000 = 0.0;
        double tuv100 = 0.0;
        double tuv010 = 0.0;
        double tuv001 = 0.0;
        double tuv200 = 0.0;
        double tuv020 = 0.0;
        double tuv002 = 0.0;
        double tuv110 = 0.0;
        double tuv101 = 0.0;
        double tuv011 = 0.0;

        for (int iz = 0; iz < THOLE_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double4 v = _thetai[2][m*THOLE_PME_ORDER+iz];
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;
            double tu20 = 0.0;
            double tu11 = 0.0;
            double tu02 = 0.0;

            for (int iy = 0; iy < THOLE_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double4 u = _thetai[1][m*THOLE_PME_ORDER+iy];
                double t0 = 0.0;
                double t1 = 0.0;
                double t2 = 0.0;

                for (int ix = 0; ix < THOLE_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].real();
                    double4 tadd = _thetai[0][m*THOLE_PME_ORDER+ix];
                    t0 += tq*tadd[0];
                    t1 += tq*tadd[1];
                    t2 += tq*tadd[2];
                }
                tu00 += t0*u[0];
                tu10 += t1*u[0];
                tu01 += t0*u[1];
                tu20 += t2*u[0];
                tu11 += t1*u[1];
                tu02 += t0*u[2];
            }
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
            tuv200 += tu20*v[0];
            tuv020 += tu02*v[0];
            tuv002 += tu00*v[2];
            tuv110 += tu11*v[0];
            tuv101 += tu10*v[1];
            tuv011 += tu01*v[1];
        }

        // Store potential and derivatives (10 components)
        // Layout: [phi, dphi/dx, dphi/dy, dphi/dz, d2phi/dx2, d2phi/dy2, d2phi/dz2, d2phi/dxdy, d2phi/dxdz, d2phi/dydz]
        _phid[10*m]   = tuv000;
        _phid[10*m+1] = tuv100;
        _phid[10*m+2] = tuv010;
        _phid[10*m+3] = tuv001;
        _phid[10*m+4] = tuv200;
        _phid[10*m+5] = tuv020;
        _phid[10*m+6] = tuv002;
        _phid[10*m+7] = tuv110;
        _phid[10*m+8] = tuv101;
        _phid[10*m+9] = tuv011;
    }
}

double ReferencePMETholeDipoleForce::computeReciprocalSpaceFixedMultipoleForceAndEnergy(
    const vector<TholeDipoleParticleData>& particleData,
    vector<Vec3>& forces, vector<Vec3>& torques) const
{
    const int deriv1[] = {1, 4, 7, 8};
    const int deriv2[] = {2, 7, 5, 9};
    const int deriv3[] = {3, 8, 9, 6};

    vector<double> cphi(10*_numParticles);
    transformPotentialToCartesianCoordinates(_phi, cphi);

    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    double energy = 0.0;
    for (unsigned int i = 0; i < _numParticles; i++) {
        double multipole[4];
        multipole[0] = particleData[i].charge;
        multipole[1] = particleData[i].dipole[0];
        multipole[2] = particleData[i].dipole[1];
        multipole[3] = particleData[i].dipole[2];

        const double* phi = &cphi[10*i];
        torques[i][0] += _electric*(multipole[3]*phi[2] - multipole[2]*phi[3]);
        torques[i][1] += _electric*(multipole[1]*phi[3] - multipole[3]*phi[1]);
        torques[i][2] += _electric*(multipole[2]*phi[1] - multipole[1]*phi[2]);

        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];

        Vec3 f = Vec3(0.0, 0.0, 0.0);
        double particleEnergy = 0.0;
        for (int k = 0; k < 4; k++) {
            particleEnergy += multipole[k]*_phi[10*i+k];
            f[0] += multipole[k]*_phi[10*i+deriv1[k]];
            f[1] += multipole[k]*_phi[10*i+deriv2[k]];
            f[2] += multipole[k]*_phi[10*i+deriv3[k]];
        }
        energy += particleEnergy;
        f *= _electric;
        Vec3 recipForce = Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                               f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                               f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);
        if (i == 0) {
            fprintf(stderr, "PME P-P Recip (p0): f_frac=[%.4f,0,0] recipF=[%.3f,0,0]\n", f[0], recipForce[0]);
        }
        forces[i] -= recipForce;
    }

    return 0.5 * _electric * energy;
}

double ReferencePMETholeDipoleForce::computeReciprocalSpaceInducedDipoleForceAndEnergy(
    const vector<TholeDipoleParticleData>& particleData,
    vector<Vec3>& forces, vector<Vec3>& torques) const
{
    // Derivative index mappings: deriv1/2/3[k] = d/dx, d/dy, d/dz of component k
    const int deriv1[] = {1, 4, 7, 8};
    const int deriv2[] = {2, 7, 5, 9};
    const int deriv3[] = {3, 8, 9, 6};

    // Transform induced potential from fractional to Cartesian coordinates
    vector<double> cphid(10*_numParticles);
    transformPotentialToCartesianCoordinates(_phid, cphid);

    Vec3 cartToFrac[3], fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cartToFrac[j][i] = fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    double energy = 0.0;
    for (unsigned int i = 0; i < _numParticles; i++) {
        Vec3 inducedDipole;
        inducedDipole[0] = _inducedDipole[i][0]*cartToFrac[0][0] + _inducedDipole[i][1]*cartToFrac[0][1] + _inducedDipole[i][2]*cartToFrac[0][2];
        inducedDipole[1] = _inducedDipole[i][0]*cartToFrac[1][0] + _inducedDipole[i][1]*cartToFrac[1][1] + _inducedDipole[i][2]*cartToFrac[1][2];
        inducedDipole[2] = _inducedDipole[i][0]*cartToFrac[2][0] + _inducedDipole[i][1]*cartToFrac[2][1] + _inducedDipole[i][2]*cartToFrac[2][2];

        // P-I reciprocal energy: μ_ind · E_perm_recip
        energy += inducedDipole[0]*_phi[10*i+1] + inducedDipole[1]*_phi[10*i+2] + inducedDipole[2]*_phi[10*i+3];

        // I-I reciprocal energy: μ_ind · E_ind_recip (for Mutual)
        // Factor 0.5 to avoid double counting (each pair counted twice in full sum)
        if (_polarizationType == Mutual) {
            energy += 0.5 * (inducedDipole[0]*_phid[10*i+1] + inducedDipole[1]*_phid[10*i+2] + inducedDipole[2]*_phid[10*i+3]);
        }

        // Torque on permanent dipoles from induced potential
        const double* phi = &cphid[10*i];
        torques[i][0] += 0.5*_electric*(particleData[i].dipole[2]*phi[2] - particleData[i].dipole[1]*phi[3]);
        torques[i][1] += 0.5*_electric*(particleData[i].dipole[0]*phi[3] - particleData[i].dipole[2]*phi[1]);
        torques[i][2] += 0.5*_electric*(particleData[i].dipole[1]*phi[1] - particleData[i].dipole[0]*phi[2]);

        double multipole[4];
        multipole[0] = particleData[i].charge;
        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];

        Vec3 f(0.0, 0.0, 0.0);

        // Force on induced dipole from permanent multipole field gradient
        Vec3 f_ind(0.0, 0.0, 0.0);
        for (int k = 0; k < 3; k++) {
            int j1 = deriv1[k+1];
            int j2 = deriv2[k+1];
            int j3 = deriv3[k+1];
            f_ind[0] += 2.0*inducedDipole[k]*_phi[10*i+j1];
            f_ind[1] += 2.0*inducedDipole[k]*_phi[10*i+j2];
            f_ind[2] += 2.0*inducedDipole[k]*_phi[10*i+j3];
        }

        // Mutual: induced-induced reciprocal force
        if (_polarizationType == Mutual) {
            for (int k = 0; k < 3; k++) {
                int j1 = deriv1[k+1];
                int j2 = deriv2[k+1];
                int j3 = deriv3[k+1];
                f_ind[0] += 2.0*inducedDipole[k]*_phid[10*i+j1];
                f_ind[1] += 2.0*inducedDipole[k]*_phid[10*i+j2];
                f_ind[2] += 2.0*inducedDipole[k]*_phid[10*i+j3];
            }
        }
        f += f_ind;

        // Force on permanent multipoles from induced potential
        Vec3 f_perm(0.0, 0.0, 0.0);
        for (int k = 0; k < 4; k++) {
            f_perm[0] += multipole[k]*2.0*_phid[10*i+deriv1[k]];
            f_perm[1] += multipole[k]*2.0*_phid[10*i+deriv2[k]];
            f_perm[2] += multipole[k]*2.0*_phid[10*i+deriv3[k]];
        }
        f += f_perm;

        if (i == 0) {
            fprintf(stderr, "PME Recip (p0) pol=%d: f_ind=[%.3f,0,0] f_perm=[%.3f,0,0]\n",
                    (int)_polarizationType, f_ind[0], f_perm[0]);
        }

        f *= (0.5*_electric);
        forces[i] -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                          f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                          f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);
    }

    return 0.5*_electric*energy;
}

void ReferencePMETholeDipoleForce::recordFixedMultipoleField()
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    for (unsigned int i = 0; i < _numParticles; i++) {
        _fixedDipoleField[i][0] = -(_phi[10*i+1]*fracToCart[0][0] + _phi[10*i+2]*fracToCart[0][1] + _phi[10*i+3]*fracToCart[0][2]);
        _fixedDipoleField[i][1] = -(_phi[10*i+1]*fracToCart[1][0] + _phi[10*i+2]*fracToCart[1][1] + _phi[10*i+3]*fracToCart[1][2]);
        _fixedDipoleField[i][2] = -(_phi[10*i+1]*fracToCart[2][0] + _phi[10*i+2]*fracToCart[2][1] + _phi[10*i+3]*fracToCart[2][2]);
    }
}

void ReferencePMETholeDipoleForce::calculatePmeDirectInducedDipolePairIxn(
    const TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData& particleJ,
    const vector<Vec3>& inducedDipoles,
    double iScale,
    vector<Vec3>& field) const
{
    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r = sqrt(r2);

    double ralpha = _alphaEwald * r;
    double bn0 = erfc(ralpha) / r;
    double alsq2 = 2.0 * _alphaEwald * _alphaEwald;
    double alsq2n = 1.0 / (SQRT_PI * _alphaEwald);
    double exp2a = exp(-(ralpha * ralpha));
    alsq2n *= alsq2;
    double bn1 = (bn0 + alsq2n * exp2a) / r2;

    alsq2n *= alsq2;
    double bn2 = (3.0 * bn1 + alsq2n * exp2a) / r2;

    double thole3 = 1.0, thole5 = 1.0;
    if (_polarizationType == Mutual &&
        particleI.polarizability > 0 && particleJ.polarizability > 0) {
        double dthole3, dthole5;
        computeTholeDampingFactors(r, particleI.polarizability, particleJ.polarizability,
                                   thole3, thole5, dthole3, dthole5);
    }

    const Vec3& uI = inducedDipoles[particleI.particleIndex];
    const Vec3& uJ = inducedDipoles[particleJ.particleIndex];

    double uJr = uJ.dot(deltaR);
    // Field at I from dipole at J
    Vec3 fieldAtI = -thole3 * uJ * bn1 + thole5 * deltaR * (bn2 * uJr);

    double uIr = uI.dot(deltaR);
    // Field at J from dipole at I
    Vec3 fieldAtJ = -thole3 * uI * bn1 + thole5 * deltaR * (bn2 * uIr);

    field[particleI.particleIndex] += fieldAtI * iScale;
    field[particleJ.particleIndex] += fieldAtJ * iScale;
}

void ReferencePMETholeDipoleForce::spreadInducedDipolesOnGrid(const vector<Vec3>& inputInducedDipole)
{
    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    vector<Vec3> transformedDipoles(_numParticles);
    for (unsigned int i = 0; i < _numParticles; i++) {
        transformedDipoles[i] = Vec3();
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                transformedDipoles[i][j] += a[j][k]*inputInducedDipole[i][k];
    }

    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = complex<double>(0, 0);

    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        Vec3 atomDipole = transformedDipoles[atomIndex];

        IntVec& gridPoint = _iGrid[atomIndex];
        for (int ix = 0; ix < THOLE_PME_ORDER; ix++) {
            int x = (gridPoint[0]+ix) % _pmeGridDimensions[0];
            double4 t = _thetai[0][atomIndex*THOLE_PME_ORDER+ix];
            for (int iy = 0; iy < THOLE_PME_ORDER; iy++) {
                int y = (gridPoint[1]+iy) % _pmeGridDimensions[1];
                double4 u = _thetai[1][atomIndex*THOLE_PME_ORDER+iy];

                double term0 = atomDipole[1]*t[0]*u[1] + atomDipole[0]*t[1]*u[0];
                double term1 = atomDipole[2]*t[0]*u[0];

                for (int iz = 0; iz < THOLE_PME_ORDER; iz++) {
                    int z = (gridPoint[2]+iz) % _pmeGridDimensions[2];
                    double4 v = _thetai[2][atomIndex*THOLE_PME_ORDER+iz];
                    complex<double>& gridValue = _pmeGrid[x*_pmeGridDimensions[1]*_pmeGridDimensions[2]+y*_pmeGridDimensions[2]+z];
                    gridValue += term0*v[0] + term1*v[1];
                }
            }
        }
    }
}

void ReferencePMETholeDipoleForce::recordInducedDipoleField(vector<Vec3>& field)
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    for (unsigned int i = 0; i < _numParticles; i++) {
        field[i][0] -= (_phid[10*i+1]*fracToCart[0][0] + _phid[10*i+2]*fracToCart[0][1] + _phid[10*i+3]*fracToCart[0][2]);
        field[i][1] -= (_phid[10*i+1]*fracToCart[1][0] + _phid[10*i+2]*fracToCart[1][1] + _phid[10*i+3]*fracToCart[1][2]);
        field[i][2] -= (_phid[10*i+1]*fracToCart[2][0] + _phid[10*i+2]*fracToCart[2][1] + _phid[10*i+3]*fracToCart[2][2]);
    }
}

double ReferencePMETholeDipoleForce::calculatePmeSelfEnergy(const vector<TholeDipoleParticleData>& particleData) const
{
    double cii = 0.0;
    double dii_perm = 0.0;
    double dii_ind = 0.0;
    double totalCharge = 0.0;

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        const TholeDipoleParticleData& particleI = particleData[ii];

        totalCharge += particleI.charge;
        cii += particleI.charge*particleI.charge;

        dii_perm += particleI.dipole.dot(particleI.dipole);

        // Induced self-energy needed since we compute I-I reciprocal energy explicitly
        if (_polarizationType == Mutual) {
            dii_ind += _inducedDipole[ii].dot(_inducedDipole[ii]);
        }
    }

    double prefac = -_alphaEwald * _electric / (_dielectric*SQRT_PI);
    double a2 = _alphaEwald * _alphaEwald;
    double twoThirds = 2.0/3.0;

    double chargeTerm = prefac*cii;
    double dipoleTerm = prefac*twoThirds*a2*(dii_perm + dii_ind);
    double energy = chargeTerm + dipoleTerm;

    double volume = _computeBoxVolume();
    double plasmaTerm = totalCharge*totalCharge*M_PI*_electric/(2.0*_dielectric*volume*_alphaEwald*_alphaEwald);
    energy -= plasmaTerm;

    return energy;
}

double ReferencePMETholeDipoleForce::calculatePmeDirectElectrostaticPairIxn(
    const TholeDipoleParticleData& particleI,
    const TholeDipoleParticleData& particleJ,
    double mScale, double iScale,
    vector<Vec3>& forces, vector<Vec3>& torques) const
{
    unsigned int iIndex = particleI.particleIndex;
    unsigned int jIndex = particleJ.particleIndex;

    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return 0.0;

    double r = sqrt(r2);

    double ralpha = _alphaEwald * r;
    double bn0 = erfc(ralpha) / r;
    double alsq2 = 2.0 * _alphaEwald * _alphaEwald;
    double alsq2n = 1.0 / (SQRT_PI * _alphaEwald);
    double exp2a = exp(-(ralpha * ralpha));
    alsq2n *= alsq2;
    double bn1 = (bn0 + alsq2n * exp2a) / r2;

    alsq2n *= alsq2;
    double bn2 = (3.0 * bn1 + alsq2n * exp2a) / r2;

    alsq2n *= alsq2;
    double bn3 = (5.0 * bn2 + alsq2n * exp2a) / r2;

    double dIr = particleI.dipole.dot(deltaR);
    double dJr = particleJ.dipole.dot(deltaR);
    double dIdJ = particleI.dipole.dot(particleJ.dipole);

    double qIqJ = particleI.charge * particleJ.charge;
    double qIdJr = particleI.charge * dJr;
    double qJdIr = particleJ.charge * dIr;

    // erfc-damped energy components (short-range, computed in direct space)
    double erfcCC = bn0 * qIqJ;
    double erfcCD = bn1 * (qJdIr - qIdJr);
    double erfcDD = bn1 * dIdJ - bn2 * dIr * dJr;
    double erfcEnergy = erfcCC + erfcCD + erfcDD;

    // For excluded pairs (mScale < 1), we need to subtract the erf (reciprocal) contribution
    // that was incorrectly included in reciprocal space.
    // Full Coulomb = erfc + erf, so erf = full - erfc
    // The undamped (full Coulomb) interaction terms:
    double rInv = 1.0 / r;
    double rInv2 = rInv * rInv;
    double rInv3 = rInv2 * rInv;
    double rInv4 = rInv2 * rInv2;
    double rInv5 = rInv3 * rInv2;
    // Full Coulomb energy components
    double fullCC = rInv * qIqJ;
    double fullCD = rInv3 * (qJdIr - qIdJr);
    double fullDD = rInv3 * dIdJ - 3.0 * rInv3 * rInv2 * dIr * dJr;
    double fullEnergy = fullCC + fullCD + fullDD;

    // PME direct energy = mScale * erfc - (1-mScale) * erf
    //                   = mScale * erfc - (1-mScale) * (full - erfc)
    //                   = erfc - (1-mScale) * full
    double pmeDirectEnergy = (erfcEnergy - (1.0 - mScale) * fullEnergy) * (_electric / _dielectric);

    Vec3 rhat = deltaR / r;

    double muIr = particleI.dipole.dot(rhat);
    double muJr = particleJ.dipole.dot(rhat);
    double muIdotMuJ = particleI.dipole.dot(particleJ.dipole);

    Vec3 muI_perp = particleI.dipole - muIr * rhat;
    Vec3 muJ_perp = particleJ.dipole - muJr * rhat;

    Vec3 force(0.0, 0.0, 0.0);

        const Vec3& uI = _inducedDipole[particleI.particleIndex];
    const Vec3& uJ = _inducedDipole[particleJ.particleIndex];

    double uIr = uI.dot(deltaR);
    double uJr = uJ.dot(deltaR);
    double uIdJ = uI.dot(particleJ.dipole);
    double uJdI = uJ.dot(particleI.dipole);
    double uIuJ = uI.dot(uJ);

    double thole_c = 1.0, thole_d0 = 1.0, thole_d1 = 1.0;
    double dthole_c = 1.0, dthole_d0 = 1.0, dthole_d1 = 1.0;
    if (particleI.polarizability > 0 && particleJ.polarizability > 0) {
        computePmeTholeDampingFactors(r, particleI.polarizability, particleJ.polarizability,
                                      thole_c, thole_d0, thole_d1,
                                      dthole_c, dthole_d0, dthole_d1);
    }

    // For P-I energy, respect _dampPermanentInducedField flag
    double thole_c_pi = _dampPermanentInducedField ? thole_c : 1.0;
    double thole_d0_pi = _dampPermanentInducedField ? thole_d0 : 1.0;
    double thole_d1_pi = _dampPermanentInducedField ? thole_d1 : 1.0;

    // Permanent-induced energy (charge-induced, perm_dipole-induced): use mScale
    // Factor of 0.5 for variational/linear response formulation: E_pol = -0.5 * μ_ind · E_perm
    // PME direct space uses: Thole_damped_undamped + (erfc_damped - undamped)
    //                      = thole_c/r^n + bn - 1/r^n = bn - (1-thole_c)/r^n

    // Charge-induced: C-Ind interaction goes like q * (μ·r) / r³
    // For PME with Thole damping:
    //   Non-excluded pairs: erfc + (thole - 1) * full = erfc - (1 - thole) * full
    //   Excluded pairs: erfc - full
    //   Combined: erfc - (1 - mScale * thole) * full
    double qIndErfc = bn1 * (particleI.charge * uJr - particleJ.charge * uIr);
    double qIndFull = rInv3 * (particleI.charge * uJr - particleJ.charge * uIr);
    double qIndEnergy = qIndErfc - (1.0 - mScale * thole_c_pi) * qIndFull;

    // Dipole-induced: D-Ind interaction has both 1/r³ and 1/r⁵ terms
    // Use separate Thole damping for each term (thole_d0 for 1/r³, thole_d1 for 1/r⁵)
    double dIndErfc = bn1 * (uIdJ + uJdI) - bn2 * (dIr * uJr + dJr * uIr);
    double dIndFull0 = rInv3 * (uIdJ + uJdI);  // 1/r³ term
    double dIndFull1 = -3.0 * rInv5 * (dIr * uJr + dJr * uIr);  // 1/r⁵ term
    double dIndEnergy = dIndErfc - (1.0 - mScale * thole_d0_pi) * dIndFull0 - (1.0 - mScale * thole_d1_pi) * dIndFull1;

    // Q-Ind: positive qIndEnergy = stabilizing, subtract to lower energy
    pmeDirectEnergy += -0.5 * qIndEnergy * (_electric / _dielectric);
    // D-Ind: negative dIndEnergy = stabilizing (tensor formula), add to lower energy
    pmeDirectEnergy += 0.5 * dIndEnergy * (_electric / _dielectric);

    // Induced-induced energy: use iScale (with Thole damping for Mutual)
    // Uses thole_d0 for 1/r³ term and thole_d1 for 1/r⁵ term
    // Same tensor formula as D-Ind: negative = stabilizing
    double indIndEnergyVal = 0.0;
    if (_polarizationType == Mutual && particleI.polarizability > 0 && particleJ.polarizability > 0) {
        indIndEnergyVal = (thole_d0 * bn1) * uIuJ - (thole_d1 * bn2) * uIr * uJr;
        pmeDirectEnergy += indIndEnergyVal * iScale * (_electric / _dielectric);
    }

    // Force calculation: erfc-damped force terms (permanent multipoles)
    // Charge-charge
    force += qIqJ * bn1 * deltaR;
    // Charge-dipole radial
    force += (particleJ.charge * muIr - particleI.charge * muJr) * (r * bn2 - bn1 / r) * deltaR;
    // Charge-dipole perpendicular
    force += bn1 * (particleI.charge * muJ_perp - particleJ.charge * muI_perp);
    // Dipole-dipole radial
    force += (bn2 * muIdotMuJ + (2.0 * bn2 - r * r * bn3) * muIr * muJr) * deltaR;
    // Dipole-dipole perpendicular
    force += bn2 * r * (muJr * muI_perp + muIr * muJ_perp);

    // Induced dipole force contributions
    // Perm-ind uses same exclusion correction pattern as perm-perm
    // Ind-ind uses iScale
    Vec3 indForceTotal(0.0, 0.0, 0.0);
    Vec3 fullIndForce(0.0, 0.0, 0.0);  // Full Coulomb perm-ind for exclusion correction
    {
        double uIr_bn = uI.dot(rhat);
        double uJr_bn = uJ.dot(rhat);
        Vec3 uI_perp = uI - uIr_bn * rhat;
        Vec3 uJ_perp = uJ - uJr_bn * rhat;

        // Charge-induced dipole force (erfc-damped) - NOT scaled by mScale
        Vec3 cIndForce(0.0, 0.0, 0.0);
        cIndForce += -(particleI.charge * uJr_bn - particleJ.charge * uIr_bn) * (r * bn2 - bn1 / r) * deltaR;
        cIndForce += -bn1 * (particleJ.charge * uI_perp - particleI.charge * uJ_perp);
        indForceTotal += cIndForce;

        // Full Coulomb charge-induced force for exclusion correction
        // F = q * (3(u·rhat)rhat - u) / r³
        fullIndForce += (particleJ.charge * (3.0 * uIr_bn * rhat - uI) -
                         particleI.charge * (3.0 * uJr_bn * rhat - uJ)) * rInv3;

        // Permanent dipole - induced dipole force (erfc-damped) - NOT scaled by mScale
        Vec3 dIndForce(0.0, 0.0, 0.0);
        dIndForce += (bn2 * (uIdJ + uJdI) + (2.0 * bn2 - r * r * bn3) * (muIr * uJr_bn + muJr * uIr_bn)) * deltaR;
        dIndForce += bn2 * r * (muIr * uJ_perp + uJr_bn * muI_perp + muJr * uI_perp + uIr_bn * muJ_perp);
        indForceTotal += dIndForce;

        // Full Coulomb dipole-induced force for exclusion correction
        // Same tensor form as dipole-dipole: F = 3*(d·rhat*u + u·rhat*d + (d·u)rhat - 5*(d·rhat)(u·rhat)rhat)/r⁴
        fullIndForce += rInv4 * (3.0 * (muIr * uJ + uJr_bn * particleI.dipole + muJr * uI + uIr_bn * particleJ.dipole
                                       + (uIdJ + uJdI) * rhat)
                                - 15.0 * (muIr * uJr_bn + muJr * uIr_bn) * rhat);

        // Induced-induced dipole (only for Mutual, scaled by iScale)
        Vec3 indIndForce(0.0, 0.0, 0.0);
        if (_polarizationType == Mutual && particleI.polarizability > 0 && particleJ.polarizability > 0) {
            indIndForce += (dthole_d0 * bn2 * uIuJ + (2.0 * dthole_d1 * bn2 - r * r * bn3) * uIr_bn * uJr_bn) * deltaR;
            indIndForce += dthole_d1 * bn2 * r * (uJr_bn * uI_perp + uIr_bn * uJ_perp);
            indForceTotal += iScale * indIndForce;
        }

        if (particleI.particleIndex == 0) {
            fprintf(stderr, "PME Direct (pair 0-1) pol=%d:\n", (int)_polarizationType);
            fprintf(stderr, "  erfc P-P: [%.3f, 0, 0]\n", force[0]);
            fprintf(stderr, "  erfc C-I: [%.3f, 0, 0]\n", cIndForce[0]);
            fprintf(stderr, "  erfc D-I: [%.3f, 0, 0]\n", dIndForce[0]);
            fprintf(stderr, "  erfc I-I: [%.3f, 0, 0]\n", indIndForce[0]);
        }
    }
    force += indForceTotal;

    // Full Coulomb (undamped) force for exclusion correction
    Vec3 fullForce(0.0, 0.0, 0.0);

    // Charge-charge: F = q1*q2/r² * rhat = q1*q2/r³ * deltaR
    fullForce += qIqJ * rInv3 * deltaR;

    // Charge-dipole: F = q*(3(μ·rhat)rhat - μ)/r³
    fullForce += (particleJ.charge * (3.0 * muIr * rhat - particleI.dipole) -
                  particleI.charge * (3.0 * muJr * rhat - particleJ.dipole)) * rInv3;

    // Dipole-dipole: F = 3*(μ1·rhat*μ2 + μ2·rhat*μ1 + (μ1·μ2)rhat - 5*(μ1·rhat)(μ2·rhat)rhat)/r⁴
    fullForce += rInv4 * (3.0 * (muIr * particleJ.dipole + muJr * particleI.dipole + muIdotMuJ * rhat)
                         - 15.0 * muIr * muJr * rhat);

    // Add perm-ind full Coulomb for exclusion correction
    fullForce += fullIndForce;

    // erfc-damped field for torques (permanent multipoles only - induced dipoles are isotropic)
    Vec3 fieldAtI_erfc = -particleJ.charge * bn1 * r * rhat + (bn2 * r2 * muJr * rhat - bn1 * particleJ.dipole);
    Vec3 fieldAtJ_erfc = particleI.charge * bn1 * r * rhat + (bn2 * r2 * muIr * rhat - bn1 * particleI.dipole);

    // Full Coulomb field for exclusion correction (permanent multipoles only)
    Vec3 fieldAtI_full = -particleJ.charge * rInv2 * rhat + (3.0 * muJr * rhat - particleJ.dipole) * rInv3;
    Vec3 fieldAtJ_full = particleI.charge * rInv2 * rhat + (3.0 * muIr * rhat - particleI.dipole) * rInv3;

    // PME direct field = erfc_field - (1-mScale) * full_field
    Vec3 fieldAtI = fieldAtI_erfc - (1.0 - mScale) * fieldAtI_full;
    Vec3 fieldAtJ = fieldAtJ_erfc - (1.0 - mScale) * fieldAtJ_full;

    Vec3 forceTotal = (force - (1.0 - mScale) * fullForce) * (_electric / _dielectric);

    if (particleI.particleIndex == 0) {
        fprintf(stderr, "  full P-I: [%.3f, 0, 0] mScale=%.1f excl=[%.3f,0,0]\n",
                fullIndForce[0], mScale, ((1.0-mScale)*fullForce)[0]);
        fprintf(stderr, "  forceTotal: [%.3f, 0, 0]\n", forceTotal[0]);
    }

    forces[iIndex] -= forceTotal;
    forces[jIndex] += forceTotal;

    Vec3 torqueI = particleI.dipole.cross(fieldAtI) * (_electric / _dielectric);
    Vec3 torqueJ = particleJ.dipole.cross(fieldAtJ) * (_electric / _dielectric);

    torques[iIndex] += torqueI;
    torques[jIndex] += torqueJ;

    return pmeDirectEnergy;
}
