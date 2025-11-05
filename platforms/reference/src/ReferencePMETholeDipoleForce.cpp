
#include "ReferencePMETholeDipoleForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "pocketfft_hdronly.h"
#include "openmm/internal/MSVC_erfc.h"
#include <cmath>
#include <iostream>

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
    _phid.resize(4*_numParticles);
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

    for (int dim = 0; dim < 3; dim++) {
        _pmeBsplineModuli[dim].resize(_pmeGridDimensions[dim]);
    }

    double data[order];
    double ddata[order];
    vector<double> bsarray(maxSize + 1, 0.0);

    double x = 0.0;
    data[order-1] = 0.0;
    data[1] = x;
    data[0] = 1.0 - x;

    for (int k = 3; k < order; k++) {
        double div = 1.0 / (k - 1.0);
        data[k-1] = div * x * data[k-2];
        for (int l = 1; l < (k-1); l++) {
            data[k-l-1] = div * (l*data[k-l-2] + (k-l)*data[k-l-1]);
        }
        data[0] = div * data[0];
    }

    ddata[0] = -data[0];
    for (int k = 1; k < order; k++) {
        ddata[k] = data[k-1] - data[k];
    }

    double div = 1.0 / (order - 1);
    data[order-1] = div * x * data[order-2];
    for (int l = 1; l < (order-1); l++) {
        data[order-l-1] = div * (l*data[order-l-2] + (order-l)*data[order-l-1]);
    }
    data[0] = div * data[0];

    for (int i = 1; i <= order; i++) {
        bsarray[i] = data[i-1];
    }

    for (int dim = 0; dim < 3; dim++) {
        int ndata = _pmeGridDimensions[dim];
        double factor = 2.0 * M_PI / ndata;

        for (int i = 0; i < ndata; i++) {
            double sc = 0.0, ss = 0.0;
            for (int j = 0; j < ndata; j++) {
                double arg = factor * i * j;
                sc += bsarray[j] * cos(arg);
                ss += bsarray[j] * sin(arg);
            }
            _pmeBsplineModuli[dim][i] = sc*sc + ss*ss;
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

    calculatePmeSelfTorque(particleData, torques);
    double recipEnergy = computeReciprocalSpaceFixedMultipoleForceAndEnergy(particleData, forces, torques);
    double selfEnergy = calculatePmeSelfEnergy(particleData);

    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    double inducedRecipEnergy = 0.0;
    for (int i = 0; i < _numParticles; i++) {
        Vec3 u_frac;
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                u_frac[j] += a[j][k] * _inducedDipole[i][k];

        inducedRecipEnergy += u_frac[0] * _phi[10*i+1] + u_frac[1] * _phi[10*i+2] + u_frac[2] * _phi[10*i+3];
    }
    inducedRecipEnergy *= -1.0 * _electric;

    energy = directEnergy + recipEnergy + inducedRecipEnergy + selfEnergy;
    return energy;
}

void ReferencePMETholeDipoleForce::calculateFixedDipoleField(const vector<TholeDipoleParticleData>& particleData)
{
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

    double totalGridMagAfterIFFT = 0.0;
    for (int i = 0; i < _totalGridSize; i++) {
        totalGridMagAfterIFFT += std::abs(_pmeGrid[i]);
    }

    computeFixedPotentialFromGrid();
    recordFixedMultipoleField();

    int mid = _numParticles / 2;
    Vec3 fieldAfterRecip_mid = _fixedDipoleField[mid];

    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    for (unsigned int jj = 0; jj < _numParticles; jj++) {
        Vec3 selfEnergy = particleData[jj].dipole*term;
        _fixedDipoleField[jj] += selfEnergy;
    }
    Vec3 fieldAfterSelf_mid = _fixedDipoleField[mid];

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
    Vec3 fieldAfterDirect_mid = _fixedDipoleField[mid];

}

void ReferencePMETholeDipoleForce::calculateInducedDipoleFields(const vector<TholeDipoleParticleData>& particleData,
                                                                const vector<Vec3>& inducedDipoles,
                                                                vector<Vec3>& inducedDipoleField)
{

    for (unsigned int i = 0; i < _numParticles; i++)
        inducedDipoleField[i] = Vec3(0.0, 0.0, 0.0);

    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = 0; j < _numParticles; j++) {
            if (i == j)
                continue;

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

    Vec3 fim = -particleJ.dipole * bn1 - deltaR * (bn1 * particleJ.charge - bn2 * djr);
    Vec3 fjm = -particleI.dipole * bn1 + deltaR * (bn1 * particleI.charge + bn2 * dir);

    _fixedDipoleField[particleI.particleIndex] += fim * mScale;
    _fixedDipoleField[particleJ.particleIndex] += fjm * mScale;
}

void ReferencePMETholeDipoleForce::computeBSplinePoint(double* data, double* ddata,
                                                       double* d2data, double* d3data,
                                                       double w, int order)
{
    data[order-1] = 0.0;
    data[1] = w;
    data[0] = 1.0 - w;

    for (int k = 3; k < order; k++) {
        double div = 1.0 / (k - 1.0);
        data[k-1] = div * w * data[k-2];
        for (int l = 1; l < (k-1); l++) {
            data[k-l-1] = div * ((w + l) * data[k-l-2] + (k - l - w) * data[k-l-1]);
        }
        data[0] = div * (1.0 - w) * data[0];
    }

    ddata[0] = -data[0];
    for (int k = 1; k < order; k++) {
        ddata[k] = data[k-1] - data[k];
    }

    double div = 1.0 / (order - 1);
    data[order-1] = div * w * data[order-2];
    for (int l = 1; l < (order-1); l++) {
        data[order-l-1] = div * ((w + l) * data[order-l-2] + (order - l - w) * data[order-l-1]);
    }
    data[0] = div * (1.0 - w) * data[0];

    d2data[0] = -ddata[0];
    for (int k = 1; k < order; k++) {
        d2data[k] = ddata[k-1] - ddata[k];
    }

    d3data[0] = -d2data[0];
    for (int k = 1; k < order; k++) {
        d3data[k] = d2data[k-1] - d2data[k];
    }
}

void ReferencePMETholeDipoleForce::updateGridIndexAndFraction(const vector<TholeDipoleParticleData>& particleData)
{
    for (int i = 0; i < _numParticles; i++) {
        Vec3 position = particleData[i].position;

        for (int d = 0; d < 3; d++) {
            double t = position[0]*_recipBoxVectors[0][d] +
                      position[1]*_recipBoxVectors[1][d] +
                      position[2]*_recipBoxVectors[2][d];

            t = (t - floor(t)) * _pmeGridDimensions[d];
            int ti = (int)t;

            _particleFraction[i][d] = t - ti;
            _iGrid[i][d] = ti % _pmeGridDimensions[d];
        }
    }
}

void ReferencePMETholeDipoleForce::computePmeBSplines(const vector<TholeDipoleParticleData>& particleData)
{
    updateGridIndexAndFraction(particleData);

    const int order = THOLE_PME_ORDER;
    double data[order], ddata[order], d2data[order], d3data[order];

    for (int i = 0; i < _numParticles; i++) {
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
    for (int i = 0; i < _numParticles; i++) {
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
	double boxfactor = M_PI * fabs(_computeBoxVolume());

	double esum = 0.0;

	int maxkx = (nx + 1) / 2;
	int maxky = (ny + 1) / 2;
	int maxkz = (nz + 1) / 2;

	for (int kx = 0; kx < nx; kx++) {
		double mx = (kx < maxkx) ? kx : (kx - nx);
		double mhx = mx * _recipBoxVectors[0][0];
		double bx = boxfactor * _pmeBsplineModuli[0][kx];

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

				double eterm = exp(-factor*m2) / denom;

				_pmeGrid[index] = complex<double>(d1*eterm, d2*eterm);

				double struct2 = d1*d1 + d2*d2;
				double ets2 = eterm * struct2;
				esum += ets2;
			}
		}
	}

	return 0.5 * esum;
}

void ReferencePMETholeDipoleForce::computeFixedPotentialFromGrid()
{
    const int order = THOLE_PME_ORDER;

    for (int m = 0; m < _numParticles; m++) {
        int x0index = _iGrid[m][0];
        int y0index = _iGrid[m][1];
        int z0index = _iGrid[m][2];

        // Initialize all 10 potential components
        double tuv000 = 0.0, tuv100 = 0.0, tuv010 = 0.0, tuv001 = 0.0;  // φ and first derivatives
        double tuv200 = 0.0, tuv020 = 0.0, tuv002 = 0.0;                // second derivatives (diagonal)
        double tuv110 = 0.0, tuv101 = 0.0, tuv011 = 0.0;                // second derivatives (mixed)

        for (int ix = 0; ix < order; ix++) {
            int xindex = (x0index + ix) % _pmeGridDimensions[0];
            double4 tx = _thetai[0][m*order + ix];
            
            for (int iy = 0; iy < order; iy++) {
                int yindex = (y0index + iy) % _pmeGridDimensions[1];
                double4 ty = _thetai[1][m*order + iy];
                
                double tu00 = 0.0, tu01 = 0.0, tu02 = 0.0;  // z basis: value, 1st deriv, 2nd deriv
                
                for (int iz = 0; iz < order; iz++) {
                    int zindex = (z0index + iz) % _pmeGridDimensions[2];
                    int index = xindex*_pmeGridDimensions[1]*_pmeGridDimensions[2] +
                               yindex*_pmeGridDimensions[2] + zindex;
                    
                    double gridvalue = _pmeGrid[index].real();
                    double4 tz = _thetai[2][m*order + iz];
                    
                    tu00 += gridvalue * tz[0];  // B_z value
                    tu01 += gridvalue * tz[1];  // B_z first derivative  
                    tu02 += gridvalue * tz[2];  // B_z second derivative
                }
                
                // Potential: B_x * B_y * B_z
                tuv000 += tx[0] * ty[0] * tu00;
                
                tuv100 += tx[1] * ty[0] * tu00;  // ∂φ/∂x
                tuv010 += tx[0] * ty[1] * tu00;  // ∂φ/∂y
                tuv001 += tx[0] * ty[0] * tu01;  // ∂φ/∂z
                
                tuv200 += tx[2] * ty[0] * tu00;  // ∂²φ/∂x²
                tuv020 += tx[0] * ty[2] * tu00;  // ∂²φ/∂y²
                tuv002 += tx[0] * ty[0] * tu02;  // ∂²φ/∂z²
                
                tuv110 += tx[1] * ty[1] * tu00;  // ∂²φ/∂x∂y
                tuv101 += tx[1] * ty[0] * tu01;  // ∂²φ/∂x∂z
                tuv011 += tx[0] * ty[1] * tu01;  // ∂²φ/∂y∂z
            }
        }
        
        // Store in the _phi array with proper indexing
        _phi[10*m + 0] = tuv000;  // potential
        _phi[10*m + 1] = tuv100;  // ∂φ/∂x
        _phi[10*m + 2] = tuv010;  // ∂φ/∂y
        _phi[10*m + 3] = tuv001;  // ∂φ/∂z
        _phi[10*m + 4] = tuv200;  // ∂²φ/∂x²
        _phi[10*m + 5] = tuv020;  // ∂²φ/∂y²
        _phi[10*m + 6] = tuv002;  // ∂²φ/∂z²
        _phi[10*m + 7] = tuv110;  // ∂²φ/∂x∂y
        _phi[10*m + 8] = tuv101;  // ∂²φ/∂x∂z
        _phi[10*m + 9] = tuv011;  // ∂²φ/∂y∂z
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

        for (int iz = 0; iz < THOLE_PME_ORDER; iz++) {
            int k = gridPoint[2]+iz-(gridPoint[2]+iz >= _pmeGridDimensions[2] ? _pmeGridDimensions[2] : 0);
            double4 v = _thetai[2][m*THOLE_PME_ORDER+iz];
            double tu00 = 0.0;
            double tu10 = 0.0;
            double tu01 = 0.0;

            for (int iy = 0; iy < THOLE_PME_ORDER; iy++) {
                int j = gridPoint[1]+iy-(gridPoint[1]+iy >= _pmeGridDimensions[1] ? _pmeGridDimensions[1] : 0);
                double4 u = _thetai[1][m*THOLE_PME_ORDER+iy];
                double t0 = 0.0;
                double t1 = 0.0;

                for (int ix = 0; ix < THOLE_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].real();
                    double4 tadd = _thetai[0][m*THOLE_PME_ORDER+ix];
                    t0 += tq*tadd[0];
                    t1 += tq*tadd[1];
                }
                tu00 += t0*u[0];
                tu10 += t1*u[0];
                tu01 += t0*u[1];
            }
            tuv000 += tu00*v[0];
            tuv100 += tu10*v[0];
            tuv010 += tu01*v[0];
            tuv001 += tu00*v[1];
        }

        // Store potential (4 components: potential + 3 field components)
        _phid[4*m] = tuv000;
        _phid[4*m+1] = tuv100;
        _phid[4*m+2] = tuv010;
        _phid[4*m+3] = tuv001;
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

    for (int i = 0; i < _numParticles; i++) {
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
        for (int k = 0; k < 4; k++) {
            f[0]   += multipole[k]*_phi[10*i+deriv1[k]];
            f[1]   += multipole[k]*_phi[10*i+deriv2[k]];
            f[2]   += multipole[k]*_phi[10*i+deriv3[k]];
        }
        f *= (_electric);
        forces[i] -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                          f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                          f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);
    }

    return _fixedMultipoleRecipEnergy * _electric;
}

void ReferencePMETholeDipoleForce::recordFixedMultipoleField()
{
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    for (int i = 0; i < _numParticles; i++) {
        double fieldScale = 1.0 / _electric;
        _fixedDipoleField[i][0] = fieldScale * (-(_phi[10*i+1]*fracToCart[0][0] + _phi[10*i+2]*fracToCart[0][1] + _phi[10*i+3]*fracToCart[0][2]));
        _fixedDipoleField[i][1] = fieldScale * (-(_phi[10*i+1]*fracToCart[1][0] + _phi[10*i+2]*fracToCart[1][1] + _phi[10*i+3]*fracToCart[1][2]));
        _fixedDipoleField[i][2] = fieldScale * (-(_phi[10*i+1]*fracToCart[2][0] + _phi[10*i+2]*fracToCart[2][1] + _phi[10*i+3]*fracToCart[2][2]));
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

    // Calculate Thole damping factors for mutual polarization
    double damp1 = 1.0, damp2 = 1.0;

    if (_polarizationType == Mutual &&
        particleI.polarizability > 0 && particleJ.polarizability > 0) {

        if (_tholeDampingType == TholeDipoleForce::NoDamping) {
            damp1 = damp2 = 1.0;
        }
        else {
            const double a = _tholeDampingParameter;
            double r_pol_scale;
            if (fabs(particleI.polarizability * particleJ.polarizability) > 1e-12) {
                r_pol_scale = pow(particleI.polarizability * particleJ.polarizability, 1.0/6.0);
            }
            else {
                r_pol_scale = 1.0;
            }
            const double u = r / r_pol_scale;

            if (_tholeDampingType == TholeDipoleForce::Exponential) {
                const double ar = a * r;
                const double exp_ar = (ar < 50.0) ? exp(-ar) : 0.0;
                damp1 = 1.0 - exp_ar * (1.0 + ar + 0.5 * ar * ar);
                damp2 = damp1 - exp_ar * (ar * ar * ar / 6.0);
            }
            else if (_tholeDampingType == TholeDipoleForce::Amoeba) {
                const double au3 = a * u * u * u;
                const double exp_au3 = (au3 < 50.0) ? exp(-au3) : 0.0;
                damp1 = 1.0 - exp_au3;
                damp2 = 1.0 - (1.0 + au3) * exp_au3;
            }
            else { // TholeDipoleForce::Linear
                const double s = a * r_pol_scale;
                if (r >= s) {
                    damp1 = damp2 = 1.0;
                } else {
                    const double v = r / s;
                    const double v2 = v * v;
                    const double v3 = v2 * v;
                    damp1 = (4.0 - 3.0 * v) * v3;
                    damp2 = v3 * v;
                }
            }
        }
    }

    const Vec3& uI = inducedDipoles[particleI.particleIndex];
    const Vec3& uJ = inducedDipoles[particleJ.particleIndex];

    double uJr = uJ.dot(deltaR);
    // Field at I from dipole at J
    Vec3 fieldAtI = -damp1 * uJ * bn1 + damp2 * deltaR * (bn2 * uJr);

    double uIr = uI.dot(deltaR);
    // Field at J from dipole at I
    Vec3 fieldAtJ = -damp1 * uI * bn1 + damp2 * deltaR * (bn2 * uIr);

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
    for (int i = 0; i < _numParticles; i++) {
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

    for (int i = 0; i < _numParticles; i++) {
        double fieldScale = 1.0 / _electric;
        Vec3 recipField;
        recipField[0] = fieldScale * (-(_phid[4*i+1]*fracToCart[0][0] + _phid[4*i+2]*fracToCart[0][1] + _phid[4*i+3]*fracToCart[0][2]));
        recipField[1] = fieldScale * (-(_phid[4*i+1]*fracToCart[1][0] + _phid[4*i+2]*fracToCart[1][1] + _phid[4*i+3]*fracToCart[1][2]));
        recipField[2] = fieldScale * (-(_phid[4*i+1]*fracToCart[2][0] + _phid[4*i+2]*fracToCart[2][1] + _phid[4*i+3]*fracToCart[2][2]));

        field[i] += recipField;
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
        
        dii_ind += _inducedDipole[ii].dot(_inducedDipole[ii])*0.5;
    }

    double prefac = -_alphaEwald * _electric / (_dielectric*SQRT_PI);
    double a2 = _alphaEwald * _alphaEwald;
    double twoThirds = 2.0/3.0;

    double chargeTerm = prefac*cii;
    double dipoleTerm = prefac*twoThirds*a2*(dii_perm + dii_ind);
    double energy = chargeTerm + dipoleTerm;

    double volume = _computeBoxVolume();
    double plasmaTerm = totalCharge*totalCharge*M_PI/(2.0*volume*_alphaEwald*_alphaEwald);
    energy += plasmaTerm;

    return energy;
}

void ReferencePMETholeDipoleForce::calculatePmeSelfTorque(const vector<TholeDipoleParticleData>& particleData,
                                                          vector<Vec3>& torques) const
{
    double term = (2.0/3.0)*(_electric/_dielectric)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        const TholeDipoleParticleData& particleI = particleData[ii];
        Vec3 ui = _inducedDipole[ii];
        Vec3 torque = particleI.dipole.cross(ui)*term;
        torques[ii] += torque;
    }
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

    double erfcEnergy = bn0 * qIqJ + bn1 * (qIdJr - qJdIr) + bn2 * dIr * dJr - bn1 * dIdJ;
    double pmeDirectEnergy = erfcEnergy * mScale * (_electric / _dielectric);

    Vec3 rhat = deltaR / r;
    double rInv = 1.0 / r;

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

	double inducedEnergy = bn1 * (particleI.charge * uJr - particleJ.charge * uIr)
                         + bn1 * (uIdJ + uJdI)
                         + bn2 * (dIr * uJr + dJr * uIr);

	if (_polarizationType == Mutual) {
	    // Calculate Thole damping factors for I-I interaction
	    double damp1 = 1.0, damp2 = 1.0;

	    if (particleI.polarizability > 0 && particleJ.polarizability > 0) {
	        if (_tholeDampingType != TholeDipoleForce::NoDamping) {
	            const double a = _tholeDampingParameter;
	            double r_pol_scale;
	            if (fabs(particleI.polarizability * particleJ.polarizability) > 1e-12) {
	                r_pol_scale = pow(particleI.polarizability * particleJ.polarizability, 1.0/6.0);
	            }
	            else {
	                r_pol_scale = 1.0;
	            }
	            const double u = r / r_pol_scale;

	            if (_tholeDampingType == TholeDipoleForce::Exponential) {
	                const double ar = a * r;
	                const double exp_ar = (ar < 50.0) ? exp(-ar) : 0.0;
	                damp1 = 1.0 - exp_ar * (1.0 + ar + 0.5 * ar * ar);
	                damp2 = damp1 - exp_ar * (ar * ar * ar / 6.0);
	            }
	            else if (_tholeDampingType == TholeDipoleForce::Amoeba) {
	                const double au3 = a * u * u * u;
	                const double exp_au3 = (au3 < 50.0) ? exp(-au3) : 0.0;
	                damp1 = 1.0 - exp_au3;
	                damp2 = 1.0 - (1.0 + au3) * exp_au3;
	            }
	            else { // TholeDipoleForce::Linear
	                const double s = a * r_pol_scale;
	                if (r >= s) {
	                    damp1 = damp2 = 1.0;
	                } else {
	                    const double v = r / s;
	                    const double v2 = v * v;
	                    const double v3 = v2 * v;
	                    damp1 = (4.0 - 3.0 * v) * v3;
	                    damp2 = v3 * v;
	                }
	            }
	        }
	    }

	    // Apply Thole damping to I-I energy
	    inducedEnergy += - (damp1 * bn1) * uIuJ + (damp2 * bn2) * uIr * uJr;
	}

	pmeDirectEnergy -= inducedEnergy * iScale * (_electric / _dielectric);

    force += qIqJ * bn1 * deltaR;

    force += bn2 * (particleI.charge * muJr - particleJ.charge * muIr) * deltaR;
    force += bn1 * rInv * (particleI.charge * muJ_perp - particleJ.charge * muI_perp);

    force += (bn3 * muIr * muJr - bn2 * muIdotMuJ) * deltaR;
    force += bn2 * rInv * (muJr * muI_perp + muIr * muJ_perp);

    Vec3 fieldAtI = -particleJ.charge * bn1 * rhat + (bn2 * muJr * rhat - bn1 * particleJ.dipole);
    Vec3 fieldAtJ = particleI.charge * bn1 * rhat + (bn2 * muIr * rhat - bn1 * particleI.dipole);

    Vec3 forceTotal = force * mScale * (_electric / _dielectric);
    forces[iIndex] -= forceTotal;
    forces[jIndex] += forceTotal;

    Vec3 torqueI = particleI.dipole.cross(fieldAtI) * mScale * (_electric / _dielectric);
    Vec3 torqueJ = particleJ.dipole.cross(fieldAtJ) * mScale * (_electric / _dielectric);

    torques[iIndex] += torqueI;
    torques[jIndex] += torqueJ;

    return pmeDirectEnergy;
}

