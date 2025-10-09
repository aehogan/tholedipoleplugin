/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Mark Friedrichs                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

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
               _pmeGridSize(0), _totalGridSize(0), _alphaEwald(0.0)
{
    _pmeGrid = NULL;
    _pmeGridDimensions = IntVec(-1, -1, -1);

    // Set nonbonded method to PME
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

    double determinant = vectors[0][0]*vectors[1][1]*vectors[2][2];
    assert(determinant > 0);
    double scale = 1.0/determinant;
    _recipBoxVectors[0] = Vec3(vectors[1][1]*vectors[2][2], 0, 0)*scale;
    _recipBoxVectors[1] = Vec3(-vectors[1][0]*vectors[2][2], vectors[0][0]*vectors[2][2], 0)*scale;
    _recipBoxVectors[2] = Vec3(vectors[1][0]*vectors[2][1]-vectors[1][1]*vectors[2][0], -vectors[0][0]*vectors[2][1], vectors[0][0]*vectors[1][1])*scale;
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

    // For charge + dipole (no quadrupole):
    // phi has 10 components per particle: charge(1) + dipole(3) + derivatives(6)
    _phi.resize(10*_numParticles);
    _phid.resize(4*_numParticles);     // For induced dipole field
    _phidp.resize(10*_numParticles);   // For induced dipole force/energy
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
    deltaR -= _periodicBoxVectors[2]*floor(deltaR[2]*_recipBoxVectors[2][2]+0.5);
    deltaR -= _periodicBoxVectors[1]*floor(deltaR[1]*_recipBoxVectors[1][1]+0.5);
    deltaR -= _periodicBoxVectors[0]*floor(deltaR[0]*_recipBoxVectors[0][0]+0.5);
}

void ReferencePMETholeDipoleForce::initializeBSplineModuli()
{
    // Initialize the B-spline moduli

    int maxSize = -1;
    for (unsigned int ii = 0; ii < 3; ii++) {
       _pmeBsplineModuli[ii].resize(_pmeGridDimensions[ii]);
        maxSize = maxSize  > _pmeGridDimensions[ii] ? maxSize : _pmeGridDimensions[ii];
    }

    double array[THOLE_PME_ORDER];
    double x = 0.0;
    array[0] = 1.0 - x;
    array[1] = x;
    for (int k = 2; k < THOLE_PME_ORDER; k++) {
        double denom = 1.0/k;
        array[k] = x*array[k-1]*denom;
        for (int i = 1; i < k; i++) {
            array[k-i] = ((x+i)*array[k-i-1] + ((k-i+1)-x)*array[k-i])*denom;
        }
        array[0] = (1.0-x)*array[0]*denom;
    }

    vector<double> bsarray(maxSize+1, 0.0);
    for (int i = 2; i <= THOLE_PME_ORDER+1; i++) {
        bsarray[i] = array[i-2];
    }

    for (int dim = 0; dim < 3; dim++) {
        int size = _pmeGridDimensions[dim];

        // Get the modulus of the discrete Fourier transform

        double factor = 2.0*M_PI/size;
        for (int i = 0; i < size; i++) {
            double sum1 = 0.0;
            double sum2 = 0.0;
            for (int j = 1; j <= size; j++) {
                double arg = factor*i*(j-1);
                sum1 += bsarray[j]*cos(arg);
                sum2 += bsarray[j]*sin(arg);
            }
            _pmeBsplineModuli[dim][i] = (sum1*sum1 + sum2*sum2);
        }

        // Fix for exponential Euler spline interpolation failure

        double eps = 1.0e-7;
        if (_pmeBsplineModuli[dim][0] < eps) {
            _pmeBsplineModuli[dim][0] = 0.5*_pmeBsplineModuli[dim][1];
        }
        for (int i = 1; i < size-1; i++) {
            if (_pmeBsplineModuli[dim][i] < eps) {
                _pmeBsplineModuli[dim][i] = 0.5*(_pmeBsplineModuli[dim][i-1]+_pmeBsplineModuli[dim][i+1]);
            }
        }
        if (_pmeBsplineModuli[dim][size-1] < eps) {
            _pmeBsplineModuli[dim][size-1] = 0.5*_pmeBsplineModuli[dim][size-2];
        }

        // Compute and apply the optimal zeta coefficient

        int jcut = 50;
        for (int i = 1; i <= size; i++) {
            int k = i - 1;
            if (i > size/2)
                k = k - size;
            double zeta;
            if (k == 0)
                zeta = 1.0;
            else {
                double sum1 = 1.0;
                double sum2 = 1.0;
                factor = M_PI*k/size;
                for (int j = 1; j <= jcut; j++) {
                    double arg = factor/(factor+M_PI*j);
                    sum1 = sum1 + pow(arg,   THOLE_PME_ORDER);
                    sum2 = sum2 + pow(arg, 2*THOLE_PME_ORDER);
                }
                for (int j = 1; j <= jcut; j++) {
                    double arg  = factor/(factor-M_PI*j);
                    sum1 += pow(arg,   THOLE_PME_ORDER);
                    sum2 += pow(arg, 2*THOLE_PME_ORDER);
                }
                zeta = sum2/sum1;
            }
            _pmeBsplineModuli[dim][i-1] = _pmeBsplineModuli[dim][i-1]*(zeta*zeta);
        }
    }
}

// Override calculateElectrostatic to add PME reciprocal space

double ReferencePMETholeDipoleForce::calculateElectrostatic(const vector<TholeDipoleParticleData>& particleData,
                                                            vector<Vec3>& torques, vector<Vec3>& forces)
{
    double energy = 0.0;

    // Calculate pairwise direct space interactions (with cutoff)
    double directEnergy = 0.0;
    for (unsigned int i = 0; i < _numParticles; i++) {
        for (unsigned int j = i + 1; j < _numParticles; j++) {
            double mScale = 1.0;
            double iScale = 1.0;

            // Get scaling factors if within cutoff for exclusions
            if (j <= _maxScaleIndex[i]) {
                mScale = getScaleFactor(i, j, M_SCALE);
                iScale = getScaleFactor(i, j, I_SCALE);
            }

            directEnergy += calculatePmeDirectElectrostaticPairIxn(particleData[i], particleData[j],
                                                            mScale, iScale, forces, torques);
        }
    }

    // Add PME reciprocal space contributions
    calculatePmeSelfTorque(particleData, torques);
    double recipEnergy = computeReciprocalSpaceFixedMultipoleForceAndEnergy(particleData, forces, torques);
    double selfEnergy = calculatePmeSelfEnergy(particleData);

    std::cout << "PME Debug: Direct=" << directEnergy << " Reciprocal=" << recipEnergy
              << " Self=" << selfEnergy << " Total=" << (directEnergy+recipEnergy+selfEnergy) << std::endl;

    energy = directEnergy + recipEnergy + selfEnergy;
    return energy;
}

void ReferencePMETholeDipoleForce::calculateFixedDipoleField(const vector<TholeDipoleParticleData>& particleData)
{
    // First calculate reciprocal space fixed dipole fields
    resizePmeArrays();
    computeAmoebaBsplines(particleData);
    initializePmeGrid();
    spreadFixedMultipolesOntoGrid(particleData);

    // Perform FFT
    vector<size_t> shape = {(size_t) _pmeGridDimensions[0], (size_t) _pmeGridDimensions[1], (size_t) _pmeGridDimensions[2]};
    vector<size_t> axes = {0, 1, 2};
    vector<ptrdiff_t> stride = {(ptrdiff_t) (_pmeGridDimensions[1]*_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) (_pmeGridDimensions[2]*sizeof(complex<double>)),
                                (ptrdiff_t) sizeof(complex<double>)};
    pocketfft::c2c(shape, stride, stride, axes, true, _pmeGrid, _pmeGrid, 1.0, 0);
    performAmoebaReciprocalConvolution();
    pocketfft::c2c(shape, stride, stride, axes, false, _pmeGrid, _pmeGrid, 1.0, 0);

    double totalGridMagAfterIFFT = 0.0;
    for (int i = 0; i < _totalGridSize; i++) {
        totalGridMagAfterIFFT += std::abs(_pmeGrid[i]);
    }
    std::cout << "Grid after inverse FFT: totalMag=" << totalGridMagAfterIFFT << std::endl;

    computeFixedPotentialFromGrid();
    recordFixedMultipoleField();
    Vec3 fieldAfterRecip = _fixedDipoleField[0];

    // Include self-energy portion of the dipole field
    double term = (4.0/3.0)*(_alphaEwald*_alphaEwald*_alphaEwald)/SQRT_PI;
    Vec3 fieldAfterSelf = _fixedDipoleField[0];
    for (unsigned int jj = 0; jj < _numParticles; jj++) {
        Vec3 selfEnergy = particleData[jj].dipole*term;
        _fixedDipoleField[jj] += selfEnergy;
        if (jj == 0) {
            std::cout << "Self-energy term for particle 0: " << selfEnergy << std::endl;
        }
    }
    fieldAfterSelf = _fixedDipoleField[0];

    // Include direct space fixed dipole fields (call base class)
    Vec3 fieldBeforeDirect = _fixedDipoleField[0];
    ReferenceTholeDipoleForce::calculateFixedDipoleField(particleData);
    Vec3 fieldAfterDirect = _fixedDipoleField[0];
    std::cout << "Field components at particle 0:" << std::endl;
    std::cout << "  Reciprocal: " << fieldAfterRecip << std::endl;
    std::cout << "  After self: " << fieldAfterSelf << " (self = " << (fieldAfterSelf - fieldAfterRecip) << ")" << std::endl;
    std::cout << "  After direct: " << fieldAfterDirect << " (direct = " << (fieldAfterDirect - fieldBeforeDirect) << ")" << std::endl;
    std::cout << "  Total magnitude: " << sqrt(fieldAfterDirect.dot(fieldAfterDirect)) << std::endl;
}

void ReferencePMETholeDipoleForce::calculateInducedDipoleFields(const vector<TholeDipoleParticleData>& particleData,
                                                                const vector<Vec3>& inducedDipoles,
                                                                vector<Vec3>& inducedDipoleField)
{
    // TODO: Stage 3 - Implement with reciprocal space induced field contributions
    // For now, call base class (direct space only)
    ReferenceTholeDipoleForce::calculateInducedDipoleFields(particleData, inducedDipoles, inducedDipoleField);
}

void ReferencePMETholeDipoleForce::calculateFixedDipoleFieldPairIxn(const TholeDipoleParticleData& particleI,
                                                                    const TholeDipoleParticleData& particleJ,
                                                                    double mScale, double iScale)
{
    // Compute direct space (erfc-damped) contribution to fixed field for PME
    static bool printed = false;
    if (!printed && particleI.particleIndex == 0 && particleJ.particleIndex == 1) {
        std::cout << "PME calculateFixedDipoleFieldPairIxn called" << std::endl;
        printed = true;
    }

    if (particleI.particleIndex == particleJ.particleIndex)
        return;

    Vec3 deltaR = particleJ.position - particleI.position;
    getPeriodicDelta(deltaR);
    double r2 = deltaR.dot(deltaR);

    if (r2 > _cutoffDistanceSquared)
        return;

    double r = sqrt(r2);

    // Calculate erfc damping terms
    double ralpha = _alphaEwald * r;
    double bn0 = erfc(ralpha) / r;
    double alsq2 = 2.0 * _alphaEwald * _alphaEwald;
    double alsq2n = 1.0 / (SQRT_PI * _alphaEwald);
    double exp2a = exp(-(ralpha * ralpha));
    alsq2n *= alsq2;
    double bn1 = (bn0 + alsq2n * exp2a) / r2;

    alsq2n *= alsq2;
    double bn2 = (3.0 * bn1 + alsq2n * exp2a) / r2;

    // Also need undamped terms (1/r, 1/r³, 1/r⁵) for subtraction
    double rInv = 1.0 / r;
    double rInv2 = rInv * rInv;
    double rInv3 = rInv2 * rInv;
    double rInv5 = rInv3 * rInv2;

    // Dipole dot products
    double djr = particleJ.dipole.dot(deltaR);
    double dir = particleI.dipole.dot(deltaR);

    // Erfc-damped field (fim, fjm from AMOEBA, without quadrupole terms)
    // Field at I from J:
    Vec3 fim = -particleJ.dipole * bn1 - deltaR * (bn1 * particleJ.charge - bn2 * djr);

    // Field at J from I:
    Vec3 fjm = -particleI.dipole * bn1 + deltaR * (bn1 * particleI.charge + bn2 * dir);

    // For PME: direct space uses erfc-damped field only
    // (reciprocal space provides the complementary erf part)
    _fixedDipoleField[particleI.particleIndex] += fim * mScale;
    _fixedDipoleField[particleJ.particleIndex] += fjm * mScale;
}

void ReferencePMETholeDipoleForce::computeBSplinePoint(vector<double4>& thetai, double w)
{
#define ARRAY(x,y) array[(x)-1+((y)-1)*THOLE_PME_ORDER]

    double array[THOLE_PME_ORDER*THOLE_PME_ORDER];

    // Initialization to get to 2nd order recursion
    ARRAY(2,2) = w;
    ARRAY(2,1) = 1.0 - w;

    // Perform one pass to get to 3rd order recursion
    ARRAY(3,3) = 0.5 * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5 * ((1.0+w)*ARRAY(2,1)+(2.0-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5 * (1.0-w) * ARRAY(2,1);

    // Compute standard B-spline recursion to desired order
    for (int i = 4; i <= THOLE_PME_ORDER; i++) {
        int k = i - 1;
        double denom = 1.0 / k;
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1)+(i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1.0-w) * ARRAY(k,1);
    }

    // Get coefficients for the B-spline first derivative
    int k = THOLE_PME_ORDER - 1;
    ARRAY(k,THOLE_PME_ORDER) = ARRAY(k,THOLE_PME_ORDER-1);
    for (int i = THOLE_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // Get coefficients for the B-spline second derivative
    k = THOLE_PME_ORDER - 2;
    ARRAY(k,THOLE_PME_ORDER-1) = ARRAY(k,THOLE_PME_ORDER-2);
    for (int i = THOLE_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,THOLE_PME_ORDER) = ARRAY(k,THOLE_PME_ORDER-1);
    for (int i = THOLE_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // Get coefficients for the B-spline third derivative
    k = THOLE_PME_ORDER - 3;
    ARRAY(k,THOLE_PME_ORDER-2) = ARRAY(k,THOLE_PME_ORDER-3);
    for (int i = THOLE_PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,THOLE_PME_ORDER-1) = ARRAY(k,THOLE_PME_ORDER-2);
    for (int i = THOLE_PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,THOLE_PME_ORDER) = ARRAY(k,THOLE_PME_ORDER-1);
    for (int i = THOLE_PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // Copy coefficients from temporary to permanent storage
    for (int i = 1; i <= THOLE_PME_ORDER; i++)
        thetai[i-1] = double4(ARRAY(THOLE_PME_ORDER,i), ARRAY(THOLE_PME_ORDER-1,i),
                                      ARRAY(THOLE_PME_ORDER-2,i), ARRAY(THOLE_PME_ORDER-3,i));

#undef ARRAY
}

void ReferencePMETholeDipoleForce::computeAmoebaBsplines(const vector<TholeDipoleParticleData>& particleData)
{
    // Get the B-spline coefficients for each particle

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        Vec3 position = particleData[ii].position;
        getPeriodicDelta(position);
        IntVec igrid;
        for (unsigned int jj = 0; jj < 3; jj++) {
            double w  = position[0]*_recipBoxVectors[0][jj]+position[1]*_recipBoxVectors[1][jj]+position[2]*_recipBoxVectors[2][jj];
            double fr = _pmeGridDimensions[jj]*(w-(int)(w+0.5)+0.5);
            int ifr   = static_cast<int>(floor(fr));
            w         = fr - ifr;
            igrid[jj] = ifr - THOLE_PME_ORDER + 1;
            igrid[jj] += igrid[jj] < 0 ? _pmeGridDimensions[jj] : 0;
            vector<double4> thetaiTemp(THOLE_PME_ORDER);
            computeBSplinePoint(thetaiTemp, w);
            for (unsigned int kk = 0; kk < THOLE_PME_ORDER; kk++)
                _thetai[jj][ii*THOLE_PME_ORDER+kk] = thetaiTemp[kk];
        }

        // Record the grid point
        _iGrid[ii] = igrid;
    }
}

void ReferencePMETholeDipoleForce::transformDipolesToFractionalCoordinates(const vector<TholeDipoleParticleData>& particleData)
{
    // Build matrix for transforming dipoles to fractional coordinates
    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[j][i] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    // Transform the multipoles (charge + dipole, no quadrupole)
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
    // Build matrix for transforming potential from fractional to Cartesian
    Vec3 a[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            a[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    // Transform the potential (10 components: charge + 3 dipole + 6 derivatives)
    for (int i = 0; i < _numParticles; i++) {
        cphi[10*i] = fphi[10*i];  // Charge potential (no transformation)

        // Transform dipole potential (components 1-3)
        cphi[10*i+1] = a[0][0]*fphi[10*i+1] + a[0][1]*fphi[10*i+2] + a[0][2]*fphi[10*i+3];
        cphi[10*i+2] = a[1][0]*fphi[10*i+1] + a[1][1]*fphi[10*i+2] + a[1][2]*fphi[10*i+3];
        cphi[10*i+3] = a[2][0]*fphi[10*i+1] + a[2][1]*fphi[10*i+2] + a[2][2]*fphi[10*i+3];

        // Transform derivative components (4-9)
        // For dipoles, second derivatives follow simpler pattern than quadrupoles
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

    // Clear the grid
    for (int gridIndex = 0; gridIndex < _totalGridSize; gridIndex++)
        _pmeGrid[gridIndex] = complex<double>(0, 0);

    // Loop over atoms and spread them on the grid
    for (int atomIndex = 0; atomIndex < _numParticles; atomIndex++) {
        double atomCharge = _transformed[atomIndex].charge;
        Vec3 atomDipole = _transformed[atomIndex].dipole;

        IntVec& gridPoint = _iGrid[atomIndex];
        for (int ix = 0; ix < THOLE_PME_ORDER; ix++) {
            int x = (gridPoint[0]+ix) % _pmeGridDimensions[0];
            double4 t = _thetai[0][atomIndex*THOLE_PME_ORDER+ix];
            for (int iy = 0; iy < THOLE_PME_ORDER; iy++) {
                int y = (gridPoint[1]+iy) % _pmeGridDimensions[1];
                double4 u = _thetai[1][atomIndex*THOLE_PME_ORDER+iy];

                // For charge + dipole (no quadrupole):
                // term0 = charge*t[0]*u[0] + dipole_y*t[0]*u[1] + dipole_x*t[1]*u[0]
                // term1 = dipole_z*t[0]*u[0]
                double term0 = atomCharge*t[0]*u[0] + atomDipole[1]*t[0]*u[1] + atomDipole[0]*t[1]*u[0];
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

    double totalGridMag = 0.0;
    for (int i = 0; i < _totalGridSize; i++) {
        totalGridMag += std::abs(_pmeGrid[i]);
    }
    std::cout << "Grid after spreading: totalMag=" << totalGridMag << " gridSize=" << _totalGridSize << std::endl;
}

void ReferencePMETholeDipoleForce::performAmoebaReciprocalConvolution()
{
    double expFactor   = (M_PI*M_PI)/(_alphaEwald*_alphaEwald);
    double scaleFactor = 1.0/(M_PI*_periodicBoxVectors[0][0]*_periodicBoxVectors[1][1]*_periodicBoxVectors[2][2]);

    for (int index = 0; index < _totalGridSize; index++)
    {
        int kx = index/(_pmeGridDimensions[1]*_pmeGridDimensions[2]);
        int remainder = index-kx*_pmeGridDimensions[1]*_pmeGridDimensions[2];
        int ky = remainder/_pmeGridDimensions[2];
        int kz = remainder-ky*_pmeGridDimensions[2];

        if (kx == 0 && ky == 0 && kz == 0) {
            _pmeGrid[index] = complex<double>(0, 0);
            continue;
        }

        int mx = (kx < (_pmeGridDimensions[0]+1)/2) ? kx : (kx-_pmeGridDimensions[0]);
        int my = (ky < (_pmeGridDimensions[1]+1)/2) ? ky : (ky-_pmeGridDimensions[1]);
        int mz = (kz < (_pmeGridDimensions[2]+1)/2) ? kz : (kz-_pmeGridDimensions[2]);

        double mhx = mx*_recipBoxVectors[0][0];
        double mhy = mx*_recipBoxVectors[1][0]+my*_recipBoxVectors[1][1];
        double mhz = mx*_recipBoxVectors[2][0]+my*_recipBoxVectors[2][1]+mz*_recipBoxVectors[2][2];

        double bx = _pmeBsplineModuli[0][kx];
        double by = _pmeBsplineModuli[1][ky];
        double bz = _pmeBsplineModuli[2][kz];

        double m2 = mhx*mhx+mhy*mhy+mhz*mhz;
        double denom = m2*bx*by*bz;
        double eterm = scaleFactor*exp(-expFactor*m2)/denom;

        _pmeGrid[index] *= eterm;
    }

    double totalGridMag = 0.0;
    for (int i = 0; i < _totalGridSize; i++) {
        totalGridMag += std::abs(_pmeGrid[i]);
    }
    std::cout << "Grid after convolution: totalMag=" << totalGridMag << std::endl;
}

void ReferencePMETholeDipoleForce::computeFixedPotentialFromGrid()
{
    // Extract the permanent multipole potential at each site

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
                double4 t = double4(0.0, 0.0, 0.0, 0.0);

                for (int ix = 0; ix < THOLE_PME_ORDER; ix++) {
                    int i = gridPoint[0]+ix-(gridPoint[0]+ix >= _pmeGridDimensions[0] ? _pmeGridDimensions[0] : 0);
                    int gridIndex = i*_pmeGridDimensions[1]*_pmeGridDimensions[2] + j*_pmeGridDimensions[2] + k;
                    double tq = _pmeGrid[gridIndex].real();
                    double4 tadd = _thetai[0][m*THOLE_PME_ORDER+ix];
                    t[0] += tq*tadd[0];
                    t[1] += tq*tadd[1];
                    t[2] += tq*tadd[2];
                }
                tu00 += t[0]*u[0];
                tu10 += t[1]*u[0];
                tu01 += t[0]*u[1];
                tu20 += t[2]*u[0];
                tu11 += t[1]*u[1];
                tu02 += t[0]*u[2];
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

        // Store potential (10 components for charge + dipole)
        _phi[10*m] = tuv000;
        _phi[10*m+1] = tuv100;
        _phi[10*m+2] = tuv010;
        _phi[10*m+3] = tuv001;
        _phi[10*m+4] = tuv200;
        _phi[10*m+5] = tuv020;
        _phi[10*m+6] = tuv002;
        _phi[10*m+7] = tuv110;
        _phi[10*m+8] = tuv101;
        _phi[10*m+9] = tuv011;

        if (m == 0) {
            std::cout << "Phi[0]: " << tuv000 << " " << tuv100 << " " << tuv010 << " " << tuv001 << std::endl;
        }
    }
}

void ReferencePMETholeDipoleForce::computeInducedPotentialFromGrid()
{
    // TODO: Stage 3 - Implement induced potential interpolation
}

double ReferencePMETholeDipoleForce::computeReciprocalSpaceFixedMultipoleForceAndEnergy(
    const vector<TholeDipoleParticleData>& particleData,
    vector<Vec3>& forces, vector<Vec3>& torques) const
{
    // Derivative indices for force calculation
    const int deriv1[] = {1, 4, 7, 8};  // x-derivatives (reduced from AMOEBA's 10)
    const int deriv2[] = {2, 7, 5, 9};  // y-derivatives
    const int deriv3[] = {3, 8, 9, 6};  // z-derivatives

    vector<double> cphi(10*_numParticles);
    transformPotentialToCartesianCoordinates(_phi, cphi);

    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    double energy = 0.0;

    for (int i = 0; i < _numParticles; i++) {
        // Compute the torque for charge + dipole
        double multipole[4];
        multipole[0] = particleData[i].charge;
        multipole[1] = particleData[i].dipole[0];
        multipole[2] = particleData[i].dipole[1];
        multipole[3] = particleData[i].dipole[2];

        const double* phi = &cphi[10*i];
        // Torque from dipole cross field
        torques[i][0] += _electric*(multipole[3]*phi[2] - multipole[2]*phi[3]);
        torques[i][1] += _electric*(multipole[1]*phi[3] - multipole[3]*phi[1]);
        torques[i][2] += _electric*(multipole[2]*phi[1] - multipole[1]*phi[2]);

        // Compute the force and energy
        // Use transformed dipoles for force calculation
        multipole[1] = _transformed[i].dipole[0];
        multipole[2] = _transformed[i].dipole[1];
        multipole[3] = _transformed[i].dipole[2];

        Vec3 f = Vec3(0.0, 0.0, 0.0);
        for (int k = 0; k < 4; k++) {
            energy += multipole[k]*_phi[10*i+k];
            f[0]   += multipole[k]*_phi[10*i+deriv1[k]];
            f[1]   += multipole[k]*_phi[10*i+deriv2[k]];
            f[2]   += multipole[k]*_phi[10*i+deriv3[k]];
        }
        f *= (_electric);
        forces[i] -= Vec3(f[0]*fracToCart[0][0] + f[1]*fracToCart[0][1] + f[2]*fracToCart[0][2],
                          f[0]*fracToCart[1][0] + f[1]*fracToCart[1][1] + f[2]*fracToCart[1][2],
                          f[0]*fracToCart[2][0] + f[1]*fracToCart[2][1] + f[2]*fracToCart[2][2]);
    }

    double rawEnergy = energy;
    double scaledEnergy_WithElectric = 0.5*_electric*energy;
    double scaledEnergy_NoElectric = 0.5*energy;  // TEST
    std::cout << "Reciprocal energy: raw=" << rawEnergy << " WithElectric=" << scaledEnergy_WithElectric
              << " NoElectric=" << scaledEnergy_NoElectric << std::endl;

    return scaledEnergy_NoElectric;  // Remove _electric (field was divided by _electric)
}

void ReferencePMETholeDipoleForce::recordFixedMultipoleField()
{
    // Transform field from fractional to Cartesian coordinates
    Vec3 fracToCart[3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            fracToCart[i][j] = _pmeGridDimensions[j]*_recipBoxVectors[i][j];

    for (int i = 0; i < _numParticles; i++) {
        // The grid potential/field is in "full" units, so divide by _electric to get "reduced" units
        // matching the direct space field calculation
        double fieldScale = 1.0 / _electric;
        _fixedDipoleField[i][0] = fieldScale * (-(_phi[10*i+1]*fracToCart[0][0] + _phi[10*i+2]*fracToCart[0][1] + _phi[10*i+3]*fracToCart[0][2]));
        _fixedDipoleField[i][1] = fieldScale * (-(_phi[10*i+1]*fracToCart[1][0] + _phi[10*i+2]*fracToCart[1][1] + _phi[10*i+3]*fracToCart[1][2]));
        _fixedDipoleField[i][2] = fieldScale * (-(_phi[10*i+1]*fracToCart[2][0] + _phi[10*i+2]*fracToCart[2][1] + _phi[10*i+3]*fracToCart[2][2]));
        if (i == 0) {
            std::cout << "Reciprocal field at particle 0 (scaled by 1/_electric=" << fieldScale << "): " << _fixedDipoleField[i] << std::endl;
        }
    }
}

void ReferencePMETholeDipoleForce::calculateReciprocalSpaceInducedDipoleField()
{
    // TODO: Stage 3 - Implement reciprocal induced field
}

void ReferencePMETholeDipoleForce::calculateDirectInducedDipolePairIxn(unsigned int iIndex, unsigned int jIndex,
                                                                       double preFactor, const Vec3& delta,
                                                                       const vector<Vec3>& inducedDipole,
                                                                       vector<Vec3>& field) const
{
    // TODO: Stage 3 - Implement direct induced dipole interaction
}

void ReferencePMETholeDipoleForce::calculateDirectInducedDipolePairIxns(const TholeDipoleParticleData& particleI,
                                                                        const TholeDipoleParticleData& particleJ)
{
    // TODO: Stage 3 - Implement direct induced dipole pair interaction
}

void ReferencePMETholeDipoleForce::spreadInducedDipolesOnGrid(const vector<Vec3>& inputInducedDipole)
{
    // TODO: Stage 3 - Spread induced dipoles onto grid
}

void ReferencePMETholeDipoleForce::recordInducedDipoleField(vector<Vec3>& field)
{
    // TODO: Stage 3 - Record induced dipole field
}

double ReferencePMETholeDipoleForce::calculatePmeSelfEnergy(const vector<TholeDipoleParticleData>& particleData) const
{
    double cii = 0.0;
    double dii = 0.0;
    double totalCharge = 0.0;

    for (unsigned int ii = 0; ii < _numParticles; ii++) {
        const TholeDipoleParticleData& particleI = particleData[ii];

        totalCharge += particleI.charge;
        cii += particleI.charge*particleI.charge;

        // For dipoles: Try WITHOUT induced dipole to test
        double dii_perm = particleI.dipole.dot(particleI.dipole);
        double dii_ind = particleI.dipole.dot(_inducedDipole[ii]*0.5);
        //dii += dii_perm + dii_ind;  // AMOEBA formula
        dii += dii_perm;  // TEST: only permanent dipoles

        if (ii == 0) {
            std::cout << "Particle 0: dipole=" << particleI.dipole << " induced=" << _inducedDipole[ii]
                      << " dii_perm=" << dii_perm << " dii_ind=" << dii_ind << std::endl;
        }
    }
    std::cout << "Self-energy test: using ONLY permanent dipoles (no induced)" << std::endl;

    // TEST: Try without _electric factor
    double prefac_noElectric = -_alphaEwald / (_dielectric*SQRT_PI);
    double prefac = -_alphaEwald * _electric / (_dielectric*SQRT_PI);
    double a2 = _alphaEwald * _alphaEwald;
    double twoThirds = 2.0/3.0;

    std::cout << "Self-energy prefac components: alpha=" << _alphaEwald << " electric=" << _electric
              << " dielectric=" << _dielectric << " SQRT_PI=" << SQRT_PI << std::endl;
    std::cout << "  prefac WITH electric=" << prefac << " WITHOUT electric=" << prefac_noElectric << std::endl;
    std::cout << "  AMOEBA uses: prefac = -alphaEwald * _electric / (dielectric*SQRT_PI)" << std::endl;

    // For charge + dipole (no quadrupole term)
    // Use prefac WITHOUT _electric (field was divided by _electric)
    double chargeTerm = prefac_noElectric*cii;
    double dipoleTerm = prefac_noElectric*twoThirds*a2*dii;
    double energy = chargeTerm + dipoleTerm;

    // Correction for the neutralizing plasma
    double volume = _periodicBoxVectors[0][0] * _periodicBoxVectors[1][1] * _periodicBoxVectors[2][2];
    double plasmaTerm = totalCharge*totalCharge*M_PI/(2.0*volume*_alphaEwald*_alphaEwald);
    energy -= plasmaTerm;

    std::cout << "Self-energy debug: cii=" << cii << " dii=" << dii << " chargeTerm=" << chargeTerm
              << " dipoleTerm=" << dipoleTerm << " plasmaTerm=" << plasmaTerm << " total=" << energy << std::endl;

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

    // Calculate erfc damping terms (matching field calculation)
    double ralpha = _alphaEwald * r;
    double bn0 = erfc(ralpha) / r;
    double alsq2 = 2.0 * _alphaEwald * _alphaEwald;
    double alsq2n = 1.0 / (SQRT_PI * _alphaEwald);
    double exp2a = exp(-(ralpha * ralpha));
    alsq2n *= alsq2;
    double bn1 = (bn0 + alsq2n * exp2a) / r2;

    alsq2n *= alsq2;
    double bn2 = (3.0 * bn1 + alsq2n * exp2a) / r2;

    // Calculate bn3 for force derivatives
    alsq2n *= alsq2;
    double bn3 = (5.0 * bn2 + alsq2n * exp2a) / r2;

    // Dipole dot products
    double dIr = particleI.dipole.dot(deltaR);
    double dJr = particleJ.dipole.dot(deltaR);
    double dIdJ = particleI.dipole.dot(particleJ.dipole);

    // PME erfc-damped energy components
    double qIqJ = particleI.charge * particleJ.charge;
    double qIdJr = particleI.charge * dJr;
    double qJdIr = particleJ.charge * dIr;

    // Erfc-damped energy (simplified for charge + dipole, no quadrupoles)
    double erfcEnergy = bn0 * qIqJ + bn1 * (qIdJr - qJdIr) + bn2 * dIr * dJr - bn1 * dIdJ;

    // PME direct space energy (negated to match convention)
    double pmeDirectEnergy = -erfcEnergy * mScale * (_electric / _dielectric);

    // Calculate PME direct space forces with proper tensor derivatives
    // Key insight: ∇(μ·r̂) = (μ - (μ·r̂)r̂)/r = mu_perp/r
    // This requires perpendicular components for orientational terms

    Vec3 rhat = deltaR / r;
    double rInv = 1.0 / r;

    // Projections
    double muIr = particleI.dipole.dot(rhat);
    double muJr = particleJ.dipole.dot(rhat);
    double muIdotMuJ = particleI.dipole.dot(particleJ.dipole);

    // Perpendicular components: mu_perp = μ - (μ·r̂)r̂
    Vec3 muI_perp = particleI.dipole - muIr * rhat;
    Vec3 muJ_perp = particleJ.dipole - muJr * rhat;

    // Initialize force
    Vec3 force(0.0, 0.0, 0.0);

    // (1) Charge-charge: E = bn0*qI*qJ
    //     F = -∇E = qI*qJ*bn1*deltaR (using dbn0/dr = -r*bn1)
    force += qIqJ * bn1 * deltaR;

    // (2) Charge-dipole: E = bn1*(qI*muJr - qJ*muIr)
    //     Radial: bn2*(qI*muJr - qJ*muIr)*deltaR
    //     Orientational: bn1*rInv*(qI*muJ_perp - qJ*muI_perp)
    force += bn2 * (particleI.charge * muJr - particleJ.charge * muIr) * deltaR;
    force += bn1 * rInv * (particleI.charge * muJ_perp - particleJ.charge * muI_perp);

    // (3) Dipole-dipole: E = bn2*muIr*muJr - bn1*muIdotMuJ
    //     Radial: (bn3*muIr*muJr - bn2*muIdotMuJ)*deltaR
    //     Orientational: bn2*rInv*(muJr*muI_perp + muIr*muJ_perp)
    force += (bn3 * muIr * muJr - bn2 * muIdotMuJ) * deltaR;
    force += bn2 * rInv * (muJr * muI_perp + muIr * muJ_perp);

    // Fields for torques
    Vec3 fieldAtI = -particleJ.charge * bn1 * rhat + (bn2 * muJr * rhat - bn1 * particleJ.dipole);
    Vec3 fieldAtJ = particleI.charge * bn1 * rhat + (bn2 * muIr * rhat - bn1 * particleI.dipole);

    // Apply mScale and convert to proper units
    // force already represents -∇erfcEnergy, apply mScale and unit conversion
    Vec3 forceTotal = force * mScale * (_electric / _dielectric);
    forces[iIndex] -= forceTotal;
    forces[jIndex] += forceTotal;

    // Torques: τ = μ × E
    Vec3 torqueI = particleI.dipole.cross(fieldAtI) * mScale * (_electric / _dielectric);
    Vec3 torqueJ = particleJ.dipole.cross(fieldAtJ) * mScale * (_electric / _dielectric);

    torques[iIndex] += torqueI;
    torques[jIndex] += torqueJ;

    return pmeDirectEnergy;
}

double ReferencePMETholeDipoleForce::computeReciprocalSpaceInducedDipoleForceAndEnergy(
    const vector<TholeDipoleParticleData>& particleData,
    vector<Vec3>& forces, vector<Vec3>& torques) const
{
    // TODO: Stage 3 - Implement reciprocal induced dipole energy/forces
    return 0.0;
}
