#ifndef OPENMM_REFERENCEPMETHOLEDIPOLEFORCE_H_
#define OPENMM_REFERENCEPMETHOLEDIPOLEFORCE_H_

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

#include "ReferenceTholeDipoleForce.h"
#include "openmm/Vec3.h"
#include <complex>
#include <vector>

namespace TholeDipolePlugin {

using OpenMM::Vec3;
using std::vector;

/**
 * 3-element int vector for grid indices
 */
class IntVec {
public:
    IntVec() {
        data[0] = data[1] = data[2] = 0;
    }
    IntVec(int x, int y, int z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
    int operator[](int index) const {
        return data[index];
    }
    int& operator[](int index) {
        return data[index];
    }
private:
    int data[3];
};

/**
 * 4-element double vector for B-spline coefficients
 */
class double4 {
public:
    double4() {
        data[0] = data[1] = data[2] = data[3] = 0.0;
    }
    double4(double x, double y, double z, double w) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
    double operator[](int index) const {
        return data[index];
    }
    double& operator[](int index) {
        return data[index];
    }
    double4 operator+(const double4& rhs) const {
        return double4(data[0] + rhs[0], data[1] + rhs[1], data[2] + rhs[2], data[3] + rhs[3]);
    }
    double4& operator+=(const double4& rhs) {
        data[0] += rhs[0];
        data[1] += rhs[1];
        data[2] += rhs[2];
        data[3] += rhs[3];
        return *this;
    }
    double4& operator-=(const double4& rhs) {
        data[0] -= rhs[0];
        data[1] -= rhs[1];
        data[2] -= rhs[2];
        data[3] -= rhs[3];
        return *this;
    }
    double4 operator*(double rhs) const {
        return double4(data[0]*rhs, data[1]*rhs, data[2]*rhs, data[3]*rhs);
    }
private:
    double data[4];
};

/**
 * This class implements PME for the Thole dipole force.
 * It extends ReferenceTholeDipoleForce to add reciprocal space calculations
 * using Particle Mesh Ewald (PME) for charge and dipole interactions.
 */
class ReferencePMETholeDipoleForce : public ReferenceTholeDipoleForce {

public:

    /**
     * Constructor
     */
    ReferencePMETholeDipoleForce();

    /**
     * Destructor
     */
    ~ReferencePMETholeDipoleForce();

    /**
     * Get cutoff distance.
     *
     * @return cutoff distance
     */
    double getCutoffDistance() const;

    /**
     * Set cutoff distance.
     *
     * @param cutoffDistance the cutoff distance
     */
    void setCutoffDistance(double cutoffDistance);

    /**
     * Get alpha used in Ewald summation.
     *
     * @return alpha
     */
    double getAlphaEwald() const;

    /**
     * Set alpha used in Ewald summation.
     *
     * @param alphaEwald the Ewald alpha parameter
     */
    void setAlphaEwald(double alphaEwald);

    /**
     * Get PME grid dimensions.
     *
     * @param pmeGridDimensions output vector containing PME grid dimensions
     */
    void getPmeGridDimensions(vector<int>& pmeGridDimensions) const;

    /**
     * Set PME grid dimensions.
     *
     * @param pmeGridDimensions input PME grid dimensions
     */
    void setPmeGridDimensions(vector<int>& pmeGridDimensions);

    /**
     * Set periodic box size.
     *
     * @param vectors the vectors defining the periodic box
     */
     void setPeriodicBoxSize(OpenMM::Vec3* vectors);

protected:

    /**
     * Calculate electrostatic forces and energy.
     * Overrides base class to add reciprocal space contributions.
     *
     * @param particleData vector of particle data
     * @param torques      output torques
     * @param forces       output forces
     *
     * @return energy
     */
    double calculateElectrostatic(const vector<TholeDipoleParticleData>& particleData,
                                   vector<Vec3>& torques, vector<Vec3>& forces) override;

    /**
     * Calculate fixed dipole field.
     * Overrides base class to add reciprocal space field contributions.
     *
     * @param particleData vector of particle data
     */
    void calculateFixedDipoleField(const vector<TholeDipoleParticleData>& particleData) override;

    /**
     * Calculate induced dipole fields during mutual polarization.
     * This shadows the base class function to add reciprocal space field contributions.
     *
     * @param particleData      vector of particle data
     * @param inducedDipoles    current induced dipole values
     * @param inducedDipoleField output field from induced dipoles
     */
    void calculateInducedDipoleFields(const vector<TholeDipoleParticleData>& particleData,
                                      const vector<Vec3>& inducedDipoles,
                                      vector<Vec3>& inducedDipoleField);

private:

    static const int THOLE_PME_ORDER;
    static const double SQRT_PI;

    double _alphaEwald;
    double _cutoffDistance;
    double _cutoffDistanceSquared;

    Vec3 _recipBoxVectors[3];
    Vec3 _periodicBoxVectors[3];

    int _totalGridSize;
    IntVec _pmeGridDimensions;

    unsigned int _pmeGridSize;
    std::complex<double>* _pmeGrid;

    vector<double> _pmeBsplineModuli[3];
    vector<double4> _thetai[3];
    vector<IntVec> _iGrid;
    vector<double> _phi;
    vector<double> _phid;
    vector<double> _phidp;
    vector<double4> _pmeBsplineTheta;
    vector<double4> _pmeBsplineDtheta;
    vector<TholeDipoleParticleData> _transformed;

    /**
     * Resize PME arrays.
     */
    void resizePmeArrays();

    /**
     * Zero PME grid.
     */
    void initializePmeGrid();

    /**
     * Modify input vector of differences in particle positions for periodic boundary conditions.
     *
     * @param deltaR input/output vector of difference in particle positions
     */
    void getPeriodicDelta(Vec3& deltaR) const;

    /**
     * Initialize B-spline moduli.
     */
    void initializeBSplineModuli();

    /**
     * Calculate direct-space field at site I due fixed multipoles at site J and vice versa.
     * Uses erfc damping for PME.
     *
     * @param particleI positions and parameters for particle I
     * @param particleJ positions and parameters for particle J
     * @param mScale    multipole scale factor for i-j interaction
     * @param iScale    induced dipole scale factor for i-j interaction
     */
    void calculateFixedDipoleFieldPairIxn(const TholeDipoleParticleData& particleI,
                                          const TholeDipoleParticleData& particleJ,
                                          double mScale, double iScale);

    /**
     * Compute B-spline coefficients for a single atom along a single axis.
     *
     * @param thetai output spline coefficients
     * @param w      offset from grid point
     */
    void computeBSplinePoint(vector<double4>& thetai, double w);

    /**
     * Compute B-spline coefficients for all atoms.
     *
     * @param particleData vector of particle data
     */
    void computeAmoebaBsplines(const vector<TholeDipoleParticleData>& particleData);

    /**
     * Transform dipoles from Cartesian to fractional coordinates.
     *
     * @param particleData vector of particle data
     */
    void transformDipolesToFractionalCoordinates(const vector<TholeDipoleParticleData>& particleData);

    /**
     * Transform potential from fractional to Cartesian coordinates.
     *
     * @param fphi input fractional potential
     * @param cphi output Cartesian potential
     */
    void transformPotentialToCartesianCoordinates(const vector<double>& fphi, vector<double>& cphi) const;

    /**
     * Spread fixed charges and dipoles onto PME grid.
     *
     * @param particleData vector of particle data
     */
    void spreadFixedMultipolesOntoGrid(const vector<TholeDipoleParticleData>& particleData);

    /**
     * Perform reciprocal convolution using FFT.
     */
    void performAmoebaReciprocalConvolution();

    /**
     * Compute reciprocal potential at each particle site from grid.
     */
    void computeFixedPotentialFromGrid();

    /**
     * Compute reciprocal potential for induced dipoles from grid.
     */
    void computeInducedPotentialFromGrid();

    /**
     * Calculate reciprocal space energy and force due to fixed charges and dipoles.
     *
     * @param particleData vector of particle data
     * @param forces       output forces
     * @param torques      output torques
     *
     * @return energy
     */
    double computeReciprocalSpaceFixedMultipoleForceAndEnergy(const vector<TholeDipoleParticleData>& particleData,
                                                              vector<Vec3>& forces, vector<Vec3>& torques) const;

    /**
     * Set reciprocal space fixed dipole fields.
     */
    void recordFixedMultipoleField();

    /**
     * Compute reciprocal space induced dipole field.
     */
    void calculateReciprocalSpaceInducedDipoleField();

    /**
     * Calculate direct space field due to induced dipole.
     *
     * @param iIndex        particle I index
     * @param jIndex        particle J index
     * @param preFactor     factor used in calculating field
     * @param delta         delta in particle positions
     * @param inducedDipole induced dipole
     * @param field         output field
     */
    void calculateDirectInducedDipolePairIxn(unsigned int iIndex, unsigned int jIndex,
                                             double preFactor, const Vec3& delta,
                                             const vector<Vec3>& inducedDipole,
                                             vector<Vec3>& field) const;

    /**
     * Calculate direct space induced dipole field for particle pair.
     *
     * @param particleI    particle I data
     * @param particleJ    particle J data
     */
    void calculateDirectInducedDipolePairIxns(const TholeDipoleParticleData& particleI,
                                              const TholeDipoleParticleData& particleJ);

    /**
     * Spread induced dipoles onto grid.
     *
     * @param inputInducedDipole induced dipole values
     */
    void spreadInducedDipolesOnGrid(const vector<Vec3>& inputInducedDipole);

    /**
     * Set reciprocal space induced dipole field.
     *
     * @param field reciprocal space induced dipole field
     */
    void recordInducedDipoleField(vector<Vec3>& field);

    /**
     * Compute PME self energy.
     *
     * @param particleData vector of particle data
     *
     * @return self energy
     */
    double calculatePmeSelfEnergy(const vector<TholeDipoleParticleData>& particleData) const;

    /**
     * Compute PME self torques.
     *
     * @param particleData vector of particle data
     * @param torques      output torques
     */
    void calculatePmeSelfTorque(const vector<TholeDipoleParticleData>& particleData, vector<Vec3>& torques) const;

    /**
     * Calculate direct space electrostatic interaction between particles I and J with erfc damping.
     *
     * @param particleI      particle I data
     * @param particleJ      particle J data
     * @param mScale         multipole scale factor
     * @param iScale         induced scale factor
     * @param forces         output forces
     * @param torques        output torques
     *
     * @return interaction energy
     */
    double calculatePmeDirectElectrostaticPairIxn(const TholeDipoleParticleData& particleI,
                                                  const TholeDipoleParticleData& particleJ,
                                                  double mScale, double iScale,
                                                  vector<Vec3>& forces, vector<Vec3>& torques) const;

    /**
     * Calculate reciprocal space energy/force/torque for induced dipole interaction.
     *
     * @param particleData vector of particle data
     * @param forces       output forces
     * @param torques      output torques
     *
     * @return energy
     */
     double computeReciprocalSpaceInducedDipoleForceAndEnergy(const vector<TholeDipoleParticleData>& particleData,
                                                              vector<Vec3>& forces, vector<Vec3>& torques) const;

};

} // namespace TholeDipolePlugin

#endif // OPENMM_REFERENCEPMETHOLEDIPOLEFORCE_H_
