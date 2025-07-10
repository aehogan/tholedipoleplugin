/* -------------------------------------------------------------------------- *
 *                            OpenMMTholeDipole                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2024 Stanford University and the Authors.           *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#ifndef __ReferenceTholeDipoleForce_H__
#define __ReferenceTholeDipoleForce_H__

#include "TholeDipoleForce.h"
#include "openmm/Vec3.h"
#include <map>
#include <complex>

namespace TholeDipolePlugin {

using namespace OpenMM;

typedef std::map<unsigned int, double> MapIntRealOpenMM;
typedef MapIntRealOpenMM::iterator MapIntRealOpenMMI;
typedef MapIntRealOpenMM::const_iterator MapIntRealOpenMMCI;

/**
 * This class implements the Thole/Applequist polarizable dipole model in the reference platform.
 * It provides a simplified version of the AMOEBA multipole force with only charge and dipole
 * moments (no quadrupoles), and a single induced dipole per atom.
 */
class ReferenceTholeDipoleForce {

public:
    
    /**
     * This is an enumeration of the different methods that may be used for handling long range forces.
     */
    enum NonbondedMethod {
        /**
         * No cutoff is applied to the interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 1
    };

    enum PolarizationType {
        /**
         * Direct polarization approximation.  The induced dipoles depend only on the fixed dipoles, not on other
         * induced dipoles.
         */
        Direct = 0,
        /**
         * Full mutually induced polarization.  The dipoles are iterated until they converge to the accuracy specified
         * by getMutualInducedTargetEpsilon().
         */
        Mutual = 1,
        /**
         * Extrapolated perturbation theory approximation.  The dipoles are iterated a few times, and then an analytic
         * approximation is used to extrapolate to the fully converged values.
         */
        Extrapolated = 2
    };

    /**
     * Constructor
     */
    ReferenceTholeDipoleForce();

    /**
     * Constructor
     * 
     * @param nonbondedMethod nonbonded method
     */
    ReferenceTholeDipoleForce(NonbondedMethod nonbondedMethod);

    /**
     * Destructor
     */
    virtual ~ReferenceTholeDipoleForce();

    /**
     * Get nonbonded method.
     * 
     * @return nonbonded method
     */
    NonbondedMethod getNonbondedMethod() const;

    /**
     * Set nonbonded method.
     * 
     * @param nonbondedMethod nonbonded method
     */
    void setNonbondedMethod(NonbondedMethod nonbondedMethod);

    /**
     * Get polarization type.
     * 
     * @return polarization type
     */
    PolarizationType getPolarizationType() const;

    /**
     * Set polarization type.
     * 
     * @param polarizationType polarization type
     */
    void setPolarizationType(PolarizationType polarizationType);

    /**
     * Get flag indicating if mutual induced dipoles are converged.
     *
     * @return nonzero if converged
     */
    int getMutualInducedDipoleConverged() const;

    /**
     * Get the number of iterations used in computing mutual induced dipoles.
     *
     * @return number of iterations
     */
    int getMutualInducedDipoleIterations() const;

    /**
     * Get the final epsilon for mutual induced dipoles.
     *
     * @return epsilon
     */
    double getMutualInducedDipoleEpsilon() const;

    /**
     * Set the coefficients for the μ_0, μ_1, μ_2, μ_n terms in the extrapolation
     * theory algorithm for induced dipoles
     *
     * @param coefficients a vector whose mth entry specifies the coefficient for μ_m
     */
    void setExtrapolationCoefficients(const std::vector<double> &coefficients);

    /**
     * Set the target epsilon for converging mutual induced dipoles.
     *
     * @param targetEpsilon target epsilon for converging mutual induced dipoles
     */
    void setMutualInducedDipoleTargetEpsilon(double targetEpsilon);

    /**
     * Get the target epsilon for converging mutual induced dipoles.
     *
     * @return target epsilon for converging mutual induced dipoles
     */
    double getMutualInducedDipoleTargetEpsilon() const;

    /**
     * Set the maximum number of iterations to be executed in converging mutual induced dipoles.
     *
     * @param maximumMutualInducedDipoleIterations maximum number of iterations to be executed in converging mutual induced dipoles
     */
    void setMaximumMutualInducedDipoleIterations(int maximumMutualInducedDipoleIterations);

    /**
     * Get the maximum number of iterations to be executed in converging mutual induced dipoles.
     *
     * @return maximum number of iterations to be executed in converging mutual induced dipoles
     */
    int getMaximumMutualInducedDipoleIterations() const;

    /**
     * Calculate force and energy.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarizabilities          polarizabilities for each particle
     * @param tholeDampingFactors       Thole damping factors for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indices of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indices of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indices of particle specifying the molecular frame y-axis for each particle
     * @param multipoleCovalentInfo     covalent info needed to set scaling factors
     * @param forces                    add forces to this vector
     *
     * @return energy
     */
    double calculateForceAndEnergy(const std::vector<OpenMM::Vec3>& particlePositions,
                                   const std::vector<double>& charges,
                                   const std::vector<double>& dipoles,
                                   const std::vector<double>& polarizabilities,
                                   const std::vector<double>& tholeDampingFactors,
                                   const std::vector<int>& axisTypes,
                                   const std::vector<int>& multipoleAtomZs,
                                   const std::vector<int>& multipoleAtomXs,
                                   const std::vector<int>& multipoleAtomYs,
                                   const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                   std::vector<OpenMM::Vec3>& forces);

    /**
     * Calculate particle induced dipoles.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarizabilities          polarizabilities for each particle
     * @param tholeDampingFactors       Thole damping factors for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indices of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indices of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indices of particle specifying the molecular frame y-axis for each particle
     * @param multipoleCovalentInfo     covalent info needed to set scaling factors
     * @param outputInducedDipoles      output induced dipoles
     */
    void calculateInducedDipoles(const std::vector<OpenMM::Vec3>& particlePositions,
                                 const std::vector<double>& charges,
                                 const std::vector<double>& dipoles,
                                 const std::vector<double>& polarizabilities,
                                 const std::vector<double>& tholeDampingFactors,
                                 const std::vector<int>& axisTypes,
                                 const std::vector<int>& multipoleAtomZs,
                                 const std::vector<int>& multipoleAtomXs,
                                 const std::vector<int>& multipoleAtomYs,
                                 const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                 std::vector<Vec3>& outputInducedDipoles);

    /**
     * Calculate particle permanent dipoles rotated in the lab frame.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarizabilities          polarizabilities for each particle
     * @param tholeDampingFactors       Thole damping factors for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indices of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indices of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indices of particle specifying the molecular frame y-axis for each particle
     * @param multipoleCovalentInfo     covalent info needed to set scaling factors
     * @param outputRotatedPermanentDipoles output permanent dipoles
     */
    void calculateLabFramePermanentDipoles(const std::vector<Vec3>& particlePositions,
                                           const std::vector<double>& charges,
                                           const std::vector<double>& dipoles,
                                           const std::vector<double>& polarizabilities,
                                           const std::vector<double>& tholeDampingFactors,
                                           const std::vector<int>& axisTypes,
                                           const std::vector<int>& multipoleAtomZs,
                                           const std::vector<int>& multipoleAtomXs,
                                           const std::vector<int>& multipoleAtomYs,
                                           const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                           std::vector<Vec3>& outputRotatedPermanentDipoles);

    /**
     * Calculate particle total dipoles.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarizabilities          polarizabilities for each particle
     * @param tholeDampingFactors       Thole damping factors for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indices of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indices of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indices of particle specifying the molecular frame y-axis for each particle
     * @param multipoleCovalentInfo     covalent info needed to set scaling factors
     * @param outputTotalDipoles        output total dipoles
     */
    void calculateTotalDipoles(const std::vector<Vec3>& particlePositions,
                               const std::vector<double>& charges,
                               const std::vector<double>& dipoles,
                               const std::vector<double>& polarizabilities,
                               const std::vector<double>& tholeDampingFactors,
                               const std::vector<int>& axisTypes,
                               const std::vector<int>& multipoleAtomZs,
                               const std::vector<int>& multipoleAtomXs,
                               const std::vector<int>& multipoleAtomYs,
                               const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                               std::vector<Vec3>& outputTotalDipoles);

    /**
     * Calculate system multipole moments.
     *
     * @param masses                    particle masses
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarizabilities          polarizabilities for each particle
     * @param tholeDampingFactors       Thole damping factors for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indices of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indices of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indices of particle specifying the molecular frame y-axis for each particle
     * @param multipoleCovalentInfo     covalent info needed to set scaling factors
     * @param outputMultipoleMoments    output multipole moments
     */
    void calculateTholeDipoleSystemMultipoleMoments(const std::vector<double>& masses,
                                                    const std::vector<OpenMM::Vec3>& particlePositions,
                                                    const std::vector<double>& charges,
                                                    const std::vector<double>& dipoles,
                                                    const std::vector<double>& polarizabilities,
                                                    const std::vector<double>& tholeDampingFactors,
                                                    const std::vector<int>& axisTypes,
                                                    const std::vector<int>& multipoleAtomZs,
                                                    const std::vector<int>& multipoleAtomXs,
                                                    const std::vector<int>& multipoleAtomYs,
                                                    const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                                    std::vector<double>& outputMultipoleMoments);

    /**
     * Calculate electrostatic potential at a set of grid points.
     *
     * @param particlePositions         Cartesian coordinates of particles
     * @param charges                   scalar charges for each particle
     * @param dipoles                   molecular frame dipoles for each particle
     * @param polarizabilities          polarizabilities for each particle
     * @param tholeDampingFactors       Thole damping factors for each particle
     * @param axisTypes                 axis type (Z-then-X, ...) for each particle
     * @param multipoleAtomZs           indices of particle specifying the molecular frame z-axis for each particle
     * @param multipoleAtomXs           indices of particle specifying the molecular frame x-axis for each particle
     * @param multipoleAtomYs           indices of particle specifying the molecular frame y-axis for each particle
     * @param multipoleCovalentInfo     covalent info needed to set scaling factors
     * @param inputGrid                 input grid points to compute potential
     * @param outputPotential           output electrostatic potential
     */
    void calculateElectrostaticPotential(const std::vector<OpenMM::Vec3>& particlePositions,
                                         const std::vector<double>& charges,
                                         const std::vector<double>& dipoles,
                                         const std::vector<double>& polarizabilities,
                                         const std::vector<double>& tholeDampingFactors,
                                         const std::vector<int>& axisTypes,
                                         const std::vector<int>& multipoleAtomZs,
                                         const std::vector<int>& multipoleAtomXs,
                                         const std::vector<int>& multipoleAtomYs,
                                         const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo,
                                         const std::vector<Vec3>& inputGrid,
                                         std::vector<double>& outputPotential);

protected:

    /**
     * Particle parameters and coordinates for TholeDipole calculations
     */
    class TholeDipoleParticleData {
    public:
        unsigned int particleIndex;
        Vec3 position;
        double charge;
        Vec3 dipole;
        Vec3 inducedDipole;
        double polarizability;
        double tholeDamping;
        int axisType;
        int multipoleAtomZ;
        int multipoleAtomX;
        int multipoleAtomY;
    };

    enum ScaleType { M_SCALE, I_SCALE, LAST_SCALE_TYPE_INDEX };

    unsigned int _numParticles;
    NonbondedMethod _nonbondedMethod;
    PolarizationType _polarizationType;

    double _electric;
    double _dielectric;

    std::vector<std::vector<MapIntRealOpenMM>> _scaleMaps;
    std::vector<unsigned int> _maxScaleIndex;
    double _mScale[5];
    double _iScale[5];

    std::vector<Vec3> _fixedDipoleField;
    std::vector<Vec3> _inducedDipole;

    int _mutualInducedDipoleConverged;
    int _mutualInducedDipoleIterations;
    int _maximumMutualInducedDipoleIterations;
    std::vector<double> _extrapolationCoefficients;
    double _mutualInducedDipoleEpsilon;
    double _mutualInducedDipoleTargetEpsilon;
    double _debye;

    /**
     * Helper constructor method to centralize initialization of objects.
     */
    void initialize();

    /**
     * Load particle data.
     *
     * @param particlePositions   particle coordinates
     * @param charges             charges
     * @param dipoles             dipoles
     * @param polarizabilities    polarizabilities
     * @param tholeDampingFactors Thole damping factors
     * @param particleData        output data struct
     */
    void loadParticleData(const std::vector<OpenMM::Vec3>& particlePositions,
                          const std::vector<double>& charges,
                          const std::vector<double>& dipoles,
                          const std::vector<double>& polarizabilities,
                          const std::vector<double>& tholeDampingFactors,
                          std::vector<TholeDipoleParticleData>& particleData) const;

    /**
     * Setup scale factors given covalent info.
     *
     * @param multipoleCovalentInfo vector of vectors containing the covalent info
     */
    void setupScaleMaps(const std::vector<std::vector<std::vector<int>>>& multipoleCovalentInfo);

    /**
     * Get scale factor for particleI & particleJ
     * 
     * @param particleI           index of particleI whose scale factor is to be retrieved
     * @param particleJ           index of particleJ whose scale factor is to be retrieved
     * @param scaleType           scale type (M_SCALE, I_SCALE)
     *
     * @return scaleFactor 
     */
    double getScaleFactor(unsigned int particleI, unsigned int particleJ, ScaleType scaleType) const;

    /**
     * Apply rotation matrix to molecular dipole to get corresponding lab frame values.
     * 
     * @param particleData            vector of parameters for particles
     * @param axisTypes               axis types for particles
     * @param multipoleAtomZs         vector of z-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomXs         vector of x-particle indices used to map molecular frame to lab frame
     * @param multipoleAtomYs         vector of y-particle indices used to map molecular frame to lab frame
     */
    void applyRotationMatrix(std::vector<TholeDipoleParticleData>& particleData,
                             const std::vector<int>& axisTypes,
                             const std::vector<int>& multipoleAtomZs,
                             const std::vector<int>& multipoleAtomXs,
                             const std::vector<int>& multipoleAtomYs) const;

    /**
     * Zero fixed dipole fields.
     */
    virtual void zeroFixedDipoleFields();

    /**
     * Calculate fixed dipole field at each site.
     * 
     * @param particleData      vector of particle positions and parameters
     */
    virtual void calculateFixedDipoleField(const std::vector<TholeDipoleParticleData>& particleData);

    /**
     * Calculate field at particle I due to fixed dipole at particle J and vice versa.
     * 
     * @param particleI       particle I data
     * @param particleJ       particle J data
     * @param dScale          d-scale factor for this interaction
     * @param pScale          p-scale factor for this interaction  
     */
    virtual void calculateFixedDipoleFieldPairIxn(const TholeDipoleParticleData& particleI, 
                                                   const TholeDipoleParticleData& particleJ,
                                                   double dScale, double pScale);

    /**
     * Converge induced dipoles using DIIS for mutual polarization.
     * 
     * @param particleData      vector of particle positions and parameters
     */
    virtual void convergeInducedDipolesByDIIS(const std::vector<TholeDipoleParticleData>& particleData);

    /**
     * Converge induced dipoles using extrapolated perturbation theory.
     * 
     * @param particleData      vector of particle positions and parameters
     */
    virtual void convergeInducedDipolesByExtrapolation(const std::vector<TholeDipoleParticleData>& particleData);

    /**
     * Calculate induced dipoles.
     * 
     * @param particleData      vector of particle positions and parameters
     */
    virtual void calculateInducedDipoles(const std::vector<TholeDipoleParticleData>& particleData);

    /**
     * Calculate electrostatic forces
     * 
     * @param particleData            vector of parameters for particles
     * @param forces                  output forces 
     *
     * @return energy
     */
    virtual double calculateElectrostatic(const std::vector<TholeDipoleParticleData>& particleData,
                                          std::vector<OpenMM::Vec3>& forces);

    /**
     * Apply periodic boundary conditions to difference in positions
     * 
     * @param deltaR  difference in particle positions; modified on output after applying PBC
     */
    virtual void getPeriodicDelta(Vec3& deltaR) const {};
};

/**
 * This class implements PME for the Thole dipole force.
 */
class ReferencePmeTholeDipoleForce : public ReferenceTholeDipoleForce {
public:
    /**
     * Constructor
     */
    ReferencePmeTholeDipoleForce();

    /**
     * Destructor
     */
    ~ReferencePmeTholeDipoleForce();

    /**
     * Set cutoff distance.
     *
     * @param cutoffDistance cutoff distance
     */
    void setCutoffDistance(double cutoffDistance);

    /**
     * Get cutoff distance.
     *
     * @return cutoff distance
     */
    double getCutoffDistance() const;

    /**
     * Set alpha parameter for Ewald.
     *
     * @param alphaEwald alpha parameter for Ewald
     */
    void setAlphaEwald(double alphaEwald);

    /**
     * Get alpha parameter for Ewald.
     *
     * @return alpha parameter for Ewald
     */
    double getAlphaEwald() const;

    /**
     * Set PME grid dimensions.
     *
     * @param pmeGridDimensions PME grid dimensions
     */
    void setPmeGridDimensions(const std::vector<int>& pmeGridDimensions);

    /**
     * Get PME grid dimensions.
     *
     * @param pmeGridDimensions PME grid dimensions
     */
    void getPmeGridDimensions(std::vector<int>& pmeGridDimensions) const;

    /**
     * Set periodic box size.
     *
     * @param boxVectors box vectors
     */
    void setPeriodicBoxSize(Vec3* boxVectors);

protected:
    double _cutoffDistance;
    double _alphaEwald;
    std::vector<int> _pmeGridDimensions;
    Vec3 _periodicBoxVectors[3];

    /**
     * Apply periodic boundary conditions to difference in positions
     */
    void getPeriodicDelta(Vec3& deltaR) const override;
};

} // namespace TholeDipolePlugin

#endif // __ReferenceTholeDipoleForce_H__