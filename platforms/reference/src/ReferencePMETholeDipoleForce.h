#ifndef OPENMM_REFERENCEPMETHOLEDIPOLEFORCE_H_
#define OPENMM_REFERENCEPMETHOLEDIPOLEFORCE_H_

#include "ReferenceTholeDipoleForce.h"
#include "openmm/Vec3.h"
#include <complex>
#include <vector>

namespace TholeDipolePlugin {

using OpenMM::Vec3;
using std::vector;

class IntVec {
public:
    IntVec() : data{0, 0, 0} {}
    IntVec(int x, int y, int z) : data{x, y, z} {}
    int operator[](int i) const { return data[i]; }
    int& operator[](int i) { return data[i]; }
private:
    int data[3];
};

class double4 {
public:
    double4() : data{0, 0, 0, 0} {}
    double4(double x, double y, double z, double w) : data{x, y, z, w} {}
    double operator[](int i) const { return data[i]; }
    double& operator[](int i) { return data[i]; }
private:
    double data[4];
};

class ReferencePMETholeDipoleForce : public ReferenceTholeDipoleForce {

public:
    ReferencePMETholeDipoleForce();
    ~ReferencePMETholeDipoleForce();

    double getCutoffDistance() const;
    void setCutoffDistance(double cutoffDistance);
    double getAlphaEwald() const;
    void setAlphaEwald(double alphaEwald);
    void getPmeGridDimensions(vector<int>& pmeGridDimensions) const;
    void setPmeGridDimensions(vector<int>& pmeGridDimensions);
    void setPeriodicBoxSize(OpenMM::Vec3* vectors);

    // Test accessors for PME internals
    const vector<double>& getPmeBsplineModuli(int dim) const { return _pmeBsplineModuli[dim]; }
    const vector<double4>& getThetai(int dim) const { return _thetai[dim]; }
    const vector<IntVec>& getIGrid() const { return _iGrid; }
    const vector<double>& getPhi() const { return _phi; }
    const vector<double>& getPhid() const { return _phid; }
    const std::complex<double>* getPmeGrid() const { return _pmeGrid.data(); }
    int getTotalGridSize() const { return _totalGridSize; }
    const vector<TholeDipoleParticleData>& getTransformed() const { return _transformed; }
    double getFixedMultipoleRecipEnergy() const { return _fixedMultipoleRecipEnergy; }
    double getInducedDipoleRecipEnergy() const { return _inducedDipoleRecipEnergy; }

protected:
    double calculateElectrostatic(const vector<TholeDipoleParticleData>& particleData,
                                   vector<Vec3>& torques, vector<Vec3>& forces) override;
    void calculateFixedDipoleField(const vector<TholeDipoleParticleData>& particleData) override;
    void calculateInducedDipoleFields(const vector<TholeDipoleParticleData>& particleData,
                                      const vector<Vec3>& inducedDipoles,
                                      vector<Vec3>& inducedDipoleField) override;

private:

    static const int THOLE_PME_ORDER;
    static const double SQRT_PI;

    double _alphaEwald;
    double _cutoffDistance;
    double _cutoffDistanceSquared;

    Vec3 _recipBoxVectors[3];
    Vec3 _periodicBoxVectors[3];

    size_t _totalGridSize;
    IntVec _pmeGridDimensions;

    std::vector<std::complex<double>> _pmeGrid;

    vector<double> _pmeBsplineModuli[3];
    vector<double4> _thetai[3];
    vector<IntVec> _iGrid;
    vector<double> _phi;
    vector<double> _phid;
    vector<TholeDipoleParticleData> _transformed;
    vector<Vec3> _particleFraction;

    double _fixedMultipoleRecipEnergy;
    double _inducedDipoleRecipEnergy;


    double _computeBoxVolume() const {
        const Vec3& a = _periodicBoxVectors[0];
        const Vec3& b = _periodicBoxVectors[1];
        const Vec3& c = _periodicBoxVectors[2];

        return a[0]*(b[1]*c[2] - b[2]*c[1])
             - a[1]*(b[0]*c[2] - b[2]*c[0])
             + a[2]*(b[0]*c[1] - b[1]*c[0]);
    }

    void resizePmeArrays();
    void initializePmeGrid();
    void performFFT(bool forward);
    void getPeriodicDelta(Vec3& deltaR) const;
    void initializeBSplineModuli();
    void calculateFixedDipoleFieldPairIxn(const TholeDipoleParticleData& particleI,
                                          const TholeDipoleParticleData& particleJ,
                                          double mScale);
    void computeBSplinePoint(double* data, double* ddata, double* d2data, double* d3data, double w, int order);
    void updateGridIndexAndFraction(const vector<TholeDipoleParticleData>& particleData);
    void computePmeBSplines(const vector<TholeDipoleParticleData>& particleData);
    void transformDipolesToFractionalCoordinates(const vector<TholeDipoleParticleData>& particleData);
    void transformPotentialToCartesianCoordinates(const vector<double>& fphi, vector<double>& cphi) const;
    void spreadFixedMultipolesOntoGrid(const vector<TholeDipoleParticleData>& particleData);
    double performPmeReciprocalConvolution();
    void computeFixedPotentialFromGrid();
    void computeInducedPotentialFromGrid();
    double computeReciprocalSpaceFixedMultipoleForceAndEnergy(const vector<TholeDipoleParticleData>& particleData,
                                                              vector<Vec3>& forces, vector<Vec3>& torques) const;
    double computeReciprocalSpaceInducedDipoleForceAndEnergy(const vector<TholeDipoleParticleData>& particleData,
                                                              vector<Vec3>& forces, vector<Vec3>& torques) const;
    void recordFixedMultipoleField();
    void calculatePmeDirectInducedDipolePairIxn(const TholeDipoleParticleData& particleI,
                                                const TholeDipoleParticleData& particleJ,
                                                const vector<Vec3>& inducedDipoles,
                                                double iScale,
                                                vector<Vec3>& field) const;
    void spreadInducedDipolesOnGrid(const vector<Vec3>& inputInducedDipole);
    void recordInducedDipoleField(vector<Vec3>& field);
    double calculatePmeSelfEnergy(const vector<TholeDipoleParticleData>& particleData) const;
    void calculatePmeSelfTorque(const vector<TholeDipoleParticleData>& particleData, vector<Vec3>& torques) const;
    double calculatePmeDirectElectrostaticPairIxn(const TholeDipoleParticleData& particleI,
                                                  const TholeDipoleParticleData& particleJ,
                                                  double mScale, double iScale,
                                                  vector<Vec3>& forces, vector<Vec3>& torques) const;

};

} // namespace TholeDipolePlugin

#endif // OPENMM_REFERENCEPMETHOLEDIPOLEFORCE_H_

