%module tholedipoleplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "TholeDipoleForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import simtk.openmm as mm
import simtk.unit as unit
%}

/*
 * Add units to function outputs.
*/
%pythonappend TholeDipolePlugin::TholeDipoleForce::getBondParameters(int index, int& particle1, int& particle2,
                                                             double& length, double& k) const %{
    val[2] = unit.Quantity(val[2], unit.nanometer)
    val[3] = unit.Quantity(val[3], unit.kilojoule_per_mole/unit.nanometer**4)
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}


namespace TholeDipolePlugin {

class TholeDipoleForce : public OpenMM::Force {
public:
    TholeDipoleForce();

    int getNumBonds() const;

    int addBond(int particle1, int particle2, double length, double k);

    void setBondParameters(int index, int particle1, int particle2, double length, double k);

    void updateParametersInContext(OpenMM::Context& context);

    /*
     * The reference parameters to this function are output values.
     * Marking them as such will cause swig to return a tuple.
    */
    %apply int& OUTPUT {int& particle1};
    %apply int& OUTPUT {int& particle2};
    %apply double& OUTPUT {double& length};
    %apply double& OUTPUT {double& k};
    void getBondParameters(int index, int& particle1, int& particle2, double& length, double& k) const;
    %clear int& particle1;
    %clear int& particle2;
    %clear double& length;
    %clear double& k;

    /*
     * Add methods for casting a Force to a TholeDipoleForce.
    */
    %extend {
        static TholeDipolePlugin::TholeDipoleForce& cast(OpenMM::Force& force) {
            return dynamic_cast<TholeDipolePlugin::TholeDipoleForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<TholeDipolePlugin::TholeDipoleForce*>(&force) != NULL);
        }
    }
};

}
