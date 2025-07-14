#
# Testing
#

# Create individual tests for each test function
SET(INDIVIDUAL_TESTS
    TholeDipoleAmmoniaDirectPolarization
    TholeDipoleAmmoniaMutualPolarization
    TholeDipoleWaterPMEDirectPolarization
    TholeDipoleWaterPMEMutualPolarization
    ParticleInducedDipoles
    ParticleLabFramePermanentDipoles
    ParticleTotalDipoles
    Triclinic
    ZBisect
    ZOnly
    ZOnlySimple
    NeutralizingPlasmaCorrection
)

FOREACH(TEST_NAME ${INDIVIDUAL_TESTS})
    ADD_EXECUTABLE(Test${TEST_NAME} TestReferenceTholeDipoleForce.cpp)
    TARGET_COMPILE_DEFINITIONS(Test${TEST_NAME} PRIVATE RUN_ONLY_${TEST_NAME})
    TARGET_LINK_LIBRARIES(Test${TEST_NAME} ${SHARED_TARGET})
    SET_TARGET_PROPERTIES(Test${TEST_NAME} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ADD_TEST(${TEST_NAME} ${EXECUTABLE_OUTPUT_PATH}/Test${TEST_NAME})
ENDFOREACH(TEST_NAME ${INDIVIDUAL_TESTS})

# Also keep the original test that runs all tests
ADD_EXECUTABLE(TestReferenceTholeDipoleForceAll TestReferenceTholeDipoleForce.cpp)
TARGET_LINK_LIBRARIES(TestReferenceTholeDipoleForceAll ${SHARED_TARGET})
SET_TARGET_PROPERTIES(TestReferenceTholeDipoleForceAll PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
ADD_TEST(AllTests ${EXECUTABLE_OUTPUT_PATH}/TestReferenceTholeDipoleForceAll)
