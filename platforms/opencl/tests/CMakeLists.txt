#
# Testing
#

INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})

# Create individual tests for each test file
SET(INDIVIDUAL_TESTS
    TholeDipoleAmmonia
    TholeDipoleWaterPME
    Triclinic
    ZOnly
    AxisTypes
    EdgeCases
    TwoPointCharges
    TwoDipoles
    ParticleAccessors
    1DChainSizeConsistency
    CustomPointCharges
    PMECutoffConvergence
)

FOREACH(TEST_NAME ${INDIVIDUAL_TESTS})
    ADD_EXECUTABLE(TestOpenCL${TEST_NAME} TestOpenCL${TEST_NAME}.cpp OpenCLTestCommon.cpp)
    TARGET_LINK_LIBRARIES(TestOpenCL${TEST_NAME} ${SHARED_THOLEDIPOLE_TARGET} ${SHARED_TARGET} TholeDipolePluginReference)
    SET_TARGET_PROPERTIES(TestOpenCL${TEST_NAME} PROPERTIES LINK_FLAGS "${EXTRA_COMPILE_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS}")
    ADD_TEST(TestOpenCL${TEST_NAME}Single ${EXECUTABLE_OUTPUT_PATH}/TestOpenCL${TEST_NAME} single)
    ADD_TEST(TestOpenCL${TEST_NAME}Mixed ${EXECUTABLE_OUTPUT_PATH}/TestOpenCL${TEST_NAME} mixed)
    ADD_TEST(TestOpenCL${TEST_NAME}Double ${EXECUTABLE_OUTPUT_PATH}/TestOpenCL${TEST_NAME} double)
ENDFOREACH(TEST_NAME ${INDIVIDUAL_TESTS})
