/* -------------------------------------------------------------------------- *
 *                              OpenMMTholeDipole                             *
 * -------------------------------------------------------------------------- *
 * GPU kernels for Thole/Applequist polarizable dipole model                  *
 * -------------------------------------------------------------------------- */

#define WARPS_PER_GROUP (THREAD_BLOCK_SIZE/TILE_SIZE)

#ifdef USE_PERIODIC
/**
 * Apply minimum image convention for triclinic boxes.
 * Computes reciprocal box vectors inline and uses them for proper fractional coordinate wrapping.
 */
DEVICE real3 applyPeriodicDelta(real3 deltaR, real4 periodicBoxSize, real4 invPeriodicBoxSize,
                                 real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    real3 a = make_real3(periodicBoxVecX.x, periodicBoxVecX.y, periodicBoxVecX.z);
    real3 b = make_real3(periodicBoxVecY.x, periodicBoxVecY.y, periodicBoxVecY.z);
    real3 c = make_real3(periodicBoxVecZ.x, periodicBoxVecZ.y, periodicBoxVecZ.z);
#ifdef DEBUG_TRICLINIC
    if (GLOBAL_ID == 0) {
        printf("applyPeriodicDelta: a=(%.6f,%.6f,%.6f) b=(%.6f,%.6f,%.6f) c=(%.6f,%.6f,%.6f)\n",
               a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
        printf("applyPeriodicDelta: input deltaR=(%.6f,%.6f,%.6f)\n", deltaR.x, deltaR.y, deltaR.z);
    }
#endif

    // Compute determinant = a · (b × c)
    real det = a.x * (b.y * c.z - b.z * c.y)
             - a.y * (b.x * c.z - b.z * c.x)
             + a.z * (b.x * c.y - b.y * c.x);
    real invDet = RECIP(det);

    // Compute reciprocal box vectors (rows of inverse matrix)
    // These match the Reference implementation exactly
    real3 recipVec0 = make_real3((b.y * c.z - b.z * c.y) * invDet,
                                  (a.z * c.y - a.y * c.z) * invDet,
                                  (a.y * b.z - a.z * b.y) * invDet);
    real3 recipVec1 = make_real3((b.z * c.x - b.x * c.z) * invDet,
                                  (a.x * c.z - a.z * c.x) * invDet,
                                  (a.z * b.x - a.x * b.z) * invDet);
    real3 recipVec2 = make_real3((b.x * c.y - b.y * c.x) * invDet,
                                  (a.y * c.x - a.x * c.y) * invDet,
                                  (a.x * b.y - a.y * b.x) * invDet);

    // Compute fractional coordinates: lambda_i = deltaR · recipVec[i]
    real lambda2 = deltaR.x * recipVec2.x + deltaR.y * recipVec2.y + deltaR.z * recipVec2.z;
    real lambda1 = deltaR.x * recipVec1.x + deltaR.y * recipVec1.y + deltaR.z * recipVec1.z;
    real lambda0 = deltaR.x * recipVec0.x + deltaR.y * recipVec0.y + deltaR.z * recipVec0.z;

    // Wrap to nearest image
    real n2 = floor(lambda2 + 0.5f);
    deltaR = deltaR - c * n2;

    real n1 = floor(lambda1 + 0.5f);
    deltaR = deltaR - b * n1;

    real n0 = floor(lambda0 + 0.5f);
    deltaR = deltaR - a * n0;

#ifdef DEBUG_TRICLINIC
    if (GLOBAL_ID == 0) {
        printf("applyPeriodicDelta: output deltaR=(%.6f,%.6f,%.6f) n=(%.0f,%.0f,%.0f)\n",
               deltaR.x, deltaR.y, deltaR.z, n0, n1, n2);
    }
#endif
    return deltaR;
}
#endif

#ifdef USE_EWALD
/**
 * Compute Ewald-damped B_n coefficients for real-space PME.
 * bn0 = erfc(alpha*r)/r
 * bn1 = (bn0 + 2*alpha/sqrt(pi)*exp(-alpha^2*r^2))/r^2
 * bn2 = (3*bn1 + (2*alpha)^3/sqrt(pi)*exp(-alpha^2*r^2))/r^2
 */
DEVICE void computeEwaldCoefficients(real r, real* bn0, real* bn1, real* bn2) {
    real ralpha = EWALD_ALPHA * r;
    real exp2a = EXP(-ralpha * ralpha);
    *bn0 = erfc(ralpha) / r;
    real alsq2 = 2.0f * EWALD_ALPHA * EWALD_ALPHA;
    real alsq2n = RECIP(SQRT_PI * EWALD_ALPHA);
    alsq2n *= alsq2;
    real r2 = r * r;
    *bn1 = (*bn0 + alsq2n * exp2a) / r2;
    alsq2n *= alsq2;
    *bn2 = (3.0f * (*bn1) + alsq2n * exp2a) / r2;
}
#endif

typedef struct {
    real4 posq;
    real3 field;
    real3 dipole;
    float thole, damp;
} AtomData;

inline DEVICE AtomData loadAtomData(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labFrameDipole,
        GLOBAL const float2* RESTRICT dampingAndThole) {
    AtomData data;
    data.posq = posq[atom];
    data.dipole = make_real3(labFrameDipole[3*atom], labFrameDipole[3*atom+1], labFrameDipole[3*atom+2]);
    float2 temp = dampingAndThole[atom];
    data.damp = temp.x;
    data.thole = temp.y;
    return data;
}

KERNEL void computeLabFrameMoments(GLOBAL const real4* RESTRICT posq, GLOBAL const int4* RESTRICT multipoleParticles,
        GLOBAL const float* RESTRICT localDipoles, GLOBAL real* RESTRICT labFrameDipoles,
        GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT inverseAtomIndex
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        // atom is GPU index, origAtom is original particle index
        int origAtom = atomIndex[atom];
        int4 particles = multipoleParticles[origAtom];
        int axisType = particles.w;
        // atomX, atomY, atomZ are original indices - convert to GPU indices for posq access
        int origAtomX = particles.x;
        int origAtomY = particles.y;
        int origAtomZ = particles.z;
        int atomX = (origAtomX >= 0) ? inverseAtomIndex[origAtomX] : -1;
        int atomY = (origAtomY >= 0) ? inverseAtomIndex[origAtomY] : -1;
        int atomZ = (origAtomZ >= 0) ? inverseAtomIndex[origAtomZ] : -1;

        real3 dipole = make_real3(localDipoles[3*origAtom], localDipoles[3*origAtom+1], localDipoles[3*origAtom+2]);

        if (axisType == 5) {
            // NoAxisType - dipole stays in local frame (lab frame)
            labFrameDipoles[3*atom] = dipole.x;
            labFrameDipoles[3*atom+1] = dipole.y;
            labFrameDipoles[3*atom+2] = dipole.z;
        }
        else {
            // Build local frame axes
            real4 thisPos = posq[atom];
            real4 posZ = posq[atomZ];
            real3 zAxis = make_real3(posZ.x-thisPos.x, posZ.y-thisPos.y, posZ.z-thisPos.z);
#ifdef USE_PERIODIC
            zAxis = applyPeriodicDelta(zAxis, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
            real invLength = RSQRT(dot(zAxis, zAxis));
            zAxis = zAxis * invLength;

            real3 xAxis;
            if (axisType == 4) {
                // ZOnly - just need Z axis, pick arbitrary X
                if (fabs(zAxis.x) < 0.866f)
                    xAxis = make_real3(1, 0, 0);
                else
                    xAxis = make_real3(0, 1, 0);
                real3 yAxis = cross(zAxis, xAxis);
                invLength = RSQRT(dot(yAxis, yAxis));
                yAxis = yAxis * invLength;
                xAxis = cross(yAxis, zAxis);
            }
            else if (axisType == 0) {
                // ZThenX
                real4 posX = posq[atomX];
                xAxis = make_real3(posX.x-thisPos.x, posX.y-thisPos.y, posX.z-thisPos.z);
#ifdef USE_PERIODIC
                xAxis = applyPeriodicDelta(xAxis, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                real3 yAxis = cross(zAxis, xAxis);
                invLength = RSQRT(dot(yAxis, yAxis));
                yAxis = yAxis * invLength;
                xAxis = cross(yAxis, zAxis);
            }
            else if (axisType == 1) {
                // Bisector - z-axis bisects directions to atomZ and atomX
                real4 posX = posq[atomX];
                real3 v1 = make_real3(posZ.x-thisPos.x, posZ.y-thisPos.y, posZ.z-thisPos.z);
                real3 v2 = make_real3(posX.x-thisPos.x, posX.y-thisPos.y, posX.z-thisPos.z);
#ifdef USE_PERIODIC
                v1 = applyPeriodicDelta(v1, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                v2 = applyPeriodicDelta(v2, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                invLength = RSQRT(dot(v1, v1));
                v1 = v1 * invLength;
                invLength = RSQRT(dot(v2, v2));
                v2 = v2 * invLength;
                zAxis = v1 + v2;
                invLength = RSQRT(dot(zAxis, zAxis));
                zAxis = zAxis * invLength;
                // Orthogonalize v2 (xAxis direction) to new zAxis
                xAxis = v2 - zAxis * dot(zAxis, v2);
                invLength = RSQRT(dot(xAxis, xAxis));
                xAxis = xAxis * invLength;
            }
            else if (axisType == 2) {
                // ZBisect
                real4 posX = posq[atomX];
                real4 posY = posq[atomY];
                real3 v1 = make_real3(posX.x-thisPos.x, posX.y-thisPos.y, posX.z-thisPos.z);
                real3 v2 = make_real3(posY.x-thisPos.x, posY.y-thisPos.y, posY.z-thisPos.z);
#ifdef USE_PERIODIC
                v1 = applyPeriodicDelta(v1, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                v2 = applyPeriodicDelta(v2, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                invLength = RSQRT(dot(v1, v1));
                v1 = v1 * invLength;
                invLength = RSQRT(dot(v2, v2));
                v2 = v2 * invLength;
                xAxis = v1 + v2;
                invLength = RSQRT(dot(xAxis, xAxis));
                xAxis = xAxis * invLength;
                // Recompute zAxis with minimum image (already loaded above, but wasn't wrapped for ZBisect)
                real3 zVec = make_real3(posZ.x-thisPos.x, posZ.y-thisPos.y, posZ.z-thisPos.z);
#ifdef USE_PERIODIC
                zVec = applyPeriodicDelta(zVec, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                invLength = RSQRT(dot(zVec, zVec));
                zAxis = zVec * invLength;
                real3 yAxis = cross(zAxis, xAxis);
                invLength = RSQRT(dot(yAxis, yAxis));
                yAxis = yAxis * invLength;
                xAxis = cross(yAxis, zAxis);
            }
            else {
                // ThreeFold - z-axis is average of directions to atomZ, atomX, atomY
                real4 posX = posq[atomX];
                real4 posY = posq[atomY];
                real3 v1 = make_real3(posZ.x-thisPos.x, posZ.y-thisPos.y, posZ.z-thisPos.z);
                real3 v2 = make_real3(posX.x-thisPos.x, posX.y-thisPos.y, posX.z-thisPos.z);
                real3 v3 = make_real3(posY.x-thisPos.x, posY.y-thisPos.y, posY.z-thisPos.z);
#ifdef USE_PERIODIC
                v1 = applyPeriodicDelta(v1, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                v2 = applyPeriodicDelta(v2, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
                v3 = applyPeriodicDelta(v3, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                invLength = RSQRT(dot(v1, v1));
                v1 = v1 * invLength;
                invLength = RSQRT(dot(v2, v2));
                v2 = v2 * invLength;
                invLength = RSQRT(dot(v3, v3));
                v3 = v3 * invLength;
                zAxis = v1 + v2 + v3;
                invLength = RSQRT(dot(zAxis, zAxis));
                zAxis = zAxis * invLength;
                // Orthogonalize v2 (xAxis direction) to new zAxis
                xAxis = v2 - zAxis * dot(zAxis, v2);
                invLength = RSQRT(dot(xAxis, xAxis));
                xAxis = xAxis * invLength;
            }

            // Transform dipole from molecular to lab frame
            real3 yAxis = cross(zAxis, xAxis);
            labFrameDipoles[3*atom] = dipole.x*xAxis.x + dipole.y*yAxis.x + dipole.z*zAxis.x;
            labFrameDipoles[3*atom+1] = dipole.x*xAxis.y + dipole.y*yAxis.y + dipole.z*zAxis.y;
            labFrameDipoles[3*atom+2] = dipole.x*xAxis.z + dipole.y*yAxis.z + dipole.z*zAxis.z;
        }
    }
}

KERNEL void recordInducedDipoles(GLOBAL const long long* RESTRICT field, GLOBAL real* RESTRICT inducedDipole,
        GLOBAL const float* RESTRICT polarizability) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real alpha = polarizability[atom];
        real scale = RECIP((real) 0x100000000);
        inducedDipole[3*atom] = alpha * scale * field[atom];
        inducedDipole[3*atom+1] = alpha * scale * field[atom + PADDED_NUM_ATOMS];
        inducedDipole[3*atom+2] = alpha * scale * field[atom + 2*PADDED_NUM_ATOMS];
    }
}

KERNEL void mapTorqueToForce(GLOBAL mm_ulong* RESTRICT forceBuffer, GLOBAL const mm_long* RESTRICT torque,
        GLOBAL const real4* RESTRICT posq, GLOBAL const int4* RESTRICT multipoleParticles,
        GLOBAL const int* RESTRICT atomIndex, GLOBAL const int* RESTRICT inverseAtomIndex
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {
    const real minSin = 1e-8;

    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        // atom is GPU index, origAtom is original particle index
        int origAtom = atomIndex[atom];
        int4 particles = multipoleParticles[origAtom];
        int axisType = particles.w;
        // atomX, atomY, atomZ are original indices - convert to GPU indices for posq and forceBuffer access
        int origAtomX = particles.x;
        int origAtomY = particles.y;
        int origAtomZ = particles.z;
        int atomX = (origAtomX >= 0) ? inverseAtomIndex[origAtomX] : -1;
        int atomY = (origAtomY >= 0) ? inverseAtomIndex[origAtomY] : -1;
        int atomZ = (origAtomZ >= 0) ? inverseAtomIndex[origAtomZ] : -1;

        if (axisType == 5)  // NoAxisType
            continue;

        real scale = RECIP((real) 0x100000000);
        real3 t = make_real3(scale*torque[atom], scale*torque[atom + PADDED_NUM_ATOMS], scale*torque[atom + 2*PADDED_NUM_ATOMS]);

        real4 thisPos = posq[atom];
        real4 posZCoord = posq[atomZ];
        real3 vectorU = make_real3(posZCoord.x-thisPos.x, posZCoord.y-thisPos.y, posZCoord.z-thisPos.z);
#ifdef USE_PERIODIC
        vectorU = applyPeriodicDelta(vectorU, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
        real normU = SQRT(dot(vectorU, vectorU));
        vectorU = vectorU / normU;

        // For ZOnly, create arbitrary perpendicular vector for V
        real3 vectorV;
        real normV;
        if (axisType == 4) {  // ZOnly
            if (fabs(vectorU.x) < 0.866f)
                vectorV = make_real3(1, 0, 0);
            else
                vectorV = make_real3(0, 1, 0);
            // Orthogonalize V to U
            vectorV = vectorV - vectorU * dot(vectorU, vectorV);
            normV = SQRT(dot(vectorV, vectorV));
            vectorV = vectorV / normV;
            normV = 1;  // Arbitrary distance for dummy atom
        }
        else {
            real4 posXCoord = posq[atomX];
            vectorV = make_real3(posXCoord.x-thisPos.x, posXCoord.y-thisPos.y, posXCoord.z-thisPos.z);
#ifdef USE_PERIODIC
            vectorV = applyPeriodicDelta(vectorV, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
            normV = SQRT(dot(vectorV, vectorV));
            vectorV = vectorV / normV;
        }

        // W is either U×V or from atomY for ZBisect/ThreeFold
        real3 vectorW;
        real normW;
        if (atomY >= 0 && (axisType == 2 || axisType == 3)) {  // ZBisect or ThreeFold
            real4 posYCoord = posq[atomY];
            vectorW = make_real3(posYCoord.x-thisPos.x, posYCoord.y-thisPos.y, posYCoord.z-thisPos.z);
#ifdef USE_PERIODIC
            vectorW = applyPeriodicDelta(vectorW, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
            normW = SQRT(dot(vectorW, vectorW));
            vectorW = vectorW / normW;
        }
        else {
            vectorW = cross(vectorU, vectorV);
            normW = SQRT(dot(vectorW, vectorW));
            if (normW > 0) vectorW = vectorW / normW;
        }

        // Cross products for force calculation
        real3 vectorUV = cross(vectorV, vectorU);
        real3 vectorUW = cross(vectorW, vectorU);
        real3 vectorVW = cross(vectorW, vectorV);

        real normUV = SQRT(dot(vectorUV, vectorUV));
        if (normUV > 0) vectorUV = vectorUV / normUV;
        real normUW = SQRT(dot(vectorUW, vectorUW));
        if (normUW > 0) vectorUW = vectorUW / normUW;
        real normVW = SQRT(dot(vectorVW, vectorVW));
        if (normVW > 0) vectorVW = vectorVW / normVW;

        // Calculate sines of angles
        real cosUV = dot(vectorU, vectorV);
        real sinUV = SQRT(1 - cosUV*cosUV);
        if (sinUV < minSin) sinUV = minSin;

        real cosUW = dot(vectorU, vectorW);
        real sinUW = SQRT(1 - cosUW*cosUW);
        if (sinUW < minSin) sinUW = minSin;

        real cosVW = dot(vectorV, vectorW);
        real sinVW = SQRT(1 - cosVW*cosVW);
        if (sinVW < minSin) sinVW = minSin;

        // Project torque onto local axes and negate (dphi = -torque projected)
        real dphiU = -dot(vectorU, t);
        real dphiV = -dot(vectorV, t);
        real dphiW = -dot(vectorW, t);

        if (axisType == 4) {  // ZOnly
            real3 forceU = vectorUV * dphiV / (normU * sinUV) + vectorUW * dphiW / normU;
            ATOMIC_ADD(&forceBuffer[atomZ], (mm_ulong) realToFixedPoint(-forceU.x));
            ATOMIC_ADD(&forceBuffer[atomZ+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceU.y));
            ATOMIC_ADD(&forceBuffer[atomZ+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceU.z));
            ATOMIC_ADD(&forceBuffer[atom], (mm_ulong) realToFixedPoint(forceU.x));
            ATOMIC_ADD(&forceBuffer[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceU.y));
            ATOMIC_ADD(&forceBuffer[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceU.z));
        }
        else if (axisType == 0 || axisType == 1) {  // ZThenX or Bisector
            real factor1 = dphiV / (normU * sinUV);
            real factor2 = dphiW / normU;
            real factor3 = -dphiU / (normV * sinUV);
            real factor4 = 0;

            if (axisType == 1) {  // Bisector
                factor2 *= 0.5;
                factor4 = 0.5 * dphiW / normV;
            }

            real3 forceU = vectorUV * factor1 + vectorUW * factor2;
            real3 forceV = vectorUV * factor3 + vectorVW * factor4;

            ATOMIC_ADD(&forceBuffer[atomZ], (mm_ulong) realToFixedPoint(-forceU.x));
            ATOMIC_ADD(&forceBuffer[atomZ+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceU.y));
            ATOMIC_ADD(&forceBuffer[atomZ+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceU.z));
            ATOMIC_ADD(&forceBuffer[atomX], (mm_ulong) realToFixedPoint(-forceV.x));
            ATOMIC_ADD(&forceBuffer[atomX+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceV.y));
            ATOMIC_ADD(&forceBuffer[atomX+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceV.z));
            ATOMIC_ADD(&forceBuffer[atom], (mm_ulong) realToFixedPoint(forceU.x + forceV.x));
            ATOMIC_ADD(&forceBuffer[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceU.y + forceV.y));
            ATOMIC_ADD(&forceBuffer[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceU.z + forceV.z));
        }
        else if (axisType == 2) {  // ZBisect
            real3 vectorR = vectorV + vectorW;
            real normR = SQRT(dot(vectorR, vectorR));
            if (normR > 0) vectorR = vectorR / normR;

            real3 vectorS = cross(vectorU, vectorR);
            real normS = SQRT(dot(vectorS, vectorS));
            if (normS > 0) vectorS = vectorS / normS;

            real3 vectorUR = cross(vectorR, vectorU);
            real normURval = SQRT(dot(vectorUR, vectorUR));
            if (normURval > 0) vectorUR = vectorUR / normURval;

            real3 vectorUS = cross(vectorS, vectorU);
            real normUSval = SQRT(dot(vectorUS, vectorUS));
            if (normUSval > 0) vectorUS = vectorUS / normUSval;

            real cosUR = dot(vectorU, vectorR);
            real sinUR = SQRT(1 - cosUR*cosUR);
            if (sinUR < minSin) sinUR = minSin;

            real cosVS = dot(vectorV, vectorS);
            real sinVS = SQRT(1 - cosVS*cosVS);
            if (sinVS < minSin) sinVS = minSin;

            real cosWS = dot(vectorW, vectorS);
            real sinWS = SQRT(1 - cosWS*cosWS);
            if (sinWS < minSin) sinWS = minSin;

            real3 t1 = vectorV - vectorS * cosVS;
            real normt1 = SQRT(dot(t1, t1));
            if (normt1 > 0) t1 = t1 / normt1;

            real3 t2 = vectorW - vectorS * cosWS;
            real normt2 = SQRT(dot(t2, t2));
            if (normt2 > 0) t2 = t2 / normt2;

            real ut1cos = dot(vectorU, t1);
            real ut1sin = SQRT(1 - ut1cos*ut1cos);
            if (ut1sin < minSin) ut1sin = minSin;

            real ut2cos = dot(vectorU, t2);
            real ut2sin = SQRT(1 - ut2cos*ut2cos);
            if (ut2sin < minSin) ut2sin = minSin;

            real dphiR = -dot(vectorR, t);
            real dphiS = -dot(vectorS, t);

            real factor1 = dphiR / (normU * sinUR);
            real factor2 = dphiS / normU;
            real factor3 = dphiU / (normV * (ut1sin + ut2sin));
            real factor4 = dphiU / (normW * (ut1sin + ut2sin));

            real3 forceU = vectorUR * factor1 + vectorUS * factor2;
            real3 forceV = (vectorS * sinVS - t1 * cosVS) * factor3;
            real3 forceW = (vectorS * sinWS - t2 * cosWS) * factor4;

            ATOMIC_ADD(&forceBuffer[atomZ], (mm_ulong) realToFixedPoint(-forceU.x));
            ATOMIC_ADD(&forceBuffer[atomZ+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceU.y));
            ATOMIC_ADD(&forceBuffer[atomZ+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceU.z));
            ATOMIC_ADD(&forceBuffer[atomX], (mm_ulong) realToFixedPoint(-forceV.x));
            ATOMIC_ADD(&forceBuffer[atomX+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceV.y));
            ATOMIC_ADD(&forceBuffer[atomX+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceV.z));
            ATOMIC_ADD(&forceBuffer[atomY], (mm_ulong) realToFixedPoint(-forceW.x));
            ATOMIC_ADD(&forceBuffer[atomY+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceW.y));
            ATOMIC_ADD(&forceBuffer[atomY+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forceW.z));
            ATOMIC_ADD(&forceBuffer[atom], (mm_ulong) realToFixedPoint(forceU.x + forceV.x + forceW.x));
            ATOMIC_ADD(&forceBuffer[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceU.y + forceV.y + forceW.y));
            ATOMIC_ADD(&forceBuffer[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(forceU.z + forceV.z + forceW.z));
        }
        else if (axisType == 3) {  // ThreeFold
            real3 du = vectorUW * dphiW / (normU * sinUW) + vectorUV * dphiV / (normU * sinUV)
                     - vectorUW * dphiU / (normU * sinUW) - vectorUV * dphiU / (normU * sinUV);

            real3 dv = vectorVW * dphiW / (normV * sinVW) - vectorUV * dphiU / (normV * sinUV)
                     - vectorVW * dphiV / (normV * sinVW) + vectorUV * dphiV / (normV * sinUV);

            real3 dw = -vectorUW * dphiU / (normW * sinUW) - vectorVW * dphiV / (normW * sinVW)
                     + vectorUW * dphiW / (normW * sinUW) + vectorVW * dphiW / (normW * sinVW);

            du = du / 3;
            dv = dv / 3;
            dw = dw / 3;

            ATOMIC_ADD(&forceBuffer[atomZ], (mm_ulong) realToFixedPoint(-du.x));
            ATOMIC_ADD(&forceBuffer[atomZ+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-du.y));
            ATOMIC_ADD(&forceBuffer[atomZ+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-du.z));
            ATOMIC_ADD(&forceBuffer[atomX], (mm_ulong) realToFixedPoint(-dv.x));
            ATOMIC_ADD(&forceBuffer[atomX+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-dv.y));
            ATOMIC_ADD(&forceBuffer[atomX+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-dv.z));
            ATOMIC_ADD(&forceBuffer[atomY], (mm_ulong) realToFixedPoint(-dw.x));
            ATOMIC_ADD(&forceBuffer[atomY+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-dw.y));
            ATOMIC_ADD(&forceBuffer[atomY+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-dw.z));
            ATOMIC_ADD(&forceBuffer[atom], (mm_ulong) realToFixedPoint(du.x + dv.x + dw.x));
            ATOMIC_ADD(&forceBuffer[atom+PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(du.y + dv.y + dw.y));
            ATOMIC_ADD(&forceBuffer[atom+2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(du.z + dv.z + dw.z));
        }
    }
}

DEVICE void computeTholeDamping(real r, float damp1, float damp2, real* thole3, real* thole5, real* thole3_dr, real* thole5_dr) {
#ifdef NO_DAMPING
    *thole3 = 1;
    *thole5 = 1;
    *thole3_dr = 0;
    *thole5_dr = 0;
#else
    float damp = damp1 * damp2;
    if (damp == 0) {
        *thole3 = 1;
        *thole5 = 1;
        *thole3_dr = 0;
        *thole5_dr = 0;
    }
    else {
#ifdef AMOEBA_DAMPING
        real ratio = r / damp;
        real ratio2 = ratio * ratio;
        real ratio3 = ratio2 * ratio;
        real a_ratio3 = THOLE_PARAMETER * ratio3;
        real dampExp = EXP(-a_ratio3);
        *thole3 = 1 - dampExp;
        *thole5 = 1 - (1 + a_ratio3) * dampExp;
        // Derivatives w.r.t. r
        *thole3_dr = dampExp * THOLE_PARAMETER * 3 * ratio2 / damp;
        *thole5_dr = dampExp * THOLE_PARAMETER * 3 * ratio2 * a_ratio3 / damp;
#elif defined(EXPONENTIAL_DAMPING)
        real ar = THOLE_PARAMETER * r;
        real ar2 = ar * ar;
        real ar3 = ar2 * ar;
        real dampExp = EXP(-ar);
        *thole3 = 1 - dampExp * (1 + ar + 0.5f * ar2);
        *thole5 = *thole3 - dampExp * (ar3 / 6);
        // Derivatives w.r.t. r
        real a = THOLE_PARAMETER;
        *thole3_dr = 0.5f * a * a * a * r * r * dampExp;
        *thole5_dr = a * a * a * a * r * r * r * dampExp / 6;
#elif defined(LINEAR_DAMPING)
        real s = damp;
        if (r < s) {
            real v = r / s;
            real v2 = v * v;
            real v3 = v2 * v;
            real v4 = v3 * v;
            *thole3 = 4 * v3 - 3 * v4;
            *thole5 = v4;
            *thole3_dr = (12 * v2 - 12 * v3) / s;
            *thole5_dr = 4 * v3 / s;
        }
        else {
            *thole3 = 1;
            *thole5 = 1;
            *thole3_dr = 0;
            *thole5_dr = 0;
        }
#endif
    }
#endif
}

DEVICE void calculateFixedDipoleFieldPairIxn(AtomData* atom1, LOCAL_ARG AtomData* atom2, real3 deltaR, float mScale, real3* field1, real3* field2) {
    real r2 = dot(deltaR, deltaR);
#ifdef USE_CUTOFF
    if (r2 > CUTOFF_SQUARED) {
        *field1 = make_real3(0);
        *field2 = make_real3(0);
        return;
    }
#endif
    real r = SQRT(r2);
    real rI = RECIP(r);
    real r2I = rI * rI;

    real thole3, thole5, thole3_dr, thole5_dr;
    computeTholeDamping(r, atom1->damp, atom2->damp, &thole3, &thole5, &thole3_dr, &thole5_dr);

    real dir = dot(atom1->dipole, deltaR);
    real dkr = dot(atom2->dipole, deltaR);

#ifdef USE_EWALD
    // PME real-space: use erfc-damped coefficients
    real bn0, bn1, bn2;
    computeEwaldCoefficients(r, &bn0, &bn1, &bn2);

    // erfc-damped field (from AMOEBA)
    real3 fim = -atom2->dipole * bn1 - deltaR * (bn1 * atom2->posq.w - bn2 * dkr);
    real3 fjm = -atom1->dipole * bn1 + deltaR * (bn1 * atom1->posq.w + bn2 * dir);

    // Thole-damped correction term (subtract the short-range part that will be added via reciprocal space)
#ifdef DAMP_PERM_IND_FIELD
    real dampedMScale3 = thole3 * mScale;
    real dampedMScale5 = thole5 * mScale;
#else
    real dampedMScale3 = mScale;  // No Thole damping on perm->ind field
    real dampedMScale5 = mScale;
#endif

    real rInv3 = rI * r2I;
    real rInv5 = rInv3 * r2I;

    real drr3 = (1.0f - dampedMScale3) * rInv3;
    real drr5 = 3.0f * (1.0f - dampedMScale5) * rInv5;

    real3 fid = -atom2->dipole * drr3 - deltaR * (drr3 * atom2->posq.w - drr5 * dkr);
    real3 fjd = -atom1->dipole * drr3 + deltaR * (drr3 * atom1->posq.w + drr5 * dir);

    *field1 = fim - fid;
    *field2 = fjm - fjd;
#else
    // NoCutoff: simple 1/r^n terms with Thole damping (if enabled)
#ifdef DAMP_PERM_IND_FIELD
    real t3 = thole3;
    real t5 = thole5;
#else
    real t3 = 1.0f;
    real t5 = 1.0f;
#endif
    real rr3 = rI * r2I * t3;
    real rr5 = 3 * rr3 * r2I * t5 / t3;

    real factor1 = -rr3 * atom2->posq.w + rr5 * dkr;
    *field1 = mScale * (deltaR * factor1 - rr3 * atom2->dipole);

    real factor2 = rr3 * atom1->posq.w + rr5 * dir;
    *field2 = mScale * (deltaR * factor2 - rr3 * atom1->dipole);
#endif
}

KERNEL void computeFixedField(GLOBAL mm_ulong* RESTRICT field, GLOBAL const real4* RESTRICT posq,
        GLOBAL const int2* RESTRICT covalentFlags, GLOBAL const int2* RESTRICT exclusionTiles,
        int startTileIndex, int numTileIndices,
        GLOBAL const real* RESTRICT labFrameDipole, GLOBAL const float2* RESTRICT dampingAndThole,
        GLOBAL const float* RESTRICT pairScaleFactors
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {

    const unsigned int totalWarps = GLOBAL_SIZE / TILE_SIZE;
    const unsigned int warp = GLOBAL_ID / TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE - 1);
    const unsigned int tbx = LOCAL_ID - tgx;

    LOCAL AtomData localData[THREAD_BLOCK_SIZE];

    int pos = (int) (startTileIndex + warp * (mm_long)numTileIndices / totalWarps);
    int end = (int) (startTileIndex + (warp+1) * (mm_long)numTileIndices / totalWarps);

    while (pos < end) {
        // Convert linear tile index to upper triangular (x, y) coordinates
        int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Fix occasional roundoff error
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        int atom1 = x * TILE_SIZE + tgx;
        AtomData data1 = loadAtomData(atom1, posq, labFrameDipole, dampingAndThole);
        data1.field = make_real3(0);

        if (x == y) {
            localData[LOCAL_ID] = data1;
            SYNC_WARPS;

            // Each thread computes contributions TO itself from all other atoms
            for (int j = 0; j < TILE_SIZE; j++) {
                int atom2 = y * TILE_SIZE + j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2) {
                    float mScale = pairScaleFactors[atom1 * PADDED_NUM_ATOMS + atom2];
#ifdef USE_EWALD
                    // PME: process all pairs (exclusion correction needed for mScale=0)
                    {
#elif defined(MUTUAL_POLARIZATION)
                    // NoCutoff Mutual: process all pairs (I-I forces use iScale=1 even for bonded pairs)
                    {
#else
                    // NoCutoff Direct: skip excluded pairs
                    if (mScale != 0) {
#endif
                        real3 deltaR = make_real3(localData[tbx+j].posq.x - data1.posq.x,
                                                  localData[tbx+j].posq.y - data1.posq.y,
                                                  localData[tbx+j].posq.z - data1.posq.z);
#ifdef USE_PERIODIC
                        deltaR = applyPeriodicDelta(deltaR, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                        real3 f1, f2;
                        calculateFixedDipoleFieldPairIxn(&data1, &localData[tbx+j], deltaR, mScale, &f1, &f2);
                        data1.field = data1.field + f1;
                    }
                }
            }
            SYNC_WARPS;

            if (atom1 < NUM_ATOMS) {
                ATOMIC_ADD(&field[atom1], (mm_ulong) realToFixedPoint(data1.field.x));
                ATOMIC_ADD(&field[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.y));
                ATOMIC_ADD(&field[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.z));
            }
        }
        else {
            int atom2 = y * TILE_SIZE + tgx;
            localData[LOCAL_ID] = loadAtomData(atom2, posq, labFrameDipole, dampingAndThole);
            localData[LOCAL_ID].field = make_real3(0);
            SYNC_WARPS;

            // Use rotating index to avoid race condition: each thread accesses different localData element
            unsigned int tj = tgx;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int localAtom2 = tbx + tj;
                int globalAtom2 = y * TILE_SIZE + tj;
                if (atom1 < NUM_ATOMS && globalAtom2 < NUM_ATOMS) {
                    float mScale = pairScaleFactors[atom1 * PADDED_NUM_ATOMS + globalAtom2];
#ifdef USE_EWALD
                    // PME: process all pairs (exclusion correction needed for mScale=0)
                    {
#elif defined(MUTUAL_POLARIZATION)
                    // NoCutoff Mutual: process all pairs (I-I forces use iScale=1 even for bonded pairs)
                    {
#else
                    // NoCutoff Direct: skip excluded pairs
                    if (mScale != 0) {
#endif
                        real3 deltaR = make_real3(localData[localAtom2].posq.x - data1.posq.x,
                                                  localData[localAtom2].posq.y - data1.posq.y,
                                                  localData[localAtom2].posq.z - data1.posq.z);
#ifdef USE_PERIODIC
                        deltaR = applyPeriodicDelta(deltaR, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                        real3 f1, f2;
                        calculateFixedDipoleFieldPairIxn(&data1, &localData[localAtom2], deltaR, mScale, &f1, &f2);
                        data1.field = data1.field + f1;
                        localData[localAtom2].field = localData[localAtom2].field + f2;
                    }
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            SYNC_WARPS;

            if (atom1 < NUM_ATOMS) {
                ATOMIC_ADD(&field[atom1], (mm_ulong) realToFixedPoint(data1.field.x));
                ATOMIC_ADD(&field[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.y));
                ATOMIC_ADD(&field[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.z));
            }
            atom2 = y * TILE_SIZE + tgx;
            if (atom2 < NUM_ATOMS) {
                ATOMIC_ADD(&field[atom2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.x));
                ATOMIC_ADD(&field[atom2 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.y));
                ATOMIC_ADD(&field[atom2 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.z));
            }
        }
        pos++;
    }
}

typedef struct {
    real4 posq;
    real3 force, torque;
    real3 dipole, inducedDipole;
    float thole, damp;
} ElecAtomData;

inline DEVICE ElecAtomData loadElecAtomData(int atom, GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labFrameDipole,
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const float2* RESTRICT dampingAndThole) {
    ElecAtomData data;
    data.posq = posq[atom];
    data.dipole = make_real3(labFrameDipole[3*atom], labFrameDipole[3*atom+1], labFrameDipole[3*atom+2]);
    data.inducedDipole = make_real3(inducedDipole[3*atom], inducedDipole[3*atom+1], inducedDipole[3*atom+2]);
    float2 temp = dampingAndThole[atom];
    data.damp = temp.x;
    data.thole = temp.y;
    return data;
}

DEVICE real calculateElectrostaticPairIxn(ElecAtomData* atom1, LOCAL_ARG ElecAtomData* atom2, real3 deltaR, float mScale,
        real3* force1, real3* force2, real3* torque1, real3* torque2) {
    real r2 = dot(deltaR, deltaR);
#ifdef USE_CUTOFF
    if (r2 > CUTOFF_SQUARED) {
        *force1 = make_real3(0);
        *force2 = make_real3(0);
        *torque1 = make_real3(0);
        *torque2 = make_real3(0);
        return 0;
    }
#endif
#ifdef DEBUG_TRICLINIC
    if (GLOBAL_ID == 0) {
        printf("  Pair r=%.6f deltaR=(%.6f,%.6f,%.6f)\n", SQRT(r2), deltaR.x, deltaR.y, deltaR.z);
    }
#endif
    real r = SQRT(r2);
    real rI = RECIP(r);
    real r2I = rI * rI;
    real r3I = rI * r2I;
    real r4I = r2I * r2I;
    real3 rhat = deltaR * rI;

    real thole3, thole5, thole3_dr, thole5_dr;
    computeTholeDamping(r, atom1->damp, atom2->damp, &thole3, &thole5, &thole3_dr, &thole5_dr);

    real qi = atom1->posq.w;
    real qk = atom2->posq.w;
    real3 mi = atom1->dipole;
    real3 mk = atom2->dipole;
    real3 ui = atom1->inducedDipole;
    real3 uk = atom2->inducedDipole;

    // Use dot products with unit vector rhat for forces
    real mi_rhat = dot(mi, rhat);
    real mk_rhat = dot(mk, rhat);
    real ui_rhat = dot(ui, rhat);
    real uk_rhat = dot(uk, rhat);
    real mik = dot(mi, mk);
    real uik = dot(ui, uk);
    real mi_uk = dot(mi, uk);
    real ui_mk = dot(ui, mk);

    // Dot products with deltaR for field/energy
    real dir = dot(mi, deltaR);
    real dkr = dot(mk, deltaR);
    real uir = dot(ui, deltaR);
    real ukr = dot(uk, deltaR);

#ifdef USE_EWALD
    // ===== PME REAL-SPACE: Ewald-damped terms =====
    real bn0, bn1, bn2;
    computeEwaldCoefficients(r, &bn0, &bn1, &bn2);

    // Permanent multipole energy: erfc-damped terms
    real e_cc_ewald = qi * qk * bn0;
    real e_cd_ewald = -bn1 * (qi * dkr - qk * dir);
    real e_dd_ewald = bn1 * mik - bn2 * dir * dkr;

    // Exclusion correction: subtract full Coulomb for excluded pairs
    // PME direct energy = erfc_energy - (1-mScale)*full_coulomb_energy
    real excl = 1.0f - mScale;
    real e_cc_excl = excl * rI * qi * qk;
    real e_cd_excl = excl * r3I * (qk * dir - qi * dkr);
    real e_dd_excl = excl * r3I * mik - excl * 3.0f * r3I * r2I * dir * dkr;

    real permEnergy = ENERGY_SCALE_FACTOR * (e_cc_ewald + e_cd_ewald + e_dd_ewald - e_cc_excl - e_cd_excl - e_dd_excl);

    // Permanent-Induced energy (P-I): Thole-damped only if DAMP_PERM_IND_FIELD is defined
    // P-I energy = 0.5 * (erfc_damped - (1 - mScale*thole)*full_coulomb) for charge-induced and dipole-induced
#ifdef DAMP_PERM_IND_FIELD
    real t3_pi_e = thole3;
    real t5_pi_e = thole5;
#else
    real t3_pi_e = 1.0f;
    real t5_pi_e = 1.0f;
#endif
    real e_qi_erfc = bn1 * (qi * ukr - qk * uir);
    real e_qi_full = r3I * (qi * ukr - qk * uir);
    real e_qi = e_qi_erfc - (1.0f - mScale * t3_pi_e) * e_qi_full;

    real e_pi_erfc = bn1 * (mi_uk + ui_mk) - bn2 * (dir * ukr + dkr * uir);
    real e_pi_full0 = r3I * (mi_uk + ui_mk);
    real e_pi_full1 = -3.0f * r3I * r2I * (dir * ukr + dkr * uir);
    real e_pi = e_pi_erfc - (1.0f - mScale * t3_pi_e) * e_pi_full0 - (1.0f - mScale * t5_pi_e) * e_pi_full1;

    // P-I energy with -0.5 factor for charge-induced, +0.5 for dipole-induced
    real indEnergy = ENERGY_SCALE_FACTOR * 0.5f * (-e_qi + e_pi);

    real energy = permEnergy + indEnergy;

    // Forces: Ewald-damped real-space
    real3 force = make_real3(0);

    // Ewald-damped permanent multipole force
    // bn3 = (5*bn2 + alsq2n*exp2a)/r2 where alsq2n = 8*alpha^5/sqrt(pi)
    real bn3 = (5.0f * bn2 + 8.0f * EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA / SQRT_PI * EXP(-EWALD_ALPHA * EWALD_ALPHA * r2)) / r2;

    force += qi * qk * bn1 * deltaR;
    force += bn1 * (qi * mk - qk * mi) + bn2 * (qk * dir - qi * dkr) * deltaR;
    force += bn2 * (dir * mk + dkr * mi + mik * deltaR) - bn3 * dir * dkr * deltaR;

    // Induced dipole forces: erfc-damped (for PME direct space)
    // Perpendicular components of dipoles
    real3 ui_perp = ui - ui_rhat * rhat;
    real3 uk_perp = uk - uk_rhat * rhat;
    real3 mi_perp = mi - mi_rhat * rhat;
    real3 mk_perp = mk - mk_rhat * rhat;

    // Charge-induced force (erfc-damped)
    force -= (qi * uk_rhat - qk * ui_rhat) * (r * bn2 - bn1 * rI) * deltaR;
    force -= bn1 * (qk * ui_perp - qi * uk_perp);

    // Dipole-induced force (erfc-damped)
    force += (bn2 * (mi_uk + ui_mk) + (2.0f * bn2 - r2 * bn3) * (mi_rhat * uk_rhat + mk_rhat * ui_rhat)) * deltaR;
    force += bn2 * r * (mi_rhat * uk_perp + uk_rhat * mi_perp + mk_rhat * ui_perp + ui_rhat * mk_perp);

#ifdef MUTUAL_POLARIZATION
    // Induced-induced force (Thole-damped erfc - mixed Thole and erfc)
    force += (thole3 * bn2 * uik + thole5 * (2.0f * bn2 - r2 * bn3) * ui_rhat * uk_rhat) * deltaR;
    force += thole5 * bn2 * r * (uk_rhat * ui_perp + ui_rhat * uk_perp);
    // Damping derivative term
    force -= r3I * (thole3_dr * uik - 3.0f * thole5_dr * ui_rhat * uk_rhat) * rhat;
#endif

    // Full Coulomb induced forces for exclusion correction
    real3 fullIndForce = make_real3(0);
    // Charge-induced full Coulomb
    fullIndForce += (qk * (3.0f * ui_rhat * rhat - ui) - qi * (3.0f * uk_rhat * rhat - uk)) * r3I;
    // Dipole-induced full Coulomb
    fullIndForce += r4I * (3.0f * (mi_rhat * uk + uk_rhat * mi + mk_rhat * ui + ui_rhat * mk + (mi_uk + ui_mk) * rhat)
                         - 15.0f * (mi_rhat * uk_rhat + mk_rhat * ui_rhat) * rhat);

    // Exclusion correction forces: subtract full Coulomb for excluded pairs
    real rInv5 = r3I * r2I;
    real rInv7 = rInv5 * r2I;
    force -= excl * r3I * qi * qk * deltaR;
    force -= excl * r3I * (qi * mk - qk * mi) + excl * 3.0f * rInv5 * (qk * dir - qi * dkr) * deltaR;
    force -= excl * 3.0f * rInv5 * (dir * mk + dkr * mi + mik * deltaR) - excl * 15.0f * rInv7 * dir * dkr * deltaR;
    // Induced force exclusion correction
    force -= excl * fullIndForce;

    *force1 = -ENERGY_SCALE_FACTOR * force;
    *force2 = ENERGY_SCALE_FACTOR * force;

    // Torques: erfc-damped field from permanent multipoles
    real3 fieldAtI = -qk * bn1 * deltaR - bn1 * mk + bn2 * dkr * deltaR;
    real3 fieldAtK = qi * bn1 * deltaR - bn1 * mi + bn2 * dir * deltaR;

    // Add erfc-damped induced field (Thole-damped only if DAMP_PERM_IND_FIELD)
    // For non-excluded: mScale * thole * erfc
    // For excluded: (erfc - full) for exclusion correction
    real3 indFieldAtI_erfc = (bn2 * r2 * uk_rhat * rhat - bn1 * uk);
    real3 indFieldAtK_erfc = (bn2 * r2 * ui_rhat * rhat - bn1 * ui);
    real3 indFieldAtI_full = (3.0f * uk_rhat * rhat - uk) * r3I;
    real3 indFieldAtK_full = (3.0f * ui_rhat * rhat - ui) * r3I;
    fieldAtI += mScale * t3_pi_e * indFieldAtI_erfc - excl * (indFieldAtI_full - indFieldAtI_erfc);
    fieldAtK += mScale * t3_pi_e * indFieldAtK_erfc - excl * (indFieldAtK_full - indFieldAtK_erfc);

    // Subtract exclusion correction: full Coulomb field for excluded pairs (permanent multipoles)
    fieldAtI -= excl * r3I * (-qk * deltaR - mk) + excl * 3.0f * rInv5 * dkr * deltaR;
    fieldAtK -= excl * r3I * (qi * deltaR - mi) + excl * 3.0f * rInv5 * dir * deltaR;

    *torque1 = ENERGY_SCALE_FACTOR * cross(mi, fieldAtI);
    *torque2 = ENERGY_SCALE_FACTOR * cross(mk, fieldAtK);

#else
    // ===== NoCutoff: simple 1/r^n terms =====
    real e_cc = qi * qk * rI;
    real e_cd = (qk * mi_rhat - qi * mk_rhat) * r2I;
    real e_dd = (mik - 3 * mi_rhat * mk_rhat) * r3I;
    real energy = ENERGY_SCALE_FACTOR * mScale * (e_cc + e_cd + e_dd);

    // P-P forces (separate for debugging)
    real3 f_cc = qi * qk * r2I * rhat;
    real3 f_cd = (qk * (3 * mi_rhat * rhat - mi) - qi * (3 * mk_rhat * rhat - mk)) * r3I;
    real3 f_dd = (3 * (mi_rhat * mk + mk_rhat * mi + mik * rhat) - 15 * mi_rhat * mk_rhat * rhat) * r4I;

    // P-I forces: use damping only if DAMP_PERM_IND_FIELD is defined
#ifdef DAMP_PERM_IND_FIELD
    real t3_pi = thole3;
    real t5_pi = thole5;
    real t3_dr_pi = thole3_dr;
    real t5_dr_pi = thole5_dr;
#else
    real t3_pi = 1.0f;
    real t5_pi = 1.0f;
    real t3_dr_pi = 0.0f;
    real t5_dr_pi = 0.0f;
#endif

    // Charge-induced (f_ci) and dipole-induced (f_di) forces
    real3 f_ci = t3_pi * (qk * (3 * ui_rhat * rhat - ui) - qi * (3 * uk_rhat * rhat - uk)) * r3I;
    f_ci += t3_dr_pi * r2I * (qi * uk_rhat - qk * ui_rhat) * rhat;

    real3 f_di = t5_pi * 3 * (mi_rhat * uk + uk_rhat * mi + ui_rhat * mk + mk_rhat * ui) * r4I;
    f_di -= t5_pi * 15 * (mi_rhat * uk_rhat + ui_rhat * mk_rhat) * rhat * r4I;
    f_di += t3_pi * 3 * (mi_uk + ui_mk) * rhat * r4I;
    f_di -= r3I * (t3_dr_pi * (mi_uk + ui_mk) - 3 * t5_dr_pi * (mi_rhat * uk_rhat + ui_rhat * mk_rhat)) * rhat;

    real3 force = f_cc + f_cd + f_dd + f_ci + f_di;

    // Apply mScale to permanent and P-I forces
    *force1 = -ENERGY_SCALE_FACTOR * mScale * force;
    *force2 = ENERGY_SCALE_FACTOR * mScale * force;

#ifdef MUTUAL_POLARIZATION
    // I-I forces: use iScale (which is 1.0 for TholeDipole)
    // These are computed separately because they use a different scale factor
    real3 forceII = make_real3(0);
    forceII += thole5 * 3 * (ui_rhat * uk + uk_rhat * ui) * r4I;
    forceII -= thole5 * 15 * ui_rhat * uk_rhat * rhat * r4I;
    forceII += thole3 * 3 * uik * rhat * r4I;
    forceII -= r3I * (thole3_dr * uik - 3 * thole5_dr * ui_rhat * uk_rhat) * rhat;
    // iScale = 1.0 for all pairs in TholeDipole
    *force1 -= ENERGY_SCALE_FACTOR * forceII;
    *force2 += ENERGY_SCALE_FACTOR * forceII;
#endif

    real3 fieldAtI = -qk * r2I * rhat + (3 * mk_rhat * rhat - mk) * r3I;
    real3 fieldAtK = qi * r2I * rhat + (3 * mi_rhat * rhat - mi) * r3I;
    fieldAtI += (t5_pi * 3 * uk_rhat * rhat - t3_pi * uk) * r3I;
    fieldAtK += (t5_pi * 3 * ui_rhat * rhat - t3_pi * ui) * r3I;

    *torque1 = ENERGY_SCALE_FACTOR * mScale * cross(mi, fieldAtI);
    *torque2 = ENERGY_SCALE_FACTOR * mScale * cross(mk, fieldAtK);
#endif

    return energy;
}

KERNEL void computeElectrostatics(GLOBAL mm_ulong* RESTRICT forceBuffer, GLOBAL mm_ulong* RESTRICT torqueBuffer,
        GLOBAL mixed* RESTRICT energyBuffer, GLOBAL const real4* RESTRICT posq,
        GLOBAL const int2* RESTRICT covalentFlags, GLOBAL const int2* RESTRICT exclusionTiles,
        int startTileIndex, int numTileIndices,
        GLOBAL const real* RESTRICT labFrameDipole, GLOBAL const real* RESTRICT inducedDipole,
        GLOBAL const float2* RESTRICT dampingAndThole, GLOBAL const float* RESTRICT pairScaleFactors
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {

#ifdef DEBUG_TRICLINIC
    if (GLOBAL_ID == 0) {
        printf("computeElectrostatics: periodicBoxVecX=(%.6f,%.6f,%.6f)\n", periodicBoxVecX.x, periodicBoxVecX.y, periodicBoxVecX.z);
        printf("computeElectrostatics: periodicBoxVecY=(%.6f,%.6f,%.6f)\n", periodicBoxVecY.x, periodicBoxVecY.y, periodicBoxVecY.z);
        printf("computeElectrostatics: periodicBoxVecZ=(%.6f,%.6f,%.6f)\n", periodicBoxVecZ.x, periodicBoxVecZ.y, periodicBoxVecZ.z);
    }
#endif

    const unsigned int totalWarps = GLOBAL_SIZE / TILE_SIZE;
    const unsigned int warp = GLOBAL_ID / TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE - 1);
    const unsigned int tbx = LOCAL_ID - tgx;

    LOCAL ElecAtomData localData[THREAD_BLOCK_SIZE];
    mixed energy = 0;

    int pos = (int) (startTileIndex + warp * (mm_long)numTileIndices / totalWarps);
    int end = (int) (startTileIndex + (warp+1) * (mm_long)numTileIndices / totalWarps);

    while (pos < end) {
        // Convert linear tile index to upper triangular (x, y) coordinates
        // Formula from OpenMM: maps tile index to upper triangular matrix position
        int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Fix occasional roundoff error
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        unsigned int atom1 = x * TILE_SIZE + tgx;
        ElecAtomData data1 = loadElecAtomData(atom1, posq, labFrameDipole, inducedDipole, dampingAndThole);
        data1.force = make_real3(0);
        data1.torque = make_real3(0);

        if (x == y) {
            localData[LOCAL_ID].posq = data1.posq;
            localData[LOCAL_ID].dipole = data1.dipole;
            localData[LOCAL_ID].inducedDipole = data1.inducedDipole;
            localData[LOCAL_ID].damp = data1.damp;
            localData[LOCAL_ID].thole = data1.thole;
            localData[LOCAL_ID].force = make_real3(0);
            localData[LOCAL_ID].torque = make_real3(0);
            SYNC_WARPS;

            // Each thread computes contributions TO itself from all other atoms
            // Energy is counted once per pair (atom1 < atom2)
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                unsigned int atom2 = y * TILE_SIZE + j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2) {
                    float mScale = pairScaleFactors[atom1 * PADDED_NUM_ATOMS + atom2];
#ifdef USE_EWALD
                    // PME: process all pairs (exclusion correction needed for mScale=0)
                    {
#elif defined(MUTUAL_POLARIZATION)
                    // NoCutoff Mutual: process all pairs (I-I forces use iScale=1 even for bonded pairs)
                    {
#else
                    // NoCutoff Direct: skip excluded pairs
                    if (mScale != 0) {
#endif
                        real3 deltaR = make_real3(localData[tbx+j].posq.x - data1.posq.x,
                                                  localData[tbx+j].posq.y - data1.posq.y,
                                                  localData[tbx+j].posq.z - data1.posq.z);
#ifdef USE_PERIODIC
                        deltaR = applyPeriodicDelta(deltaR, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                        real3 f1, f2, t1, t2;
                        real e = calculateElectrostaticPairIxn(&data1, &localData[tbx+j], deltaR, mScale, &f1, &f2, &t1, &t2);
#ifdef DEBUG_TRICLINIC
                        real r_final = SQRT(dot(deltaR, deltaR));
                        // Print pairs that ARE within cutoff and show energy contribution
                        if (r_final < 0.7f && atom1 < atom2 && atom1 < 3) {
                            printf("Pair %d-%d: r=%.4f mScale=%.1f e=%.4f pos1=(%.3f,%.3f,%.3f) pos2=(%.3f,%.3f,%.3f)\n",
                                   atom1, atom2, r_final, mScale, e,
                                   data1.posq.x, data1.posq.y, data1.posq.z,
                                   localData[tbx+j].posq.x, localData[tbx+j].posq.y, localData[tbx+j].posq.z);
                        }
#endif
                        // Energy counted once per pair
                        if (atom1 < atom2)
                            energy += e;
                        // Force and torque on this atom
                        data1.force = data1.force + f1;
                        data1.torque = data1.torque + t1;
                    }
                }
            }
            SYNC_WARPS;

            if (atom1 < NUM_ATOMS) {
                ATOMIC_ADD(&forceBuffer[atom1], (mm_ulong) realToFixedPoint(data1.force.x));
                ATOMIC_ADD(&forceBuffer[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.force.y));
                ATOMIC_ADD(&forceBuffer[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.force.z));
                ATOMIC_ADD(&torqueBuffer[atom1], (mm_ulong) realToFixedPoint(data1.torque.x));
                ATOMIC_ADD(&torqueBuffer[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.torque.y));
                ATOMIC_ADD(&torqueBuffer[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.torque.z));
            }
        }
        else {
            unsigned int atom2 = y * TILE_SIZE + tgx;
            localData[LOCAL_ID] = loadElecAtomData(atom2, posq, labFrameDipole, inducedDipole, dampingAndThole);
            localData[LOCAL_ID].force = make_real3(0);
            localData[LOCAL_ID].torque = make_real3(0);
            SYNC_WARPS;

            // Use rotating index to avoid race condition: each thread accesses different localData element
            unsigned int tj = tgx;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int localAtom2 = tbx + tj;
                unsigned int globalAtom2 = y * TILE_SIZE + tj;
                if (atom1 < NUM_ATOMS && globalAtom2 < NUM_ATOMS) {
                    float mScale = pairScaleFactors[atom1 * PADDED_NUM_ATOMS + globalAtom2];
#ifdef USE_EWALD
                    // PME: process all pairs (exclusion correction needed for mScale=0)
                    {
#elif defined(MUTUAL_POLARIZATION)
                    // NoCutoff Mutual: process all pairs (I-I forces use iScale=1 even for bonded pairs)
                    {
#else
                    // NoCutoff Direct: skip excluded pairs
                    if (mScale != 0) {
#endif
                        real3 deltaR = make_real3(localData[localAtom2].posq.x - data1.posq.x,
                                                  localData[localAtom2].posq.y - data1.posq.y,
                                                  localData[localAtom2].posq.z - data1.posq.z);
#ifdef USE_PERIODIC
                        deltaR = applyPeriodicDelta(deltaR, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                        real3 f1, f2, t1, t2;
                        real e = calculateElectrostaticPairIxn(&data1, &localData[localAtom2], deltaR, mScale, &f1, &f2, &t1, &t2);
                        energy += e;
                        data1.force = data1.force + f1;
                        data1.torque = data1.torque + t1;
                        localData[localAtom2].force = localData[localAtom2].force + f2;
                        localData[localAtom2].torque = localData[localAtom2].torque + t2;
                    }
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            SYNC_WARPS;

            if (atom1 < NUM_ATOMS) {
                ATOMIC_ADD(&forceBuffer[atom1], (mm_ulong) realToFixedPoint(data1.force.x));
                ATOMIC_ADD(&forceBuffer[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.force.y));
                ATOMIC_ADD(&forceBuffer[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.force.z));
                ATOMIC_ADD(&torqueBuffer[atom1], (mm_ulong) realToFixedPoint(data1.torque.x));
                ATOMIC_ADD(&torqueBuffer[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.torque.y));
                ATOMIC_ADD(&torqueBuffer[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.torque.z));
            }
            atom2 = y * TILE_SIZE + tgx;
            if (atom2 < NUM_ATOMS) {
                ATOMIC_ADD(&forceBuffer[atom2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.x));
                ATOMIC_ADD(&forceBuffer[atom2 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.y));
                ATOMIC_ADD(&forceBuffer[atom2 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].force.z));
                ATOMIC_ADD(&torqueBuffer[atom2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].torque.x));
                ATOMIC_ADD(&torqueBuffer[atom2 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].torque.y));
                ATOMIC_ADD(&torqueBuffer[atom2 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].torque.z));
            }
        }
        pos++;
    }

    energyBuffer[GLOBAL_ID] += energy;
}

#ifdef MUTUAL_POLARIZATION

KERNEL void computeInducedField(GLOBAL mm_ulong* RESTRICT inducedField, GLOBAL const real4* RESTRICT posq,
        GLOBAL const int2* RESTRICT exclusionTiles, GLOBAL const real* RESTRICT inducedDipole,
        int startTileIndex, int numTileIndices, GLOBAL const float2* RESTRICT dampingAndThole
#ifdef USE_PERIODIC
        , real4 periodicBoxSize, real4 invPeriodicBoxSize,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ
#endif
        ) {

    const unsigned int totalWarps = GLOBAL_SIZE / TILE_SIZE;
    const unsigned int warp = GLOBAL_ID / TILE_SIZE;
    const unsigned int tgx = LOCAL_ID & (TILE_SIZE - 1);
    const unsigned int tbx = LOCAL_ID - tgx;

    LOCAL AtomData localData[THREAD_BLOCK_SIZE];

    int pos = (int) (startTileIndex + warp * (mm_long)numTileIndices / totalWarps);
    int end = (int) (startTileIndex + (warp+1) * (mm_long)numTileIndices / totalWarps);

    while (pos < end) {
        // Convert linear tile index to upper triangular (x, y) coordinates
        int y = (int) floor(NUM_BLOCKS+0.5f-SQRT((NUM_BLOCKS+0.5f)*(NUM_BLOCKS+0.5f)-2*pos));
        int x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        if (x < y || x >= NUM_BLOCKS) { // Fix occasional roundoff error
            y += (x < y ? -1 : 1);
            x = (pos-y*NUM_BLOCKS+y*(y+1)/2);
        }

        int atom1 = x * TILE_SIZE + tgx;
        real4 p = posq[atom1];
        float2 dt = dampingAndThole[atom1];
        AtomData data1;
        data1.posq = p;
        data1.dipole = make_real3(inducedDipole[3*atom1], inducedDipole[3*atom1+1], inducedDipole[3*atom1+2]);
        data1.damp = dt.x;
        data1.thole = dt.y;
        data1.field = make_real3(0);

        if (x == y) {
            localData[LOCAL_ID] = data1;
            SYNC_WARPS;

            // Each thread computes field contributions TO itself from all other atoms
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                unsigned int atom2 = y * TILE_SIZE + j;
                if (atom1 < NUM_ATOMS && atom2 < NUM_ATOMS && atom1 != atom2) {
                    real3 deltaR = make_real3(localData[tbx+j].posq.x - data1.posq.x,
                                              localData[tbx+j].posq.y - data1.posq.y,
                                              localData[tbx+j].posq.z - data1.posq.z);
#ifdef USE_PERIODIC
                    deltaR = applyPeriodicDelta(deltaR, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                    real r2 = dot(deltaR, deltaR);
#ifdef USE_CUTOFF
                    if (r2 > CUTOFF_SQUARED)
                        continue;
#endif
                    real r = SQRT(r2);
                    real rI = RECIP(r);
                    real r2I = rI * rI;

                    real thole3, thole5, thole3_dr, thole5_dr;
                    computeTholeDamping(r, data1.damp, localData[tbx+j].damp, &thole3, &thole5, &thole3_dr, &thole5_dr);

#ifdef USE_EWALD
                    // Use erfc-damped coefficients for PME
                    real ralpha = EWALD_ALPHA * r;
                    real bn0 = erfc(ralpha) * rI;
                    real alsq2 = 2.0f * EWALD_ALPHA * EWALD_ALPHA;
                    real alsq2n = RECIP(SQRT_PI * EWALD_ALPHA);
                    real exp2a = EXP(-(ralpha * ralpha));
                    alsq2n *= alsq2;
                    real bn1 = (bn0 + alsq2n * exp2a) * r2I;
                    alsq2n *= alsq2;
                    real bn2 = (3.0f * bn1 + alsq2n * exp2a) * r2I;

                    real rr3 = thole3 * bn1;
                    real rr5 = thole5 * bn2;
#else
                    real rr3 = rI * r2I * thole3;
                    real rr5 = 3 * rr3 * r2I * thole5 / thole3;
#endif

                    real dkr = dot(localData[tbx+j].dipole, deltaR);

                    data1.field = data1.field + deltaR * rr5 * dkr - localData[tbx+j].dipole * rr3;
                }
            }
            SYNC_WARPS;

            if (atom1 < NUM_ATOMS) {
                ATOMIC_ADD(&inducedField[atom1], (mm_ulong) realToFixedPoint(data1.field.x));
                ATOMIC_ADD(&inducedField[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.y));
                ATOMIC_ADD(&inducedField[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.z));
            }
        }
        else {
            int atom2 = y * TILE_SIZE + tgx;
            real4 p2 = posq[atom2];
            float2 dt2 = dampingAndThole[atom2];
            localData[LOCAL_ID].posq = p2;
            localData[LOCAL_ID].dipole = make_real3(inducedDipole[3*atom2], inducedDipole[3*atom2+1], inducedDipole[3*atom2+2]);
            localData[LOCAL_ID].damp = dt2.x;
            localData[LOCAL_ID].thole = dt2.y;
            localData[LOCAL_ID].field = make_real3(0);
            SYNC_WARPS;

            // Use rotating index to avoid race condition: each thread accesses different localData element
            unsigned int tj = tgx;
            for (unsigned int j = 0; j < TILE_SIZE; j++) {
                int localAtom2 = tbx + tj;
                unsigned int globalAtom2 = y * TILE_SIZE + tj;
                if (atom1 < NUM_ATOMS && globalAtom2 < NUM_ATOMS) {
                    real3 deltaR = make_real3(localData[localAtom2].posq.x - data1.posq.x,
                                              localData[localAtom2].posq.y - data1.posq.y,
                                              localData[localAtom2].posq.z - data1.posq.z);
#ifdef USE_PERIODIC
                    deltaR = applyPeriodicDelta(deltaR, periodicBoxSize, invPeriodicBoxSize, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ);
#endif
                    real r2 = dot(deltaR, deltaR);
#ifdef USE_CUTOFF
                    if (r2 > CUTOFF_SQUARED) {
                        tj = (tj + 1) & (TILE_SIZE - 1);
                        continue;
                    }
#endif
                    real r = SQRT(r2);
                    real rI = RECIP(r);
                    real r2I = rI * rI;

                    real thole3, thole5, thole3_dr, thole5_dr;
                    computeTholeDamping(r, data1.damp, localData[localAtom2].damp, &thole3, &thole5, &thole3_dr, &thole5_dr);

#ifdef USE_EWALD
                    // Use erfc-damped coefficients for PME
                    real ralpha = EWALD_ALPHA * r;
                    real bn0 = erfc(ralpha) * rI;
                    real alsq2 = 2.0f * EWALD_ALPHA * EWALD_ALPHA;
                    real alsq2n = RECIP(SQRT_PI * EWALD_ALPHA);
                    real exp2a = EXP(-(ralpha * ralpha));
                    alsq2n *= alsq2;
                    real bn1 = (bn0 + alsq2n * exp2a) * r2I;
                    alsq2n *= alsq2;
                    real bn2 = (3.0f * bn1 + alsq2n * exp2a) * r2I;

                    real rr3 = thole3 * bn1;
                    real rr5 = thole5 * bn2;
#else
                    real rr3 = rI * r2I * thole3;
                    real rr5 = 3 * rr3 * r2I * thole5 / thole3;
#endif

                    real dkr = dot(localData[localAtom2].dipole, deltaR);
                    real dir = dot(data1.dipole, deltaR);

                    // Field at atom1 from dipole at atom2 (using deltaR = pos2 - pos1)
                    data1.field = data1.field + deltaR * rr5 * dkr - localData[localAtom2].dipole * rr3;
                    // Field at atom2 from dipole at atom1 (using -deltaR = pos1 - pos2)
                    localData[localAtom2].field = localData[localAtom2].field + deltaR * rr5 * dir - data1.dipole * rr3;
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
            }
            SYNC_WARPS;

            if (atom1 < NUM_ATOMS) {
                ATOMIC_ADD(&inducedField[atom1], (mm_ulong) realToFixedPoint(data1.field.x));
                ATOMIC_ADD(&inducedField[atom1 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.y));
                ATOMIC_ADD(&inducedField[atom1 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(data1.field.z));
            }
            atom2 = y * TILE_SIZE + tgx;
            if (atom2 < NUM_ATOMS) {
                ATOMIC_ADD(&inducedField[atom2], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.x));
                ATOMIC_ADD(&inducedField[atom2 + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.y));
                ATOMIC_ADD(&inducedField[atom2 + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(localData[LOCAL_ID].field.z));
            }
        }
        pos++;
    }
}

KERNEL void recordInducedDipolesForDIIS(GLOBAL const long long* RESTRICT field, GLOBAL const long long* RESTRICT inducedField,
        GLOBAL const float* RESTRICT polarizability, GLOBAL real* RESTRICT inducedDipole,
        GLOBAL float2* RESTRICT errors, GLOBAL real* RESTRICT prevDipoles, GLOBAL real* RESTRICT prevErrors,
        GLOBAL float* RESTRICT diisMatrix, int iteration) {
    LOCAL real errorBuffer[THREAD_BLOCK_SIZE];
    real scale = RECIP((real) 0x100000000);
    real totalError = 0;
    int prevIndex = iteration % MAX_PREV_DIIS_DIPOLES;

    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real alpha = polarizability[atom];
        real3 newDipole;
        newDipole.x = alpha * scale * (field[atom] + inducedField[atom]);
        newDipole.y = alpha * scale * (field[atom + PADDED_NUM_ATOMS] + inducedField[atom + PADDED_NUM_ATOMS]);
        newDipole.z = alpha * scale * (field[atom + 2*PADDED_NUM_ATOMS] + inducedField[atom + 2*PADDED_NUM_ATOMS]);

        real3 oldDipole = make_real3(inducedDipole[3*atom], inducedDipole[3*atom+1], inducedDipole[3*atom+2]);
        real3 error = newDipole - oldDipole;

        prevDipoles[prevIndex*3*NUM_ATOMS + 3*atom] = newDipole.x;
        prevDipoles[prevIndex*3*NUM_ATOMS + 3*atom+1] = newDipole.y;
        prevDipoles[prevIndex*3*NUM_ATOMS + 3*atom+2] = newDipole.z;
        prevErrors[prevIndex*3*NUM_ATOMS + 3*atom] = error.x;
        prevErrors[prevIndex*3*NUM_ATOMS + 3*atom+1] = error.y;
        prevErrors[prevIndex*3*NUM_ATOMS + 3*atom+2] = error.z;

        // Weight error by 1/alpha to match PCG metric (r·z = r·r/alpha)
        real alphaInv = (alpha > 1e-12f) ? RECIP(alpha) : 0.0f;
        totalError += alphaInv * (error.x*error.x + error.y*error.y + error.z*error.z);

        // Update induced dipoles to new value
        inducedDipole[3*atom] = newDipole.x;
        inducedDipole[3*atom+1] = newDipole.y;
        inducedDipole[3*atom+2] = newDipole.z;
    }

    // Block-level reduction to sum errors across all threads in this block
    errorBuffer[LOCAL_ID] = totalError;
    SYNC_THREADS;
    for (int offset = 1; offset < LOCAL_SIZE; offset *= 2) {
        if (LOCAL_ID + offset < LOCAL_SIZE && (LOCAL_ID & (2*offset-1)) == 0)
            errorBuffer[LOCAL_ID] += errorBuffer[LOCAL_ID + offset];
        SYNC_THREADS;
    }
    if (LOCAL_ID == 0)
        errors[GROUP_ID] = make_float2((float) errorBuffer[0], 0);
}

KERNEL void computeDIISMatrix(GLOBAL const real* RESTRICT prevErrors, int iteration, GLOBAL float* RESTRICT diisMatrix) {
    int numPrev = min(iteration+1, MAX_PREV_DIIS_DIPOLES);
    for (int i = 0; i < numPrev; i++) {
        for (int j = i; j < numPrev; j++) {
            float sum = 0;
            for (int k = 0; k < 3*NUM_ATOMS; k++) {
                int indexI = (i % MAX_PREV_DIIS_DIPOLES) * 3 * NUM_ATOMS + k;
                int indexJ = (j % MAX_PREV_DIIS_DIPOLES) * 3 * NUM_ATOMS + k;
                sum += prevErrors[indexI] * prevErrors[indexJ];
            }
            diisMatrix[i*MAX_PREV_DIIS_DIPOLES + j] = sum;
            diisMatrix[j*MAX_PREV_DIIS_DIPOLES + i] = sum;
        }
    }
}

KERNEL void solveDIISMatrix(int numPrev, GLOBAL const float* RESTRICT diisMatrix, GLOBAL float* RESTRICT coefficients) {
    LOCAL real b[MAX_PREV_DIIS_DIPOLES+1][MAX_PREV_DIIS_DIPOLES+1];
    LOCAL real piv[MAX_PREV_DIIS_DIPOLES+1];
    LOCAL real x[MAX_PREV_DIIS_DIPOLES+1];

    if (numPrev == 1) {
        if (LOCAL_ID == 0)
            coefficients[0] = 1;
        return;
    }

    int rank = numPrev + 1;
    for (int index = LOCAL_ID; index < numPrev*numPrev; index += LOCAL_SIZE) {
        int i = index / numPrev;
        int j = index - i*numPrev;
        b[i+1][j+1] = diisMatrix[i*MAX_PREV_DIIS_DIPOLES + j];
    }
    for (int i = LOCAL_ID; i < rank; i += LOCAL_SIZE) {
        b[i][0] = -1;
        piv[i] = i;
    }
    SYNC_THREADS;

    if (LOCAL_ID == 0) {
        real mean = 0;
        for (int i = 0; i < numPrev; i++)
            for (int j = 0; j < numPrev; j++)
                mean += fabs(b[i+1][j+1]);
        mean /= numPrev*numPrev;
        b[0][0] = 0;
        for (int i = 1; i < rank; i++)
            b[0][i] = -mean;

        int pivsign = 1;
        for (int j = 0; j < rank; j++) {
            for (int i = 0; i < rank; i++) {
                int kmax = min(i, j);
                real s = 0;
                for (int k = 0; k < kmax; k++)
                    s += b[i][k] * b[k][j];
                b[i][j] -= s;
            }

            int p = j;
            for (int i = j+1; i < rank; i++)
                if (fabs(b[i][j]) > fabs(b[p][j]))
                    p = i;
            if (p != j) {
                int k = 0;
                for (k = 0; k < rank; k++) {
                    real t = b[p][k];
                    b[p][k] = b[j][k];
                    b[j][k] = t;
                }
                k = piv[p];
                piv[p] = piv[j];
                piv[j] = k;
                pivsign = -pivsign;
            }

            if ((j < rank) && (b[j][j] != 0))
                for (int i = j+1; i < rank; i++)
                    b[i][j] /= b[j][j];
        }
        for (int i = 0; i < rank; i++)
            if (b[i][i] == 0) {
                for (int j = 0; j < rank-1; j++)
                    coefficients[j] = 0;
                coefficients[rank-1] = 1;
                return;
            }

        for (int i = 0; i < rank; i++)
            x[i] = (piv[i] == 0 ? -1 : 0);
        for (int k = 0; k < rank; k++)
            for (int i = k+1; i < rank; i++)
                x[i] -= x[k] * b[i][k];

        for (int k = rank-1; k >= 0; k--) {
            x[k] /= b[k][k];
            for (int i = 0; i < k; i++)
                x[i] -= x[k] * b[i][k];
        }

        real lastCoeff = 1;
        for (int i = 0; i < rank-1; i++) {
            real c = x[i+1] * mean;
            coefficients[i] = c;
            lastCoeff -= c;
        }
        coefficients[rank-1] = lastCoeff;
    }
}

KERNEL void updateInducedFieldByDIIS(GLOBAL real* RESTRICT inducedDipole, GLOBAL const real* RESTRICT prevDipoles,
        GLOBAL const float* RESTRICT coefficients, int numPrev) {
    for (int atom = GLOBAL_ID; atom < NUM_ATOMS; atom += GLOBAL_SIZE) {
        real3 newDipole = make_real3(0);
        for (int i = 0; i < numPrev; i++) {
            int prevIndex = i % MAX_PREV_DIIS_DIPOLES;
            float c = coefficients[i];
            newDipole.x += c * prevDipoles[prevIndex*3*NUM_ATOMS + 3*atom];
            newDipole.y += c * prevDipoles[prevIndex*3*NUM_ATOMS + 3*atom+1];
            newDipole.z += c * prevDipoles[prevIndex*3*NUM_ATOMS + 3*atom+2];
        }
        inducedDipole[3*atom] = newDipole.x;
        inducedDipole[3*atom+1] = newDipole.y;
        inducedDipole[3*atom+2] = newDipole.z;
    }
}

#endif // MUTUAL_POLARIZATION

KERNEL void computePotentialAtPoints(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labFrameDipole,
        GLOBAL const real* RESTRICT inducedDipole, GLOBAL const real4* RESTRICT points, GLOBAL real* RESTRICT potential,
        int numPoints, real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ) {
    for (int point = GLOBAL_ID; point < numPoints; point += GLOBAL_SIZE) {
        real4 p = points[point];
        real phi = 0;
        for (int atom = 0; atom < NUM_ATOMS; atom++) {
            real4 atomPos = posq[atom];
            real3 deltaR = make_real3(atomPos.x - p.x, atomPos.y - p.y, atomPos.z - p.z);
            real r = SQRT(dot(deltaR, deltaR));
            if (r > 0) {
                real rI = RECIP(r);
                real r2I = rI * rI;
                real q = atomPos.w;
                real3 d = make_real3(labFrameDipole[3*atom] + inducedDipole[3*atom],
                                     labFrameDipole[3*atom+1] + inducedDipole[3*atom+1],
                                     labFrameDipole[3*atom+2] + inducedDipole[3*atom+2]);
                real dr = dot(d, deltaR);
                phi += ENERGY_SCALE_FACTOR * (q * rI + dr * rI * r2I);
            }
        }
        potential[point] = phi;
    }
}

// ==================== PME Kernels ====================
#ifdef USE_EWALD

#define ARRAY(x,y) array[(x)-1+((y)-1)*PME_ORDER]

/**
 * Calculate the spline coefficients for a single atom along a single axis.
 * thetai[k] = (B-spline value, 1st derivative, 2nd derivative, 3rd derivative)
 */
DEVICE void computeBSplinePoint(real4* thetai, real w, real* array) {
    // Initialize for 2nd order recursion
    ARRAY(2,2) = w;
    ARRAY(2,1) = 1 - w;

    // Build to 3rd order
    ARRAY(3,3) = 0.5f * w * ARRAY(2,2);
    ARRAY(3,2) = 0.5f * ((1+w)*ARRAY(2,1) + (2-w)*ARRAY(2,2));
    ARRAY(3,1) = 0.5f * (1-w) * ARRAY(2,1);

    // Standard B-spline recursion to order 5
    for (int i = 4; i <= PME_ORDER; i++) {
        int k = i - 1;
        real denom = RECIP((real) k);
        ARRAY(i,i) = denom * w * ARRAY(k,k);
        for (int j = 1; j <= i-2; j++)
            ARRAY(i,i-j) = denom * ((w+j)*ARRAY(k,i-j-1) + (i-j-w)*ARRAY(k,i-j));
        ARRAY(i,1) = denom * (1-w) * ARRAY(k,1);
    }

    // First derivative
    int k = PME_ORDER - 1;
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // Second derivative
    k = PME_ORDER - 2;
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // Third derivative
    k = PME_ORDER - 3;
    ARRAY(k,PME_ORDER-2) = ARRAY(k,PME_ORDER-3);
    for (int i = PME_ORDER-3; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER-1) = ARRAY(k,PME_ORDER-2);
    for (int i = PME_ORDER-2; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);
    ARRAY(k,PME_ORDER) = ARRAY(k,PME_ORDER-1);
    for (int i = PME_ORDER-1; i >= 2; i--)
        ARRAY(k,i) = ARRAY(k,i-1) - ARRAY(k,i);
    ARRAY(k,1) = -ARRAY(k,1);

    // Copy to output
    for (int i = 1; i <= PME_ORDER; i++)
        thetai[i-1] = make_real4(ARRAY(PME_ORDER,i), ARRAY(PME_ORDER-1,i), ARRAY(PME_ORDER-2,i), ARRAY(PME_ORDER-3,i));
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Wrap position into primary unit cell using triclinic box vectors.
 */
DEVICE void wrapPositionIntoCell(real4* pos, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
                                  real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // Wrap Z: compute lambda_z = pos · recipBoxVec[2]
    real scale = floor((*pos).x*recipBoxVecZ.x + (*pos).y*recipBoxVecZ.y + (*pos).z*recipBoxVecZ.z + 0.5f);
    (*pos).x -= periodicBoxVecZ.x * scale;
    (*pos).y -= periodicBoxVecZ.y * scale;
    (*pos).z -= periodicBoxVecZ.z * scale;

    // Wrap Y: compute lambda_y = pos · recipBoxVec[1]
    scale = floor((*pos).x*recipBoxVecY.x + (*pos).y*recipBoxVecY.y + (*pos).z*recipBoxVecY.z + 0.5f);
    (*pos).x -= periodicBoxVecY.x * scale;
    (*pos).y -= periodicBoxVecY.y * scale;
    (*pos).z -= periodicBoxVecY.z * scale;

    // Wrap X: compute lambda_x = pos · recipBoxVec[0]
    scale = floor((*pos).x*recipBoxVecX.x + (*pos).y*recipBoxVecX.y + (*pos).z*recipBoxVecX.z + 0.5f);
    (*pos).x -= periodicBoxVecX.x * scale;
    (*pos).y -= periodicBoxVecX.y * scale;
    (*pos).z -= periodicBoxVecX.z * scale;
}

/**
 * Convert fixed dipoles from Cartesian (lab frame) to fractional coordinates.
 */
KERNEL void pmeTransformMultipoles(GLOBAL const real* RESTRICT labDipole,
        GLOBAL real* RESTRICT fracDipole, real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // Build transformation matrix: a[i][j] = gridSize[i] * recipBoxVec[j][i]
    LOCAL real a[3][3];
    if (LOCAL_ID == 0) {
        a[0][0] = GRID_SIZE_X * recipBoxVecX.x;
        a[0][1] = GRID_SIZE_X * recipBoxVecY.x;
        a[0][2] = GRID_SIZE_X * recipBoxVecZ.x;
        a[1][0] = GRID_SIZE_Y * recipBoxVecX.y;
        a[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
        a[1][2] = GRID_SIZE_Y * recipBoxVecZ.y;
        a[2][0] = GRID_SIZE_Z * recipBoxVecX.z;
        a[2][1] = GRID_SIZE_Z * recipBoxVecY.z;
        a[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
    }
    SYNC_THREADS;

    // Transform dipoles
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        for (int j = 0; j < 3; j++) {
            real dipole = 0;
            for (int k = 0; k < 3; k++)
                dipole += a[j][k] * labDipole[3*i+k];
            fracDipole[3*i+j] = dipole;
        }
    }
}

/**
 * Convert potential from fractional to Cartesian coordinates.
 * fphi layout: [phi, dphi/du, dphi/dv, dphi/dw, d2phi/du2, d2phi/dv2, d2phi/dw2, d2phi/dudv, d2phi/dudw, d2phi/dvdw]
 *              indices [0-9] are strided by NUM_ATOMS
 * cphi layout: [phi, dphi/dx, dphi/dy, dphi/dz, d2phi/dxx, d2phi/dyy, d2phi/dzz, d2phi/dxy, d2phi/dxz, d2phi/dyz]
 *              packed as 10 consecutive values per atom
 */
KERNEL void pmeTransformPotential(GLOBAL const real* RESTRICT fphi, GLOBAL real* RESTRICT cphi,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    // Build transformation matrix: a[i][j] = gridDim[j] * recipBox[i][j]
    // This transforms fractional phi gradients to Cartesian
    LOCAL real a[3][3];
    if (LOCAL_ID == 0) {
        a[0][0] = GRID_SIZE_X * recipBoxVecX.x;
        a[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
        a[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
        a[1][0] = GRID_SIZE_X * recipBoxVecY.x;
        a[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
        a[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
        a[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
        a[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
        a[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
    }
    // Build 6x6 matrix for second derivatives
    int index1[] = {0, 1, 2, 0, 0, 1};
    int index2[] = {0, 1, 2, 1, 2, 2};
    LOCAL real b[6][6];
    if (LOCAL_ID < 36) {
        int i = LOCAL_ID / 6;
        int j = LOCAL_ID - 6*i;
        b[i][j] = a[index1[i]][index1[j]] * a[index2[i]][index2[j]];
        if (index1[j] != index2[j])
            b[i][j] += (i < 3 ? b[i][j] : a[index1[i]][index2[j]] * a[index2[i]][index1[j]]);
    }
    SYNC_THREADS;

    // Transform potential
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        // Scalar potential (unchanged)
        cphi[10*i] = fphi[i];
        // First derivatives: gradient transform
        cphi[10*i+1] = a[0][0]*fphi[i+PADDED_NUM_ATOMS*1] + a[0][1]*fphi[i+PADDED_NUM_ATOMS*2] + a[0][2]*fphi[i+PADDED_NUM_ATOMS*3];
        cphi[10*i+2] = a[1][0]*fphi[i+PADDED_NUM_ATOMS*1] + a[1][1]*fphi[i+PADDED_NUM_ATOMS*2] + a[1][2]*fphi[i+PADDED_NUM_ATOMS*3];
        cphi[10*i+3] = a[2][0]*fphi[i+PADDED_NUM_ATOMS*1] + a[2][1]*fphi[i+PADDED_NUM_ATOMS*2] + a[2][2]*fphi[i+PADDED_NUM_ATOMS*3];
        // Second derivatives: Hessian transform
        for (int j = 0; j < 6; j++) {
            cphi[10*i+4+j] = 0;
            for (int k = 0; k < 6; k++)
                cphi[10*i+4+j] += b[j][k] * fphi[i+PADDED_NUM_ATOMS*(4+k)];
        }
    }
}

/**
 * Spread fixed charges and dipoles onto the PME grid.
 */
KERNEL void pmeSpreadFixedMultipoles(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT fracDipole,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real2* RESTRICT pmeGrid,
#endif
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        // Note: Don't wrap positions here - the fractional coordinate formula handles wrapping implicitly

        real atomCharge = pos.w;
        real atomDipoleX = fracDipole[m*3];
        real atomDipoleY = fracDipole[m*3+1];
        real atomDipoleZ = fracDipole[m*3+2];

        // Compute B-spline coefficients for each dimension
        // w = pos · recipBoxVec[d] is the fractional coordinate
        // (w - floor(w + 0.5) + 0.5) wraps to [0, 1)
        real w = pos.x*recipBoxVecX.x + pos.y*recipBoxVecY.x + pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w - floor(w + 0.5f) + 0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta1, w, array);

        w = pos.x*recipBoxVecX.y + pos.y*recipBoxVecY.y + pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta2, w, array);

        w = pos.x*recipBoxVecX.z + pos.y*recipBoxVecY.z + pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta3, w, array);

        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Spread onto grid
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xindex = igrid1 + ix;
            xindex -= (xindex >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            int xbase = xindex * GRID_SIZE_Y * GRID_SIZE_Z;
            real4 t = theta1[ix];

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int yindex = igrid2 + iy;
                yindex -= (yindex >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                int ybase = xbase + yindex * GRID_SIZE_Z;
                real4 u = theta2[iy];

                // Charge + dipole terms (no quadrupole)
                // term0 = q*t.x*u.x + dy*t.x*u.y + dx*t.y*u.x
                // term1 = dz*t.x*u.x
                real term0 = atomCharge*t.x*u.x + atomDipoleY*t.x*u.y + atomDipoleX*t.y*u.x;
                real term1 = atomDipoleZ*t.x*u.x;

                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3 + iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    size_t index = ybase + zindex;
                    real4 v = theta3[iz];
                    real add = term0*v.x + term1*v.y;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[2*index], (mm_ulong) realToFixedPoint(add));
#else
                    ATOMIC_ADD(&pmeGrid[index].x, add);
#endif
                }
            }
        }
    }
}

/**
 * Spread induced dipoles onto the PME grid.
 */
KERNEL void pmeSpreadInducedDipoles(GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT inducedDipole,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real2* RESTRICT pmeGrid,
#endif
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];

    LOCAL real cartToFrac[3][3];
    if (LOCAL_ID == 0) {
        cartToFrac[0][0] = GRID_SIZE_X * recipBoxVecX.x;
        cartToFrac[0][1] = GRID_SIZE_X * recipBoxVecY.x;
        cartToFrac[0][2] = GRID_SIZE_X * recipBoxVecZ.x;
        cartToFrac[1][0] = GRID_SIZE_Y * recipBoxVecX.y;
        cartToFrac[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
        cartToFrac[1][2] = GRID_SIZE_Y * recipBoxVecZ.y;
        cartToFrac[2][0] = GRID_SIZE_Z * recipBoxVecX.z;
        cartToFrac[2][1] = GRID_SIZE_Z * recipBoxVecY.z;
        cartToFrac[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
    }
    SYNC_THREADS;

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        // Note: Don't wrap positions here - the fractional coordinate formula handles wrapping implicitly

        // Transform induced dipole to fractional coordinates
        real3 cind = make_real3(inducedDipole[3*m], inducedDipole[3*m+1], inducedDipole[3*m+2]);
        real3 find = make_real3(
            cind.x*cartToFrac[0][0] + cind.y*cartToFrac[0][1] + cind.z*cartToFrac[0][2],
            cind.x*cartToFrac[1][0] + cind.y*cartToFrac[1][1] + cind.z*cartToFrac[1][2],
            cind.x*cartToFrac[2][0] + cind.y*cartToFrac[2][1] + cind.z*cartToFrac[2][2]);

        // Compute B-splines with full dot product for triclinic boxes
        real w = pos.x*recipBoxVecX.x + pos.y*recipBoxVecY.x + pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w - floor(w + 0.5f) + 0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta1, w, array);

        w = pos.x*recipBoxVecX.y + pos.y*recipBoxVecY.y + pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta2, w, array);

        w = pos.x*recipBoxVecX.z + pos.y*recipBoxVecY.z + pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta3, w, array);

        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Spread induced dipole onto grid
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xindex = igrid1 + ix;
            xindex -= (xindex >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            int xbase = xindex * GRID_SIZE_Y * GRID_SIZE_Z;
            real4 t = theta1[ix];

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int yindex = igrid2 + iy;
                yindex -= (yindex >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                int ybase = xbase + yindex * GRID_SIZE_Z;
                real4 u = theta2[iy];

                // Dipole only: dy*t.x*u.y + dx*t.y*u.x, dz*t.x*u.x
                real term0 = find.y*t.x*u.y + find.x*t.y*u.x;
                real term1 = find.z*t.x*u.x;

                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int zindex = igrid3 + iz;
                    zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    size_t index = ybase + zindex;
                    real4 v = theta3[iz];
                    real add = term0*v.x + term1*v.y;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                    ATOMIC_ADD(&pmeGrid[2*index], (mm_ulong) realToFixedPoint(add));
#else
                    ATOMIC_ADD(&pmeGrid[index].x, add);
#endif
                }
            }
        }
    }
}

/**
 * Convert fixed-point grid values to floating point.
 */
KERNEL void finishSpreadCharge(GLOBAL const mm_long* RESTRICT pmeGridLong, GLOBAL real* RESTRICT pmeGrid) {
    const unsigned int gridSize = 2*GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE)
        pmeGrid[index] = scale*pmeGridLong[index];
}

/**
 * Perform convolution in reciprocal space (after FFT forward, before FFT inverse).
 * Multiplies grid by B-spline moduli and Ewald factor.
 * Also computes reciprocal space energy: E = 0.5 * ELECTRIC * sum(eterm * |grid|^2)
 */
KERNEL void pmeReciprocalConvolution(GLOBAL real2* RESTRICT pmeGrid,
        GLOBAL const real* RESTRICT pmeBsplineModuliX,
        GLOBAL const real* RESTRICT pmeBsplineModuliY,
        GLOBAL const real* RESTRICT pmeBsplineModuliZ,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ,
        GLOBAL mm_ulong* RESTRICT energyBuffer) {
    const unsigned int gridSize = GRID_SIZE_X * GRID_SIZE_Y * GRID_SIZE_Z;
    real expFactor = (real)(M_PI * M_PI) / (EWALD_ALPHA * EWALD_ALPHA);
    // Compute volume as scalar triple product: a · (b × c)
    real volume = periodicBoxVecX.x * (periodicBoxVecY.y*periodicBoxVecZ.z - periodicBoxVecY.z*periodicBoxVecZ.y)
                - periodicBoxVecX.y * (periodicBoxVecY.x*periodicBoxVecZ.z - periodicBoxVecY.z*periodicBoxVecZ.x)
                + periodicBoxVecX.z * (periodicBoxVecY.x*periodicBoxVecZ.y - periodicBoxVecY.y*periodicBoxVecZ.x);
    real scaleFactor = RECIP((real)M_PI * volume);
    real energy = 0;

    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
        int kx = index / (GRID_SIZE_Y * GRID_SIZE_Z);
        int remainder = index - kx * GRID_SIZE_Y * GRID_SIZE_Z;
        int ky = remainder / GRID_SIZE_Z;
        int kz = remainder - ky * GRID_SIZE_Z;

        // Skip DC component (k=0)
        if (kx == 0 && ky == 0 && kz == 0) {
            pmeGrid[index] = make_real2(0, 0);
            continue;
        }

        // Convert to signed indices
        int mx = (kx < (GRID_SIZE_X+1)/2) ? kx : (kx - GRID_SIZE_X);
        int my = (ky < (GRID_SIZE_Y+1)/2) ? ky : (ky - GRID_SIZE_Y);
        int mz = (kz < (GRID_SIZE_Z+1)/2) ? kz : (kz - GRID_SIZE_Z);

        // Reciprocal vector components
        real mhx = mx * recipBoxVecX.x;
        real mhy = mx * recipBoxVecY.x + my * recipBoxVecY.y;
        real mhz = mx * recipBoxVecZ.x + my * recipBoxVecZ.y + mz * recipBoxVecZ.z;

        // B-spline moduli
        real bx = pmeBsplineModuliX[kx];
        real by = pmeBsplineModuliY[ky];
        real bz = pmeBsplineModuliZ[kz];

        // Convolution kernel
        real m2 = mhx*mhx + mhy*mhy + mhz*mhz;
        real denom = m2 * bx * by * bz;
        real eterm = scaleFactor * EXP(-expFactor * m2) / denom;

        real2 grid = pmeGrid[index];

        // Accumulate energy: E = 0.5 * ELECTRIC * sum(eterm * |grid|^2)
        real gridNormSq = grid.x*grid.x + grid.y*grid.y;
        energy += eterm * gridNormSq;

        pmeGrid[index] = make_real2(grid.x * eterm, grid.y * eterm);
    }

    // Reduce energy within thread and accumulate to global buffer (0.5 factor included in host)
    ATOMIC_ADD(energyBuffer, (mm_ulong)(energy * 0x100000000));
}

/**
 * Compute potential from PME grid at particle positions.
 * Only computes 10 components (phi + 3 gradient + 6 Hessian) for charge+dipole.
 * Output phi is strided: phi[m + N*component]
 */
KERNEL void pmeComputeFixedPotentialFromGrid(GLOBAL const real2* RESTRICT pmeGrid,
        GLOBAL real* RESTRICT phi, GLOBAL mm_ulong* RESTRICT field,
        GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labDipole,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];

    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
    }
    SYNC_THREADS;

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        // Note: Don't wrap positions here - the fractional coordinate formula handles wrapping implicitly

        // Compute B-splines with full dot product for triclinic boxes
        real w = pos.x*recipBoxVecX.x + pos.y*recipBoxVecY.x + pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w - floor(w + 0.5f) + 0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta1, w, array);

        w = pos.x*recipBoxVecX.y + pos.y*recipBoxVecY.y + pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta2, w, array);

        w = pos.x*recipBoxVecX.z + pos.y*recipBoxVecY.z + pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta3, w, array);

        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Interpolate potential from grid (10 components)
        real tuv000 = 0, tuv100 = 0, tuv010 = 0, tuv001 = 0;
        real tuv200 = 0, tuv020 = 0, tuv002 = 0;
        real tuv110 = 0, tuv101 = 0, tuv011 = 0;

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1 + ix;
            i -= (i >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 v = theta1[ix];

            real tu00 = 0, tu10 = 0, tu01 = 0;
            real tu20 = 0, tu11 = 0, tu02 = 0;

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2 + iy;
                j -= (j >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];

                real t0 = 0, t1 = 0, t2 = 0;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3 + iz;
                    k -= (k >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int gridIndex = i*GRID_SIZE_Y*GRID_SIZE_Z + j*GRID_SIZE_Z + k;
                    real tq = pmeGrid[gridIndex].x;
                    real4 tadd = theta3[iz];
                    t0 += tq * tadd.x;  // value
                    t1 += tq * tadd.y;  // 1st derivative
                    t2 += tq * tadd.z;  // 2nd derivative
                }
                tu00 += u.x * t0;
                tu10 += u.y * t0;
                tu01 += u.x * t1;
                tu20 += u.z * t0;
                tu11 += u.y * t1;
                tu02 += u.x * t2;
            }
            tuv000 += v.x * tu00;
            tuv100 += v.y * tu00;
            tuv010 += v.x * tu10;
            tuv001 += v.x * tu01;
            tuv200 += v.z * tu00;
            tuv020 += v.x * tu20;
            tuv002 += v.x * tu02;
            tuv110 += v.y * tu10;
            tuv101 += v.y * tu01;
            tuv011 += v.x * tu11;
        }

        // Store fractional potential (10 components, strided by PADDED_NUM_ATOMS)
        phi[m] = tuv000;
        phi[m + PADDED_NUM_ATOMS] = tuv100;
        phi[m + PADDED_NUM_ATOMS*2] = tuv010;
        phi[m + PADDED_NUM_ATOMS*3] = tuv001;
        phi[m + PADDED_NUM_ATOMS*4] = tuv200;
        phi[m + PADDED_NUM_ATOMS*5] = tuv020;
        phi[m + PADDED_NUM_ATOMS*6] = tuv002;
        phi[m + PADDED_NUM_ATOMS*7] = tuv110;
        phi[m + PADDED_NUM_ATOMS*8] = tuv101;
        phi[m + PADDED_NUM_ATOMS*9] = tuv011;

        // Add reciprocal field to direct-space field
        // E_recip = -grad(phi) in Cartesian coordinates
        // Also add dipole self-field term: (4/3)*alpha^3/sqrt(pi) * mu
        real dipoleScale = (4.0f/3.0f) * EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA / SQRT_PI;
        real fieldx = dipoleScale*labDipole[m*3] - tuv100*fracToCart[0][0] - tuv010*fracToCart[0][1] - tuv001*fracToCart[0][2];
        real fieldy = dipoleScale*labDipole[m*3+1] - tuv100*fracToCart[1][0] - tuv010*fracToCart[1][1] - tuv001*fracToCart[1][2];
        real fieldz = dipoleScale*labDipole[m*3+2] - tuv100*fracToCart[2][0] - tuv010*fracToCart[2][1] - tuv001*fracToCart[2][2];
        ATOMIC_ADD(&field[m], (mm_ulong) realToFixedPoint(fieldx));
        ATOMIC_ADD(&field[m + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fieldy));
        ATOMIC_ADD(&field[m + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fieldz));
    }
}

/**
 * Compute induced dipole potential from PME grid.
 */
KERNEL void pmeComputeInducedPotentialFromGrid(GLOBAL const real2* RESTRICT pmeGrid,
        GLOBAL real* RESTRICT phid, GLOBAL mm_ulong* RESTRICT inducedField,
        GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT inducedDipole,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {
    real array[PME_ORDER*PME_ORDER];
    real4 theta1[PME_ORDER];
    real4 theta2[PME_ORDER];
    real4 theta3[PME_ORDER];

    LOCAL real fracToCart[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;
    }
    SYNC_THREADS;

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real4 pos = posq[m];
        // Note: Don't wrap positions here - the fractional coordinate formula handles wrapping implicitly

        // Compute B-splines with full dot product for triclinic boxes
        real w = pos.x*recipBoxVecX.x + pos.y*recipBoxVecY.x + pos.z*recipBoxVecZ.x;
        real fr = GRID_SIZE_X*(w - floor(w + 0.5f) + 0.5f);
        int ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid1 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta1, w, array);

        w = pos.x*recipBoxVecX.y + pos.y*recipBoxVecY.y + pos.z*recipBoxVecZ.y;
        fr = GRID_SIZE_Y*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid2 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta2, w, array);

        w = pos.x*recipBoxVecX.z + pos.y*recipBoxVecY.z + pos.z*recipBoxVecZ.z;
        fr = GRID_SIZE_Z*(w - floor(w + 0.5f) + 0.5f);
        ifr = (int) floor(fr);
        w = fr - ifr;
        int igrid3 = ifr - PME_ORDER + 1;
        computeBSplinePoint(theta3, w, array);

        igrid1 += (igrid1 < 0 ? GRID_SIZE_X : 0);
        igrid2 += (igrid2 < 0 ? GRID_SIZE_Y : 0);
        igrid3 += (igrid3 < 0 ? GRID_SIZE_Z : 0);

        // Interpolate potential from grid (all 10 components for force calculation)
        real tuv000 = 0, tuv100 = 0, tuv010 = 0, tuv001 = 0;
        real tuv200 = 0, tuv020 = 0, tuv002 = 0;
        real tuv110 = 0, tuv101 = 0, tuv011 = 0;

        for (int ix = 0; ix < PME_ORDER; ix++) {
            int i = igrid1 + ix;
            i -= (i >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            real4 v = theta1[ix];

            real tu00 = 0, tu10 = 0, tu01 = 0;
            real tu20 = 0, tu11 = 0, tu02 = 0;

            for (int iy = 0; iy < PME_ORDER; iy++) {
                int j = igrid2 + iy;
                j -= (j >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                real4 u = theta2[iy];

                real t0 = 0, t1 = 0, t2 = 0;
                for (int iz = 0; iz < PME_ORDER; iz++) {
                    int k = igrid3 + iz;
                    k -= (k >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
                    int gridIndex = i*GRID_SIZE_Y*GRID_SIZE_Z + j*GRID_SIZE_Z + k;
                    real tq = pmeGrid[gridIndex].x;
                    real4 tadd = theta3[iz];
                    t0 += tq * tadd.x;  // value
                    t1 += tq * tadd.y;  // 1st derivative
                    t2 += tq * tadd.z;  // 2nd derivative
                }
                tu00 += u.x * t0;
                tu10 += u.y * t0;
                tu01 += u.x * t1;
                tu20 += u.z * t0;
                tu11 += u.y * t1;
                tu02 += u.x * t2;
            }
            tuv000 += v.x * tu00;
            tuv100 += v.y * tu00;
            tuv010 += v.x * tu10;
            tuv001 += v.x * tu01;
            tuv200 += v.z * tu00;
            tuv020 += v.x * tu20;
            tuv002 += v.x * tu02;
            tuv110 += v.y * tu10;
            tuv101 += v.y * tu01;
            tuv011 += v.x * tu11;
        }

        // Store fractional potential (10 components, strided by PADDED_NUM_ATOMS)
        phid[m] = tuv000;
        phid[m + PADDED_NUM_ATOMS] = tuv100;
        phid[m + PADDED_NUM_ATOMS*2] = tuv010;
        phid[m + PADDED_NUM_ATOMS*3] = tuv001;
        phid[m + PADDED_NUM_ATOMS*4] = tuv200;
        phid[m + PADDED_NUM_ATOMS*5] = tuv020;
        phid[m + PADDED_NUM_ATOMS*6] = tuv002;
        phid[m + PADDED_NUM_ATOMS*7] = tuv110;
        phid[m + PADDED_NUM_ATOMS*8] = tuv101;
        phid[m + PADDED_NUM_ATOMS*9] = tuv011;

        // Add reciprocal induced field to direct-space induced field
        // Also add induced dipole self-field term: (4/3)*alpha^3/sqrt(pi) * mu_ind
        real dipoleScale = (4.0f/3.0f) * EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA / SQRT_PI;
        real3 cind = make_real3(inducedDipole[3*m], inducedDipole[3*m+1], inducedDipole[3*m+2]);
        real fieldx = dipoleScale*cind.x - tuv100*fracToCart[0][0] - tuv010*fracToCart[0][1] - tuv001*fracToCart[0][2];
        real fieldy = dipoleScale*cind.y - tuv100*fracToCart[1][0] - tuv010*fracToCart[1][1] - tuv001*fracToCart[1][2];
        real fieldz = dipoleScale*cind.z - tuv100*fracToCart[2][0] - tuv010*fracToCart[2][1] - tuv001*fracToCart[2][2];

        ATOMIC_ADD(&inducedField[m], (mm_ulong) realToFixedPoint(fieldx));
        ATOMIC_ADD(&inducedField[m + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fieldy));
        ATOMIC_ADD(&inducedField[m + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(fieldz));
    }
}

/**
 * Compute reciprocal space forces and torques from PME potential.
 * This kernel adds forces and torques from the reciprocal potential to the
 * force and torque arrays.
 */
KERNEL void pmeRecipForceAndTorque(GLOBAL const real* RESTRICT phi,
        GLOBAL const real* RESTRICT phid,
        GLOBAL const real4* RESTRICT posq, GLOBAL const real* RESTRICT labDipole,
        GLOBAL const real* RESTRICT fracDipole, GLOBAL const real* RESTRICT inducedDipoleCart,
        GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL mm_ulong* RESTRICT torqueBuffers,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ) {

    LOCAL real fracToCart[3][3];
    LOCAL real cartToFrac[3][3];
    if (LOCAL_ID == 0) {
        fracToCart[0][0] = GRID_SIZE_X * recipBoxVecX.x;
        fracToCart[1][0] = GRID_SIZE_X * recipBoxVecY.x;
        fracToCart[2][0] = GRID_SIZE_X * recipBoxVecZ.x;
        fracToCart[0][1] = GRID_SIZE_Y * recipBoxVecX.y;
        fracToCart[1][1] = GRID_SIZE_Y * recipBoxVecY.y;
        fracToCart[2][1] = GRID_SIZE_Y * recipBoxVecZ.y;
        fracToCart[0][2] = GRID_SIZE_Z * recipBoxVecX.z;
        fracToCart[1][2] = GRID_SIZE_Z * recipBoxVecY.z;
        fracToCart[2][2] = GRID_SIZE_Z * recipBoxVecZ.z;

        // cartToFrac is transpose of fracToCart
        cartToFrac[0][0] = fracToCart[0][0];
        cartToFrac[0][1] = fracToCart[1][0];
        cartToFrac[0][2] = fracToCart[2][0];
        cartToFrac[1][0] = fracToCart[0][1];
        cartToFrac[1][1] = fracToCart[1][1];
        cartToFrac[1][2] = fracToCart[2][1];
        cartToFrac[2][0] = fracToCart[0][2];
        cartToFrac[2][1] = fracToCart[1][2];
        cartToFrac[2][2] = fracToCart[2][2];
    }
    SYNC_THREADS;

    for (int m = GLOBAL_ID; m < NUM_ATOMS; m += GLOBAL_SIZE) {
        real q = posq[m].w;

        // Get lab frame dipole (for torques)
        real3 dipole = make_real3(labDipole[3*m], labDipole[3*m+1], labDipole[3*m+2]);

        // Get fractional dipole (for forces)
        real3 fdip = make_real3(fracDipole[3*m], fracDipole[3*m+1], fracDipole[3*m+2]);

        // Read fractional phi components (strided by PADDED_NUM_ATOMS)
        // phi[0]=potential, phi[1-3]=gradient, phi[4-6]=diagonal hessian, phi[7-9]=off-diagonal
        real phi0 = phi[m];                           // potential
        real phi1 = phi[m + PADDED_NUM_ATOMS];        // d/dx
        real phi2 = phi[m + PADDED_NUM_ATOMS*2];      // d/dy
        real phi3 = phi[m + PADDED_NUM_ATOMS*3];      // d/dz
        real phi4 = phi[m + PADDED_NUM_ATOMS*4];      // d²/dx²
        real phi5 = phi[m + PADDED_NUM_ATOMS*5];      // d²/dy²
        real phi6 = phi[m + PADDED_NUM_ATOMS*6];      // d²/dz²
        real phi7 = phi[m + PADDED_NUM_ATOMS*7];      // d²/dxdy
        real phi8 = phi[m + PADDED_NUM_ATOMS*8];      // d²/dxdz
        real phi9 = phi[m + PADDED_NUM_ATOMS*9];      // d²/dydz

        // Transform gradient from fractional to Cartesian for torques
        // cphi[1-3] = fracToCart * fphi[1-3]
        real cphi1 = fracToCart[0][0]*phi1 + fracToCart[0][1]*phi2 + fracToCart[0][2]*phi3;
        real cphi2 = fracToCart[1][0]*phi1 + fracToCart[1][1]*phi2 + fracToCart[1][2]*phi3;
        real cphi3 = fracToCart[2][0]*phi1 + fracToCart[2][1]*phi2 + fracToCart[2][2]*phi3;

        // Torque = dipole × E = dipole × (-grad(phi))
        // torque[0] = dy*(-cphi3) - dz*(-cphi2) = dz*cphi2 - dy*cphi3
        // torque[1] = dz*(-cphi1) - dx*(-cphi3) = dx*cphi3 - dz*cphi1
        // torque[2] = dx*(-cphi2) - dy*(-cphi1) = dy*cphi1 - dx*cphi2
        real torquex = ENERGY_SCALE_FACTOR * (dipole.z*cphi2 - dipole.y*cphi3);
        real torquey = ENERGY_SCALE_FACTOR * (dipole.x*cphi3 - dipole.z*cphi1);
        real torquez = ENERGY_SCALE_FACTOR * (dipole.y*cphi1 - dipole.x*cphi2);

        // Get Cartesian induced dipole and transform to fractional
        real3 uCart = make_real3(inducedDipoleCart[3*m], inducedDipoleCart[3*m+1], inducedDipoleCart[3*m+2]);
        real3 uFrac;
        uFrac.x = cartToFrac[0][0]*uCart.x + cartToFrac[0][1]*uCart.y + cartToFrac[0][2]*uCart.z;
        uFrac.y = cartToFrac[1][0]*uCart.x + cartToFrac[1][1]*uCart.y + cartToFrac[1][2]*uCart.z;
        uFrac.z = cartToFrac[2][0]*uCart.x + cartToFrac[2][1]*uCart.y + cartToFrac[2][2]*uCart.z;

        // Force gradient in fractional coordinates for fixed multipoles
        // Using deriv indices: deriv1[]={1,4,7,8}, deriv2[]={2,7,5,9}, deriv3[]={3,8,9,6}
        // f[0] = q*phi1 + fdip.x*phi4 + fdip.y*phi7 + fdip.z*phi8
        // f[1] = q*phi2 + fdip.x*phi7 + fdip.y*phi5 + fdip.z*phi9
        // f[2] = q*phi3 + fdip.x*phi8 + fdip.y*phi9 + fdip.z*phi6
        real ff0 = q*phi1 + fdip.x*phi4 + fdip.y*phi7 + fdip.z*phi8;
        real ff1 = q*phi2 + fdip.x*phi7 + fdip.y*phi5 + fdip.z*phi9;
        real ff2 = q*phi3 + fdip.x*phi8 + fdip.y*phi9 + fdip.z*phi6;

        // Add force on induced dipole from fixed potential
        // Reference uses 2*induced*phi * 0.5*_electric = induced*phi*_electric
        // So we use factor of 1.0 here (Reference's 2*0.5 cancels)
        ff0 += uFrac.x*phi4 + uFrac.y*phi7 + uFrac.z*phi8;
        ff1 += uFrac.x*phi7 + uFrac.y*phi5 + uFrac.z*phi9;
        ff2 += uFrac.x*phi8 + uFrac.y*phi9 + uFrac.z*phi6;

        // Read fractional phid components (induced potential from spreading induced dipoles)
        real phid1 = phid[m + PADDED_NUM_ATOMS];        // d/dx
        real phid2 = phid[m + PADDED_NUM_ATOMS*2];      // d/dy
        real phid3 = phid[m + PADDED_NUM_ATOMS*3];      // d/dz
        real phid4 = phid[m + PADDED_NUM_ATOMS*4];      // d²/dx²
        real phid5 = phid[m + PADDED_NUM_ATOMS*5];      // d²/dy²
        real phid6 = phid[m + PADDED_NUM_ATOMS*6];      // d²/dz²
        real phid7 = phid[m + PADDED_NUM_ATOMS*7];      // d²/dxdy
        real phid8 = phid[m + PADDED_NUM_ATOMS*8];      // d²/dxdz
        real phid9 = phid[m + PADDED_NUM_ATOMS*9];      // d²/dydz

        // Add force on permanent multipoles from induced potential
        // Reference uses 2*multipole*phid * 0.5*_electric = multipole*phid*_electric
        // So we use factor of 1.0 here (Reference's 2*0.5 cancels)
        ff0 += q*phid1 + fdip.x*phid4 + fdip.y*phid7 + fdip.z*phid8;
        ff1 += q*phid2 + fdip.x*phid7 + fdip.y*phid5 + fdip.z*phid9;
        ff2 += q*phid3 + fdip.x*phid8 + fdip.y*phid9 + fdip.z*phid6;

#ifdef MUTUAL_POLARIZATION
        // Add I-I reciprocal force: induced dipole force from induced potential
        // This force is needed for consistency even though I-I energy cancels with polarization cost
        ff0 += uFrac.x*phid4 + uFrac.y*phid7 + uFrac.z*phid8;
        ff1 += uFrac.x*phid7 + uFrac.y*phid5 + uFrac.z*phid9;
        ff2 += uFrac.x*phid8 + uFrac.y*phid9 + uFrac.z*phid6;
#endif

        // Transform phid gradient from fractional to Cartesian for torques
        real cphid1 = fracToCart[0][0]*phid1 + fracToCart[0][1]*phid2 + fracToCart[0][2]*phid3;
        real cphid2 = fracToCart[1][0]*phid1 + fracToCart[1][1]*phid2 + fracToCart[1][2]*phid3;
        real cphid3 = fracToCart[2][0]*phid1 + fracToCart[2][1]*phid2 + fracToCart[2][2]*phid3;

        // Add torque on permanent dipoles from induced potential: τ = dipole × (-grad(phid))
        torquex += ENERGY_SCALE_FACTOR * (dipole.z*cphid2 - dipole.y*cphid3);
        torquey += ENERGY_SCALE_FACTOR * (dipole.x*cphid3 - dipole.z*cphid1);
        torquez += ENERGY_SCALE_FACTOR * (dipole.y*cphid1 - dipole.x*cphid2);

        // Transform force from fractional to Cartesian
        // recipForce = fracToCart * f_frac
        real forcex = ENERGY_SCALE_FACTOR * (fracToCart[0][0]*ff0 + fracToCart[0][1]*ff1 + fracToCart[0][2]*ff2);
        real forcey = ENERGY_SCALE_FACTOR * (fracToCart[1][0]*ff0 + fracToCart[1][1]*ff1 + fracToCart[1][2]*ff2);
        real forcez = ENERGY_SCALE_FACTOR * (fracToCart[2][0]*ff0 + fracToCart[2][1]*ff1 + fracToCart[2][2]*ff2);

        // Subtract force (forces[i] -= recipForce in reference)
        ATOMIC_ADD(&forceBuffers[m], (mm_ulong) realToFixedPoint(-forcex));
        ATOMIC_ADD(&forceBuffers[m + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forcey));
        ATOMIC_ADD(&forceBuffers[m + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(-forcez));

        // PME self-torque: torque from self-interaction between permanent and induced dipoles
        // term = (2/3) * _electric * alpha^3 / sqrt(pi)
        // torque = μ_perm × (2 * μ_ind) * term
        real selfTerm = ENERGY_SCALE_FACTOR * (2.0f/3.0f) * EWALD_ALPHA * EWALD_ALPHA * EWALD_ALPHA / SQRT_PI;
        real3 ui2 = 2.0f * uCart;  // 2 * induced dipole
        // Self-torque = dipole × (2*ui) * term
        torquex += selfTerm * (dipole.y * ui2.z - dipole.z * ui2.y);
        torquey += selfTerm * (dipole.z * ui2.x - dipole.x * ui2.z);
        torquez += selfTerm * (dipole.x * ui2.y - dipole.y * ui2.x);

        // Add torque
        ATOMIC_ADD(&torqueBuffers[m], (mm_ulong) realToFixedPoint(torquex));
        ATOMIC_ADD(&torqueBuffers[m + PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(torquey));
        ATOMIC_ADD(&torqueBuffers[m + 2*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(torquez));
    }
}

#endif // USE_EWALD
