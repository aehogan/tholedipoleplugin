#include "ReferenceTests.h"
#include "TholeDipoleTestCommon.h"

void testPythonPMEDipoles() {

    std::string testName = "testPythonPMEDipoles";

    // Parameters from Python PME Implementation
    double boxDimension = 3.0; // boxlength
    int gridDimension = 25;    // PME grid size
    double alpha = 2.5;            // Ewald splitting parameter
    double cutoff = 1.0;          // Direct space cutoff
    int numberOfParticles = 125;

    cout << "Testing Python PME Dipole Implementation (" << numberOfParticles << " atoms)" << endl;

    // 1. Setup System
    System tholeDipoleSystem;
    Vec3 a(boxDimension, 0.0, 0.0);
    Vec3 b(0.0, boxDimension, 0.0);
    Vec3 c(0.0, 0.0, boxDimension);
    tholeDipoleSystem.setDefaultPeriodicBoxVectors(a, b, c);

    // 2. Setup Force
    TholeDipoleForce* tholeDipoleForce = new TholeDipoleForce();
    tholeDipoleForce->setNonbondedMethod(TholeDipoleForce::PME);
    tholeDipoleForce->setPolarizationType(TholeDipoleForce::Direct);
    tholeDipoleForce->setCutoffDistance(cutoff);
    tholeDipoleForce->setMutualInducedTargetEpsilon(1.0e-9);
    tholeDipoleForce->setMutualInducedMaxIterations(500);

    // Set PME Parameters (Alpha, Grid, Grid, Grid)
    tholeDipoleForce->setPMEParameters(alpha, gridDimension, gridDimension, gridDimension);
    tholeDipoleForce->setEwaldErrorTolerance(1.0e-4);

    // 3. Add Particles
    std::vector<Vec3> positions(125);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {5.13147691e-03, 2.19468970e-02, -7.68935399e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[0] = Vec3(3.23576302e-01, 2.74336720e-01, 2.67222174e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.07882482e-02, -1.56821984e-02, 2.29049707e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[1] = Vec3(3.57691704e-01, 3.22179569e-01, 8.97711828e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.37995406e-02, -3.17508270e-02, -3.24548244e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[2] = Vec3(2.92628669e-01, 2.47161348e-01, 1.48776531e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.49431794e-02, 2.24455325e-02, 1.11023511e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[3] = Vec3(3.03786165e-01, 3.03819310e-01, 2.11612812e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.71736769e-02, -2.06285954e-02, 1.30976124e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[4] = Vec3(3.26693206e-01, 2.78755070e-01, 2.68341464e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-6.31490235e-04, -7.41697097e-03, -1.87738777e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[5] = Vec3(2.51052593e-01, 8.92044141e-01, 2.91703532e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.83667588e-04, 1.23952952e-02, -3.84381605e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[6] = Vec3(2.91162157e-01, 9.47206700e-01, 9.53299202e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.49544635e-02, -1.69657357e-03, 4.85559786e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[7] = Vec3(2.78074258e-01, 8.89779145e-01, 1.54395710e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.26340801e-02, 1.03060128e-02, 4.50680065e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[8] = Vec3(3.02338214e-01, 9.13547343e-01, 2.05447544e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.81300766e-02, 3.75456842e-02, 1.04223375e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[9] = Vec3(2.81131660e-01, 8.76494495e-01, 2.69004267e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.74689051e-02, 3.42342438e-02, -4.16805012e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[10] = Vec3(3.20317654e-01, 1.51031239e+00, 3.14988420e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {7.24569575e-03, -4.04287483e-02, 3.85326826e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[11] = Vec3(3.31641941e-01, 1.46923996e+00, 8.63306755e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {9.44318794e-03, 5.67851924e-03, -3.41040356e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[12] = Vec3(3.15269877e-01, 1.52680996e+00, 1.44193550e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.91970296e-02, 5.43832497e-03, -1.11049426e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[13] = Vec3(2.58368462e-01, 1.52346354e+00, 2.07825197e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.56408536e-02, -1.95231927e-02, -1.01814318e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[14] = Vec3(3.51015899e-01, 1.54100040e+00, 2.68288771e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.62547814e-02, 9.31769166e-03, 1.91701799e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[15] = Vec3(3.24595060e-01, 2.15944302e+00, 2.82709784e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.56543986e-02, 1.31281542e-03, 1.66624550e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[16] = Vec3(2.58135294e-01, 2.08786516e+00, 8.68902708e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.61564337e-02, 3.46506225e-02, 5.32573448e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[17] = Vec3(2.52709018e-01, 2.05570739e+00, 1.47863767e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.45735324e-02, -3.28918171e-02, 3.29112635e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[18] = Vec3(3.42534299e-01, 2.08618054e+00, 2.07801455e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.15330594e-03, -4.97311935e-02, 4.88345419e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[19] = Vec3(2.80640502e-01, 2.10628441e+00, 2.70942618e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.00101531e-03, 4.01911373e-02, 4.83630885e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[20] = Vec3(3.48640989e-01, 2.66491630e+00, 2.75098730e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.05629946e-02, 2.31073036e-02, -3.38930986e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[21] = Vec3(2.70905048e-01, 2.70772309e+00, 9.36836242e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.20634210e-02, -7.16527253e-03, -2.95457140e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[22] = Vec3(3.12083828e-01, 2.74390373e+00, 1.55802259e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.03139225e-02, 4.27584240e-02, 6.90037314e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[23] = Vec3(2.94076379e-01, 2.70573163e+00, 2.05119921e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.51420967e-02, 2.08697395e-02, 3.39243348e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[24] = Vec3(2.94889440e-01, 2.73042312e+00, 2.72902346e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.93530247e-02, 1.65261465e-02, -3.88607828e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[25] = Vec3(8.59912546e-01, 3.33719753e-01, 2.74384394e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-5.96721233e-03, -6.17856156e-03, 2.65096095e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[26] = Vec3(9.19784694e-01, 3.46542815e-01, 9.23557352e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.14843703e-02, -1.62933617e-02, 4.27576580e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[27] = Vec3(9.07877040e-01, 2.50188500e-01, 1.50992053e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.20851039e-02, 3.59389076e-02, 3.21504113e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[28] = Vec3(9.30086040e-01, 3.08887659e-01, 2.13019728e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.61584427e-02, -1.00621290e-02, -7.56931389e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[29] = Vec3(9.49184599e-01, 2.55435744e-01, 2.64981361e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.11644348e-02, -3.20124259e-03, 3.07938209e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[30] = Vec3(9.07466205e-01, 8.54669226e-01, 2.64167940e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {8.21754591e-03, -2.93904273e-02, 2.17757562e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[31] = Vec3(8.40891165e-01, 9.06191127e-01, 9.51831858e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.35900359e-02, -4.67802065e-02, 2.44780655e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[32] = Vec3(8.85478302e-01, 9.20206074e-01, 1.44351837e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.33225557e-02, 1.53364871e-02, 4.96086327e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[33] = Vec3(8.96749560e-01, 8.54610523e-01, 2.10511631e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.99834075e-02, 1.61167867e-02, -4.50902869e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[34] = Vec3(9.32327680e-01, 9.08852894e-01, 2.65231623e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.88187174e-02, -8.84307768e-03, -1.89737245e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[35] = Vec3(9.35075916e-01, 1.50224599e+00, 2.91104123e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.13096251e-02, -8.27089391e-03, 4.89034507e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[36] = Vec3(8.61795461e-01, 1.47855827e+00, 9.41463960e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.08703658e-02, -3.63472751e-03, 2.21633532e-04};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[37] = Vec3(8.68391977e-01, 1.55001988e+00, 1.55020770e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.04470358e-02, -2.61750094e-02, 3.07791086e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[38] = Vec3(8.77640274e-01, 1.44568074e+00, 2.06900228e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.80582199e-02, 3.95048226e-03, 1.26309362e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[39] = Vec3(9.47397395e-01, 1.44518675e+00, 2.67623362e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.24814473e-02, -4.02961841e-02, -3.80912384e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[40] = Vec3(8.40665449e-01, 2.09818913e+00, 3.58599424e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.98846331e-02, -2.91751703e-02, -5.66322982e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[41] = Vec3(9.55560536e-01, 2.08101967e+00, 9.35870728e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.67494307e-02, 1.50750366e-02, 3.65459852e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[42] = Vec3(9.25872153e-01, 2.08926237e+00, 1.46292083e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.32551365e-02, 4.93033261e-02, -2.63537604e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[43] = Vec3(8.43029083e-01, 2.07202870e+00, 2.10024853e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.67520214e-02, -1.99389864e-02, 1.34442268e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[44] = Vec3(8.84915062e-01, 2.06568143e+00, 2.65265350e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.34280874e-02, 3.38859817e-03, -3.37984163e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[45] = Vec3(8.73748174e-01, 2.68347321e+00, 2.40713141e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.73803395e-02, 3.87593460e-02, -4.83881370e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[46] = Vec3(9.11691973e-01, 2.67517830e+00, 9.15846059e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.10998694e-02, 4.71046141e-02, 3.71682933e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[47] = Vec3(8.55234964e-01, 2.73325950e+00, 1.44550743e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.72878914e-02, -1.44042332e-02, 4.29763653e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[48] = Vec3(9.25219398e-01, 2.75502117e+00, 2.09157760e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.46054838e-02, -3.76076990e-02, 9.64868983e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[49] = Vec3(8.57853319e-01, 2.75280348e+00, 2.73992594e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.15177723e-02, -2.74501590e-02, 3.75124534e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[50] = Vec3(1.44196710e+00, 3.26542124e-01, 2.40928502e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.74536640e-02, 7.21467680e-03, 1.60951795e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[51] = Vec3(1.48362916e+00, 3.04795192e-01, 9.08172386e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.32350662e-02, 8.74937475e-03, 4.48252372e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[52] = Vec3(1.47578945e+00, 2.90235223e-01, 1.49437067e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.91109561e-03, 4.27454999e-02, -3.01634311e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[53] = Vec3(1.50672417e+00, 3.00067371e-01, 2.04042387e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.57153058e-02, -4.73388884e-02, 4.20149230e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[54] = Vec3(1.44625094e+00, 2.88813467e-01, 2.68468758e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.11953312e-02, -1.64456126e-02, -1.50433772e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[55] = Vec3(1.52170836e+00, 9.48507119e-01, 3.12903488e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.57780194e-02, 4.37668357e-02, 4.08011084e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[56] = Vec3(1.48678491e+00, 9.30575650e-01, 8.84314941e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.93884871e-02, -1.63660471e-02, -1.72900107e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[57] = Vec3(1.48185568e+00, 9.16156568e-01, 1.47286107e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.59345225e-02, -7.74566469e-03, -2.54966961e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[58] = Vec3(1.54587313e+00, 9.38676458e-01, 2.12515479e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.07813903e-02, 1.02932197e-02, -1.35812550e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[59] = Vec3(1.45408781e+00, 8.76126403e-01, 2.65743165e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.84494553e-02, -2.21976406e-02, 2.41760422e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[60] = Vec3(1.50774844e+00, 1.46296029e+00, 3.21228703e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.93984703e-02, 4.12132121e-02, 8.07132134e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[61] = Vec3(1.50716855e+00, 1.48018037e+00, 9.05158654e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.99598685e-02, 3.20574220e-02, -3.50651453e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[62] = Vec3(1.46792237e+00, 1.52960372e+00, 1.53333228e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.53697119e-02, 1.57815073e-02, 2.72877831e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[63] = Vec3(1.53357200e+00, 1.46849739e+00, 2.07990963e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.08963873e-02, 1.75035127e-02, -4.93972114e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[64] = Vec3(1.52260492e+00, 1.46451649e+00, 2.69648265e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-8.80951903e-04, -2.29823733e-02, -1.39576281e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[65] = Vec3(1.45048893e+00, 2.08161537e+00, 3.53323865e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.45752507e-02, -4.37294010e-03, -2.20197982e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[66] = Vec3(1.46527832e+00, 2.09054401e+00, 8.66164253e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.56581909e-02, 2.07115060e-02, -1.61109610e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[67] = Vec3(1.55194700e+00, 2.07772216e+00, 1.54916576e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.67246383e-02, 4.47119540e-02, 1.17659977e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[68] = Vec3(1.49330653e+00, 2.04435880e+00, 2.04488198e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.34933557e-02, -1.38182734e-02, 3.63353352e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[69] = Vec3(1.48426498e+00, 2.11343724e+00, 2.66473578e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.15966090e-02, -1.77026057e-02, 4.72098245e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[70] = Vec3(1.50112821e+00, 2.67562818e+00, 3.54030195e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-9.43468015e-03, -2.42651894e-02, -4.17347324e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[71] = Vec3(1.55848213e+00, 2.68903922e+00, 9.18710772e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.15113969e-02, 4.53818403e-02, -3.97120115e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[72] = Vec3(1.47163324e+00, 2.67257758e+00, 1.48783669e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.28008217e-02, 3.68314710e-02, -2.19523019e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[73] = Vec3(1.51502502e+00, 2.69300369e+00, 2.09082217e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.23098210e-02, 2.34875483e-03, -3.90911803e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[74] = Vec3(1.44246914e+00, 2.75017164e+00, 2.74373763e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.61716540e-02, 4.43200558e-02, -2.54869408e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[75] = Vec3(2.05121125e+00, 3.40495933e-01, 2.89231886e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.24551885e-02, -3.26697270e-03, -1.24890852e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[76] = Vec3(2.04157918e+00, 2.42897809e-01, 9.25126283e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.67020103e-02, 2.74580205e-02, -3.65386503e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[77] = Vec3(2.10514325e+00, 3.43070021e-01, 1.51825846e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.04778548e-02, -1.50481473e-02, -2.22576040e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[78] = Vec3(2.05986720e+00, 3.13521874e-01, 2.06865401e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.61300415e-02, 2.60210258e-02, -2.69910043e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[79] = Vec3(2.15987021e+00, 2.44873935e-01, 2.71749870e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.78095315e-02, -4.48099053e-02, -2.05693054e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[80] = Vec3(2.05077982e+00, 9.17813965e-01, 3.27912146e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.68884895e-02, 1.12179362e-02, 4.88214944e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[81] = Vec3(2.09413060e+00, 8.74452395e-01, 9.37261615e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.80597342e-02, 3.82712985e-02, 4.19472466e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[82] = Vec3(2.14830678e+00, 8.66658847e-01, 1.44000983e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.07695929e-02, 3.51548051e-02, -3.72387776e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[83] = Vec3(2.08986043e+00, 9.29353855e-01, 2.06553978e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.94353612e-02, 4.16848785e-02, 1.76234608e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[84] = Vec3(2.14726384e+00, 8.99580957e-01, 2.69113148e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.96619266e-02, -1.60189146e-02, 9.50738763e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[85] = Vec3(2.13648316e+00, 1.54291821e+00, 3.50685883e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.22219516e-03, 1.17186089e-02, -9.52605140e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[86] = Vec3(2.09295890e+00, 1.55194110e+00, 8.87707686e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.77344869e-02, -3.52277156e-02, -2.15780765e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[87] = Vec3(2.15909741e+00, 1.45186215e+00, 1.46647240e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.82622585e-02, 1.16006478e-02, -4.41060521e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[88] = Vec3(2.13350944e+00, 1.50274704e+00, 2.04407444e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {6.36645929e-03, 2.27079951e-02, 1.71126604e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[89] = Vec3(2.11934025e+00, 1.48540432e+00, 2.65628080e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.16803364e-02, -1.40132651e-02, 2.97732595e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[90] = Vec3(2.06970158e+00, 2.10298395e+00, 3.04519613e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.61912095e-02, 6.75741626e-03, -3.24171735e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[91] = Vec3(2.11535062e+00, 2.04459979e+00, 9.05577483e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.17099079e-02, -3.32518359e-02, 3.40764922e-03};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[92] = Vec3(2.10124516e+00, 2.13083350e+00, 1.45321262e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.62607891e-02, 2.60045807e-02, 2.69406387e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[93] = Vec3(2.08628922e+00, 2.06983485e+00, 2.11769190e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.56399031e-02, 2.95604589e-02, -8.02395114e-04};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[94] = Vec3(2.14509255e+00, 2.10248620e+00, 2.64420398e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.65886312e-02, -6.70306692e-03, 3.84003033e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[95] = Vec3(2.09302551e+00, 2.67821217e+00, 2.74145904e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.56312028e-02, 1.97942236e-02, 3.05396935e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[96] = Vec3(2.11777957e+00, 2.74301132e+00, 9.42293945e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.15750411e-02, -4.59092207e-02, 1.61108355e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[97] = Vec3(2.12797535e+00, 2.71262722e+00, 1.52608250e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-6.50142890e-03, -9.72128307e-03, -3.78160472e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[98] = Vec3(2.13511816e+00, 2.66915546e+00, 2.09581776e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.94130588e-03, -4.72457070e-02, -4.68082012e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[99] = Vec3(2.10308538e+00, 2.69354980e+00, 2.71960713e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.76704682e-02, -3.19403326e-03, 1.25906512e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[100] = Vec3(2.72416318e+00, 3.24909734e-01, 3.55192696e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.96115768e-02, 1.66527119e-02, -3.07969856e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[101] = Vec3(2.69486181e+00, 2.66753548e-01, 8.85201240e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.48270050e-02, -2.01420816e-02, 4.41806964e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[102] = Vec3(2.69705613e+00, 3.56092392e-01, 1.44380027e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.50747525e-02, 3.99770829e-03, 4.31702883e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[103] = Vec3(2.74906102e+00, 2.59440101e-01, 2.15773413e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.47385146e-02, -1.73031814e-02, -3.20609825e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[104] = Vec3(2.74567286e+00, 2.86957979e-01, 2.71876118e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.54143969e-02, -3.88621295e-03, 1.84891465e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[105] = Vec3(2.69601719e+00, 8.71593724e-01, 2.82607815e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.03990534e-02, -4.01816001e-02, 4.43180571e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[106] = Vec3(2.68034759e+00, 9.59503329e-01, 9.19052113e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-2.74465116e-02, 3.01276784e-02, 3.75459829e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[107] = Vec3(2.75337334e+00, 9.14559405e-01, 1.44203898e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.83029486e-02, -3.84255464e-02, 4.52602698e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[108] = Vec3(2.69447878e+00, 8.83862474e-01, 2.07290700e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.55551551e-02, 2.64664215e-02, 3.10314845e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[109] = Vec3(2.73703513e+00, 8.59773523e-01, 2.66484601e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {8.94154331e-03, 8.76157553e-03, 4.67361886e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[110] = Vec3(2.65960052e+00, 1.55809539e+00, 2.67336248e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.64657538e-02, -3.93944739e-02, -4.97908099e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[111] = Vec3(2.71892009e+00, 1.51018851e+00, 9.02252709e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-1.31946740e-02, 3.03843316e-02, -1.17629788e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[112] = Vec3(2.75429866e+00, 1.49983892e+00, 1.47940025e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.23795936e-02, -1.88716724e-03, -3.31502898e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[113] = Vec3(2.73242030e+00, 1.49285544e+00, 2.14128930e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-5.64036981e-03, -4.02840394e-02, -2.93216851e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[114] = Vec3(2.67171936e+00, 1.55323377e+00, 2.74860341e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {2.74136070e-02, -2.39733924e-03, 3.70370504e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[115] = Vec3(2.67257902e+00, 2.09810637e+00, 2.80605253e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.47502311e-02, 4.45236635e-02, -2.09913575e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[116] = Vec3(2.75949381e+00, 2.06638031e+00, 9.13400565e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.36061451e-02, 2.33395399e-02, 4.94610387e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[117] = Vec3(2.72724513e+00, 2.04180194e+00, 1.54549709e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.24149980e-02, 1.68072737e-02, -3.27388258e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[118] = Vec3(2.70014277e+00, 2.06512008e+00, 2.11135723e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.84041065e-02, -3.03915953e-02, -4.72659219e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[119] = Vec3(2.74784552e+00, 2.11451896e+00, 2.64522824e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-3.96479075e-02, 1.63042786e-02, 2.10075224e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[120] = Vec3(2.70611439e+00, 2.73759764e+00, 3.43192938e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {-4.30017811e-02, 1.92803611e-03, 1.94314886e-02};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[121] = Vec3(2.67534204e+00, 2.75656368e+00, 8.73442496e-01);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {3.86678166e-02, 2.47325906e-02, -2.90408041e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[122] = Vec3(2.66935917e+00, 2.68062986e+00, 1.50763536e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {1.18761778e-02, 1.32426630e-04, 9.71253398e-03};
        tholeDipoleForce->addParticle(-5.04000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[123] = Vec3(2.67021325e+00, 2.70286568e+00, 2.13227504e+00);

    tholeDipoleSystem.addParticle(1.0);
    {
        std::vector<double> dipole = {4.47067490e-02, 4.15354507e-02, 2.54518340e-02};
        tholeDipoleForce->addParticle(4.96000000e-01, dipole, 0.0, 5, -1, -1, -1);
    }
    positions[124] = Vec3(2.73072720e+00, 2.70444958e+00, 2.74773033e+00);


    tholeDipoleSystem.addForce(tholeDipoleForce);

    // 4. Compare with AMOEBA (Reference Implementation)
    // Create equivalent AMOEBA system
    System amoebaSystem;
    amoebaSystem.setDefaultPeriodicBoxVectors(a, b, c);
    for (int i = 0; i < numberOfParticles; i++)
        amoebaSystem.addParticle(tholeDipoleSystem.getParticleMass(i));

    AmoebaMultipoleForce* amoebaForce = createEquivalentAmoebaForce(tholeDipoleForce);
    amoebaForce->setNonbondedMethod(AmoebaMultipoleForce::PME);
    amoebaForce->setPolarizationType(AmoebaMultipoleForce::Direct);
    amoebaSystem.addForce(amoebaForce);

    compareForces(testName, tholeDipoleSystem, amoebaSystem, positions, 1e-4, 1e-3);
}

int main(int argc, char* argv[]) {
    try {
        setupKernels(argc, argv);
        testPythonPMEDipoles();
    }
    catch (const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        std::cout << "FAIL - ERROR.  Test failed." << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
