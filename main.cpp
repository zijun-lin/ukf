//
// Created by lin on 19-12-4.
//

#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "ukf.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    // Create a UKF instance
    UKF ukf;

    /***
    // GenerateSigmaPoints
    MatrixXd Xsig = MatrixXd(5, 11);
    ukf.GenerateSigmaPoints(&Xsig);

    // AugmentedSigmaPoints
    MatrixXd Xsig_aug = MatrixXd(7, 15);
    ukf.AugmentedSigmaPoints(&Xsig_aug);

    // SigmaPointPrediction
    MatrixXd Xsig_pred = MatrixXd(5, 5);
    ukf.SigmaPointPrediction(&Xsig_pred);


    // PredictMeanAndCovariance
    VectorXd x_pred = VectorXd(5);
    MatrixXd P_pred = MatrixXd(5, 5);
    ukf.PredictMeanAndCovariance(&x_pred, &P_pred);

    // PredictRadarMeasurement
    VectorXd z_out = VectorXd(3);
    MatrixXd S_out = MatrixXd(3, 3);
    ukf.PredictRadarMeasurement(&z_out, &S_out);
    ***/
    // UpdateState
    VectorXd x_out = VectorXd(5);
    MatrixXd P_out = MatrixXd(5, 5);
    ukf.UpdateState(&x_out, &P_out);


    std::cout << "Hello World !" << std::endl;
    return 0;
}