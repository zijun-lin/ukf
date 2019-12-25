
#ifndef UKF_UKF_H
#define UKF_UKF_H

#include <Eigen/Dense>

class UKF {
public:
    UKF();   // Constructor
    virtual ~UKF();  // Destructor

    void Init();  // Init Initializes Unscented Kalman filter

    //  Student assignment functions
    void GenerateSigmaPoints(Eigen::MatrixXd  *Xsig_out);
    void AugmentedSigmaPoints(Eigen::MatrixXd *Xsig_out);
    void SigmaPointPrediction(Eigen::MatrixXd *Xsig_out);

    void PredictMeanAndCovariance(Eigen::VectorXd *x_pred,
                                  Eigen::MatrixXd *P_pred);
    void PredictRadarMeasurement(Eigen::VectorXd *z_out,
                                 Eigen::MatrixXd *S_out);
    void UpdateState(Eigen::VectorXd *x_out,
                     Eigen::MatrixXd *P_out);
};


#endif //UKF_UKF_H
