#include "kalman_filter.h"
#include "FusionEKF.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_*x_;
	P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd y = z - H_ * x_;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd K = P_ * Ht * S.inverse();

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	float c1 = sqrt(px*px + py*py);

	VectorXd H_x(3);
	H_x << c1, atan2(py,px), (px*vx+py*vy)/c1;
	H_x = H_x.transpose();

	float phi = z(1);

	while (phi > M_PI) {
		phi -= 2*M_PI;
	}
	while (phi < -M_PI) {
		phi += 2*M_PI;
	}

	VectorXd z_new(3);
	z_new << z(0), phi, z(2);

	VectorXd y = z_new - H_x;
	MatrixXd H_t = H_.transpose();
	MatrixXd S = H_ * P_ * H_t + R_;
	MatrixXd K = P_ * H_t * S.inverse();

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
