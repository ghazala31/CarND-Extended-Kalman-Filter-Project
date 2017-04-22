#include "kalman_filter.h"
#include "FusionEKF.h"

#define EPSILON 0.0001

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
	P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
	float px = x_(0);
	float py = x_(1);
	float vx = x_(2);
	float vy = x_(3);

	float rho = sqrt(px*px + py*py);
	float phi = (py == 0 && px == 0) ? 0 : atan2(py,px);
	float rho_dot = rho < EPSILON ? (px*vx+py*vy)/EPSILON : (px*vx+py*vy)/rho;

	VectorXd H_x(3);
	H_x << rho, phi, rho_dot;
	H_x = H_x.transpose();

	VectorXd y = z - H_x;
	phi = y(1);

	while (phi > M_PI) {
		phi -= 2*M_PI;
	}
	while (phi < -M_PI) {
		phi += 2*M_PI;
	}

	y(1) = phi;

	MatrixXd H_t = H_.transpose();
	MatrixXd S = H_ * P_ * H_t + R_;
	MatrixXd K = P_ * H_t * S.inverse();

	//new estimate
	x_ = x_ + (K * y);
	P_ = (I_ - K * H_) * P_;
}
