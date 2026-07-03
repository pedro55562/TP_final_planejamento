#pragma once

#include <Eigen/Dense>
#include <vector>

namespace se3 {

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Bounds3d = Eigen::Matrix<double, 3, 2>;

Eigen::Matrix3d skew(const Eigen::Vector3d& w);
Eigen::Vector3d vee_so3(const Eigen::Matrix3d& W);
Eigen::Matrix3d exp_SO3(const Eigen::Vector3d& phi);
Eigen::Vector3d log_SO3(const Eigen::Matrix3d& R);
Eigen::Matrix3d jac_left_SO3(const Eigen::Vector3d& phi);
Eigen::Matrix3d inv_jac_left_SO3(const Eigen::Vector3d& phi);

Eigen::Matrix4d make_SE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& p);
Eigen::Matrix4d inv_SE3(const Eigen::Matrix4d& H);
Eigen::Matrix4d hat_SE3(const Vector6d& xi);
Vector6d vee_SE3(const Eigen::Matrix4d& A);
Eigen::Matrix4d exp_SE3(const Eigen::Matrix4d& A);
Eigen::Matrix4d log_SE3(const Eigen::Matrix4d& H);

Matrix6d make_metric_matrix(double ell);
double metric_log_SE3_left(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Matrix6d& G);
double metric_log_SE3_right(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Matrix6d& G);
double metric_log_SE3_symmetric(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Matrix6d& G);
Eigen::Vector3d transform_point(const Eigen::Matrix4d& H, const Eigen::Vector3d& q);
double metric_object_points(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const std::vector<Eigen::Vector3d>& body_points);

Eigen::Matrix4d interpolate_SE3_left(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    double s);
Eigen::Matrix4d interpolate_SE3_right(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    double s);
std::vector<Eigen::Matrix4d> interpolate_SE3_path(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    int n_points);

Eigen::Matrix4d steer_SE3(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    double step_size,
    const Matrix6d& G);

Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q);
Eigen::Matrix3d sample_SO3_uniform();
Eigen::Matrix4d sample_SE3_uniform_box(const Bounds3d& position_bounds);

}  // namespace se3
