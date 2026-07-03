#include "se3_utilities.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace se3 {
namespace {

constexpr double kEps = 1e-12;
constexpr double kSmallAngle = 1e-8;
constexpr double kNearPi = 1e-6;

double clamp(double value, double low, double high) {
    return std::max(low, std::min(value, high));
}

std::mt19937& rng() {
    static thread_local std::mt19937 generator(std::random_device{}());
    return generator;
}

}  // namespace

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
    Eigen::Matrix3d S;
    S << 0.0, -w.z(), w.y(),
         w.z(), 0.0, -w.x(),
        -w.y(), w.x(), 0.0;
    return S;
}

Eigen::Vector3d vee_so3(const Eigen::Matrix3d& W) {
    return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
}

Eigen::Matrix3d exp_SO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    const Eigen::Matrix3d W = skew(phi);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    if (theta < kSmallAngle) {
        return I + W + 0.5 * W * W;
    }

    const double A = std::sin(theta) / theta;
    const double B = (1.0 - std::cos(theta)) / (theta * theta);
    return I + A * W + B * W * W;
}

Eigen::Vector3d log_SO3(const Eigen::Matrix3d& R) {
    constexpr double pi = 3.141592653589793238462643383279502884;

    double c = 0.5 * (R.trace() - 1.0);
    c = clamp(c, -1.0, 1.0);
    const double theta = std::acos(c);

    if (theta < kSmallAngle) {
        const Eigen::Matrix3d W = 0.5 * (R - R.transpose());
        return vee_so3(W);
    }

    if (pi - theta < kNearPi) {
        const Eigen::Matrix3d A = 0.5 * (R + Eigen::Matrix3d::Identity());

        Eigen::Vector3d axis;
        axis << std::sqrt(std::max(A(0, 0), 0.0)),
                std::sqrt(std::max(A(1, 1), 0.0)),
                std::sqrt(std::max(A(2, 2), 0.0));

        if (R(2, 1) - R(1, 2) < 0.0) {
            axis.x() = -axis.x();
        }
        if (R(0, 2) - R(2, 0) < 0.0) {
            axis.y() = -axis.y();
        }
        if (R(1, 0) - R(0, 1) < 0.0) {
            axis.z() = -axis.z();
        }

        double axis_norm = axis.norm();
        if (axis_norm < kEps) {
            const Eigen::Matrix3d W = (1.0 / (2.0 * std::sin(theta))) * (R - R.transpose());
            axis = vee_so3(W);
            axis_norm = axis.norm();

            if (axis_norm < kEps) {
                axis << 1.0, 0.0, 0.0;
            } else {
                axis /= axis_norm;
            }
        } else {
            axis /= axis_norm;
        }

        return axis * theta;
    }

    const Eigen::Matrix3d W = (1.0 / (2.0 * std::sin(theta))) * (R - R.transpose());
    const Eigen::Vector3d axis = vee_so3(W);
    return axis * theta;
}

Eigen::Matrix3d jac_left_SO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    const Eigen::Matrix3d W = skew(phi);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    if (theta < kSmallAngle) {
        return I + 0.5 * W + (1.0 / 6.0) * W * W;
    }

    const double theta2 = theta * theta;
    const double A = (1.0 - std::cos(theta)) / theta2;
    const double B = (theta - std::sin(theta)) / (theta2 * theta);
    return I + A * W + B * W * W;
}

Eigen::Matrix3d inv_jac_left_SO3(const Eigen::Vector3d& phi) {
    const double theta = phi.norm();
    const Eigen::Matrix3d W = skew(phi);
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    if (theta < kSmallAngle) {
        return I - 0.5 * W + (1.0 / 12.0) * W * W;
    }

    const double half_theta = 0.5 * theta;
    const double cot_half_theta = std::cos(half_theta) / std::sin(half_theta);
    const double theta2 = theta * theta;
    const double C = (1.0 / theta2) * (1.0 - 0.5 * theta * cot_half_theta);
    return I - 0.5 * W + C * W * W;
}

Eigen::Matrix4d make_SE3(const Eigen::Matrix3d& R, const Eigen::Vector3d& p) {
    Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
    H.block<3, 3>(0, 0) = R;
    H.block<3, 1>(0, 3) = p;
    return H;
}

Eigen::Matrix4d inv_SE3(const Eigen::Matrix4d& H) {
    const Eigen::Matrix3d R = H.block<3, 3>(0, 0);
    const Eigen::Vector3d p = H.block<3, 1>(0, 3);

    Eigen::Matrix4d H_inv = Eigen::Matrix4d::Identity();
    H_inv.block<3, 3>(0, 0) = R.transpose();
    H_inv.block<3, 1>(0, 3) = -R.transpose() * p;
    return H_inv;
}

Eigen::Matrix4d hat_SE3(const Vector6d& xi) {
    const Eigen::Vector3d v = xi.head<3>();
    const Eigen::Vector3d w = xi.tail<3>();

    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    A.block<3, 3>(0, 0) = skew(w);
    A.block<3, 1>(0, 3) = v;
    return A;
}

Vector6d vee_SE3(const Eigen::Matrix4d& A) {
    Vector6d xi;
    xi.head<3>() = A.block<3, 1>(0, 3);
    xi.tail<3>() = vee_so3(A.block<3, 3>(0, 0));
    return xi;
}

Eigen::Matrix4d exp_SE3(const Eigen::Matrix4d& A) {
    const Eigen::Matrix3d w_hat = A.block<3, 3>(0, 0);
    const Eigen::Vector3d v = A.block<3, 1>(0, 3);

    const Eigen::Vector3d phi = vee_so3(w_hat);
    const Eigen::Matrix3d R = exp_SO3(phi);
    const Eigen::Matrix3d J = jac_left_SO3(phi);
    const Eigen::Vector3d p = J * v;

    return make_SE3(R, p);
}

Eigen::Matrix4d log_SE3(const Eigen::Matrix4d& H) {
    const Eigen::Matrix3d R = H.block<3, 3>(0, 0);
    const Eigen::Vector3d p = H.block<3, 1>(0, 3);

    const Eigen::Vector3d phi = log_SO3(R);
    const Eigen::Matrix3d J_inv = inv_jac_left_SO3(phi);
    const Eigen::Vector3d v = J_inv * p;

    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    A.block<3, 3>(0, 0) = skew(phi);
    A.block<3, 1>(0, 3) = v;
    return A;
}

Matrix6d make_metric_matrix(double ell) {
    if (ell <= 0.0) {
        throw std::invalid_argument("ell must be positive.");
    }

    Matrix6d G = Matrix6d::Zero();
    G.diagonal() << 1.0, 1.0, 1.0, ell * ell, ell * ell, ell * ell;
    return G;
}

double metric_log_SE3_left(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Matrix6d& G) {
    const Eigen::Matrix4d H_rel = inv_SE3(H1) * H2;
    const Eigen::Matrix4d A = log_SE3(H_rel);
    const Vector6d xi = vee_SE3(A);
    const double d2 = xi.transpose() * G * xi;
    return std::sqrt(std::max(d2, 0.0));
}

double metric_log_SE3_right(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Matrix6d& G) {
    const Eigen::Matrix4d H_rel = H2 * inv_SE3(H1);
    const Eigen::Matrix4d A = log_SE3(H_rel);
    const Vector6d xi = vee_SE3(A);
    const double d2 = xi.transpose() * G * xi;
    return std::sqrt(std::max(d2, 0.0));
}

double metric_log_SE3_symmetric(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Matrix6d& G) {
    const double d_left = metric_log_SE3_left(H1, H2, G);
    const double d_right = metric_log_SE3_right(H1, H2, G);
    return std::sqrt(0.5 * (d_left * d_left + d_right * d_right));
}

Eigen::Vector3d transform_point(const Eigen::Matrix4d& H, const Eigen::Vector3d& q) {
    const Eigen::Vector4d q_hom(q.x(), q.y(), q.z(), 1.0);
    const Eigen::Vector4d p_hom = H * q_hom;
    return p_hom.head<3>();
}

double metric_object_points(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const std::vector<Eigen::Vector3d>& body_points) {
    if (body_points.empty()) {
        throw std::invalid_argument("body_points must not be empty.");
    }

    double acc = 0.0;
    for (const Eigen::Vector3d& q : body_points) {
        const Eigen::Vector3d p1 = transform_point(H1, q);
        const Eigen::Vector3d p2 = transform_point(H2, q);
        const Eigen::Vector3d e = p2 - p1;
        acc += e.squaredNorm();
    }

    return std::sqrt(acc / static_cast<double>(body_points.size()));
}

Eigen::Matrix4d interpolate_SE3_left(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    double s) {
    s = clamp(s, 0.0, 1.0);
    const Eigen::Matrix4d A = log_SE3(inv_SE3(H1) * H2);
    return H1 * exp_SE3(s * A);
}

Eigen::Matrix4d interpolate_SE3_right(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    double s) {
    s = clamp(s, 0.0, 1.0);
    const Eigen::Matrix4d A = log_SE3(H2 * inv_SE3(H1));
    return exp_SE3(s * A) * H1;
}

std::vector<Eigen::Matrix4d> interpolate_SE3_path(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    int n_points) {
    if (n_points < 2) {
        throw std::invalid_argument("n_points must be at least 2.");
    }

    std::vector<Eigen::Matrix4d> path;
    path.reserve(static_cast<std::size_t>(n_points));
    for (int i = 0; i < n_points; ++i) {
        const double s = static_cast<double>(i) / static_cast<double>(n_points - 1);
        path.push_back(interpolate_SE3_left(H1, H2, s));
    }
    return path;
}

Eigen::Matrix4d steer_SE3(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    double step_size,
    const Matrix6d& G) {
    if (step_size <= 0.0) {
        throw std::invalid_argument("step_size must be positive.");
    }

    const double d = metric_log_SE3_left(H1, H2, G);
    if (d < kEps) {
        return H1;
    }

    const double alpha = std::min(1.0, step_size / d);
    return interpolate_SE3_left(H1, H2, alpha);
}

Eigen::Matrix3d quat_to_rot(const Eigen::Vector4d& q_input) {
    const double n = q_input.norm();
    if (n < kEps) {
        throw std::invalid_argument("Quaternion norm is too close to zero.");
    }

    const Eigen::Vector4d q = q_input / n;
    const double qw = q(0);
    const double qx = q(1);
    const double qy = q(2);
    const double qz = q(3);

    Eigen::Matrix3d R;
    R << 1.0 - 2.0 * (qy * qy + qz * qz),
         2.0 * (qx * qy - qw * qz),
         2.0 * (qx * qz + qw * qy),
         2.0 * (qx * qy + qw * qz),
         1.0 - 2.0 * (qx * qx + qz * qz),
         2.0 * (qy * qz - qw * qx),
         2.0 * (qx * qz - qw * qy),
         2.0 * (qy * qz + qw * qx),
         1.0 - 2.0 * (qx * qx + qy * qy);
    return R;
}

Eigen::Matrix3d sample_SO3_uniform() {
    std::normal_distribution<double> normal(0.0, 1.0);

    Eigen::Vector4d q;
    do {
        q << normal(rng()), normal(rng()), normal(rng()), normal(rng());
    } while (q.norm() < kEps);

    return quat_to_rot(q);
}

Eigen::Matrix4d sample_SE3_uniform_box(const Bounds3d& position_bounds) {
    Eigen::Vector3d p;
    for (int i = 0; i < 3; ++i) {
        const double low = position_bounds(i, 0);
        const double high = position_bounds(i, 1);
        std::uniform_real_distribution<double> uniform(low, high);
        p(i) = uniform(rng());
    }

    const Eigen::Matrix3d R = sample_SO3_uniform();
    return make_SE3(R, p);
}

}  // namespace se3
