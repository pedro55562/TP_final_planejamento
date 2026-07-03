#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <vector>

#include "se3_bindings.hpp"
#include "se3_utilities.hpp"

namespace py = pybind11;

namespace {

using PyArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <int N>
Eigen::Matrix<double, N, 1> vector_from_py(const py::object& obj, const char* name) {
    PyArray array = PyArray::ensure(obj);
    if (!array) {
        std::ostringstream oss;
        oss << name << " must be convertible to a numeric array.";
        throw py::value_error(oss.str());
    }

    const py::buffer_info info = array.request();
    const bool valid_1d = info.ndim == 1 && info.shape[0] == N;
    const bool valid_col = info.ndim == 2 && info.shape[0] == N && info.shape[1] == 1;
    const bool valid_row = info.ndim == 2 && info.shape[0] == 1 && info.shape[1] == N;

    if (!valid_1d && !valid_col && !valid_row) {
        std::ostringstream oss;
        oss << name << " must have shape (" << N << ",), (" << N << ", 1), or (1, " << N
            << "), got ";
        if (info.ndim == 0) {
            oss << "().";
        } else {
            oss << "(";
            for (ssize_t i = 0; i < info.ndim; ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << info.shape[static_cast<std::size_t>(i)];
            }
            oss << ").";
        }
        throw py::value_error(oss.str());
    }

    const double* data = static_cast<const double*>(info.ptr);
    Eigen::Matrix<double, N, 1> out;
    for (int i = 0; i < N; ++i) {
        out(i) = data[i];
    }
    return out;
}

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> matrix_from_py(const py::object& obj, const char* name) {
    PyArray array = PyArray::ensure(obj);
    if (!array) {
        std::ostringstream oss;
        oss << name << " must be convertible to a numeric array.";
        throw py::value_error(oss.str());
    }

    const py::buffer_info info = array.request();
    if (info.ndim != 2 || info.shape[0] != Rows || info.shape[1] != Cols) {
        std::ostringstream oss;
        oss << name << " must have shape (" << Rows << ", " << Cols << "), got ";
        if (info.ndim == 0) {
            oss << "().";
        } else {
            oss << "(";
            for (ssize_t i = 0; i < info.ndim; ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << info.shape[static_cast<std::size_t>(i)];
            }
            oss << ").";
        }
        throw py::value_error(oss.str());
    }

    const double* data = static_cast<const double*>(info.ptr);
    Eigen::Matrix<double, Rows, Cols> out;
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            out(i, j) = data[i * Cols + j];
        }
    }
    return out;
}

std::vector<Eigen::Vector3d> body_points_from_py(const py::object& obj) {
    PyArray array = PyArray::ensure(obj);
    if (array) {
        const py::buffer_info info = array.request();
        const double* data = static_cast<const double*>(info.ptr);

        if (info.ndim == 1 && info.shape[0] == 0) {
            return {};
        }

        if (info.ndim == 2 && info.shape[1] == 3) {
            std::vector<Eigen::Vector3d> points;
            points.reserve(static_cast<std::size_t>(info.shape[0]));
            for (ssize_t i = 0; i < info.shape[0]; ++i) {
                points.emplace_back(data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]);
            }
            return points;
        }

        if (info.ndim == 3 && info.shape[1] == 3 && info.shape[2] == 1) {
            std::vector<Eigen::Vector3d> points;
            points.reserve(static_cast<std::size_t>(info.shape[0]));
            for (ssize_t i = 0; i < info.shape[0]; ++i) {
                const ssize_t base = i * 3;
                points.emplace_back(data[base + 0], data[base + 1], data[base + 2]);
            }
            return points;
        }

        if (info.ndim == 3 && info.shape[1] == 1 && info.shape[2] == 3) {
            std::vector<Eigen::Vector3d> points;
            points.reserve(static_cast<std::size_t>(info.shape[0]));
            for (ssize_t i = 0; i < info.shape[0]; ++i) {
                const ssize_t base = i * 3;
                points.emplace_back(data[base + 0], data[base + 1], data[base + 2]);
            }
            return points;
        }
    }

    if (!py::isinstance<py::sequence>(obj)) {
        throw py::value_error("body_points must be a sequence of 3D points.");
    }

    py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
    std::vector<Eigen::Vector3d> points;
    points.reserve(static_cast<std::size_t>(py::len(seq)));
    for (py::handle item : seq) {
        points.push_back(vector_from_py<3>(py::reinterpret_borrow<py::object>(item), "body point"));
    }
    return points;
}

}  // namespace

void bind_se3_utilities(py::module_& m) {
    m.def("skew", [](const py::object& w) {
        return se3::skew(vector_from_py<3>(w, "w"));
    }, py::arg("w"));

    m.def("vee_so3", [](const py::object& W) {
        return se3::vee_so3(matrix_from_py<3, 3>(W, "W"));
    }, py::arg("W"));

    m.def("exp_SO3", [](const py::object& phi) {
        return se3::exp_SO3(vector_from_py<3>(phi, "phi"));
    }, py::arg("phi"));

    m.def("log_SO3", [](const py::object& R) {
        return se3::log_SO3(matrix_from_py<3, 3>(R, "R"));
    }, py::arg("R"));

    m.def("jac_left_SO3", [](const py::object& phi) {
        return se3::jac_left_SO3(vector_from_py<3>(phi, "phi"));
    }, py::arg("phi"));

    m.def("inv_jac_left_SO3", [](const py::object& phi) {
        return se3::inv_jac_left_SO3(vector_from_py<3>(phi, "phi"));
    }, py::arg("phi"));

    m.def("make_SE3", [](const py::object& R, const py::object& p) {
        return se3::make_SE3(matrix_from_py<3, 3>(R, "R"), vector_from_py<3>(p, "p"));
    }, py::arg("R"), py::arg("p"));

    m.def("inv_SE3", [](const py::object& H) {
        return se3::inv_SE3(matrix_from_py<4, 4>(H, "H"));
    }, py::arg("H"));

    m.def("hat_SE3", [](const py::object& xi) {
        return se3::hat_SE3(vector_from_py<6>(xi, "xi"));
    }, py::arg("xi"));

    m.def("vee_SE3", [](const py::object& A) {
        return se3::vee_SE3(matrix_from_py<4, 4>(A, "A"));
    }, py::arg("A"));

    m.def("exp_SE3", [](const py::object& A) {
        return se3::exp_SE3(matrix_from_py<4, 4>(A, "A"));
    }, py::arg("A"));

    m.def("log_SE3", [](const py::object& H) {
        return se3::log_SE3(matrix_from_py<4, 4>(H, "H"));
    }, py::arg("H"));

    m.def("make_metric_matrix", &se3::make_metric_matrix, py::arg("ell"));

    m.def("metric_log_SE3_left", [](const py::object& H1, const py::object& H2, const py::object& G) {
        return se3::metric_log_SE3_left(
            matrix_from_py<4, 4>(H1, "H1"),
            matrix_from_py<4, 4>(H2, "H2"),
            matrix_from_py<6, 6>(G, "G"));
    }, py::arg("H1"), py::arg("H2"), py::arg("G"));

    m.def("metric_log_SE3_right", [](const py::object& H1, const py::object& H2, const py::object& G) {
        return se3::metric_log_SE3_right(
            matrix_from_py<4, 4>(H1, "H1"),
            matrix_from_py<4, 4>(H2, "H2"),
            matrix_from_py<6, 6>(G, "G"));
    }, py::arg("H1"), py::arg("H2"), py::arg("G"));

    m.def("metric_log_SE3_symmetric", [](const py::object& H1, const py::object& H2, const py::object& G) {
        return se3::metric_log_SE3_symmetric(
            matrix_from_py<4, 4>(H1, "H1"),
            matrix_from_py<4, 4>(H2, "H2"),
            matrix_from_py<6, 6>(G, "G"));
    }, py::arg("H1"), py::arg("H2"), py::arg("G"));

    m.def("transform_point", [](const py::object& H, const py::object& q) {
        return se3::transform_point(matrix_from_py<4, 4>(H, "H"), vector_from_py<3>(q, "q"));
    }, py::arg("H"), py::arg("q"));

    m.def("metric_object_points", [](const py::object& H1, const py::object& H2, const py::object& body_points) {
        return se3::metric_object_points(
            matrix_from_py<4, 4>(H1, "H1"),
            matrix_from_py<4, 4>(H2, "H2"),
            body_points_from_py(body_points));
    }, py::arg("H1"), py::arg("H2"), py::arg("body_points"));

    m.def("interpolate_SE3_left", [](const py::object& H1, const py::object& H2, double s) {
        return se3::interpolate_SE3_left(matrix_from_py<4, 4>(H1, "H1"), matrix_from_py<4, 4>(H2, "H2"), s);
    }, py::arg("H1"), py::arg("H2"), py::arg("s"));

    m.def("interpolate_SE3_right", [](const py::object& H1, const py::object& H2, double s) {
        return se3::interpolate_SE3_right(matrix_from_py<4, 4>(H1, "H1"), matrix_from_py<4, 4>(H2, "H2"), s);
    }, py::arg("H1"), py::arg("H2"), py::arg("s"));

    m.def("interpolate_SE3_path", [](const py::object& H1, const py::object& H2, int n_points) {
        return se3::interpolate_SE3_path(matrix_from_py<4, 4>(H1, "H1"), matrix_from_py<4, 4>(H2, "H2"), n_points);
    }, py::arg("H1"), py::arg("H2"), py::arg("n_points"));

    m.def("steer_SE3", [](const py::object& H1, const py::object& H2, double step_size, const py::object& G) {
        return se3::steer_SE3(
            matrix_from_py<4, 4>(H1, "H1"),
            matrix_from_py<4, 4>(H2, "H2"),
            step_size,
            matrix_from_py<6, 6>(G, "G"));
    }, py::arg("H1"), py::arg("H2"), py::arg("step_size"), py::arg("G"));

    m.def("quat_to_rot", [](const py::object& q) {
        return se3::quat_to_rot(vector_from_py<4>(q, "q"));
    }, py::arg("q"));

    m.def("sample_SO3_uniform", &se3::sample_SO3_uniform);

    m.def("sample_SE3_uniform_box", [](const py::object& position_bounds) {
        return se3::sample_SE3_uniform_box(matrix_from_py<3, 2>(position_bounds, "position_bounds"));
    }, py::arg("position_bounds"));
}
