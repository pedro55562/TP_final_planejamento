#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "declarations.h"

struct RRTSE3Options
{
    int max_iterations;
    double step_size;
    double goal_tolerance;
    double edge_resolution;
    double output_resolution;
    double connect_resolution;
    double goal_bias;
    double other_tree_bias;
    int shortcut_iterations;

    float collision_tol;
    float collision_dist_tol;
    int collision_no_iter_max;

    RRTSE3Options();
};

struct RRTSE3Result
{
    bool success;
    std::string message;
    std::vector<Eigen::Matrix4d> path;
    std::vector<Eigen::Matrix4d> path_discrete;
    int iterations;
    int total_iterations;
    int number_of_nodes_start;
    int number_of_nodes_goal;
    double execution_time;
    double planning_time;
    double shortcut_time;
    double discretization_time;
    int raw_path_size;
    int shortcut_path_size;
    int discrete_path_size;

    RRTSE3Result();
};

struct RRTSE3Node
{
    Eigen::Matrix4d H;
    int parent;

    RRTSE3Node();
    RRTSE3Node(const Eigen::Matrix4d& H_in, int parent_in);
};

struct RRTSE3Tree
{
    std::vector<RRTSE3Node> nodes;
};

int nearest_node(
    const RRTSE3Tree& tree,
    const Eigen::Matrix4d& H_query,
    const Eigen::Matrix<double, 6, 6>& G);

bool is_pose_free(
    const Eigen::Matrix4d& H,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options);

bool is_edge_free(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options);

int extend_tree(
    RRTSE3Tree& tree,
    const Eigen::Matrix4d& H_target,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options);

bool try_connect_trees(
    RRTSE3Tree& tree_from,
    RRTSE3Tree& tree_to,
    int new_node_index_from,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options,
    int& connected_index_to);

std::vector<Eigen::Matrix4d> reconstruct_path(
    const RRTSE3Tree& tree_start,
    const RRTSE3Tree& tree_goal,
    int index_start_connection,
    int index_goal_connection);

std::vector<Eigen::Matrix4d> discretize_path(
    const std::vector<Eigen::Matrix4d>& path,
    const Eigen::Matrix<double, 6, 6>& G,
    double resolution);

std::vector<Eigen::Matrix4d> shortcut_path(
    const std::vector<Eigen::Matrix4d>& path,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options);

RRTSE3Result plan_rrt_se3_bidirectional(
    const Eigen::Matrix4d& H_start,
    const Eigen::Matrix4d& H_goal,
    const Eigen::Matrix<double, 3, 2>& position_bounds,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options);
