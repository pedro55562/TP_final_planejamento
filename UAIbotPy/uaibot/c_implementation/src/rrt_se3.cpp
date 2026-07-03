#include "rrt_se3.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

#include "rigid_body_collision.hpp"
#include "se3_utilities.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::mt19937& rrt_rng()
{
    static thread_local std::mt19937 generator(std::random_device{}());
    return generator;
}

double uniform01()
{
    static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(rrt_rng());
}

int random_index(int size)
{
    std::uniform_int_distribution<int> distribution(0, size - 1);
    return distribution(rrt_rng());
}

double safe_positive(double value, double fallback)
{
    if (std::isfinite(value) && value > 0.0)
        return value;
    return fallback;
}

double clamp_probability(double value)
{
    if (!std::isfinite(value))
        return 0.0;
    return std::max(0.0, std::min(1.0, value));
}

std::vector<Eigen::Matrix4d> path_to_root(const RRTSE3Tree& tree, int index)
{
    std::vector<Eigen::Matrix4d> path;
    while (index >= 0)
    {
        path.push_back(tree.nodes[index].H);
        index = tree.nodes[index].parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

double elapsed_seconds(const Clock::time_point& start, const Clock::time_point& end)
{
    return std::chrono::duration<double>(end - start).count();
}

}  // namespace

RRTSE3Options::RRTSE3Options()
{
    max_iterations = 5000;
    step_size = 0.2;
    goal_tolerance = 0.05;
    edge_resolution = 0.03;
    output_resolution = 0.01;
    connect_resolution = 0.03;
    goal_bias = 0.05;
    other_tree_bias = 0.20;
    shortcut_iterations = 100;

    collision_tol = 1e-4f;
    collision_dist_tol = 1e-3f;
    collision_no_iter_max = 20;
}

RRTSE3Result::RRTSE3Result()
{
    success = false;
    message = "";
    path = {};
    path_discrete = {};
    iterations = 0;
    total_iterations = 0;
    number_of_nodes_start = 0;
    number_of_nodes_goal = 0;
    execution_time = 0.0;
    planning_time = 0.0;
    shortcut_time = 0.0;
    discretization_time = 0.0;
    raw_path_size = 0;
    shortcut_path_size = 0;
    discrete_path_size = 0;
}

RRTSE3Node::RRTSE3Node()
{
    H = Eigen::Matrix4d::Identity();
    parent = -1;
}

RRTSE3Node::RRTSE3Node(const Eigen::Matrix4d& H_in, int parent_in)
{
    H = H_in;
    parent = parent_in;
}

int nearest_node(
    const RRTSE3Tree& tree,
    const Eigen::Matrix4d& H_query,
    const Eigen::Matrix<double, 6, 6>& G)
{
    if (tree.nodes.empty())
        return -1;

    int best_index = 0;
    double best_distance = se3::metric_log_SE3_left(tree.nodes[0].H, H_query, G);

    for (int i = 1; i < static_cast<int>(tree.nodes.size()); ++i)
    {
        double d = se3::metric_log_SE3_left(tree.nodes[i].H, H_query, G);
        if (d < best_distance)
        {
            best_distance = d;
            best_index = i;
        }
    }

    return best_index;
}

bool is_pose_free(
    const Eigen::Matrix4d& H,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options)
{
    CollisionOracleResult result = check_rigid_body_collision(
        robot_model,
        H.cast<float>(),
        obstacles,
        options.collision_tol,
        options.collision_dist_tol,
        options.collision_no_iter_max);

    return result.is_free;
}

bool is_edge_free(
    const Eigen::Matrix4d& H1,
    const Eigen::Matrix4d& H2,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options)
{
    const double resolution = safe_positive(options.edge_resolution, options.step_size);
    const double d = se3::metric_log_SE3_left(H1, H2, G);
    const int n = std::max(1, static_cast<int>(std::ceil(d / resolution)));

    for (int i = 0; i <= n; ++i)
    {
        const double s = static_cast<double>(i) / static_cast<double>(n);
        const Eigen::Matrix4d Hs = se3::interpolate_SE3_left(H1, H2, s);
        if (!is_pose_free(Hs, robot_model, obstacles, options))
            return false;
    }

    return true;
}

int extend_tree(
    RRTSE3Tree& tree,
    const Eigen::Matrix4d& H_target,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options)
{
    const int near_index = nearest_node(tree, H_target, G);
    if (near_index < 0)
        return -1;

    const Eigen::Matrix4d& H_near = tree.nodes[near_index].H;
    const Eigen::Matrix4d H_new =
        se3::steer_SE3(H_near, H_target, safe_positive(options.step_size, 0.2), G);

    if (!is_pose_free(H_new, robot_model, obstacles, options))
        return -1;

    if (!is_edge_free(H_near, H_new, G, robot_model, obstacles, options))
        return -1;

    tree.nodes.push_back(RRTSE3Node(H_new, near_index));
    return static_cast<int>(tree.nodes.size()) - 1;
}

bool try_connect_trees(
    RRTSE3Tree& tree_from,
    RRTSE3Tree& tree_to,
    int new_node_index_from,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options,
    int& connected_index_to)
{
    connected_index_to = -1;

    if (new_node_index_from < 0 ||
        new_node_index_from >= static_cast<int>(tree_from.nodes.size()) ||
        tree_to.nodes.empty())
        return false;

    const Eigen::Matrix4d H_target = tree_from.nodes[new_node_index_from].H;
    const int nearest_to_target = nearest_node(tree_to, H_target, G);
    if (nearest_to_target < 0)
        return false;

    const double initial_distance =
        se3::metric_log_SE3_left(tree_to.nodes[nearest_to_target].H, H_target, G);
    const double connect_resolution = safe_positive(options.connect_resolution, options.step_size);
    const int max_connect_steps =
        std::max(1, static_cast<int>(std::ceil(initial_distance / connect_resolution)) + 2);

    double previous_distance = initial_distance;
    for (int step = 0; step < max_connect_steps; ++step)
    {
        const int idx_new = extend_tree(tree_to, H_target, G, robot_model, obstacles, options);
        if (idx_new == -1)
            return false;

        const double current_distance =
            se3::metric_log_SE3_left(tree_to.nodes[idx_new].H, H_target, G);

        if (current_distance < safe_positive(options.goal_tolerance, 0.05) &&
            is_edge_free(tree_to.nodes[idx_new].H, H_target, G, robot_model, obstacles, options))
        {
            connected_index_to = idx_new;
            return true;
        }

        if (current_distance >= previous_distance - 1e-10)
            return false;

        previous_distance = current_distance;
    }

    return false;
}

std::vector<Eigen::Matrix4d> reconstruct_path(
    const RRTSE3Tree& tree_start,
    const RRTSE3Tree& tree_goal,
    int index_start_connection,
    int index_goal_connection)
{
    std::vector<Eigen::Matrix4d> start_path = path_to_root(tree_start, index_start_connection);
    std::vector<Eigen::Matrix4d> goal_path = path_to_root(tree_goal, index_goal_connection);
    std::reverse(goal_path.begin(), goal_path.end());

    std::vector<Eigen::Matrix4d> path = start_path;
    path.insert(path.end(), goal_path.begin(), goal_path.end());
    return path;
}

std::vector<Eigen::Matrix4d> discretize_path(
    const std::vector<Eigen::Matrix4d>& path,
    const Eigen::Matrix<double, 6, 6>& G,
    double resolution)
{
    if (path.empty())
        return {};
    if (path.size() == 1)
        return path;

    const double safe_resolution = safe_positive(resolution, 0.03);
    std::vector<Eigen::Matrix4d> path_discrete;
    path_discrete.push_back(path.front());

    for (int k = 0; k < static_cast<int>(path.size()) - 1; ++k)
    {
        const double d = se3::metric_log_SE3_left(path[k], path[k + 1], G);
        const int n = std::max(1, static_cast<int>(std::ceil(d / safe_resolution)));
        for (int i = 1; i <= n; ++i)
        {
            const double s = static_cast<double>(i) / static_cast<double>(n);
            path_discrete.push_back(se3::interpolate_SE3_left(path[k], path[k + 1], s));
        }
    }

    return path_discrete;
}

std::vector<Eigen::Matrix4d> shortcut_path(
    const std::vector<Eigen::Matrix4d>& path,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options)
{
    std::vector<Eigen::Matrix4d> shortcut = path;
    if (shortcut.size() < 3 || options.shortcut_iterations <= 0)
        return shortcut;

    for (int k = 0; k < options.shortcut_iterations; ++k)
    {
        if (shortcut.size() < 3)
            break;

        const int i = random_index(static_cast<int>(shortcut.size()));
        const int j = random_index(static_cast<int>(shortcut.size()));
        const int a = std::min(i, j);
        const int b = std::max(i, j);

        if (b <= a + 1)
            continue;

        if (is_edge_free(shortcut[a], shortcut[b], G, robot_model, obstacles, options))
            shortcut.erase(shortcut.begin() + a + 1, shortcut.begin() + b);
    }

    return shortcut;
}

RRTSE3Result plan_rrt_se3_bidirectional(
    const Eigen::Matrix4d& H_start,
    const Eigen::Matrix4d& H_goal,
    const Eigen::Matrix<double, 3, 2>& position_bounds,
    const Eigen::Matrix<double, 6, 6>& G,
    const std::vector<GeometricPrimitives>& robot_model,
    const std::vector<GeometricPrimitives>& obstacles,
    const RRTSE3Options& options)
{
    RRTSE3Result result;
    const Clock::time_point execution_start = Clock::now();
    const Clock::time_point planning_start = execution_start;

    if (!is_pose_free(H_start, robot_model, obstacles, options))
    {
        const Clock::time_point planning_end = Clock::now();
        result.message = "Start pose is in collision.";
        result.planning_time = elapsed_seconds(planning_start, planning_end);
        result.execution_time = elapsed_seconds(execution_start, Clock::now());
        return result;
    }

    if (!is_pose_free(H_goal, robot_model, obstacles, options))
    {
        const Clock::time_point planning_end = Clock::now();
        result.message = "Goal pose is in collision.";
        result.planning_time = elapsed_seconds(planning_start, planning_end);
        result.execution_time = elapsed_seconds(execution_start, Clock::now());
        return result;
    }

    RRTSE3Tree tree_start;
    RRTSE3Tree tree_goal;
    tree_start.nodes.push_back(RRTSE3Node(H_start, -1));
    tree_goal.nodes.push_back(RRTSE3Node(H_goal, -1));

    if (is_edge_free(H_start, H_goal, G, robot_model, obstacles, options))
    {
        const Clock::time_point planning_end = Clock::now();
        const std::vector<Eigen::Matrix4d> raw_path = {H_start, H_goal};

        const Clock::time_point shortcut_start = Clock::now();
        const std::vector<Eigen::Matrix4d> path =
            shortcut_path(raw_path, G, robot_model, obstacles, options);
        const Clock::time_point shortcut_end = Clock::now();

        const Clock::time_point discretization_start = Clock::now();
        std::vector<Eigen::Matrix4d> path_discrete =
            discretize_path(path, G, options.output_resolution);
        const Clock::time_point discretization_end = Clock::now();

        result.success = true;
        result.message = "Path found.";
        result.path = path;
        result.path_discrete = path_discrete;
        result.iterations = 0;
        result.total_iterations = result.iterations;
        result.number_of_nodes_start = static_cast<int>(tree_start.nodes.size());
        result.number_of_nodes_goal = static_cast<int>(tree_goal.nodes.size());
        result.planning_time = elapsed_seconds(planning_start, planning_end);
        result.shortcut_time = elapsed_seconds(shortcut_start, shortcut_end);
        result.discretization_time = elapsed_seconds(discretization_start, discretization_end);
        result.execution_time = elapsed_seconds(execution_start, Clock::now());
        result.raw_path_size = static_cast<int>(raw_path.size());
        result.shortcut_path_size = static_cast<int>(result.path.size());
        result.discrete_path_size = static_cast<int>(result.path_discrete.size());
        return result;
    }

    const double other_tree_bias = clamp_probability(options.other_tree_bias);
    const double goal_bias = clamp_probability(options.goal_bias);
    const int max_iterations = std::max(0, options.max_iterations);

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        const bool expand_start = (iter % 2 == 0);
        RRTSE3Tree& active_tree = expand_start ? tree_start : tree_goal;
        RRTSE3Tree& other_tree = expand_start ? tree_goal : tree_start;

        Eigen::Matrix4d H_target;
        const double r = uniform01();
        if (r < other_tree_bias && !other_tree.nodes.empty())
        {
            H_target = other_tree.nodes[random_index(static_cast<int>(other_tree.nodes.size()))].H;
        }
        else if (r < other_tree_bias + goal_bias)
        {
            H_target = other_tree.nodes.front().H;
        }
        else
        {
            H_target = se3::sample_SE3_uniform_box(position_bounds);
        }

        const int new_node_index = extend_tree(
            active_tree, H_target, G, robot_model, obstacles, options);

        if (new_node_index != -1)
        {
            int connected_index_other = -1;
            const bool connected = try_connect_trees(
                active_tree,
                other_tree,
                new_node_index,
                G,
                robot_model,
                obstacles,
                options,
                connected_index_other);

            if (connected)
            {
                int index_start_connection = -1;
                int index_goal_connection = -1;

                if (expand_start)
                {
                    index_start_connection = new_node_index;
                    index_goal_connection = connected_index_other;
                }
                else
                {
                    index_start_connection = connected_index_other;
                    index_goal_connection = new_node_index;
                }

                std::vector<Eigen::Matrix4d> raw_path = reconstruct_path(
                    tree_start, tree_goal, index_start_connection, index_goal_connection);
                const Clock::time_point planning_end = Clock::now();

                const Clock::time_point shortcut_start = Clock::now();
                std::vector<Eigen::Matrix4d> path =
                    shortcut_path(raw_path, G, robot_model, obstacles, options);
                const Clock::time_point shortcut_end = Clock::now();

                const Clock::time_point discretization_start = Clock::now();
                std::vector<Eigen::Matrix4d> path_discrete =
                    discretize_path(path, G, options.output_resolution);
                const Clock::time_point discretization_end = Clock::now();

                result.success = true;
                result.message = "Path found.";
                result.path = path;
                result.path_discrete = path_discrete;
                result.iterations = iter + 1;
                result.total_iterations = result.iterations;
                result.number_of_nodes_start = static_cast<int>(tree_start.nodes.size());
                result.number_of_nodes_goal = static_cast<int>(tree_goal.nodes.size());
                result.planning_time = elapsed_seconds(planning_start, planning_end);
                result.shortcut_time = elapsed_seconds(shortcut_start, shortcut_end);
                result.discretization_time =
                    elapsed_seconds(discretization_start, discretization_end);
                result.execution_time = elapsed_seconds(execution_start, Clock::now());
                result.raw_path_size = static_cast<int>(raw_path.size());
                result.shortcut_path_size = static_cast<int>(result.path.size());
                result.discrete_path_size = static_cast<int>(result.path_discrete.size());
                return result;
            }
        }
    }

    const Clock::time_point planning_end = Clock::now();
    result.success = false;
    result.message = "Maximum number of iterations reached.";
    result.iterations = max_iterations;
    result.total_iterations = result.iterations;
    result.number_of_nodes_start = static_cast<int>(tree_start.nodes.size());
    result.number_of_nodes_goal = static_cast<int>(tree_goal.nodes.size());
    result.planning_time = elapsed_seconds(planning_start, planning_end);
    result.execution_time = elapsed_seconds(execution_start, Clock::now());
    return result;
}
