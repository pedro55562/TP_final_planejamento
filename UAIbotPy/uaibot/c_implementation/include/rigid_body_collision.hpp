#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "declarations.h"

struct CollisionOracleResult
{
    bool is_free;
    std::string message;
    float min_distance;
    int type;
    int robot_object_index;
    int obstacle_index;
    std::vector<int> info;

    CollisionOracleResult();
};

std::vector<GeometricPrimitives> transform_robot_model_to_world(
    const std::vector<GeometricPrimitives>& robot_model,
    const Eigen::Matrix4f& H_robot);

CollisionOracleResult check_rigid_body_collision(
    const std::vector<GeometricPrimitives>& robot_model,
    const Eigen::Matrix4f& H_robot,
    const std::vector<GeometricPrimitives>& obstacles,
    float tol,
    float dist_tol,
    int no_iter_max);
