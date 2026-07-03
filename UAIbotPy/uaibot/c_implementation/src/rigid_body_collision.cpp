#include "rigid_body_collision.hpp"

#include <algorithm>

CollisionOracleResult::CollisionOracleResult()
{
    is_free = true;
    message = "Ok!";
    min_distance = VERYBIGNUMBER;
    type = 0;
    robot_object_index = -1;
    obstacle_index = -1;
    info = {};
}

std::vector<GeometricPrimitives> transform_robot_model_to_world(
    const std::vector<GeometricPrimitives>& robot_model,
    const Eigen::Matrix4f& H_robot)
{
    std::vector<GeometricPrimitives> robot_world;
    robot_world.reserve(robot_model.size());

    for (const GeometricPrimitives& obj_local : robot_model)
    {
        GeometricPrimitives obj_world = obj_local.copy();
        obj_world.htm = H_robot * obj_local.htm;
        robot_world.push_back(obj_world);
    }

    return robot_world;
}

CollisionOracleResult check_rigid_body_collision(
    const std::vector<GeometricPrimitives>& robot_model,
    const Eigen::Matrix4f& H_robot,
    const std::vector<GeometricPrimitives>& obstacles,
    float tol,
    float dist_tol,
    int no_iter_max)
{
    CollisionOracleResult result;

    std::vector<GeometricPrimitives> robot_world =
        transform_robot_model_to_world(robot_model, H_robot);

    std::vector<AABB> robot_aabbs;
    robot_aabbs.reserve(robot_world.size());
    for (const GeometricPrimitives& obj_world : robot_world)
        robot_aabbs.push_back(obj_world.get_aabb());

    for (int i = 0; i < static_cast<int>(robot_world.size()); ++i)
    {
        for (int j = 0; j < static_cast<int>(obstacles.size()); ++j)
        {
            AABB aabb_obs = obstacles[j].get_aabb();
            float aabb_distance = AABB::dist_aabb(aabb_obs, robot_aabbs[i]);
            result.min_distance = std::min(result.min_distance, aabb_distance);

            if (aabb_distance < dist_tol)
            {
                PrimDistResult pdr =
                    robot_world[i].dist_to(obstacles[j], 1e-6f, 1e-6f, tol, no_iter_max);
                result.min_distance = std::min(result.min_distance, pdr.dist);

                if (pdr.dist < dist_tol)
                {
                    result.is_free = false;
                    result.message = "Collision between robot object " + std::to_string(i) +
                                     " and obstacle " + std::to_string(j) + ".";
                    result.min_distance = pdr.dist;
                    result.type = 1;
                    result.robot_object_index = i;
                    result.obstacle_index = j;
                    result.info = {1, i, j};
                    return result;
                }
            }
        }
    }

    result.is_free = true;
    result.message = "Ok!";
    result.type = 0;
    result.robot_object_index = -1;
    result.obstacle_index = -1;
    result.info = {};

    return result;
}
