#ifndef PLANNER_BASELINE_H
#define PLANNER_BASELINE_H

#include <queue>
#include <iostream>
#include <ros/ros.h>
#include <planner_base/path.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include "planner_base/planner_base.h"
#include "planner_base/planner_type.h"

namespace planner {

typedef std::shared_ptr<Node> NodePtr;
typedef std::shared_ptr<std::vector<Point3i>> point3i_vector_ptr;

class PlannerBaseline: public Planner
{

private:
    bool setup_finished;
    std::shared_ptr<Point3i> m_init;
    std::shared_ptr<Point3i> m_goal;
    std::shared_ptr<Point3i> m_size;

    point3i_vector_ptr m_local_frontier;
    boost::shared_ptr<double[]> m_local_cost_map;  // cost-to-go map
    boost::shared_ptr<int[]> m_local_info_map;  // information map

    std::vector<NodePtr> m_feasible_path;

public:
    PlannerBaseline();

    bool plan();
    void plan_result(const data_type::PlannerMessage &plan_msg);
    void setup_problem(const data_type::PlannerMessage &plan_msg);

    // planning algorithm
    void get_successor(const NodePtr &pre, std::vector<NodePtr> &successor);
    bool AStar_algorithm(std::vector<NodePtr> &path);


};

}

#endif
