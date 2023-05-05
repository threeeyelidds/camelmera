/*********************************************************************
*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2018, The Airlab.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
* Author: Delong Zhu & Yanfu Zhang
*********************************************************************/
#ifndef EXPO_BASE_H
#define EXPO_BASE_H

#include <vector>
#include <string>
#include <ros/ros.h>
#include <functional>
#include <planner_base/path.h>
#include <expo_base/nearfrontier.h>
#include <expo_base/globalfrontier.h>
#include <expo_base/frontierlock.h>
#include <expo_base/pointstate.h>
#include <visualization_msgs/Marker.h>
#include <data_type/planner_message.h>
#include <data_type/point3d.h>
#include <planner_base/planner_base.h>
#include <frontier_base/frontier_base.h>
#include <octomap_server/OctomapServer.h>



namespace expo_base {
typedef data_type::Point3i Point3i;
typedef data_type::Point3d Point3d;

class ExpoBase {
public:
    ExpoBase();

    void publish_goal(const data_type::PlannerMessage &plan_msg);

private:

    // strategy for selecting the best frontiers
    void next_best_view(data_type::PlannerMessage &plan_msg);

    // strategy for recover the trapped robot
    void recovery_behavior(data_type::PlannerMessage &plan_msg);

    bool plan_service(planner_base::path::Request &req, planner_base::path::Response &res);

    bool nearest_frontier_service(expo_base::nearfrontier::Request &req, expo_base::nearfrontier::Response &res);

    // plan to a global frontier, if it doesn't exist, delete the frontier from global set
    bool plan_to_frontier_service(expo_base::globalfrontier::Request &req, expo_base::globalfrontier::Response &res);
    bool lock_frontier_service(expo_base::frontierlock::Request &req, expo_base::frontierlock::Response &res);
    bool wait_for_frontier_update_service(expo_base::frontierlock::Request &req, expo_base::frontierlock::Response &res);
    bool point_state_service(expo_base::pointstate::Request &req, expo_base::pointstate::Response &res);
    bool is_global_frontier_service(expo_base::pointstate::Request &req, expo_base::pointstate::Response &res);

private:
    std::shared_ptr<planner::Planner> planner_ptr; // planner for local path finding
    std::shared_ptr<frontier::Frontier> frontier_ptr;
    std::shared_ptr<octomap_server::OctomapServer> map_server_ptr;  // map server todo: add map representer
    void setup_expo();

protected:
    ros::NodeHandle m_nh;
    ros::Publisher m_bbx_path_pub;
    ros::Publisher m_bbx_goal_pub;
    ros::ServiceServer m_plan_service;
    ros::ServiceServer m_nearest_frontier_service;
    ros::ServiceServer m_plan_to_frontier_service;
    ros::ServiceServer m_lock_frontier_service;
    ros::ServiceServer m_wait_frontier_service;
    ros::ServiceServer m_point_state_service;
    ros::ServiceServer m_is_global_frontier_service;

}; // end class

}; // end namespace
#endif

