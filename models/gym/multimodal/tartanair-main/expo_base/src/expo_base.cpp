/*********************************************************************
*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
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
* Author: Eitan Marder-Eppstein
*         Mike Phillips (put the planner in its own thread)
*********************************************************************/
#include "expo_base/expo_base.h"
#include <planner_base/planner_baseline.h>
#include <frontier_base/frontier_baseline.h>

#include <cmath>

namespace expo_base {

ExpoBase::ExpoBase():
    planner_ptr(nullptr),
    frontier_ptr(nullptr),
    map_server_ptr(nullptr),
    m_nh()
{
    // private node handle for parameter reading
    ros::NodeHandle private_nh = ros::NodeHandle("~");
    std::string mapFilenameParam("");
    private_nh.getParam("map_file", mapFilenameParam);

    // ros topics and services
    m_plan_service = m_nh.advertiseService("bbx_path_srv", &ExpoBase::plan_service, this);
    m_plan_to_frontier_service = m_nh.advertiseService("plan_to_frontier_srv", &ExpoBase::plan_to_frontier_service, this);
    m_nearest_frontier_service = m_nh.advertiseService("near_frontier_srv", &ExpoBase::nearest_frontier_service, this);
    m_lock_frontier_service = m_nh.advertiseService("lock_frontier_srv", &ExpoBase::lock_frontier_service, this);
    m_wait_frontier_service = m_nh.advertiseService("wait_frontier_srv", &ExpoBase::wait_for_frontier_update_service, this);
    m_point_state_service = m_nh.advertiseService("check_point_status", &ExpoBase::point_state_service, this);
    m_is_global_frontier_service = m_nh.advertiseService("is_global_frontier", &ExpoBase::is_global_frontier_service, this);
    m_bbx_path_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_path", 1);
    m_bbx_goal_pub = m_nh.advertise<visualization_msgs::Marker>("bbx_goal", 1);

    // initialize other modules
    setup_expo();

    // if necessary to read map
    if (mapFilenameParam!=""){
        if (!map_server_ptr->openFile(mapFilenameParam)){
            ROS_ERROR("Could not open file %s", mapFilenameParam.c_str());
            exit(1);
        }else {
            ROS_ERROR("Initial map was successfully loaded!");
        }
    }

}

void ExpoBase::setup_expo()
{
    // instantiate map server
    map_server_ptr = std::make_shared<octomap_server::OctomapServer>();

    // instantiate frontier detector
    frontier_ptr = std::make_shared<frontier::FrontierBaseline>();

    // register frontier detector to map server
    map_server_ptr->attach_frontier_detector(frontier_ptr);

    // instantiate planner
    planner_ptr = std::make_shared<planner::PlannerBaseline>();
}

void ExpoBase::next_best_view(data_type::PlannerMessage &plan_msg)
{
    if(plan_msg.m_init == nullptr){
        ROS_WARN("init position is not initialized!");
        return;
    }
    // sort the frontier according to your own definition about "best view"
    std::sort(plan_msg.m_frontier->begin(), plan_msg.m_frontier->end(), [plan_msg](Point3i &i, Point3i &j){
        double a = (*(plan_msg.m_init) - i).norm();
        double b = (*(plan_msg.m_init) - j).norm();
        return  a > b;
    });
}

void ExpoBase::recovery_behavior(data_type::PlannerMessage &plan_msg)
{

}

void ExpoBase::publish_goal(const data_type::PlannerMessage& plan_msg)
{

    visualization_msgs::Marker goal;
    goal.color.r = 1.0;
    goal.color.g = 0.2;
    goal.color.b = 0.8;
    goal.color.a = 1.0;
    frontier_ptr->setup_visualizer(goal);

    // visualize goal
    Point3d coor;
    frontier_ptr->key2coor(*plan_msg.m_goal, coor);
    geometry_msgs::Point point_goal;
    point_goal.x = coor.x() + plan_msg.m_local_orig->x();
    point_goal.y = coor.y() + plan_msg.m_local_orig->y();
    point_goal.z = coor.z() + plan_msg.m_local_orig->z();
    goal.points.push_back(point_goal);

    // visualize init
    frontier_ptr->key2coor(*plan_msg.m_init, coor);
    geometry_msgs::Point point_init;
    point_init.x = coor.x() + plan_msg.m_local_orig->x();
    point_init.y = coor.y() + plan_msg.m_local_orig->y();
    point_init.z = coor.z() + plan_msg.m_local_orig->z();
    goal.points.push_back(point_init);

    // publish
    m_bbx_goal_pub.publish(goal);
}

bool ExpoBase::plan_service(planner_base::path::Request &req, planner_base::path::Response &res)
{
    ROS_INFO("Path planning request %d is received!", req.round);

    // key data structure
    int round = req.round;
    data_type::PlannerMessage plan_msg;

    // get planning information from frontier detection
    frontier_ptr->detection_result(plan_msg);
    if(!plan_msg.check_info_from_frontier()){
        ROS_ERROR("Frontier infomation check failed!");
        return false; // system or logic error
    }
    if(plan_msg.m_frontier->empty()){
        ROS_WARN("No frontier is detected!");
        return false; // no planning goals
    }

    // get planning request from remote client
    if(round > int(plan_msg.m_frontier->size()) || round < 0){
        round = plan_msg.m_frontier->size();
        ROS_INFO("request round is bigger than frontier number, set round to %d", round);
    }

    // setup init node for planning
    plan_msg.m_init = std::make_shared<Point3i>();
    frontier_ptr->coor2key((*plan_msg.m_robot_pose)-(*plan_msg.m_local_orig), *plan_msg.m_init);

    // multi-goal planning (assert: round > 0)
    this->next_best_view(plan_msg); // amigo: this sould be very time consuming??
    plan_msg.m_feasible_path = std::make_shared<std::vector<Point3i>>();
    for(int i=0; i < round; i++){ 

        // set up goal
        uint frnt_num = plan_msg.m_frontier->size();
        plan_msg.m_goal = std::make_shared<Point3i>((*plan_msg.m_frontier)[frnt_num-1-i]);

        // publish init and target goal
        if(m_bbx_goal_pub.getNumSubscribers() > 0){
            publish_goal(plan_msg);
        }

        // clear history result
        plan_msg.m_feasible_path->clear();

        // carry out planning
        planner_ptr->setup_problem(plan_msg);
        planner_ptr->plan();

        // get planning result
        planner_ptr->plan_result(plan_msg);
        if(!plan_msg.m_feasible_path->empty()){
            // // for debug
            // Point3i localind = (*plan_msg.m_frontier)[frnt_num-1-i];
            // Point3d localcoor, globalcoor;
            // frontier_ptr->key2coor(localind, localcoor);
            // globalcoor = localcoor + *plan_msg.m_local_orig;
            // ROS_INFO("PlanService:: find a path to local frontier (%f, %f, %f)", globalcoor.x(), globalcoor.y(), globalcoor.z());
            break;
        }
        else // no solution found
        {
            Point3i localind = (*plan_msg.m_frontier)[frnt_num-1-i];
            Point3d localcoor, globalcoor;
            frontier_ptr->key2coor(localind, localcoor);
            globalcoor = localcoor + *plan_msg.m_local_orig;
            // delete the frontier from the global set
            ROS_WARN("The local frontier (%f, %f, %f) is not reachable! Delete from the global set! ", globalcoor.x(), globalcoor.y(), globalcoor.z());
            // delete the frontier from the global set
            frontier_ptr->find_global_frontier(globalcoor, true);
        }
    }

    if(!plan_msg.check_info_from_planner()){
        ROS_ERROR("Planner infomation check failed!");
        return false; // system or logic error
    }
    if(plan_msg.m_feasible_path->empty()){
        ROS_WARN("No path is found!");
        return false; // no feasible path
    }


    // convert to global coordinates
    for(auto it = plan_msg.m_feasible_path->begin(); it != plan_msg.m_feasible_path->end(); ++it){
        Point3d coor;
        frontier_ptr->key2coor(*it, coor);
        geometry_msgs::Point point;
        point.x = coor.x() + plan_msg.m_local_orig->x();
        point.y = coor.y() + plan_msg.m_local_orig->y();
        point.z = coor.z() + plan_msg.m_local_orig->z();
        res.path.points.push_back(point);
    }

    // publish feasible path
    if(m_bbx_path_pub.getNumSubscribers() > 0){
        res.path.color.r = 1.0;
        res.path.color.g = 0.0;
        res.path.color.b = 0.0;
        res.path.color.a = 0.7;
        frontier_ptr->setup_visualizer(res.path);
        m_bbx_path_pub.publish(res.path);
    }

    return true;
}

bool ExpoBase::plan_to_frontier_service(expo_base::globalfrontier::Request &req, expo_base::globalfrontier::Response &res)
{
    ROS_WARN("Planning to the global frontier (%f, %f, %f)!", req.global_frontier.x, req.global_frontier.y, req.global_frontier.z);

    // key data structure
    data_type::PlannerMessage plan_msg;

    // get planning information from frontier detection
    frontier_ptr->detection_result(plan_msg);
    if(!plan_msg.check_info_from_frontier()){
        ROS_ERROR("Frontier infomation check failed!");
        return false; // system or logic error
    }
    if(plan_msg.m_frontier->empty()){
        ROS_WARN("No frontier is detected!");
        ROS_WARN("The global frontier (%f, %f, %f) is not detectable! Delete from the global set! ", req.global_frontier.x, req.global_frontier.y, req.global_frontier.z);
        // delete the frontier from the global set
        Point3d frontier_pt = Point3d(req.global_frontier.x, req.global_frontier.y, req.global_frontier.z);
        frontier_ptr->find_global_frontier(frontier_pt, true);
        return false; // no planning goals
    }

    // setup init node for planning
    plan_msg.m_init = std::make_shared<Point3i>();
    frontier_ptr->coor2key((*plan_msg.m_robot_pose)-(*plan_msg.m_local_orig), *plan_msg.m_init);

    // set up goal
    plan_msg.m_goal = std::make_shared<Point3i>();
    Point3d frontier_pt = Point3d(req.global_frontier.x, req.global_frontier.y, req.global_frontier.z);
    frontier_ptr->coor2key(frontier_pt-(*plan_msg.m_local_orig), *plan_msg.m_goal);


    // publish init and target goal
    if(m_bbx_goal_pub.getNumSubscribers() > 0){
        publish_goal(plan_msg);
    }

    // clear history result
    plan_msg.m_feasible_path = std::make_shared<std::vector<Point3i>>();

    // carry out planning
    planner_ptr->setup_problem(plan_msg);
    planner_ptr->plan();

    // get planning result
    planner_ptr->plan_result(plan_msg);

    if(!plan_msg.check_info_from_planner()){
        ROS_ERROR("Planner infomation check failed!");
        return false; // system or logic error
    }
    if(plan_msg.m_feasible_path->empty()){
        ROS_WARN("The global frontier (%f, %f, %f) is not reachable! Delete from the global set! ", req.global_frontier.x, req.global_frontier.y, req.global_frontier.z);
        // delete the frontier from the global set
        frontier_ptr->find_global_frontier(frontier_pt, true);
        return true; // no feasible path
    }


    // convert to global coordinates
    for(auto it = plan_msg.m_feasible_path->begin(); it != plan_msg.m_feasible_path->end(); ++it){
        Point3d coor;
        frontier_ptr->key2coor(*it, coor);
        geometry_msgs::Point point;
        point.x = coor.x() + plan_msg.m_local_orig->x();
        point.y = coor.y() + plan_msg.m_local_orig->y();
        point.z = coor.z() + plan_msg.m_local_orig->z();
        res.path.points.push_back(point);
    }

    // publish feasible path
    if(m_bbx_path_pub.getNumSubscribers() > 0){
        res.path.color.r = 1.0;
        res.path.color.g = 0.0;
        res.path.color.b = 0.0;
        res.path.color.a = 0.7;
        frontier_ptr->setup_visualizer(res.path);
        m_bbx_path_pub.publish(res.path);
    }

    return true;
}

bool ExpoBase::lock_frontier_service(expo_base::frontierlock::Request &req, expo_base::frontierlock::Response &res)
{
    ROS_WARN("Locked the frontier update!");
    frontier_ptr->set_frontier_update_lock(true);
    return true;
}

bool ExpoBase::nearest_frontier_service(expo_base::nearfrontier::Request &req, expo_base::nearfrontier::Response &res)
{
    ROS_WARN("nearest_frontier_service::Nearest frontier request is received! Robot position (%f, %f, %f)", req.robotpos.x, req.robotpos.y, req.robotpos.z);

    Point3d robot_pos(req.robotpos.x, req.robotpos.y, req.robotpos.z);
    Point3d nearest_pt(data_type::INF, data_type::INF, data_type::INF);
    Point3d campose_pt(data_type::INF, data_type::INF, data_type::INF);

    frontier_ptr->nearest_global_frontier(robot_pos, nearest_pt, campose_pt);
    // ROS_WARN("nearest_frontier_service returned..");
    
    if(nearest_pt.x() >= data_type::INF-1)
    {
        ROS_WARN("No more global frontier. Cheers!");
        return false;
    }
    res.nearfrontier.x = nearest_pt.x();
    res.nearfrontier.y = nearest_pt.y();
    res.nearfrontier.z = nearest_pt.z();

    res.safepose.x = campose_pt.x();
    res.safepose.y = campose_pt.y();
    res.safepose.z = campose_pt.z();

    return true;
}

bool ExpoBase::wait_for_frontier_update_service(expo_base::frontierlock::Request &req, expo_base::frontierlock::Response &res)
{
    frontier_ptr->detection_ready();
    return true;

}

bool ExpoBase::point_state_service(expo_base::pointstate::Request &req, expo_base::pointstate::Response &res)
{
    Point3d global_point(req.globalpoint.x, req.globalpoint.y, req.globalpoint.z);
    bool rrr; 
    rrr = frontier_ptr->get_point_state(global_point);
    res.state.data = rrr;
    return true;
}

bool ExpoBase::is_global_frontier_service(expo_base::pointstate::Request &req, expo_base::pointstate::Response &res)
{
    Point3d global_point(req.globalpoint.x, req.globalpoint.y, req.globalpoint.z);
    bool rrr;
    rrr = frontier_ptr->find_global_frontier(global_point, false);
    // ROS_WARN("is_global_frontier_service: res %d", rrr);
    res.state.data = rrr;
    return true;
}


};
