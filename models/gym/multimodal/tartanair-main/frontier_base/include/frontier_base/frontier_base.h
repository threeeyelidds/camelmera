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
#ifndef FRONTIER_BASE_H
#define FRONTIER_BASE_H

#include <octomap/octomap.h>
#include <visualization_msgs/Marker.h>
#include "frontier_base/frontier_type.h"

namespace frontier {

class Frontier{
public:
    virtual void set_interior_para() = 0;
    virtual void set_exterior_para() = 0;

    virtual void detect(
        const std::shared_ptr<octomap::OcTree> &global_map, 
        const std::shared_ptr<octomap::OcTree> &temp_free_map,
        const octomap::point3d &robot_pose) = 0;
    virtual void detection_result(PlannerMessage &result) = 0;
    virtual void nearest_global_frontier(Point3d robot_pos, Point3d &res_frontier_coor, Point3d &res_campose) = 0;
	virtual int count_neighbor_frontiers(Point3d center_pos, double thresh) = 0;
    virtual bool find_global_frontier(Point3d frontier_coor, bool delete_after_find) = 0;
    virtual void set_frontier_update_lock(bool flag) = 0;
    virtual bool detection_ready() = 0;
    virtual bool get_point_state(Point3d global_point) = 0;

    // tool function
    virtual void setup_visualizer(visualization_msgs::Marker &marker, double scale=1.0) = 0;
    virtual void coor2key(const Point3d &coor, Point3i &key) = 0;
    virtual void key2coor(const Point3i &key, Point3d &coor) = 0;

}; // end class Frontier

}; // end namespace frontier
#endif




