#ifndef ROADMAP_H
#define ROADMAP_H

#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <dynamicEDT3D/dynamicEDTOctomap.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/Marker.h>

#include <ompl/base/StateSpace.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/ProblemDefinition.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/SpaceInformation.h>

#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/bitstar/BITstar.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/SimpleSetup.h>

#include "state_checker/octomap_sampler.h"
#include "state_checker/octomap_checker.h"
#include "objectives/clearance_objective.h"
#include "controls/kinematic_model.h"
#include "roadmap_generator/road.h"
#include "roadmap_generator/roads.h"
#include "roadmap_generator/distmap.h"
#include "roadmap_generator/endpoints.h"
#include "roadmap_generator/smooth.h"

#include <time.h>
#include <ros/ros.h>
#include <iostream>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

#include <data_type/point3i.h>
#include <data_type/point3d.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

namespace roadmap{

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef data_type::Point3d Point3d;
typedef data_type::Point3i Point3i;


typedef bg::model::point<double, 3, bg::cs::cartesian> CellPoint;


class Roadmap
{
public:
    Roadmap();
    void setup_visualizer(visualization_msgs::Marker &marker);

    /**
     * @brief octomap_callback: receive octomap message, perform euclidean distance tranformation,
     * and calculate query index bound at a specified tree depth
     * @param octomap_msg: it's necessary to transform it to OcTree type
     */
    void octomap_callback(const octomap_msgs::OctomapPtr &octomap_msg);

    bool roadmap_service(roadmap_generator::road::Request &req, roadmap_generator::road::Response &res);
    bool distmap_service(roadmap_generator::distmap::Request &req, roadmap_generator::distmap::Response &res);
    bool plan_paths_service(roadmap_generator::roads::Request &req, roadmap_generator::roads::Response &res);
    // bool endpoint_service(roadmap_generator::endpoint::Request &req, roadmap_generator::endpoint::Response &res);
    bool sample_nodes_service(roadmap_generator::endpoints::Request &req, roadmap_generator::endpoints::Response &res);
    bool path_smooth_service(roadmap_generator::smooth::Request &req, roadmap_generator::smooth::Response &res);

    double generate_geometry_roadmap(geometry_msgs::Pose &init_pose, geometry_msgs::Pose &goal_pose, geometry_msgs::PoseArray &path);
    double generate_control_roadmap(geometry_msgs::Pose &init_pose, geometry_msgs::Pose &goal_pose, geometry_msgs::PoseArray &path);
    ob::StateSamplerPtr alloc_octomap_sampler(const ompl::base::StateSpace *state);

    void height_range(double range_x_min, double range_x_max, 
                            double range_y_min, double range_y_max, 
                            double range_z_min, double range_z_max);
    bool sample_node(Point3d& resnode, double range_x_min, double range_x_max, 
                            double range_y_min, double range_y_max, 
                            double range_z_min, double range_z_max);

private:
    ros::NodeHandle m_nh;
    ros::Publisher m_roadmap_pub;
    ros::Subscriber m_octomap_sub;
    ros::ServiceServer m_plan_service;
    ros::ServiceServer m_dist_service;
    ros::ServiceServer m_plan_paths_service;
    // ros::ServiceServer m_endpoint_service;
    ros::ServiceServer m_sample_nodes_service;
    ros::ServiceServer m_path_smooth_service;

private:
    double PLANNING_TIME;
    double RRT_STAR_RANGE;       // the maximum length of a motion to be added to the tree
    double CHECKER_RESOLUTION;  // state validation checker resolution for obstacal checking
    double OMPL_X_BOUND, OMPL_Y_BOUND, OMPL_Z_BOUND;
    double DYNAMIC_EDT_MAXDIST; // dynamicEDTOctotree maxdist
    double CLEARANCE_WEIGHT; 
    bool USE_HEIGHT_MAP;
    
    boost::shared_ptr<octomap::OcTree> m_octree_ptr;
    boost::shared_ptr<DynamicEDTOctomap> m_edtmap_ptr;
    typedef boost::unique_lock<boost::mutex> ExclusiveLock;
    boost::mutex m_mutex;
    bool m_map_setup;

    boost::shared_ptr<ompl::geometric::PathSimplifier> m_pathsimplifier_ptr; // use for simplify the path
    ompl::base::SpaceInformationPtr m_spaceinfo_ptr; 

private:
    // should between 0-15
    int SAMPLE_QUERY_DEPTH;
    double CHECKER_FREE_VALUE_THRESH;
    double CHECKER_DIST_THRESH;

    // map boundary
    double MAP_X_MIN, MAP_X_MAX;
    double MAP_Y_MIN, MAP_Y_MAX;
    double MAP_Z_MIN, MAP_Z_MAX;

    // // map boundary index at query level
    // int m_query_x_min, m_query_x_max;
    // int m_query_y_min, m_query_y_max;
    // int m_query_z_min, m_query_z_max;

    // calculated after receiving the octomap
    int m_tree_max_depth;
    double m_bbx_cell_size;
    double m_tree_resolution;

    // parameters for sampling the endpoints
    double GRID_SIZE;
    int GRID_SMOOTH_STHENGTH;
    int PATH_SMOOTH_STEP;
    double PATH_SMOOTH_RATIO;
    double OPEN_HEIGHT;
    double SAMPLE_PROB_DROP; 
    double SAMPLE_PROB_LOW_THRESH; 
    cv::Mat lowMap, highMap, accProb;
    int xgridnum, ygridnum;
    double accProbMean; // monitor the accProb value
};

}



#endif // ROADMAP_H
