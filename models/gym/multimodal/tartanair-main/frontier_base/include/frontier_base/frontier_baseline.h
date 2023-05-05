#ifndef FRONTIER_BASELINE_H
#define FRONTIER_BASELINE_H

#include <ros/ros.h>
#include <boost/thread.hpp>
#include "frontier_base/frontier_base.h"
#include "visualization_msgs/Marker.h"
#include "std_msgs/ColorRGBA.h"
#include <math.h>

#include <vector>

namespace frontier{

using namespace octomap;
using namespace std;

class FrontierBaseline: public Frontier
{
public:
    FrontierBaseline();
    //~FrontierBaseline();

    // implementation of the Frontier Interface
    void set_exterior_para();  // get exterior para from map via rospara server
    void set_interior_para();  // get interior para via parsing launch file
    void detect(
        const std::shared_ptr<OcTree> &global_map, 
        const std::shared_ptr<OcTree> &temp_free_map,
        const octomap::point3d &robot_pose);

    // definition of frontiers
    void detect_local_free_node(const Point3d &local_orig, std::vector<BBXNode> &vector_free);

    void detect_local_frontier(
        std::vector<BBXNode> &vector_free, 
        node_list_ptr local_frontier,
        std::size_t valid_pose_index);
    void merge_local_frontier(node_list_ptr local_frontier);
    // inflate the cost map to keep away from obstacles
    void cost_inflation(int iter_count, int inflate_val);

    // publish bbx node
    void publish_unknown_node();
    void publish_global_frontier_node();
    void publish_local_frontier_node();
    void publish_free_node(std::vector<BBXNode> &vector_free);
    void setup_visualizer(visualization_msgs::Marker &marker, double scale=1.0);
    void publish_cost_node();

    // utility function
    void coor2key(const Point3d &coor, Point3i &key);
    void key2coor(const Point3i &key, Point3d &coor);
    bool check_front_wave(const Point3d &origin, const Point3i &neibor_key);

public:
    typedef boost::shared_lock<boost::shared_mutex> ReadLock;
    typedef boost::unique_lock<boost::shared_mutex> WriteLock;
    boost::shared_mutex m_rw_lock;
    void detection_result(data_type::PlannerMessage& result);
    // the nearest global frontier wrt one location
    void nearest_global_frontier(Point3d robot_pos, Point3d &res_frontier_coor, Point3d &res_campose);
    int count_neighbor_frontiers(Point3d center_pos, double thresh);
    bool find_global_frontier(Point3d frontier_coor, bool delete_after_find);
    bool detection_ready();
    bool get_point_state(Point3d global_point);

    void set_frontier_update_lock(bool flag);

protected:
    ros::NodeHandle m_nh;
    ros::Publisher m_free_pub;
    ros::Publisher m_unknown_pub;
    ros::Publisher m_local_frontier_pub;
    ros::Publisher m_global_frontier_pub;
    ros::Publisher m_cost_pub;


public:
    int m_bbx_depth;        // at which depth calculate the frontier
    int m_bbx_upper_orig;   // the distance above the map's origin (x-y plane)
    int m_bbx_below_orig;   // the distance below the map's origin (x-y plane)

    double m_map_x_max, m_map_x_min;
    double m_map_y_max, m_map_y_min;
    double m_map_z_max, m_map_z_min;

    int neighbor_count_thresh;  // delete frontiers that don't have enough neighbors

    // inflate the costmap to keep path away from obstacles
    int cost_inflation_iter; // number of iteration of the inflation
    double cost_inflation_val; // amount of added cost  

private:
    /**
     * @brief the follwing parameter will be inherited from octotree
     */
    bool m_exterior_para;
    double m_sensor_range;      // the adopted sensor range when bulding the tree
    double m_tree_max_depth;    // the maximum depth of the octotree
    double m_tree_resolution;   // the finest resolution of the octotree
    string m_world_frame_id;

    /**
     * @brief parameter calculated from ros parameter and exterior parameter
     */
    double m_bbx_cell_size;     // cell size in the query depth
    Point3i m_bbx_size;         // the size of 3D grid
    Point3i m_offset;           // transform w.r.t local map

    /**
     * @brief curr_orig
     */
    Point3i m_last_key;  // only key is changed by one, the frontier detection will be performed
    Point3d curr_pose;
    Point3d curr_orig;
    std::shared_ptr<octomap::OcTree> global_map_ptr;
    std::shared_ptr<octomap::OcTree> temp_free_map_ptr;
    int local_frontier_size;
    std::shared_ptr<Node_List> local_frontier_ptr;
    std::shared_ptr<Node_List> global_frontier_ptr;
    std::vector<octomap::point3d> valid_poses;

    /**
     * @brief planner related local map
     */
    boost::shared_ptr<double[]> m_local_cost_map;  // cost-to-go map
    boost::shared_ptr<int[]> m_local_info_map;  // (-1)unknown/(0)-obstacle/(1)free map

    // lock the frontier update when returning to the same pose
    // to avoid generate same unreachable frontiers again
    bool frontier_update_lock;
};

};
#endif // FRONTIER_BASELINE_H
