#ifndef FRONTIER_TYPE_H
#define FRONTIER_TYPE_H


#include <functional>
#include <list>
#include <data_type/point3i.h>
#include <data_type/point3d.h>
#include <octomap/octomap_types.h>
#include <data_type/planner_message.h>

#include <unordered_set>

namespace frontier {

const double COST_INF = data_type::INF;

typedef data_type::Point3i Point3i;
typedef data_type::Point3d Point3d;
typedef data_type::PlannerMessage PlannerMessage;

class BBXNode
{
public:
    Point3i local_index;    // key in local map
    Point3d global_coor;    // coordinates in global map
    Point3d global_orig;    // current origin of local map
    octomap::OcTreeKey global_key;
    std::size_t valid_pose_index;

    inline BBXNode():local_index(),global_coor(),global_orig(),valid_pose_index(0){}
    inline BBXNode(const Point3i& key, const Point3d& orig):local_index(key), global_orig(orig),valid_pose_index(0){}
    inline BBXNode(const Point3d& coor, const Point3d& orig):global_coor(coor), global_orig(orig),valid_pose_index(0){}
    inline BBXNode(const Point3i& key, const Point3d& coor, const Point3d& orig)
        :local_index(key), global_coor(coor), global_orig(orig),valid_pose_index(0){}

    inline BBXNode(const BBXNode& bbx_node)
        : local_index(bbx_node.local_index)
        , global_coor(bbx_node.global_coor)
        , global_orig(bbx_node.global_orig)
        , global_key(bbx_node.global_key)
        , valid_pose_index(bbx_node.valid_pose_index) {}

    bool operator == ( const BBXNode& other ) const {
        return this->global_key == other.global_key;
    }

private:
    inline void key2coor(std::function<void(const Point3i&, Point3d&)>& f_key2coor){
        f_key2coor(this->local_index, this->global_coor);
    }
    inline void coor2key(std::function<void(const Point3d&, Point3i&)>& f_coor2key){
        f_coor2key(this->global_coor, this->local_index);
    }
};

struct BBXNode_hash{
    std::size_t operator () ( const BBXNode& node) const {
        return oct_key_hash(node.global_key);
    }

    octomap::OcTreeKey::KeyHash oct_key_hash;
};

// typedef std::list<BBXNode> Node_List;
typedef std::unordered_set<BBXNode, BBXNode_hash> Node_List;
typedef std::shared_ptr<Node_List> node_list_ptr;

}


#endif // FRONTIER_MATH_H
