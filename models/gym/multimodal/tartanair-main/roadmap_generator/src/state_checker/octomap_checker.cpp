#include "state_checker/octomap_checker.h"

namespace roadmap {

roadmap::OctomapChecker::OctomapChecker(const ompl::base::SpaceInformationPtr &si,
                                        const boost::shared_ptr<octomap::OcTree> &octree,
                                        const boost::shared_ptr<DynamicEDTOctomap> &edtmap):
    StateValidityChecker(si),
    QUERY_DEPTH(13),
    FREE_VALUE_THRESH(-1),
    m_tree_initialized(false),
    m_octree_ptr(octree),
    m_edtmap_ptr(edtmap)
{
    ros::NodeHandle private_nh = ros::NodeHandle("~");
    private_nh.param("checker_free_value_thresh", FREE_VALUE_THRESH, FREE_VALUE_THRESH);
    assert(FREE_VALUE_THRESH < 0);
    private_nh.param("checker_query_depth", QUERY_DEPTH, QUERY_DEPTH);
    assert(QUERY_DEPTH > 0 && QUERY_DEPTH <= 16);
    ROS_WARN("checker_free_value_thresh is %f", FREE_VALUE_THRESH);
    ROS_WARN("checker_query_depth is %d", QUERY_DEPTH);

//    std::cout << "octree pointer number is " << m_octree_ptr.use_count() << std::endl;
}

//roadmap::OctomapChecker::OctomapChecker(const ompl::control::SpaceInformationPtr &si,
//                                        const boost::shared_ptr<octomap::OcTree> &octree):
//    StateValidityChecker(si),
//    QUERY_DEPTH(13),
//    FREE_VALUE_THRESH(-1),
//    m_tree_initialized(false),
//    m_octree_ptr(octree)
//{
//    ros::NodeHandle private_nh = ros::NodeHandle("~");
//    private_nh.param("free_value_thresh", FREE_VALUE_THRESH, FREE_VALUE_THRESH);
//    assert(FREE_VALUE_THRESH < 0);
//    private_nh.param("query_depth", QUERY_DEPTH, QUERY_DEPTH);
//    assert(QUERY_DEPTH > 0 && QUERY_DEPTH <= 16);
//    std::cout << "octree pointer number is " << m_octree_ptr.use_count() << std::endl;
//}


bool OctomapChecker::isValid(const ompl::base::State *state) const
{
    // assume m_octree_ptr is not nullptr
    const ob::RealVectorStateSpace::StateType* state_ptr = state->as<ob::RealVectorStateSpace::StateType>();

    if(!si_->satisfiesBounds(state_ptr)){
        return false;
    }
    // Note: the query resolution for state checker should be less than the finest resolution
    // even though, we cannot ensure the added edge is obstacle free. For instance, if len(x-t)
    // is less than the checker resolution, obstacle would still be gone through.

    // ####### #: free
    // ####t## t: target
    // ####o## o: obstacle
    // #####x# x: current

    // the first condition is to smaple the grid center
    // the second condition is to ensure clearance in a neighborhood
    octomap::OcTreeNode* node = m_octree_ptr->search(state_ptr->values[0],
            state_ptr->values[1],state_ptr->values[2], QUERY_DEPTH);

    if(node == nullptr){
        return false;
    }else{
        return (node->getValue() < FREE_VALUE_THRESH);
    }
}

double OctomapChecker::clearance(const ompl::base::State *state) const
{
    return edtmap_clearance(state);

}

double OctomapChecker::rtree_clearance(const ompl::base::State *state) const
{
    using namespace octomap;

    namespace bg = boost::geometry;
    namespace bgi = boost::geometry::index;
    typedef bg::model::point<double, 3, bg::cs::cartesian> CellPoint;

    // Downcast state into the RealVectorStateSpace.
    const ob::RealVectorStateSpace::StateType* state_ptr = state->as<ob::RealVectorStateSpace::StateType>();
    double clearance_dist = 2;

    // if the state coresponds to an obstacle or void voxel
    point3d cur_loc(state_ptr->values[0], state_ptr->values[1], state_ptr->values[2]);
    OcTreeNode* cur_loc_ptr = m_octree_ptr->search(cur_loc.x(), cur_loc.y(), cur_loc.z());
    if(cur_loc_ptr == nullptr || cur_loc_ptr->getValue() >= 0){
        return -clearance_dist;
    }


    // extract a cube around current robot location
    point3d low_bound = cur_loc - point3d(clearance_dist, clearance_dist, clearance_dist);
    point3d high_bound = cur_loc + point3d(clearance_dist, clearance_dist, clearance_dist);

    OcTreeKey lw = m_octree_ptr->coordToKey(low_bound);
    OcTreeKey hg = m_octree_ptr->coordToKey(high_bound);


    // build up the r-tree
    bgi::rtree<CellPoint, bgi::rstar<16>> rtree;
    for(key_type i = lw[0], x = 0; i <= hg[0]; i++, x++){
        for(key_type j = lw[1], y = 0; j<=hg[1]; j++, y++){
            for(key_type k = lw[2], z = 0; k<=hg[2]; k++, z++){

                OcTreeNode* node = m_octree_ptr->search(OcTreeKey(i, j, k));
                if(node==nullptr || node->getValue() >= 0){
                    point3d coor = m_octree_ptr->keyToCoord(OcTreeKey(i, j, k));
                    rtree.insert(CellPoint(coor.x(), coor.y(), coor.z()));
                }

            } // end k
        } // end j
    } // end i

    if(rtree.empty()){
        return clearance_dist;
    }

    // query the nearest obstacle or void voxel
    std::vector<CellPoint> result;
    rtree.query(bgi::nearest(CellPoint(cur_loc.x(), cur_loc.y(), cur_loc.z()), 1), std::back_inserter(result));
    point3d obs_loc(result[0].get<0>(),result[0].get<1>(), result[0].get<2>());

    return (cur_loc - obs_loc).norm();

}

double OctomapChecker::edtmap_clearance(const ompl::base::State *state) const
{
    using namespace octomap;

    // Downcast state into the RealVectorStateSpace.
    const ob::RealVectorStateSpace::StateType* state_ptr = state->as<ob::RealVectorStateSpace::StateType>();

    // if the state coresponds to an obstacle or void voxel
    point3d cur_loc(state_ptr->values[0], state_ptr->values[1], state_ptr->values[2]);
    float dist = m_edtmap_ptr->getDistance(cur_loc);
//    ROS_WARN("Distance quried is %f", dist);
    if(dist < 0){
        ROS_ERROR("Found points that are out of boundary!");
        dist = 0;
    }
    return dist;
}


void OctomapChecker::set_octree(const std::shared_ptr<octomap::OcTree> &octree_ptr)
{
    assert(octree_ptr != nullptr);
    m_tree_max_depth = octree_ptr->getTreeDepth();
    m_tree_resolution = octree_ptr->getResolution();
    m_bbx_cell_size = m_tree_resolution * pow(2, m_tree_max_depth - QUERY_DEPTH);
    m_tree_initialized = true;
}

void OctomapChecker::cube_center(octomap::point3d &coor)
{
    coor.x() = floor(coor.x()/m_bbx_cell_size);
    coor.y() = floor(coor.y()/m_bbx_cell_size);
    coor.z() = floor(coor.z()/m_bbx_cell_size);
}


} // NAMESPACE

