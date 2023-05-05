#ifndef OCTOMAP_CHECKER_H
#define OCTOMAP_CHECKER_H

#include <memory>
#include <ros/ros.h>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <octomap/octomap.h>
#include <data_type/point3d.h>
#include <dynamicEDT3D/dynamicEDTOctomap.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/spaces/ReedsSheppStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace roadmap {

namespace ob = ompl::base;
namespace oc = ompl::control;
typedef data_type::Point3d Point3d;


class OctomapChecker: public ob::StateValidityChecker
{
public:
    OctomapChecker(const ob::SpaceInformationPtr& si,
                   const boost::shared_ptr<octomap::OcTree>& octree,
                   const boost::shared_ptr<DynamicEDTOctomap>& edtmap);
//    OctomapChecker(const oc::SpaceInformationPtr& si, const boost::shared_ptr<octomap::OcTree>& octree);


    // Returns whether the given state's position overlaps obstacles
    bool isValid(const ob::State* state) const;

    // Returns the distance from the given state's position to obstacle.
    double clearance(const ob::State* state) const;
    double rtree_clearance(const ob::State* state) const;
    double edtmap_clearance(const ob::State* state) const;


    // Setting up the shared pointer of octree
    void set_octree(const std::shared_ptr<octomap::OcTree>& octree_ptr);
    void cube_center(octomap::point3d &coor);

private:

    int QUERY_DEPTH;
    double FREE_VALUE_THRESH;
    bool m_tree_initialized;
    double m_bbx_cell_size;     // cell size in the query depth
    double m_tree_max_depth;    // the maximum depth of the octotree
    double m_tree_resolution;   // the finest resolution of the octotree

    boost::shared_ptr<octomap::OcTree> m_octree_ptr;
    boost::shared_ptr<DynamicEDTOctomap> m_edtmap_ptr;
};

}



#endif // OCTOMAPCHECKER_H
