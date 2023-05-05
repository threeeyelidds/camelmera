#include "state_checker/octomap_sampler.h"
#include "ros/ros.h"

namespace roadmap {

OctomapSampler::OctomapSampler(const ob::StateSpace *space,
                               const boost::shared_ptr<octomap::OcTree> octree_ptr,
                               int query_depth):
    StateSampler(space),
    QUERY_DEPTH(query_depth),
    m_octree_ptr(octree_ptr),
    low(space->as<ob::RealVectorStateSpace>()->getBounds().low),
    high(space->as<ob::RealVectorStateSpace>()->getBounds().high)
{
//    std::cout << "StateSampler: octomap sampler" << std::endl;

    int tree_max_depth = m_octree_ptr->getTreeDepth();
    double tree_resolution = m_octree_ptr->getResolution();
    bbx_cell_size = tree_resolution * pow(2, tree_max_depth - QUERY_DEPTH);

//    ROS_WARN("Sampler tree_max_depth: %d", tree_max_depth);
//    ROS_WARN("Sampler tree_resolution: %f", tree_resolution);
//    ROS_WARN("Sampler bbx_cell_size: %f", bbx_cell_size);


    assert(tree_resolution > 0 && tree_max_depth <=16);
    assert(bbx_cell_size >= tree_resolution);

    // [####|####] when the bound is located exactly at cell boundary,
    // we get the stract interior by substracting tree_resolution/2
    // [#|####|#] for such case, it turns [+|####|+] after substraction
    x_low = (int)floor((low[0]+tree_resolution/2)/bbx_cell_size);
    x_high = (int)floor((high[0]-tree_resolution/2)/bbx_cell_size);

    y_low = (int)floor((low[1]+tree_resolution/2)/bbx_cell_size);
    y_high = (int)floor((high[1]-tree_resolution/2)/bbx_cell_size);

    z_low = (int)floor((low[2]+tree_resolution/2)/bbx_cell_size);
    z_high = (int)floor((high[2]-tree_resolution/2)/bbx_cell_size);

}

void OctomapSampler::sampleUniformNear(ob::State *state, const ob::State *near, const double distance)
{
    throw ompl::Exception("OctomapSampler::sampleUniformNear", "not implemented");
}

void OctomapSampler::sampleGaussian(ompl::base::State *state, const ompl::base::State *mean, const double stdDev)
{
    throw ompl::Exception("OctomapSampler::sampleGaussian", "not implemented");
}

void OctomapSampler::sampleUniform(ompl::base::State *state)
{
    // write target sample in this variable
    double* val = static_cast<ob::RealVectorStateSpace::StateType*>(state)->values;

    // close set [lower_bound, upper_bound]
    double x = (rng_.uniformInt(x_low, x_high) + 0.5) * bbx_cell_size;
    double y = (rng_.uniformInt(y_low, y_high) + 0.5) * bbx_cell_size;
    double z = (rng_.uniformInt(z_low, z_high) + 0.5) * bbx_cell_size;

    val[0] = x;
    val[1] = y;
    val[2] = z;

//    ROS_WARN("sampling (x, y, z): %f, %f, %f", x, y, z);
//    ROS_WARN("bbx, x_low, x_high: %d, %d, %d", bbx_cell_size, x_low, x_high);

//    std::cout << "sampling (x, y, z):" << x << " " << y << " " << z << std::endl;
}


}; // NAMESPACE ROADMAP

