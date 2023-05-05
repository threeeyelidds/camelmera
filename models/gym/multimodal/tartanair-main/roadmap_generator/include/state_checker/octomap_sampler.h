#ifndef OCTOMAP_SAMPLER_H
#define OCTOMAP_SAMPLER_H

#include <ompl/base/SpaceInformation.h>
#include <ompl/base/ValidStateSampler.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <octomap/octomap.h>


namespace roadmap {

namespace ob = ompl::base;

class OctomapSampler: public ob::StateSampler
{
public:
    OctomapSampler(const ompl::base::StateSpace *space,
                   const boost::shared_ptr<octomap::OcTree> octree_ptr,
                   int query_level);


    /** \brief Sample a state near another, within specified distance */
    void sampleUniformNear(ob::State *state, const ob::State *near, const double distance);

    /** \brief Sample a state using a Gaussian distribution with given \e mean and standard deviation (\e stdDev) */
    void sampleGaussian(ob::State *state, const ob::State *mean, const double stdDev);

    // Generate a sample in the valid part of the R^3 state space
    // Valid states satisfy the following constraints:
    // [x, y, z] is the cube center at different query level
    void sampleUniform(ob::State *state) override;


protected:
    ompl::RNG rng_;
    int QUERY_DEPTH;
    boost::shared_ptr<octomap::OcTree> m_octree_ptr;

    std::vector<double> low;
    std::vector<double> high;

private:
    double bbx_cell_size;
    int x_low, x_high;
    int y_low, y_high;
    int z_low, z_high;

};




}



#endif // OCTOMAP_SAMPLER_H
