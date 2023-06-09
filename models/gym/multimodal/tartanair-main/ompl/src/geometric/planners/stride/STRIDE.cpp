/** * @author: AirLab / Field Robotics Center
 *
 * @attention Copyright (C) 2016
 * @attention Carnegie Mellon University
 * @attention All rights reserved
 *
 * @attention LIMITED RIGHTS:
 * @attention The US Government is granted Limited Rights to this Data.
 *            Use, duplication, or disclosure is subject to the
 *            restrictions as stated in Agreement AFS12-1642.
 */
/*********************************************************************
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
 *********************************************************************/

/* Author: Bryant Gipson, Mark Moll, Ioan Sucan */

#include "ompl/geometric/planners/stride/STRIDE.h"
// enable sampling from the GNAT data structure
#define GNAT_SAMPLER
#include "ompl/datastructures/NearestNeighborsGNAT.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"
#include <limits>
#include <cassert>

ompl::geometric::STRIDE::STRIDE(const base::SpaceInformationPtr &si,
    bool useProjectedDistance,
    unsigned int degree, unsigned int minDegree,
    unsigned int maxDegree, unsigned int maxNumPtsPerLeaf, double estimatedDimension)
    : base::Planner(si, "STRIDE"), goalBias_(0.05), maxDistance_(0.),
    useProjectedDistance_(useProjectedDistance),
    degree_(degree), minDegree_(minDegree), maxDegree_(maxDegree),
    maxNumPtsPerLeaf_(maxNumPtsPerLeaf), estimatedDimension_(estimatedDimension),
    minValidPathFraction_(0.2)
{
    specs_.approximateSolutions = true;

    if (estimatedDimension_ < 1.)
        estimatedDimension_ = si->getStateDimension();

    Planner::declareParam<double>("range", this, &STRIDE::setRange, &STRIDE::getRange, "0.:1.:10000.");
    Planner::declareParam<double>("goal_bias", this, &STRIDE::setGoalBias, &STRIDE::getGoalBias, "0.:.05:1.");
    Planner::declareParam<bool>("use_projected_distance", this, &STRIDE::setUseProjectedDistance, &STRIDE::getUseProjectedDistance, "0,1");
    Planner::declareParam<unsigned int>("degree", this, &STRIDE::setDegree, &STRIDE::getDegree, "2:20");
    Planner::declareParam<unsigned int>("max_degree", this, &STRIDE::setMaxDegree, &STRIDE::getMaxDegree, "2:20");
    Planner::declareParam<unsigned int>("min_degree", this, &STRIDE::setMinDegree, &STRIDE::getMinDegree, "2:20");
    Planner::declareParam<unsigned int>("max_pts_per_leaf", this, &STRIDE::setMaxNumPtsPerLeaf, &STRIDE::getMaxNumPtsPerLeaf, "1:200");
    Planner::declareParam<double>("estimated_dimension", this, &STRIDE::setEstimatedDimension, &STRIDE::getEstimatedDimension, "1.:30.");
    Planner::declareParam<double>("min_valid_path_fraction", this, &STRIDE::setMinValidPathFraction, &STRIDE::getMinValidPathFraction, "0.:.05:1.");
}

ompl::geometric::STRIDE::~STRIDE(void)
{
    freeMemory();
}

void ompl::geometric::STRIDE::setup(void)
{
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configureProjectionEvaluator(projectionEvaluator_);
    sc.configurePlannerRange(maxDistance_);
    setupTree();
}

void ompl::geometric::STRIDE::setupTree(void)
{
    tree_.reset(new NearestNeighborsGNAT<Motion*>(degree_, minDegree_, maxDegree_, maxNumPtsPerLeaf_, estimatedDimension_));
    if (useProjectedDistance_)
        tree_->setDistanceFunction(boost::bind(&STRIDE::projectedDistanceFunction, this, _1, _2));
    else
        tree_->setDistanceFunction(boost::bind(&STRIDE::distanceFunction, this, _1, _2));
}

void ompl::geometric::STRIDE::clear(void)
{
    Planner::clear();
    sampler_.reset();
    freeMemory();
    setupTree();
}

void ompl::geometric::STRIDE::freeMemory(void)
{
    if (tree_)
    {
        std::vector<Motion*> motions;
        tree_->list(motions);
        for (std::size_t i = 0 ; i < motions.size() ; ++i)
        {
            if (motions[i]->state)
                si_->freeState(motions[i]->state);
            delete motions[i];
        }
        tree_.reset();
    }
}

ompl::base::PlannerStatus ompl::geometric::STRIDE::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    base::Goal                   *goal = pdef_->getGoal().get();
    base::GoalSampleableRegion *goal_s = dynamic_cast<base::GoalSampleableRegion*>(goal);

    while (const base::State *st = pis_.nextStart())
    {
        Motion *motion = new Motion(si_);
        si_->copyState(motion->state, st);
        addMotion(motion);
    }

    if (tree_->size() == 0)
    {
        OMPL_ERROR("There are no valid initial states!");
        return base::PlannerStatus::INVALID_START;
    }

    if (!sampler_)
        sampler_ = si_->allocValidStateSampler();

    OMPL_INFORM("Starting with %u states", tree_->size());

    Motion *solution  = NULL;
    Motion *approxsol = NULL;
    double  approxdif = std::numeric_limits<double>::infinity();
    base::State *xstate = si_->allocState();

    while (ptc == false)
    {
        /* Decide on a state to expand from */
        Motion *existing = selectMotion();
        assert(existing);

        /* sample random state (with goal biasing) */
        if (goal_s && rng_.uniform01() < goalBias_ && goal_s->canSample())
            goal_s->sampleGoal(xstate);
        else
            if (!sampler_->sampleNear(xstate, existing->state, maxDistance_))
                continue;

        std::pair<base::State*, double> fail(xstate, 0.0);
        bool keep = si_->checkMotion(existing->state, xstate, fail) || fail.second > minValidPathFraction_;

        if (keep)
        {
            /* create a motion */
            Motion *motion = new Motion(si_);
            si_->copyState(motion->state, xstate);
            motion->parent = existing;

            addMotion(motion);
            double dist = 0.0;
            bool solved = goal->isSatisfied(motion->state, &dist);
            if (solved)
            {
                approxdif = dist;
                solution = motion;
                break;
            }
            if (dist < approxdif)
            {
                approxdif = dist;
                approxsol = motion;
            }
        }
    }

    bool solved = false;
    bool approximate = false;
    if (solution == NULL)
    {
        solution = approxsol;
        approximate = true;
    }

    if (solution != NULL)
    {
        /* construct the solution path */
        std::vector<Motion*> mpath;
        while (solution != NULL)
        {
            mpath.push_back(solution);
            solution = solution->parent;
        }

        /* set the solution path */
        PathGeometric *path = new PathGeometric(si_);
        for (int i = mpath.size() - 1 ; i >= 0 ; --i)
            path->append(mpath[i]->state);
        pdef_->addSolutionPath(base::PathPtr(path), approximate, approxdif);
        solved = true;
    }

    si_->freeState(xstate);

    OMPL_INFORM("Created %u states", tree_->size());

    return base::PlannerStatus(solved, approximate);
}

void ompl::geometric::STRIDE::addMotion(Motion *motion)
{
    tree_->add(motion);
}

ompl::geometric::STRIDE::Motion* ompl::geometric::STRIDE::selectMotion(void)
{
    return tree_->sample(rng_);
}

void ompl::geometric::STRIDE::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion*> motions;
    tree_->list(motions);
    for (std::vector<Motion*>::iterator it=motions.begin(); it!=motions.end(); it++)
    {
        if((*it)->parent == NULL)
            data.addStartVertex(base::PlannerDataVertex((*it)->state,1));
        else
            data.addEdge(base::PlannerDataVertex((*it)->parent->state,1),base::PlannerDataVertex((*it)->state,1));
    }
}
