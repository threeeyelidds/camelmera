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
*  Copyright (c) 2011, Willow Garage, Inc.
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

/* Author: Ioan Sucan */

#include "ompl/tools/multiplan/ParallelPlan.h"
#include "ompl/geometric/PathHybridization.h"

ompl::tools::ParallelPlan::ParallelPlan(const base::ProblemDefinitionPtr &pdef) :
    pdef_(pdef), phybrid_(new geometric::PathHybridization(pdef->getSpaceInformation()))
{
}

ompl::tools::ParallelPlan::~ParallelPlan(void)
{
}

void ompl::tools::ParallelPlan::addPlanner(const base::PlannerPtr &planner)
{
    if (planner && planner->getSpaceInformation().get() != pdef_->getSpaceInformation().get())
        throw Exception("Planner instance does not match space information");
    if (planner->getProblemDefinition().get() != pdef_.get())
        planner->setProblemDefinition(pdef_);
    planners_.push_back(planner);
}

void ompl::tools::ParallelPlan::addPlannerAllocator(const base::PlannerAllocator &pa)
{
    base::PlannerPtr planner = pa(pdef_->getSpaceInformation());
    planner->setProblemDefinition(pdef_);
    planners_.push_back(planner);
}

void ompl::tools::ParallelPlan::clearPlanners(void)
{
    planners_.clear();
}

void ompl::tools::ParallelPlan::clearHybridizationPaths(void)
{
    phybrid_->clear();
}

ompl::base::PlannerStatus ompl::tools::ParallelPlan::solve(double solveTime, bool hybridize)
{
    return solve(solveTime, 1, planners_.size(), hybridize);
}

ompl::base::PlannerStatus ompl::tools::ParallelPlan::solve(double solveTime, std::size_t minSolCount, std::size_t maxSolCount, bool hybridize)
{
    return solve(base::timedPlannerTerminationCondition(solveTime, std::min(solveTime / 100.0, 0.1)), minSolCount, maxSolCount, hybridize);
}


ompl::base::PlannerStatus ompl::tools::ParallelPlan::solve(const base::PlannerTerminationCondition &ptc, bool hybridize)
{
    return solve(ptc, 1, planners_.size(), hybridize);
}

ompl::base::PlannerStatus ompl::tools::ParallelPlan::solve(const base::PlannerTerminationCondition &ptc, std::size_t minSolCount, std::size_t maxSolCount, bool hybridize)
{
    if (!pdef_->getSpaceInformation()->isSetup())
        pdef_->getSpaceInformation()->setup();
    foundSolCount_ = 0;

    time::point start = time::now();
    std::vector<boost::thread*> threads(planners_.size());
    if (hybridize)
        for (std::size_t i = 0 ; i < threads.size() ; ++i)
            threads[i] = new boost::thread(boost::bind(&ParallelPlan::solveMore, this, planners_[i].get(), minSolCount, maxSolCount, &ptc));
    else
        for (std::size_t i = 0 ; i < threads.size() ; ++i)
            threads[i] = new boost::thread(boost::bind(&ParallelPlan::solveOne, this, planners_[i].get(), minSolCount, &ptc));

    for (std::size_t i = 0 ; i < threads.size() ; ++i)
    {
        threads[i]->join();
        delete threads[i];
    }

    if (hybridize)
    {
        if (phybrid_->pathCount() > 1)
            if (const base::PathPtr &hsol = phybrid_->getHybridPath())
            {
                geometric::PathGeometric *pg = static_cast<geometric::PathGeometric*>(hsol.get());
                double difference = 0.0;
                bool approximate = !pdef_->getGoal()->isSatisfied(pg->getStates().back(), &difference);
                pdef_->addSolutionPath(hsol, approximate, difference);
            }
    }

    OMPL_INFORM("Solution found in %f seconds", time::seconds(time::now() - start));

    return base::PlannerStatus(pdef_->hasSolution(), pdef_->hasApproximateSolution());
}

void ompl::tools::ParallelPlan::solveOne(base::Planner *planner, std::size_t minSolCount, const base::PlannerTerminationCondition *ptc)
{
    OMPL_DEBUG("Starting %s", planner->getName().c_str());
    time::point start = time::now();
    if (planner->solve(*ptc))
    {
        double duration = time::seconds(time::now() - start);
        foundSolCountLock_.lock();
        unsigned int nrSol = ++foundSolCount_;
        foundSolCountLock_.unlock();
        if (nrSol >= minSolCount)
            ptc->terminate();
        OMPL_DEBUG("Solution found by %s in %lf seconds", planner->getName().c_str(), duration);
    }
}

void ompl::tools::ParallelPlan::solveMore(base::Planner *planner, std::size_t minSolCount, std::size_t maxSolCount, const base::PlannerTerminationCondition *ptc)
{
    time::point start = time::now();
    if (planner->solve(*ptc))
    {
        double duration = time::seconds(time::now() - start);
        foundSolCountLock_.lock();
        unsigned int nrSol = ++foundSolCount_;
        foundSolCountLock_.unlock();

        if (nrSol >= maxSolCount)
            ptc->terminate();

        OMPL_DEBUG("Solution found by %s in %lf seconds", planner->getName().c_str(), duration);

        const std::vector<base::PlannerSolution> &paths = pdef_->getSolutions();

        boost::mutex::scoped_lock slock(phlock_);
        start = time::now();
        unsigned int attempts = 0;
        for (std::size_t i = 0 ; i < paths.size() ; ++i)
            attempts += phybrid_->recordPath(paths[i].path_, false);

        if (phybrid_->pathCount() >= minSolCount)
            phybrid_->computeHybridPath();

        duration = time::seconds(time::now() - start);
        OMPL_DEBUG("Spent %f seconds hybridizing %u solution paths (attempted %u connections between paths)", duration, (unsigned int)phybrid_->pathCount(), attempts);
    }
}
