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
*  Copyright (c) 2010, Rice University
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
*   * Neither the name of the Rice University nor the names of its
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

#include "ompl/base/Planner.h"
#include "ompl/util/Exception.h"
#include "ompl/base/goals/GoalSampleableRegion.h"
#include <sstream>
#include <boost/thread.hpp>

ompl::base::Planner::Planner(const SpaceInformationPtr &si, const std::string &name) :
    si_(si), pis_(this), name_(name), setup_(false)
{
    if (!si_)
        throw Exception(name_, "Invalid space information instance for planner");
}

const ompl::base::PlannerSpecs& ompl::base::Planner::getSpecs(void) const
{
    return specs_;
}

const std::string& ompl::base::Planner::getName(void) const
{
    return name_;
}

void ompl::base::Planner::setName(const std::string &name)
{
    name_ = name;
}

const ompl::base::SpaceInformationPtr&  ompl::base::Planner::getSpaceInformation(void) const
{
    return si_;
}

const ompl::base::ProblemDefinitionPtr& ompl::base::Planner::getProblemDefinition(void) const
{
    return pdef_;
}

void ompl::base::Planner::setProblemDefinition(const ProblemDefinitionPtr &pdef)
{
    pdef_ = pdef;
    pis_.update();
}

const ompl::base::PlannerInputStates& ompl::base::Planner::getPlannerInputStates(void) const
{
    return pis_;
}

void ompl::base::Planner::setup(void)
{
    if (!si_->isSetup())
    {
        //OMPL_DEBUG("%s: Space information setup was not yet called. Calling now.", getName().c_str());
        si_->setup();
    }

    if (setup_)
        OMPL_WARN("%s: Planner setup called multiple times", getName().c_str());
    else
        setup_ = true;
}

void ompl::base::Planner::checkValidity(void)
{
    if (!isSetup())
        setup();
    pis_.checkValidity();
}

bool ompl::base::Planner::isSetup(void) const
{
    return setup_;
}

void ompl::base::Planner::clear(void)
{
    pis_.clear();
    pis_.update();
}

void ompl::base::Planner::getPlannerData(PlannerData & /*data*/) const
{
}

ompl::base::PlannerStatus ompl::base::Planner::solve(const PlannerTerminationConditionFn &ptc, double checkInterval)
{
    return solve(PlannerTerminationCondition(ptc, checkInterval));
}

ompl::base::PlannerStatus ompl::base::Planner::solve(double solveTime)
{
    if (solveTime < 1.0)
        return solve(timedPlannerTerminationCondition(solveTime));
    else
        return solve(timedPlannerTerminationCondition(solveTime, std::min(solveTime / 100.0, 0.1)));
}

void ompl::base::Planner::printProperties(std::ostream &out) const
{
    out << "Planner " + getName() + " specs:" << std::endl;
    out << "Multithreaded:                 " << (getSpecs().multithreaded ? "Yes" : "No") << std::endl;
    out << "Reports approximate solutions: " << (getSpecs().approximateSolutions ? "Yes" : "No") << std::endl;
    out << "Can optimize solutions:        " << (getSpecs().optimizingPaths ? "Yes" : "No") << std::endl;
    out << "Aware of the following parameters:";
    std::vector<std::string> params;
    params_.getParamNames(params);
    for (unsigned int i = 0 ; i < params.size() ; ++i)
        out << " " << params[i];
    out << std::endl;
}

void ompl::base::Planner::printSettings(std::ostream &out) const
{
    out << "Declared parameters for planner " << getName() << ":" << std::endl;
    params_.print(out);
}

void ompl::base::PlannerInputStates::clear(void)
{
    if (tempState_)
    {
        si_->freeState(tempState_);
        tempState_ = NULL;
    }
    addedStartStates_ = 0;
    sampledGoalsCount_ = 0;
    pdef_ = NULL;
    si_ = NULL;
}

void ompl::base::PlannerInputStates::restart(void)
{
    addedStartStates_ = 0;
    sampledGoalsCount_ = 0;
}

bool ompl::base::PlannerInputStates::update(void)
{
    if (!planner_)
        throw Exception("No planner set for PlannerInputStates");
    return use(planner_->getProblemDefinition());
}

void ompl::base::PlannerInputStates::checkValidity(void) const
{
    std::string error;

    if (!pdef_)
        error = "Problem definition not specified";
    else
    {
        if (pdef_->getStartStateCount() <= 0)
            error = "No start states specified";
        else
            if (!pdef_->getGoal())
                error = "No goal specified";
    }

    if (!error.empty())
    {
        if (planner_)
            throw Exception(planner_->getName(), error);
        else
            throw Exception(error);
    }
}

bool ompl::base::PlannerInputStates::use(const ProblemDefinitionPtr &pdef)
{
    if (pdef)
        return use(pdef.get());
    else
    {
        clear();
        return true;
    }
}

bool ompl::base::PlannerInputStates::use(const ProblemDefinition *pdef)
{
    if (pdef_ != pdef)
    {
        clear();
        pdef_ = pdef;
        si_ = pdef->getSpaceInformation().get();
        return true;
    }
    return false;
}

const ompl::base::State* ompl::base::PlannerInputStates::nextStart(void)
{
    if (pdef_ == NULL || si_ == NULL)
    {
        std::string error = "Missing space information or problem definition";
        if (planner_)
            throw Exception(planner_->getName(), error);
        else
            throw Exception(error);
    }

    while (addedStartStates_ < pdef_->getStartStateCount())
    {
        const base::State *st = pdef_->getStartState(addedStartStates_);
        addedStartStates_++;
        bool bounds = si_->satisfiesBounds(st);
        bool valid = bounds ? si_->isValid(st) : false;
        if (bounds && valid)
            return st;
        else
        {
            OMPL_WARN("%s: Skipping invalid start state (invalid %s)",
                      planner_ ? planner_->getName().c_str() : "PlannerInputStates",
                      bounds ? "state": "bounds");
            std::stringstream ss;
            si_->printState(st, ss);
            OMPL_DEBUG("%s: Discarded start state %s",
                       planner_ ? planner_->getName().c_str() : "PlannerInputStates",
                       ss.str().c_str());
        }
    }
    return NULL;
}

const ompl::base::State* ompl::base::PlannerInputStates::nextGoal(void)
{
    static PlannerTerminationCondition ptc = plannerAlwaysTerminatingCondition();
    return nextGoal(ptc);
}

const ompl::base::State* ompl::base::PlannerInputStates::nextGoal(const PlannerTerminationCondition &ptc)
{
    if (pdef_ == NULL || si_ == NULL)
    {
        std::string error = "Missing space information or problem definition";
        if (planner_)
            throw Exception(planner_->getName(), error);
        else
            throw Exception(error);
    }

    const GoalSampleableRegion *goal = pdef_->getGoal()->hasType(GOAL_SAMPLEABLE_REGION) ? pdef_->getGoal()->as<GoalSampleableRegion>() : NULL;

    if (goal)
    {
        time::point start_wait;
        bool first = true;
        bool attempt = true;
        while (attempt)
        {
            attempt = false;

            if (sampledGoalsCount_ < goal->maxSampleCount() && goal->canSample())
            {
                if (tempState_ == NULL)
                    tempState_ = si_->allocState();
                do
                {
                    goal->sampleGoal(tempState_);
                    sampledGoalsCount_++;
                    bool bounds = si_->satisfiesBounds(tempState_);
                    bool valid = bounds ? si_->isValid(tempState_) : false;
                    if (bounds && valid)
                    {
                        if (!first) // if we waited, show how long
                        {
                            OMPL_DEBUG("%s: Waited %lf seconds for the first goal sample.",
                                       planner_ ? planner_->getName().c_str() : "PlannerInputStates",
                                       time::seconds(time::now() - start_wait));
                        }
                        return tempState_;
                    }
                    else
                    {
                        OMPL_WARN("%s: Skipping invalid goal state (invalid %s)",
                                  planner_ ? planner_->getName().c_str() : "PlannerInputStates",
                                  bounds ? "state": "bounds");
                        std::stringstream ss;
                        si_->printState(tempState_, ss);
                        OMPL_DEBUG("%s: Discarded goal state %s",
                                   planner_ ? planner_->getName().c_str() : "PlannerInputStates",
                                   ss.str().c_str());
                    }
                }
                while (!ptc && sampledGoalsCount_ < goal->maxSampleCount() && goal->canSample());
            }
            if (goal->couldSample() && !ptc)
            {
                if (first)
                {
                    first = false;
                    start_wait = time::now();
                    OMPL_DEBUG("%s: Waiting for goal region samples ...",
                               planner_ ? planner_->getName().c_str() : "PlannerInputStates");
                }
                boost::this_thread::sleep(time::seconds(0.01));
                attempt = !ptc;
            }
        }
    }

    return NULL;
}

bool ompl::base::PlannerInputStates::haveMoreStartStates(void) const
{
    if (pdef_)
        return addedStartStates_ < pdef_->getStartStateCount();
    return false;
}

bool ompl::base::PlannerInputStates::haveMoreGoalStates(void) const
{
    if (pdef_ && pdef_->getGoal())
        if (pdef_->getGoal()->hasType(GOAL_SAMPLEABLE_REGION))
            return sampledGoalsCount_ < pdef_->getGoal()->as<GoalSampleableRegion>()->maxSampleCount();
    return false;
}
