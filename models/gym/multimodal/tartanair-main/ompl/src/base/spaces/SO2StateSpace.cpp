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

#include "ompl/base/spaces/SO2StateSpace.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include "ompl/tools/config/MagicConstants.h"
#include <boost/math/constants/constants.hpp>

// Define for boost version < 1.47
#ifndef BOOST_ASSERT_MSG
#define BOOST_ASSERT_MSG(expr, msg) assert(expr)
#endif

void ompl::base::SO2StateSpace::StateType::setValue(double unnormalized_value) {
  double v = fmod(unnormalized_value, 2.0 * boost::math::constants::pi<double>());
  if (v <= -boost::math::constants::pi<double>())
    v += 2.0 * boost::math::constants::pi<double>();
  else
    if (v > boost::math::constants::pi<double>())
      v -= 2.0 * boost::math::constants::pi<double>();
  value = v;
}

void ompl::base::SO2StateSampler::sampleUniform(State *state)
{
    state->as<SO2StateSpace::StateType>()->value =
        rng_.uniformReal(-boost::math::constants::pi<double>(), boost::math::constants::pi<double>());
}

void ompl::base::SO2StateSampler::sampleUniformNear(State *state, const State *near, const double distance)
{
    state->as<SO2StateSpace::StateType>()->value = rng_.uniformReal(near->as<SO2StateSpace::StateType>()->value - distance,
                                                                       near->as<SO2StateSpace::StateType>()->value + distance);
    space_->enforceBounds(state);
}

void ompl::base::SO2StateSampler::sampleGaussian(State *state, const State *mean, const double stdDev)
{
    state->as<SO2StateSpace::StateType>()->value = rng_.gaussian(mean->as<SO2StateSpace::StateType>()->value, stdDev);
    space_->enforceBounds(state);
}

unsigned int ompl::base::SO2StateSpace::getDimension(void) const
{
    return 1;
}

double ompl::base::SO2StateSpace::getMaximumExtent(void) const
{
    return boost::math::constants::pi<double>();
}

double ompl::base::SO2StateSpace::getMeasure() const
{
    return 2.0 * boost::math::constants::pi<double>();
}

void ompl::base::SO2StateSpace::enforceBounds(State *state) const
{
    double v = fmod(state->as<StateType>()->value, 2.0 * boost::math::constants::pi<double>());
    if (v <= -boost::math::constants::pi<double>())
        v += 2.0 * boost::math::constants::pi<double>();
    else
        if (v > boost::math::constants::pi<double>())
            v -= 2.0 * boost::math::constants::pi<double>();
    state->as<StateType>()->value = v;
}

bool ompl::base::SO2StateSpace::satisfiesBounds(const State *state) const
{
    return (state->as<StateType>()->value <= boost::math::constants::pi<double>()) &&
           (state->as<StateType>()->value > -boost::math::constants::pi<double>());
}

void ompl::base::SO2StateSpace::copyState(State *destination, const State *source) const
{
    destination->as<StateType>()->value = source->as<StateType>()->value;
}

unsigned int ompl::base::SO2StateSpace::getSerializationLength(void) const
{
    return sizeof(double);
}

void ompl::base::SO2StateSpace::serialize(void *serialization, const State *state) const
{
    memcpy(serialization, &state->as<StateType>()->value, sizeof(double));
}

void ompl::base::SO2StateSpace::deserialize(State *state, const void *serialization) const
{
    memcpy(&state->as<StateType>()->value, serialization, sizeof(double));
}

double ompl::base::SO2StateSpace::distance(const State *state1, const State *state2) const
{
    // assuming the states 1 & 2 are within bounds
    double d = fabs(state1->as<StateType>()->value - state2->as<StateType>()->value);
    BOOST_ASSERT_MSG(satisfiesBounds(state1) && satisfiesBounds(state2),
        "The states passed to SO2StateSpace::distance are not within bounds. Call "
        "SO2StateSpace::enforceBounds() in, e.g., ompl::control::ODESolver::PostPropagationEvent, "
        "ompl::control::StatePropagator, or ompl::base::StateValidityChecker");
    return (d > boost::math::constants::pi<double>()) ? 2.0 * boost::math::constants::pi<double>() - d : d;
}

bool ompl::base::SO2StateSpace::equalStates(const State *state1, const State *state2) const
{
    return fabs(state1->as<StateType>()->value - state2->as<StateType>()->value) < std::numeric_limits<double>::epsilon() * 2.0;
}

void ompl::base::SO2StateSpace::interpolate(const State *from, const State *to, const double t, State *state) const
{
    double diff = to->as<StateType>()->value - from->as<StateType>()->value;
    if (fabs(diff) <= boost::math::constants::pi<double>())
        state->as<StateType>()->value = from->as<StateType>()->value + diff * t;
    else
    {
        double &v = state->as<StateType>()->value;
        if (diff > 0.0)
            diff = 2.0 * boost::math::constants::pi<double>() - diff;
        else
            diff = -2.0 * boost::math::constants::pi<double>() - diff;
        v = from->as<StateType>()->value - diff * t;
        // input states are within bounds, so the following check is sufficient
        if (v > boost::math::constants::pi<double>())
            v -= 2.0 * boost::math::constants::pi<double>();
        else
            if (v < -boost::math::constants::pi<double>())
                v += 2.0 * boost::math::constants::pi<double>();
    }
}

ompl::base::StateSamplerPtr ompl::base::SO2StateSpace::allocDefaultStateSampler(void) const
{
    return StateSamplerPtr(new SO2StateSampler(this));
}

ompl::base::State* ompl::base::SO2StateSpace::allocState(void) const
{
    return new StateType();
}

void ompl::base::SO2StateSpace::freeState(State *state) const
{
    delete static_cast<StateType*>(state);
}

void ompl::base::SO2StateSpace::registerProjections(void)
{
    class SO2DefaultProjection : public ProjectionEvaluator
    {
    public:

        SO2DefaultProjection(const StateSpace *space) : ProjectionEvaluator(space)
        {
        }

        virtual unsigned int getDimension(void) const
        {
            return 1;
        }

        virtual void defaultCellSizes(void)
        {
            cellSizes_.resize(1);
            cellSizes_[0] = boost::math::constants::pi<double>() / magic::PROJECTION_DIMENSION_SPLITS;
            bounds_.resize(1);
            bounds_.low[0] = -boost::math::constants::pi<double>();
            bounds_.high[0] = boost::math::constants::pi<double>();
        }

        virtual void project(const State *state, EuclideanProjection &projection) const
        {
            projection(0) = state->as<SO2StateSpace::StateType>()->value;
        }
    };

    registerDefaultProjection(ProjectionEvaluatorPtr(dynamic_cast<ProjectionEvaluator*>(new SO2DefaultProjection(this))));
}

double* ompl::base::SO2StateSpace::getValueAddressAtIndex(State *state, const unsigned int index) const
{
    return index == 0 ? &(state->as<StateType>()->value) : NULL;
}

void ompl::base::SO2StateSpace::printState(const State *state, std::ostream &out) const
{
    out << "SO2State [";
    if (state)
        out << state->as<StateType>()->value;
    else
        out << "NULL";
    out << ']' << std::endl;
}

void ompl::base::SO2StateSpace::printSettings(std::ostream &out) const
{
    out << "SO2 state space '" << getName() << "'" << std::endl;
}
