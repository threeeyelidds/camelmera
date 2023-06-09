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

/* Author: Luis G. Torres */

#ifndef OMPL_BASE_OBJECTIVES_PATH_LENGTH_OPTIMIZATION_OBJECTIVE_
#define OMPL_BASE_OBJECTIVES_PATH_LENGTH_OPTIMIZATION_OBJECTIVE_

#include "ompl/base/OptimizationObjective.h"
#include "ompl/base/Path.h"

namespace ompl
{
    namespace base
    {
        /** \brief An optimization objective which corresponds to optimizing path length. */
        class PathLengthOptimizationObjective : public OptimizationObjective
        {
        public:
            PathLengthOptimizationObjective(const SpaceInformationPtr &si);

            /** \brief Motion cost for this objective is defined as
                the configuration space distance between \e s1 and \e
                s2, using the method SpaceInformation::distance(). */
            virtual Cost motionCost(const State *s1, const State *s2) const;

            /** \brief the motion cost heuristic for this objective is
                simply the configuration space distance between \e s1
                and \e s2, since this is the optimal cost between any
                two states assuming no obstacles. */
            virtual Cost motionCostHeuristic(const State *s1, const State *s2) const;

            /** \brief Get the cost that corresponds to an entire path. This implementation assumes \e Path is of type \e PathGeometric.*/
            virtual Cost pathCost(const Path *path) const;

          /** \brief Allocate a state sampler for the path-length objective (i.e., direct ellipsoidal sampling). */
            virtual InformedStateSamplerPtr allocInformedStateSampler(const StateSpace* space, const ProblemDefinitionPtr probDefn, const Cost* bestCost) const;
  
        };
    }
}

#endif
